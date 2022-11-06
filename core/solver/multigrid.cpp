/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/solver/multigrid.hpp>


#include <complex>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/base/dispatch_helper.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/solver/ir_kernels.hpp"
#include "core/solver/multigrid_kernels.hpp"
#include "core/solver/solver_base.hpp"


namespace gko {
namespace solver {
namespace multigrid {


GKO_REGISTER_OPERATION(initialize, ir::initialize);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(kcycle_step_1, multigrid::kcycle_step_1);
GKO_REGISTER_OPERATION(kcycle_step_2, multigrid::kcycle_step_2);
GKO_REGISTER_OPERATION(kcycle_check_stop, multigrid::kcycle_check_stop);


}  // namespace multigrid


namespace {


/**
 * casting does the casting type or take the real part of complex if the input
 * is complex but require real output.
 *
 * @tparam ValueType  the output type
 * @tparam T  the input type
 *
 * @param x  input
 *
 * @return the ValueType value
 */
template <typename ValueType, typename T>
std::enable_if_t<is_complex_s<ValueType>::value == is_complex_s<T>::value,
                 ValueType>
casting(const T& x)
{
    return static_cast<ValueType>(x);
}

/**
 * @copydoc casting(const T&)
 */
template <typename ValueType, typename T>
std::enable_if_t<!is_complex_s<ValueType>::value && is_complex_s<T>::value,
                 ValueType>
casting(const T& x)
{
    return static_cast<ValueType>(real(x));
}

/**
 * as_vec gives a shortcut for casting pointer to dense.
 */
template <typename ValueType>
auto as_vec(std::shared_ptr<LinOp> x)
{
    return std::static_pointer_cast<matrix::Dense<ValueType>>(x);
}


/**
 * as_real_vec gives a shortcut for casting pointer to dense with real type.
 */
template <typename ValueType>
auto as_real_vec(std::shared_ptr<LinOp> x)
{
    return std::static_pointer_cast<matrix::Dense<remove_complex<ValueType>>>(
        x);
}


/**
 * handle_list generate the smoother for each MultigridLevel
 *
 * @tparam ValueType  the type of MultigridLevel
 */
template <typename ValueType>
void handle_list(
    size_type index, std::shared_ptr<const LinOp>& matrix,
    std::vector<std::shared_ptr<const LinOpFactory>>& smoother_list,
    std::vector<std::shared_ptr<const LinOp>>& smoother, size_type iteration,
    std::complex<double> relaxation_factor)
{
    auto list_size = smoother_list.size();
    if (list_size != 0) {
        auto temp_index = list_size == 1 ? 0 : index;
        GKO_ENSURE_IN_BOUNDS(temp_index, list_size);
        auto item = smoother_list.at(temp_index);
        if (item == nullptr) {
            smoother.emplace_back(nullptr);
        } else {
            auto solver = item->generate(matrix);
            if (solver->apply_uses_initial_guess() == true) {
                smoother.emplace_back(give(solver));
            } else {
                auto ir = build_smoother<ValueType>(
                    give(solver), iteration,
                    casting<ValueType>(relaxation_factor));
                smoother.emplace_back(give(ir->generate(matrix)));
            }
        }
    } else {
        smoother.emplace_back(nullptr);
    }
}


template <typename Vec>
void clear_and_reserve(Vec& vec, size_type size)
{
    vec.clear();
    vec.reserve(size);
}


}  // namespace


namespace multigrid {
namespace experimental {


void MultigridState::generate(const LinOp* system_matrix_in,
                              const gko::solver::Multigrid* multigrid_in,
                              const size_type nrhs_in)
{
    system_matrix = system_matrix_in;
    multigrid = multigrid_in;
    nrhs = nrhs_in;
    auto current_nrows = system_matrix->get_size()[0];
    auto mg_level_list = multigrid->get_mg_level_list();
    auto list_size = mg_level_list.size();
    auto cycle = multigrid->get_cycle();
    clear_and_reserve(r_list, list_size);
    clear_and_reserve(g_list, list_size);
    clear_and_reserve(e_list, list_size);
    clear_and_reserve(one_list, list_size);
    clear_and_reserve(next_one_list, list_size);
    clear_and_reserve(neg_one_list, list_size);
    // Allocate memory first such that reusing allocation in each iter.
    for (int i = 0; i < mg_level_list.size(); i++) {
        auto next_nrows = mg_level_list.at(i)->get_coarse_op()->get_size()[0];
        auto mg_level = mg_level_list.at(i);

        run<gko::multigrid::EnableMultigridLevel, float, double,
            std::complex<float>, std::complex<double>>(
            mg_level,
            [&, this](auto mg_level, auto i, auto cycle, auto current_nrows,
                      auto next_nrows) {
                using value_type = typename std::decay_t<
                    gko::detail::pointee<decltype(mg_level)>>::value_type;
                using vec = matrix::Dense<value_type>;
                this->allocate_memory<value_type>(i, cycle, current_nrows,
                                                  next_nrows);
                auto exec = as<LinOp>(multigrid->get_mg_level_list().at(i))
                                ->get_executor();
            },
            i, cycle, current_nrows, next_nrows);

        current_nrows = next_nrows;
    }
}

template <typename VT>
void MultigridState::allocate_memory(int level, multigrid::cycle cycle,
                                     size_type current_nrows,
                                     size_type next_nrows)
{
    using vec = matrix::Dense<VT>;
    using norm_vec = matrix::Dense<remove_complex<VT>>;

    auto exec =
        as<LinOp>(multigrid->get_mg_level_list().at(level))->get_executor();
    r_list.emplace_back(vec::create(exec, dim<2>{current_nrows, nrhs}));
    if (level != 0) {
        // allocate the previous level
        g_list.emplace_back(vec::create(exec, dim<2>{current_nrows, nrhs}));
        e_list.emplace_back(vec::create(exec, dim<2>{current_nrows, nrhs}));
        next_one_list.emplace_back(initialize<vec>({gko::one<VT>()}, exec));
    }
    if (level + 1 == multigrid->get_mg_level_list().size()) {
        // the last level allocate the g, e for coarsest solver
        g_list.emplace_back(vec::create(exec, dim<2>{next_nrows, nrhs}));
        e_list.emplace_back(vec::create(exec, dim<2>{next_nrows, nrhs}));
        next_one_list.emplace_back(initialize<vec>({gko::one<VT>()}, exec));
    }
    one_list.emplace_back(initialize<vec>({gko::one<VT>()}, exec));
    neg_one_list.emplace_back(initialize<vec>({-gko::one<VT>()}, exec));
}

void MultigridState::run_cycle(multigrid::cycle cycle, size_type level,
                               const std::shared_ptr<const LinOp>& matrix,
                               const LinOp* b, LinOp* x, bool x_is_zero,
                               bool is_first, bool is_end)
{
    if (level == multigrid->get_mg_level_list().size()) {
        multigrid->get_coarsest_solver()->apply(b, x);
        return;
    }
    auto mg_level = multigrid->get_mg_level_list().at(level);
    run<gko::multigrid::EnableMultigridLevel, float, double,
        std::complex<float>, std::complex<double>>(
        mg_level, [&, this](auto mg_level) {
            using value_type = typename std::decay_t<
                gko::detail::pointee<decltype(mg_level)>>::value_type;
            this->run_cycle<value_type>(cycle, level, matrix, b, x, x_is_zero,
                                        is_first, is_end);
        });
}

template <typename VT>
void MultigridState::run_cycle(multigrid::cycle cycle, size_type level,
                               const std::shared_ptr<const LinOp>& matrix,
                               const LinOp* b, LinOp* x, bool x_is_zero,
                               bool is_first, bool is_end)
{
    auto total_level = multigrid->get_mg_level_list().size();

    auto r = r_list.at(level);
    auto g = g_list.at(level);
    auto e = e_list.at(level);
    LinOp* x_ptr = x;
    const LinOp* b_ptr = b;
    // get mg_level
    auto mg_level = multigrid->get_mg_level_list().at(level);
    // get the pre_smoother
    auto pre_smoother = multigrid->get_pre_smoother_list().at(level);
    // get the mid_smoother
    std::shared_ptr<const LinOp> mid_smoother{nullptr};
    auto mid_case = multigrid->get_parameters().mid_case;
    if (mid_case == multigrid::mid_smooth_type::standalone) {
        mid_smoother = multigrid->get_mid_smoother_list().at(level);
    }
    // get the post_smoother
    auto post_smoother = multigrid->get_post_smoother_list().at(level);
    auto one = one_list.at(level).get();
    auto next_one = next_one_list.at(level).get();
    auto neg_one = neg_one_list.at(level).get();
    // origin or next or first
    bool use_pre = is_first || mid_case == multigrid::mid_smooth_type::both ||
                   mid_case == multigrid::mid_smooth_type::pre_smoother;
    if (use_pre && pre_smoother) {
        if (x_is_zero) {
            // when level is zero, the x_ptr is already filled by zero
            if (level != 0) {
                dynamic_cast<matrix::Dense<VT>*>(x_ptr)->fill(zero<VT>());
            }
            if (auto pre_allow_zero_input =
                    std::dynamic_pointer_cast<const ApplyWithInitialGuess>(
                        pre_smoother)) {
                pre_allow_zero_input->apply_with_initial_guess(
                    b_ptr, x_ptr, initial_guess_mode::zero);
            } else {
                pre_smoother->apply(b_ptr, x_ptr);
            }
        } else {
            pre_smoother->apply(b_ptr, x_ptr);
        }
        // split the check
        // Thus, when the IR only contains iter limit, there's no additional
        // residual computation
        r->copy_from(b);  // n * b
        matrix->apply(neg_one, x_ptr, one, r.get());
    } else if (level != 0) {
        // move the residual computation at level 0 to out-of-cycle if there
        // is no pre-smoother at level 0
        r->copy_from(b);
        matrix->apply(neg_one, x_ptr, one, r.get());
    }
    // first cycle
    mg_level->get_restrict_op()->apply(r.get(), g.get());
    // next level
    if (level + 1 == total_level) {
        // the coarsest solver use the last level valuetype
        as_vec<VT>(e)->fill(zero<VT>());
    }
    auto next_level_matrix =
        (level + 1 < total_level)
            ? multigrid->get_mg_level_list().at(level + 1)->get_fine_op()
            : mg_level->get_coarse_op();
    this->run_cycle(cycle, level + 1, next_level_matrix, g.get(), e.get(), true,
                    true, cycle == multigrid::cycle::v);
    if (level < multigrid->get_mg_level_list().size() - 1) {
        // additional work for non-v_cycle
        // next level
        if (cycle == multigrid::cycle::f) {
            // f_cycle call v_cycle in the second cycle
            this->run_cycle(multigrid::cycle::v, level + 1, next_level_matrix,
                            g.get(), e.get(), false, false, true);
        } else if (cycle == multigrid::cycle::w) {
            this->run_cycle(cycle, level + 1, next_level_matrix, g.get(),
                            e.get(), false, false, true);
        }
    }
    // prolong
    mg_level->get_prolong_op()->apply(next_one, e.get(), next_one, x_ptr);

    // end or origin previous
    bool use_post = is_end || mid_case == multigrid::mid_smooth_type::both ||
                    mid_case == multigrid::mid_smooth_type::post_smoother;
    // post-smooth
    if (use_post && post_smoother) {
        post_smoother->apply(b_ptr, x_ptr);
    }

    // put the mid smoother into the end of previous cycle
    // only W/F cycle
    bool use_mid =
        (cycle == multigrid::cycle::w || cycle == multigrid::cycle::f) &&
        !is_end && mid_case == multigrid::mid_smooth_type::standalone;
    if (use_mid && mid_smoother) {
        mid_smoother->apply(b_ptr, x_ptr);
    }
}

}  // namespace experimental
}  // namespace multigrid


void Multigrid::generate()
{
    // generate coarse matrix until reaching max_level or min_coarse_rows
    auto num_rows = this->get_system_matrix()->get_size()[0];
    size_type level = 0;
    auto matrix = this->get_system_matrix();
    auto exec = this->get_executor();
    // Always generate smoother with size = level.
    while (level < parameters_.max_levels &&
           num_rows > parameters_.min_coarse_rows) {
        auto index = level_selector_(level, lend(matrix));
        GKO_ENSURE_IN_BOUNDS(index, parameters_.mg_level.size());
        auto mg_level_factory = parameters_.mg_level.at(index);
        // coarse generate
        auto mg_level = as<gko::multigrid::MultigridLevel>(
            share(mg_level_factory->generate(matrix)));
        if (mg_level->get_coarse_op()->get_size()[0] == num_rows) {
            // do not reduce dimension
            break;
        }

        run<gko::multigrid::EnableMultigridLevel, float, double,
            std::complex<float>, std::complex<double>>(
            mg_level,
            [this](auto mg_level, auto index, auto matrix) {
                using value_type = typename std::decay_t<
                    gko::detail::pointee<decltype(mg_level)>>::value_type;
                handle_list<value_type>(
                    index, matrix, parameters_.pre_smoother, pre_smoother_list_,
                    parameters_.smoother_iters, parameters_.smoother_relax);
                if (parameters_.mid_case ==
                    multigrid::mid_smooth_type::standalone) {
                    handle_list<value_type>(
                        index, matrix, parameters_.mid_smoother,
                        mid_smoother_list_, parameters_.smoother_iters,
                        parameters_.smoother_relax);
                }
                if (!parameters_.post_uses_pre) {
                    handle_list<value_type>(
                        index, matrix, parameters_.post_smoother,
                        post_smoother_list_, parameters_.smoother_iters,
                        parameters_.smoother_relax);
                }
            },
            index, mg_level->get_fine_op());

        mg_level_list_.emplace_back(mg_level);
        matrix = mg_level_list_.back()->get_coarse_op();
        num_rows = matrix->get_size()[0];
        level++;
    }
    if (parameters_.post_uses_pre) {
        post_smoother_list_ = pre_smoother_list_;
    }
    // Generate at least one level
    GKO_ASSERT_EQ(level > 0, true);
    auto last_mg_level = mg_level_list_.back();

    // generate coarsest solver
    run<gko::multigrid::EnableMultigridLevel, float, double,
        std::complex<float>, std::complex<double>>(
        last_mg_level,
        [this](auto mg_level, auto level, auto matrix) {
            using value_type = typename std::decay_t<
                gko::detail::pointee<decltype(mg_level)>>::value_type;
            auto exec = this->get_executor();
            if (parameters_.coarsest_solver.size() == 0) {
                coarsest_solver_ = matrix::Identity<value_type>::create(
                    exec, matrix->get_size()[0]);
            } else {
                auto temp_index = solver_selector_(level, lend(matrix));
                GKO_ENSURE_IN_BOUNDS(temp_index,
                                     parameters_.coarsest_solver.size());
                auto solver = parameters_.coarsest_solver.at(temp_index);
                if (solver == nullptr) {
                    coarsest_solver_ = matrix::Identity<value_type>::create(
                        exec, matrix->get_size()[0]);
                } else {
                    coarsest_solver_ = solver->generate(matrix);
                }
            }
        },
        level, matrix);
}


void Multigrid::apply_impl(const LinOp* b, LinOp* x) const
{
    this->apply_with_initial_guess(b, x, this->get_default_initial_guess());
}


void Multigrid::apply_with_initial_guess_impl(const LinOp* b, LinOp* x,
                                              initial_guess_mode guess) const
{
    if (!this->get_system_matrix()) {
        return;
    }

    auto lambda = [this, guess](auto mg_level, auto b, auto x) {
        using value_type = typename std::decay_t<
            gko::detail::pointee<decltype(mg_level)>>::value_type;
        experimental::precision_dispatch_real_complex_distributed<value_type>(
            [this, guess](auto dense_b, auto dense_x) {
                prepare_initial_guess(dense_b, dense_x, guess);
                this->apply_dense_impl(dense_b, dense_x, guess);
            },
            b, x);
    };
    auto first_mg_level = this->get_mg_level_list().front();
    run<gko::multigrid::EnableMultigridLevel, float, double,
        std::complex<float>, std::complex<double>>(first_mg_level, lambda, b,
                                                   x);
}


void Multigrid::apply_impl(const LinOp* alpha, const LinOp* b,
                           const LinOp* beta, LinOp* x) const
{
    this->apply_with_initial_guess(alpha, b, beta, x,
                                   this->get_default_initial_guess());
}


void Multigrid::apply_with_initial_guess_impl(const LinOp* alpha,
                                              const LinOp* b, const LinOp* beta,
                                              LinOp* x,
                                              initial_guess_mode guess) const
{
    if (!this->get_system_matrix()) {
        return;
    }

    auto lambda = [this, guess](auto mg_level, auto alpha, auto b, auto beta,
                                auto x) {
        using value_type = typename std::decay_t<
            gko::detail::pointee<decltype(mg_level)>>::value_type;
        experimental::precision_dispatch_real_complex_distributed<value_type>(
            [this, guess](auto dense_alpha, auto dense_b, auto dense_beta,
                          auto dense_x) {
                prepare_initial_guess(dense_b, dense_x, guess);
                auto x_clone = dense_x->clone();
                this->apply_dense_impl(dense_b, x_clone.get(), guess);
                dense_x->scale(dense_beta);
                dense_x->add_scaled(dense_alpha, x_clone.get());
            },
            alpha, b, beta, x);
    };
    auto first_mg_level = this->get_mg_level_list().front();
    run<gko::multigrid::EnableMultigridLevel, float, double,
        std::complex<float>, std::complex<double>>(first_mg_level, lambda,
                                                   alpha, b, beta, x);
}


template <typename VectorType>
void Multigrid::apply_dense_impl(const VectorType* b, VectorType* x,
                                 initial_guess_mode guess) const
{
    using ws = workspace_traits<Multigrid>;
    this->setup_workspace();
    if (state.nrhs != b->get_size()[1]) {
        state.generate(this->get_system_matrix().get(), this, b->get_size()[1]);
    }
    auto lambda = [&, this](auto mg_level, auto b, auto x) {
        using value_type = typename std::decay_t<
            gko::detail::pointee<decltype(mg_level)>>::value_type;
        auto exec = this->get_executor();
        static auto neg_one_op =
            initialize<matrix::Dense<value_type>>({-one<value_type>()}, exec);
        static auto one_op =
            initialize<matrix::Dense<value_type>>({one<value_type>()}, exec);
        constexpr uint8 RelativeStoppingId{1};
        auto& stop_status =
            this->template create_workspace_array<stopping_status>(
                ws::stop, b->get_size()[1]);
        bool one_changed{};
        exec->run(multigrid::make_initialize(&stop_status));
        // compute the residual at the r_list(0);
        // auto r = state.r_list.at(0);
        // r->copy_from(b)
        // system_matrix->apply(lend(neg_one_op), x, lend(one_op), r.get());
        auto stop_criterion = this->get_stop_criterion_factory()->generate(
            this->get_system_matrix(),
            std::shared_ptr<const LinOp>(b, null_deleter<const LinOp>{}), x,
            nullptr);
        int iter = -1;
        while (true) {
            ++iter;
            this->template log<log::Logger::iteration_complete>(this, iter,
                                                                nullptr, x);
            if (stop_criterion->update()
                    .num_iterations(iter)
                    // .residual(r.get())
                    .solution(x)
                    .check(RelativeStoppingId, true, &stop_status,
                           &one_changed)) {
                break;
            }

            state.run_cycle(this->get_parameters().cycle, 0,
                            this->get_system_matrix(), b, x,
                            guess == initial_guess_mode::zero);
            // r->copy_from(b);
            // system_matrix->apply(lend(neg_one_op), x, lend(one_op),
            // r.get());
        }
    };

    auto first_mg_level = this->get_mg_level_list().front();

    run<gko::multigrid::EnableMultigridLevel, float, double,
        std::complex<float>, std::complex<double>>(first_mg_level, lambda, b,
                                                   x);
}


int workspace_traits<Multigrid>::num_arrays(const Solver&) { return 1; }


int workspace_traits<Multigrid>::num_vectors(const Solver&) { return 0; }


std::vector<std::string> workspace_traits<Multigrid>::op_names(const Solver&)
{
    return {};
}


std::vector<std::string> workspace_traits<Multigrid>::array_names(const Solver&)
{
    return {"stop"};
}


std::vector<int> workspace_traits<Multigrid>::scalars(const Solver&)
{
    return {};
}


std::vector<int> workspace_traits<Multigrid>::vectors(const Solver&)
{
    return {};
}


}  // namespace solver
}  // namespace gko
