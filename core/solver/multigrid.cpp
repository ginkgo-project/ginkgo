/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#include "core/components/fill_array.hpp"
#include "core/solver/ir_kernels.hpp"
#include "core/solver/multigrid_kernels.hpp"


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
 * run uses template to go through the list and select the valid
 * tempalate and run it.
 *
 * @tparam Base  the Base class with one template
 * @tparam T  the object type
 * @tparam func  the validation
 * @tparam ...Args  the variadic arguments.
 */
template <template <typename> class Base, typename T, typename func,
          typename... Args>
void run(T obj, func, Args... args)
{
    GKO_NOT_IMPLEMENTED;
}

/**
 * run uses template to go through the list and select the valid
 * tempalate and run it.
 *
 * @tparam Base  the Base class with one template
 * @tparam K  the template type
 * @tparam ...Types  other types in the list.
 * @tparam T  the object type
 * @tparam func  the validation
 * @tparam ...Args  the variadic arguments.
 */
template <template <typename> class Base, typename K, typename... Types,
          typename T, typename func, typename... Args>
void run(T obj, func f, Args... args)
{
    if (auto dobj = std::dynamic_pointer_cast<const Base<K>>(obj)) {
        f(dobj, args...);
    } else {
        run<Base, Types...>(obj, f, args...);
    }
}


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
casting(const T &x)
{
    return static_cast<ValueType>(x);
}

/**
 * @copydoc casting(const T&)
 */
template <typename ValueType, typename T>
std::enable_if_t<!is_complex_s<ValueType>::value && is_complex_s<T>::value,
                 ValueType>
casting(const T &x)
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
    size_type index, std::shared_ptr<const LinOp> &matrix,
    std::vector<std::shared_ptr<const LinOpFactory>> &smoother_list,
    std::vector<std::shared_ptr<const LinOp>> &smoother, size_type iteration,
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


struct MultigridState {
    MultigridState(const LinOp *system_matrix_in, const Multigrid *multigrid_in,
                   const size_type nrhs_in)
        : system_matrix(system_matrix_in),
          multigrid(multigrid_in),
          nrhs(nrhs_in)
    {
        auto current_nrows = system_matrix->get_size()[0];
        auto mg_level_list = multigrid->get_mg_level_list();
        auto list_size = mg_level_list.size();
        auto cycle = multigrid->get_cycle();
        r_list.reserve(list_size);
        g_list.reserve(list_size);
        e_list.reserve(list_size);
        one_list.reserve(list_size);
        neg_one_list.reserve(list_size);
        if (cycle == multigrid_cycle::kfcg || cycle == multigrid_cycle::kgcr) {
            kcycle_state.reserve_space(this);
        }
        // Allocate memory first such that repeating allocation in each iter.
        for (int i = 0; i < mg_level_list.size(); i++) {
            auto next_nrows =
                mg_level_list.at(i)->get_coarse_op()->get_size()[0];
            auto mg_level = mg_level_list.at(i);

            run<gko::multigrid::EnableMultigridLevel, float, double,
                std::complex<float>, std::complex<double>>(
                mg_level,
                [this](auto mg_level, auto i, auto cycle, auto current_nrows,
                       auto next_nrows) {
                    using value_type = typename std::decay_t<
                        detail::pointee<decltype(mg_level)>>::value_type;
                    this->allocate_memory<value_type>(i, cycle, current_nrows,
                                                      next_nrows);
                },
                i, cycle, current_nrows, next_nrows);

            current_nrows = next_nrows;
        }
    }

    template <typename VT>
    void allocate_memory(int level, multigrid_cycle cycle,
                         size_type current_nrows, size_type next_nrows)
    {
        using vec = matrix::Dense<VT>;
        using norm_vec = matrix::Dense<remove_complex<VT>>;

        auto exec =
            as<LinOp>(multigrid->get_mg_level_list().at(level))->get_executor();
        r_list.emplace_back(vec::create(exec, dim<2>{current_nrows, nrhs}));
        g_list.emplace_back(vec::create(exec, dim<2>{next_nrows, nrhs}));
        e_list.emplace_back(vec::create(exec, dim<2>{next_nrows, nrhs}));
        one_list.emplace_back(initialize<vec>({gko::one<VT>()}, exec));
        neg_one_list.emplace_back(initialize<vec>({-gko::one<VT>()}, exec));
        if ((cycle == multigrid_cycle::kfcg ||
             cycle == multigrid_cycle::kgcr) &&
            level % multigrid->get_parameters().kcycle_base == 0) {
            kcycle_state.allocate_memory<VT>(level, cycle, current_nrows,
                                             next_nrows);
        }
    }

    void run_cycle(multigrid_cycle cycle, size_type level,
                   const std::shared_ptr<const LinOp> &matrix, const LinOp *b,
                   LinOp *x, bool is_first = true, bool is_end = true)
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
                    detail::pointee<decltype(mg_level)>>::value_type;
                this->run_cycle<value_type>(cycle, level, matrix, b, x,
                                            is_first, is_end);
            });
    }

    template <typename VT>
    void run_cycle(multigrid_cycle cycle, size_type level,
                   const std::shared_ptr<const LinOp> &matrix, const LinOp *b,
                   LinOp *x, bool is_first, bool is_end)
    {
        auto total_level = multigrid->get_mg_level_list().size();

        auto r = as_vec<VT>(r_list.at(level));
        auto g = as_vec<VT>(g_list.at(level));
        auto e = as_vec<VT>(e_list.at(level));
        // get mg_level
        auto mg_level = multigrid->get_mg_level_list().at(level);
        // get the pre_smoother
        auto pre_smoother = multigrid->get_pre_smoother_list().at(level);
        // get the mid_smoother
        std::shared_ptr<const LinOp> mid_smoother{nullptr};
        auto mid_case = multigrid->get_parameters().mid_case;
        if (mid_case == multigrid_mid_smooth_type::standalone) {
            mid_smoother = multigrid->get_mid_smoother_list().at(level);
        }
        // get the post_smoother
        auto post_smoother = multigrid->get_post_smoother_list().at(level);
        auto one = one_list.at(level).get();
        auto neg_one = neg_one_list.at(level).get();
        // origin or next or first
        bool use_pre = is_first ||
                       mid_case == multigrid_mid_smooth_type::origin ||
                       mid_case == multigrid_mid_smooth_type::next;
        if (use_pre && pre_smoother) {
            pre_smoother->apply(b, x);
            // compute residual
            r->copy_from(b);  // n * b
            matrix->apply(neg_one, x, one, r.get());
        } else if (level != 0) {
            // move the residual computation at level 0 to out-of-cycle if there
            // is no pre-smoother at level 0
            r->copy_from(b);
            matrix->apply(neg_one, x, one, r.get());
        }
        // first cycle
        mg_level->get_restrict_op()->apply(r.get(), g.get());
        // next level
        e->fill(zero<VT>());
        this->run_cycle(cycle, level + 1, mg_level->get_coarse_op(), g.get(),
                        e.get(), true, cycle == multigrid_cycle::v);
        if (level < multigrid->get_mg_level_list().size() - 1) {
            // additional work for non-v_cycle
            // next level
            if (cycle == multigrid_cycle::f) {
                // f_cycle call v_cycle in the second cycle
                this->run_cycle(multigrid_cycle::v, level + 1,
                                mg_level->get_coarse_op(), g.get(), e.get(),
                                false, true);
            } else if (cycle == multigrid_cycle::w) {
                this->run_cycle(cycle, level + 1, mg_level->get_coarse_op(),
                                g.get(), e.get(), false, true);
            } else if ((cycle == multigrid_cycle::kfcg ||
                        cycle == multigrid_cycle::kgcr) &&
                       level % multigrid->get_parameters().kcycle_base == 0) {
                kcycle_state.kstep<VT>(cycle, level, g, e);
            }
        }
        // prolong
        mg_level->get_prolong_op()->apply(one, e.get(), one, x);

        // end or origin previous
        bool use_post = is_end ||
                        mid_case == multigrid_mid_smooth_type::origin ||
                        mid_case == multigrid_mid_smooth_type::previous;
        // post-smooth
        if (use_post && post_smoother) {
            post_smoother->apply(b, x);
        }

        // put the mid smoother into the end of previous cycle
        // only W/F cycle
        bool use_mid =
            (cycle == multigrid_cycle::w || cycle == multigrid_cycle::f) &&
            !is_end && mid_case == multigrid_mid_smooth_type::standalone;
        if (use_mid && mid_smoother) {
            mid_smoother->apply(b, x);
        }
    }

    struct KCycleMultiGridState {
        void reserve_space(MultigridState *mg_state_in)
        {
            mg_state = mg_state_in;
            auto k_num = (mg_state->multigrid->get_mg_level_list().size() - 1) /
                         mg_state->multigrid->get_parameters().kcycle_base;
            alpha_list.reserve(k_num);
            beta_list.reserve(k_num);
            gamma_list.reserve(k_num);
            rho_list.reserve(k_num);
            zeta_list.reserve(k_num);
            v_list.reserve(k_num);
            w_list.reserve(k_num);
            d_list.reserve(k_num);
            old_norm_list.reserve(k_num);
            new_norm_list.reserve(k_num);
        }

        template <typename VT>
        void allocate_memory(int level, multigrid_cycle cycle,
                             size_type current_nrows, size_type next_nrows)
        {
            using vec = matrix::Dense<VT>;
            using norm_vec = matrix::Dense<remove_complex<VT>>;
            auto scalar_size = dim<2>{1, mg_state->nrhs};
            auto vector_size = dim<2>{next_nrows, mg_state->nrhs};
            auto exec =
                as<LinOp>(mg_state->multigrid->get_mg_level_list().at(level))
                    ->get_executor();
            // 1 x nrhs
            alpha_list.emplace_back(vec::create(exec, scalar_size));
            beta_list.emplace_back(vec::create(exec, scalar_size));
            gamma_list.emplace_back(vec::create(exec, scalar_size));
            rho_list.emplace_back(vec::create(exec, scalar_size));
            zeta_list.emplace_back(vec::create(exec, scalar_size));
            // next level's nrows x nrhs
            v_list.emplace_back(vec::create(exec, vector_size));
            w_list.emplace_back(vec::create(exec, vector_size));
            d_list.emplace_back(vec::create(exec, vector_size));
            // 1 x nrhs norm_vec
            old_norm_list.emplace_back(norm_vec::create(exec, scalar_size));
            new_norm_list.emplace_back(norm_vec::create(exec, scalar_size));
        }

        template <typename VT>
        void kstep(multigrid_cycle cycle, size_type level,
                   const std::shared_ptr<matrix::Dense<VT>> &g,
                   const std::shared_ptr<matrix::Dense<VT>> &e)
        {
            auto mg_level = mg_state->multigrid->get_mg_level_list().at(level);
            auto exec = as<LinOp>(mg_level)->get_executor();
            // otherwise, use v_cycle
            // do some work in coarse level - do not need prolong
            bool is_fcg = cycle == multigrid_cycle::kfcg;
            auto k_idx =
                level / mg_state->multigrid->get_parameters().kcycle_base;
            auto alpha = as_vec<VT>(alpha_list.at(k_idx));
            auto beta = as_vec<VT>(beta_list.at(k_idx));
            auto gamma = as_vec<VT>(gamma_list.at(k_idx));
            auto rho = as_vec<VT>(rho_list.at(k_idx));
            auto zeta = as_vec<VT>(zeta_list.at(k_idx));
            auto v = as_vec<VT>(v_list.at(k_idx));
            auto w = as_vec<VT>(w_list.at(k_idx));
            auto d = as_vec<VT>(d_list.at(k_idx));
            auto old_norm = as_real_vec<VT>(old_norm_list.at(k_idx));
            auto new_norm = as_real_vec<VT>(new_norm_list.at(k_idx));
            auto coarse = mg_level->get_coarse_op();
            auto rel_tol_val = static_cast<remove_complex<VT>>(
                mg_state->multigrid->get_parameters().kcycle_rel_tol);

            // first iteration
            coarse->apply(e.get(), v.get());
            std::shared_ptr<const matrix::Dense<VT>> t = is_fcg ? e : v;
            t->compute_dot(v.get(), rho.get());
            t->compute_dot(g.get(), alpha.get());

            if (!std::isnan(rel_tol_val) && rel_tol_val >= 0) {
                // calculate the r norm
                g->compute_norm2(old_norm.get());
            }
            // kcycle_step_1 update g, d
            // temp = alpha/rho
            // g = g - temp * v
            // d = e = temp * e
            exec->run(multigrid::make_kcycle_step_1(
                alpha.get(), rho.get(), v.get(), g.get(), d.get(), e.get()));
            // check ||new_r|| <= t * ||old_r|| only when t >= 0 && t != nan
            bool is_stop = true;

            if (!std::isnan(rel_tol_val) && rel_tol_val >= 0) {
                // calculate the updated r norm
                g->compute_norm2(new_norm.get());
                // is_stop = true when all new_norm <= t * old_norm.
                exec->run(multigrid::make_kcycle_check_stop(
                    old_norm.get(), new_norm.get(), rel_tol_val, is_stop));
            }
            // rel_tol < 0: run two iteraion
            // rel_tol is nan: run one iteraions
            // others: new_norm <= rel_tol * old_norm -> run second
            // iteraion.
            if (rel_tol_val < 0 || (rel_tol_val >= 0 && !is_stop)) {
                // second iteration
                // Apply on d for keeping the answer on e
                mg_state->run_cycle(cycle, level + 1, mg_level->get_coarse_op(),
                                    g.get(), d.get(), false, true);
                coarse->apply(d.get(), w.get());
                t = is_fcg ? d : w;
                t->compute_dot(v.get(), gamma.get());
                t->compute_dot(w.get(), beta.get());
                t->compute_dot(g.get(), zeta.get());
                // kcycle_step_2 update e
                // scalar_d = zeta/(beta - gamma^2/rho)
                // scalar_e = 1 - gamma/alpha*scalar_d
                // e = scalar_e * e + scalar_d * d
                exec->run(multigrid::make_kcycle_step_2(
                    alpha.get(), rho.get(), gamma.get(), beta.get(), zeta.get(),
                    d.get(), e.get()));
            }
        }
        MultigridState *mg_state;
        // 1 x nrhs
        std::vector<std::shared_ptr<LinOp>> alpha_list;
        std::vector<std::shared_ptr<LinOp>> beta_list;
        std::vector<std::shared_ptr<LinOp>> gamma_list;
        std::vector<std::shared_ptr<LinOp>> rho_list;
        std::vector<std::shared_ptr<LinOp>> zeta_list;
        std::vector<std::shared_ptr<LinOp>> old_norm_list;
        std::vector<std::shared_ptr<LinOp>> new_norm_list;
        // next level's nrows x nrhs
        std::vector<std::shared_ptr<LinOp>> v_list;
        std::vector<std::shared_ptr<LinOp>> w_list;
        std::vector<std::shared_ptr<LinOp>> d_list;
    };

    // current level's nrows x nrhs
    std::vector<std::shared_ptr<LinOp>> r_list;
    // next level's nrows x nrhs
    std::vector<std::shared_ptr<LinOp>> g_list;
    std::vector<std::shared_ptr<LinOp>> e_list;
    // constant 1 x 1
    std::vector<std::shared_ptr<const LinOp>> one_list;
    std::vector<std::shared_ptr<const LinOp>> neg_one_list;
    const LinOp *system_matrix;
    const Multigrid *multigrid;
    KCycleMultiGridState kcycle_state;
    size_type nrhs;
};


}  // namespace


void Multigrid::generate()
{
    // generate coarse matrix until reaching max_level or min_coarse_rows
    auto num_rows = system_matrix_->get_size()[0];
    size_type level = 0;
    auto matrix = system_matrix_;
    auto exec = this->get_executor();
    // Always generate smoother with size = level.
    while (level < parameters_.max_levels &&
           num_rows > parameters_.min_coarse_rows) {
        auto index = mg_level_index_(level, lend(matrix));
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
                    detail::pointee<decltype(mg_level)>>::value_type;
                handle_list<value_type>(
                    index, matrix, parameters_.pre_smoother, pre_smoother_list_,
                    parameters_.smoother_iters, parameters_.smoother_relax);
                if (parameters_.mid_case ==
                    multigrid_mid_smooth_type::standalone) {
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
            index, matrix);

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
                detail::pointee<decltype(mg_level)>>::value_type;
            auto exec = this->get_executor();
            if (parameters_.coarsest_solver.size() == 0) {
                coarsest_solver_ = matrix::Identity<value_type>::create(
                    exec, matrix->get_size()[0]);
            } else {
                auto temp_index = solver_index_(level, lend(matrix));
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


void Multigrid::apply_impl(const LinOp *b, LinOp *x) const
{
    auto lambda = [this](auto mg_level, auto b, auto x) {
        using value_type = typename std::decay_t<
            detail::pointee<decltype(mg_level)>>::value_type;
        auto exec = this->get_executor();
        auto neg_one_op =
            initialize<matrix::Dense<value_type>>({-one<value_type>()}, exec);
        auto one_op =
            initialize<matrix::Dense<value_type>>({one<value_type>()}, exec);
        constexpr uint8 RelativeStoppingId{1};
        Array<stopping_status> stop_status(exec, b->get_size()[1]);
        bool one_changed{};
        auto state =
            MultigridState(system_matrix_.get(), this, b->get_size()[1]);
        exec->run(multigrid::make_initialize(&stop_status));
        // compute the residual at the r_list(0);
        auto r = state.r_list.at(0);
        r->copy_from(b);
        system_matrix_->apply(lend(neg_one_op), x, lend(one_op), r.get());
        auto stop_criterion = stop_criterion_factory_->generate(
            system_matrix_,
            std::shared_ptr<const LinOp>(b, null_deleter<const LinOp>{}), x,
            r.get());
        int iter = -1;
        while (true) {
            ++iter;
            this->template log<log::Logger::iteration_complete>(this, iter,
                                                                r.get(), x);
            if (stop_criterion->update()
                    .num_iterations(iter)
                    .residual(r.get())
                    .solution(x)
                    .check(RelativeStoppingId, true, &stop_status,
                           &one_changed)) {
                break;
            }

            state.run_cycle(this->get_parameters().cycle, 0, system_matrix_, b,
                            x);
            r->copy_from(b);
            system_matrix_->apply(lend(neg_one_op), x, lend(one_op), r.get());
        }
    };

    auto first_mg_level = this->get_mg_level_list().front();

    run<gko::multigrid::EnableMultigridLevel, float, double,
        std::complex<float>, std::complex<double>>(first_mg_level, lambda, b,
                                                   x);
}


void Multigrid::apply_impl(const LinOp *alpha, const LinOp *b,
                           const LinOp *beta, LinOp *x) const
{
    auto lambda = [this](auto mg_level, auto alpha, auto b, auto beta, auto x) {
        using value_type = typename std::decay_t<
            detail::pointee<decltype(mg_level)>>::value_type;
        auto dense_x = as<matrix::Dense<value_type>>(x);
        auto x_clone = dense_x->clone();
        this->apply(b, x_clone.get());
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, x_clone.get());
    };
    auto first_mg_level = this->get_mg_level_list().front();

    run<gko::multigrid::EnableMultigridLevel, float, double,
        std::complex<float>, std::complex<double>>(first_mg_level, lambda,
                                                   alpha, b, beta, x);
}


}  // namespace solver
}  // namespace gko
