/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/solver/multigrid_kernels.hpp"

namespace gko {
namespace solver {
namespace multigrid {


GKO_REGISTER_OPERATION(initialize_v, multigrid::initialize_v);


}  // namespace multigrid


template <typename ValueType>
void Multigrid<ValueType>::generate()
{
    // generate coarse matrix until reaching max_level or min_coarse_rows
    auto num_rows = system_matrix_->get_size()[0];
    size_type level = 0;
    auto matrix = system_matrix_;
    auto exec = this->get_executor();
    while (level < parameters_.max_levels &&
           num_rows > parameters_.min_coarse_rows) {
        auto index = rstr_prlg_index_(num_rows, level);
        GKO_ENSURE_IN_BOUNDS(index, parameters_.rstr_prlg.size());
        auto rstr_prlg_factory = parameters_.rstr_prlg.at(index);
        // pre_smooth_generate
        if (parameters_.pre_smoother.size() != 0) {
            auto temp_index = parameters_.pre_smoother.size() == 1 ? 0 : index;
            auto pre_smoother = parameters_.pre_smoother.at(temp_index);
            if (pre_smoother == nullptr) {
                pre_smoother_list_.emplace_back(nullptr);
            } else {
                pre_smoother_list_.emplace_back(
                    give(pre_smoother->generate(matrix)));
            }
        } else {
            pre_smoother_list_.emplace_back(nullptr);
        }
        if (parameters_.pre_relaxation.get_num_elems() != 0) {
            auto temp_index =
                parameters_.pre_relaxation.get_num_elems() == 1 ? 0 : index;
            auto data = parameters_.pre_relaxation.get_data();
            pre_relaxation_list_.emplace_back(give(vector_type::create(
                exec, gko::dim<2>{1},
                Array<ValueType>::view(exec, 1, data + temp_index), 1)));
        }
        // post_smooth_generate
        if (!parameters_.post_uses_pre) {
            if (parameters_.post_smoother.size() != 0) {
                auto temp_index =
                    parameters_.post_smoother.size() == 1 ? 0 : index;
                auto post_smoother = parameters_.post_smoother.at(temp_index);
                if (post_smoother == nullptr) {
                    post_smoother_list_.emplace_back(nullptr);
                } else {
                    post_smoother_list_.emplace_back(
                        give(post_smoother->generate(matrix)));
                }
            } else {
                post_smoother_list_.emplace_back(nullptr);
            }
            if (parameters_.post_relaxation.get_num_elems() != 0) {
                auto temp_index =
                    parameters_.post_relaxation.get_num_elems() == 1 ? 0
                                                                     : index;
                auto data = parameters_.post_relaxation.get_data();
                post_relaxation_list_.emplace_back(give(vector_type::create(
                    exec, gko::dim<2>{1},
                    Array<ValueType>::view(exec, 1, data + temp_index), 1)));
            }
        }
        // coarse generate
        auto rstr = rstr_prlg_factory->generate(matrix);
        rstr_prlg_list_.emplace_back(give(rstr));
        matrix = rstr_prlg_list_.back()->get_coarse_operator();
        num_rows = matrix->get_size()[0];
        level++;
    }
    // generate coarsest solver
    if (parameters_.coarsest_solver != nullptr) {
        coarsest_solver_ = parameters_.coarsest_solver->generate(matrix);
    } else {
        coarsest_solver_ =
            matrix::Identity<ValueType>::create(exec, matrix->get_size()[0]);
    }
}

template <typename ValueType>
void Multigrid<ValueType>::v_cycle(
    size_type level, std::shared_ptr<const LinOp> matrix,
    const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &r_list,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &g_list,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &e_list) const
{
    auto r = r_list.at(level);
    auto g = g_list.at(level);
    auto e = e_list.at(level);
    // pre-smooth
    // r = b - Ax

    r->copy_from(b);
    matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    // x += relaxation * Smoother(r)
    auto pre_smoother = pre_smoother_list_.at(level);
    std::shared_ptr<matrix::Dense<ValueType>> pre_relaxation;
    if (parameters_.pre_relaxation.get_num_elems() == 0) {
        pre_relaxation = one_op_;
    } else {
        pre_relaxation = pre_relaxation_list_.at(level);
    }
    if (pre_smoother) {
        pre_smoother->apply(pre_relaxation.get(), r.get(), one_op_.get(), x);
        // compute residual
        r->copy_from(b);  // n * b
        matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    }
    // restrict
    rstr_prlg_list_.at(level)->restrict_apply(r.get(), g.get());
    // next level or solve it
    if (level + 1 == rstr_prlg_list_.size()) {
        coarsest_solver_->apply(g.get(), e.get());
    } else {
        this->v_cycle(level + 1,
                      rstr_prlg_list_.at(level)->get_coarse_operator(), g.get(),
                      e.get(), r_list, g_list, e_list);
    }
    // prolong
    rstr_prlg_list_.at(level)->prolong_applyadd(e.get(), x);
    // post-smooth
    std::shared_ptr<LinOp> post_smoother;
    std::shared_ptr<matrix::Dense<ValueType>> post_relaxation;

    if (parameters_.post_uses_pre) {
        post_smoother = pre_smoother;
        post_relaxation = pre_relaxation;
    } else {
        post_smoother = post_smoother_list_.at(level);
        if (parameters_.post_relaxation.get_num_elems() == 0) {
            post_relaxation = one_op_;
        } else {
            post_relaxation = post_relaxation_list_.at(level);
        }
    }
    if (post_smoother) {
        r->copy_from(b);
        matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
        post_smoother->apply(post_relaxation.get(), r.get(), one_op_.get(), x);
    }
}

template <typename ValueType>
void Multigrid<ValueType>::prepare_vcycle(
    const size_type nrhs,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &r,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &g,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &e) const
{
    auto current_nrows = system_matrix_->get_size()[0];
    auto exec = this->get_executor();
    for (int i = 0; i < rstr_prlg_list_.size(); i++) {
        auto next_nrows =
            rstr_prlg_list_.at(i)->get_coarse_operator()->get_size()[0];
        r.at(i) = matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{current_nrows, nrhs});
        g.at(i) = matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{next_nrows, nrhs});
        e.at(i) = matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{next_nrows, nrhs});
        current_nrows = next_nrows;
    }
}


template <typename ValueType>
void Multigrid<ValueType>::run_cycle(
    multigrid_cycle cycle, size_type level, std::shared_ptr<const LinOp> matrix,
    const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &r_list,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &g_list,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &e_list) const
{
    auto r = r_list.at(level);
    auto g = g_list.at(level);
    auto e = e_list.at(level);
    r->copy_from(b);
    matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    // x += relaxation * Smoother(r)
    auto pre_smoother = pre_smoother_list_.at(level);
    std::shared_ptr<matrix::Dense<ValueType>> pre_relaxation;
    if (parameters_.pre_relaxation.get_num_elems() == 0) {
        pre_relaxation = one_op_;
    } else {
        pre_relaxation = pre_relaxation_list_.at(level);
    }
    if (pre_smoother) {
        pre_smoother->apply(pre_relaxation.get(), r.get(), one_op_.get(), x);
        // compute residual
        r->copy_from(b);  // n * b
        matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    }
    // first cycle
    rstr_prlg_list_.at(level)->restrict_apply(r.get(), g.get());
    // next level or solve it
    if (level + 1 == rstr_prlg_list_.size()) {
        coarsest_solver_->apply(g.get(), e.get());
    } else {
        this->run_cycle(cycle, level + 1,
                        rstr_prlg_list_.at(level)->get_coarse_operator(),
                        g.get(), e.get(), r_list, g_list, e_list);
    }
    // additional work for non-v_cycle
    if (cycle == multigrid_cycle::f || cycle == multigrid_cycle::w) {
        // second cycle - f_cycle, w_cycle
        // prolong
        rstr_prlg_list_.at(level)->prolong_applyadd(e.get(), x);
        // compute residual
        r->copy_from(b);  // n * b
        matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
        // re-smooth
        if (pre_smoother) {
            pre_smoother->apply(pre_relaxation.get(), r.get(), one_op_.get(),
                                x);
            // compute residual
            r->copy_from(b);  // n * b
            matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
        }

        rstr_prlg_list_.at(level)->restrict_apply(r.get(), g.get());
        // next level or solve it
        if (level + 1 == rstr_prlg_list_.size()) {
            coarsest_solver_->apply(g.get(), e.get());
        } else {
            if (cycle == multigrid_cycle::f) {
                // f_cycle call v_cycle in the second cycle
                this->run_cycle(
                    multigrid_cycle::v, level + 1,
                    rstr_prlg_list_.at(level)->get_coarse_operator(), g.get(),
                    e.get(), r_list, g_list, e_list);
            } else {
                this->run_cycle(
                    cycle, level + 1,
                    rstr_prlg_list_.at(level)->get_coarse_operator(), g.get(),
                    e.get(), r_list, g_list, e_list);
            }
        }
    } else if (cycle == multigrid_cycle::kfcg ||
               cycle == multigrid_cycle::kgcr) {
        // do some work in coarse level - do not need prolong
        GKO_NOT_IMPLEMENTED;
    }

    // prolong
    rstr_prlg_list_.at(level)->prolong_applyadd(e.get(), x);

    // post-smooth
    std::shared_ptr<LinOp> post_smoother;
    std::shared_ptr<matrix::Dense<ValueType>> post_relaxation;

    if (parameters_.post_uses_pre) {
        post_smoother = pre_smoother;
        post_relaxation = pre_relaxation;
    } else {
        post_smoother = post_smoother_list_.at(level);
        if (parameters_.post_relaxation.get_num_elems() == 0) {
            post_relaxation = one_op_;
        } else {
            post_relaxation = post_relaxation_list_.at(level);
        }
    }
    if (post_smoother) {
        r->copy_from(b);
        matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
        post_smoother->apply(post_relaxation.get(), r.get(), one_op_.get(), x);
    }
}


template <typename ValueType>
void Multigrid<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    if (cycle_ == multigrid_cycle::kfcg || cycle_ == multigrid_cycle::kgcr) {
        GKO_NOT_IMPLEMENTED;
    }

    auto exec = this->get_executor();
    constexpr uint8 RelativeStoppingId{1};
    Array<stopping_status> stop_status(exec, b->get_size()[1]);
    bool one_changed{};
    auto dense_x = gko::as<matrix::Dense<ValueType>>(x);
    auto dense_b = gko::as<matrix::Dense<ValueType>>(b);
    auto r = dense_b->clone();
    system_matrix_->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());
    std::vector<std::shared_ptr<vector_type>> r_list(rstr_prlg_list_.size());
    std::vector<std::shared_ptr<vector_type>> g_list(rstr_prlg_list_.size());
    std::vector<std::shared_ptr<vector_type>> e_list(rstr_prlg_list_.size());
    this->prepare_vcycle(b->get_size()[1], r_list, g_list, e_list);
    exec->run(multigrid::make_initialize_v(e_list, &stop_status));
    int iter = -1;
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(this, iter, r.get(),
                                                            dense_x);
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }
        this->run_cycle(cycle_, 0, system_matrix_, dense_b, dense_x, r_list,
                        g_list, e_list);
        r->copy_from(dense_b);
        system_matrix_->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    }
}


template <typename ValueType>
void Multigrid<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                      const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_MULTIGRID(_type) class Multigrid<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID);


}  // namespace solver
}  // namespace gko
