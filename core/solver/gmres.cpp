/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/gmres.hpp"


#include <iostream>


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/name_demangling.hpp"
#include "core/base/range_accessors.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/dense.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/solver/gmres_kernels.hpp"


namespace gko {
namespace solver {


namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(initialize_1, gmres::initialize_1<ValueType>);
};


template <typename... TplArgs>
struct TemplatedOperationRange {
    GKO_REGISTER_OPERATION(initialize_2, gmres::initialize_2<TplArgs...>);
    GKO_REGISTER_OPERATION(step_1, gmres::step_1<TplArgs...>);
    GKO_REGISTER_OPERATION(step_2, gmres::step_2<TplArgs...>);
};


template <typename... TplArgs>
struct TemplatedOperationDenseRange {
    GKO_REGISTER_OPERATION(simple_apply, dense::simple_apply<TplArgs...>);
    GKO_REGISTER_OPERATION(apply, dense::apply<TplArgs...>);
};


}  // namespace


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    ASSERT_IS_SQUARE_MATRIX(system_matrix_);

    this->template log<log::Logger::apply>(GKO_FUNCTION_NAME);

    using std::swap;
    using Vector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    auto r = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);
    auto e1 = Vector::create_with_config_of(dense_x);
    auto Q = Vector::create(exec, dim<2>{system_matrix_->get_size()[1],
                                         (default_max_num_iterations + 1) *
                                             dense_b->get_size()[1]});
    auto H = Vector::create(
        exec, dim<2>{default_max_num_iterations + 1,
                     default_max_num_iterations * dense_b->get_size()[1]});
    auto sn = Vector::create(
        exec, dim<2>{default_max_num_iterations, dense_b->get_size()[1]});
    auto cs = Vector::create(
        exec, dim<2>{default_max_num_iterations, dense_b->get_size()[1]});
    auto beta = Vector::create(
        exec, dim<2>{default_max_num_iterations + 1, dense_b->get_size()[1]});
    auto r_norm = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto b_norm = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    Array<size_type> iter_nums(this->get_executor(), dense_b->get_size()[1]);

    // Define range accessors for Q and H.
    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;
    row_major_range range_Q{Q->get_values(), Q->get_size()[0], Q->get_size()[1],
                            Q->get_stride()};
    row_major_range range_H{H->get_values(), H->get_size()[0], H->get_size()[1],
                            H->get_stride()};

    // std::cout << "Size of range_Q = " << range_Q.length(0) << " "
    //           << range_Q.length(1) << std::endl;
    // std::cout << "Size of range_H = " << range_H.length(0) << " "
    //           << range_H.length(1) << std::endl;

    bool one_changed{};
    Array<stopping_status> stop_status(this->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    exec->run(TemplatedOperation<ValueType>::make_initialize_1_operation(
        dense_b, r.get(), e1.get(), sn.get(), cs.get(), b_norm.get(),
        &iter_nums, &stop_status));
    // r = dense_b
    // sn = cs = 0
    // e1 = {1, 0, ..., 0}

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    // r = b - Ax

    exec->run(
        TemplatedOperationRange<ValueType,
                                range<accessor::row_major<ValueType, 2>>>::
            make_initialize_2_operation(r.get(), r_norm.get(), beta.get(),
                                        range_Q));
    // beta = {r_norm, 0, ..., 0}
    // Q(:, 1) = r / r_norm

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());

    size_type iter = 0;
    for (; iter < default_max_num_iterations; ++iter) {
        preconditioner_->apply(r.get(), z.get());
        // std::cout << "Iter = " << iter << std::endl;

        if (stop_criterion->update()
                .num_iterations(iter)
                .residual_norm(r_norm.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            this->template log<log::Logger::converged>(iter + 1, r.get());
            // std::cout << "Stopping criterion stops at iter " << iter
            //           << std::endl;
            break;
        }

        for (int i = 0; i < dense_b->get_size()[1]; ++i) {
            iter_nums.get_data()[i] +=
                (1 - stop_status.get_const_data()[i].has_stopped());
        }

        // Start Arnoldi function
        auto range_Q_k = range_Q(span{0, system_matrix_->get_size()[0]},
                                 span{dense_b->get_size()[1] * iter,
                                      dense_b->get_size()[1] * (iter + 1)});
        auto range_H_k = range_H(span{0, iter + 2},
                                 span{dense_b->get_size()[1] * iter,
                                      dense_b->get_size()[1] * (iter + 1)});
        auto q = Vector::create_with_config_of(dense_b);

        // std::cout << "b = " << std::endl;
        // for (int i = 0; i < dense_b->get_size()[0]; ++i) {
        //     for (int j = 0; j < dense_b->get_size()[1]; ++j) {
        //         std::cout << dense_b->at(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }

        exec->run(TemplatedOperationDenseRange<
                  ValueType, range<accessor::row_major<ValueType, 2>>>::
                      make_simple_apply_operation(
                          as<matrix::Dense<ValueType>>(system_matrix_.get()),
                          range_Q_k, q.get()));

        // std::cout << "q = " << std::endl;
        // for (int i = 0; i < q->get_size()[0]; ++i) {
        //     for (int j = 0; j < q->get_size()[1]; ++j) {
        //         std::cout << q->at(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }

        exec->run(
            TemplatedOperationRange<ValueType,
                                    range<accessor::row_major<ValueType, 2>>>::
                make_step_1_operation(q.get(), sn.get(), cs.get(), beta.get(),
                                      range_Q, range_H_k, r_norm.get(),
                                      b_norm.get(), iter, &stop_status));
        // std::cout << "Q = " << std::endl;
        // for (int i = 0; i < dense_b->get_size()[0]; ++i) {
        //     for (int j = 0; j < dense_b->get_size()[1] * (iter + 2); ++j) {
        //         std::cout << range_Q(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << "H = " << std::endl;
        // for (int i = 0; i < iter + 2; ++i) {
        //     for (int j = 0; j < dense_b->get_size()[1] * (iter + 1); ++j) {
        //         std::cout << range_H(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << "beta = " << std::endl;
        // for (int i = 0; i < beta->get_size()[0]; ++i) {
        //     for (int j = 0; j < beta->get_size()[1]; ++j) {
        //         std::cout << beta->at(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }

        this->template log<log::Logger::iteration_complete>(iter + 1);
    }

    // Solve x
    auto y = Vector::create(
        exec, dim<2>{default_max_num_iterations, dense_b->get_size()[1]});
    auto range_Q_small = range_Q(span{0, system_matrix_->get_size()[0]},
                                 span{0, dense_b->get_size()[1] * (iter + 1)});
    auto range_H_small =
        range_H(span{0, iter}, span{0, dense_b->get_size()[1] * (iter)});

    // std::cout << "Size of range_H_small = " << range_H_small.length(0) << " "
    //           << range_H_small.length(1) << std::endl;
    // std::cout << "Size of range_Q_small = " << range_Q_small.length(0) << " "
    //           << range_Q_small.length(1) << std::endl;

    // std::cout << "range_Q_small = " << std::endl;
    // for (int i = 0; i < range_Q_small.length(0); ++i) {
    //     for (int j = 0; j < range_Q_small.length(1); ++j) {
    //         std::cout << range_Q_small(i, j) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "range_H_small = " << std::endl;
    // for (int i = 0; i < range_H_small.length(0); ++i) {
    //     for (int j = 0; j < range_H_small.length(1); ++j) {
    //         std::cout << range_H_small(i, j) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    exec->run(
        TemplatedOperationRange<ValueType,
                                range<accessor::row_major<ValueType, 2>>>::
            make_step_2_operation(beta.get(), range_H_small, &iter_nums,
                                  y.get(), range_Q_small, dense_x));

    // std::cout << "Print x:" << std::endl;
    // for (int i = 0; i < dense_x->get_size()[0]; ++i) {
    //     for (int j = 0; j < dense_x->get_size()[1]; ++j) {
    //         std::cout << dense_x->at(i, j) << " ";
    //     }
    //     std::cout << std::endl;
    // }
}


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                  const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_GMRES(_type) class Gmres<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES);
#undef GKO_DECLARE_GMRES


}  // namespace solver
}  // namespace gko
