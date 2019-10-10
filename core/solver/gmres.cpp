/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/solver/gmres.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/solver/gmres_kernels.hpp"


namespace gko {
namespace solver {


namespace gmres {


GKO_REGISTER_OPERATION(initialize_1, gmres::initialize_1);
GKO_REGISTER_OPERATION(initialize_2, gmres::initialize_2);
GKO_REGISTER_OPERATION(step_1, gmres::step_1);
GKO_REGISTER_OPERATION(step_2, gmres::step_2);


}  // namespace gmres


namespace {


template <typename ValueType>
void apply_preconditioner(
    const LinOp *preconditioner, matrix::Dense<ValueType> *krylov_bases,
    std::shared_ptr<matrix::Dense<ValueType>> &preconditioned_vector,
    const size_type iter)
{
    std::shared_ptr<matrix::Dense<ValueType>> target_basis =
        krylov_bases->create_submatrix(
            span{0, krylov_bases->get_size()[0]},
            span{iter * preconditioned_vector->get_size()[1],
                 (iter + 1) * preconditioned_vector->get_size()[1]});

    // Apply preconditioner
    auto identity_pointer =
        dynamic_cast<const matrix::Identity<ValueType> *>(preconditioner);
    if (identity_pointer) {
        preconditioned_vector = target_basis;
    } else {
        preconditioner->apply(target_basis.get(), preconditioned_vector.get());
    }
}


}  // namespace


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix_);

    using Vector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    auto residual = Vector::create_with_config_of(dense_b);
    auto krylov_bases = Vector::create(
        exec, dim<2>{system_matrix_->get_size()[1],
                     (krylov_dim_ + 1) * dense_b->get_size()[1]});
    auto next_krylov_basis = Vector::create_with_config_of(dense_b);
    std::shared_ptr<matrix::Dense<ValueType>> preconditioned_vector =
        Vector::create_with_config_of(dense_b);
    auto hessenberg = Vector::create(
        exec, dim<2>{krylov_dim_ + 1, krylov_dim_ * dense_b->get_size()[1]});
    auto givens_sin =
        Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});
    auto givens_cos =
        Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});
    auto residual_norm_collection =
        Vector::create(exec, dim<2>{krylov_dim_ + 1, dense_b->get_size()[1]});
    auto residual_norm =
        Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto b_norm = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    Array<size_type> final_iter_nums(this->get_executor(),
                                     dense_b->get_size()[1]);
    auto y = Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});

    bool one_changed{};
    Array<stopping_status> stop_status(this->get_executor(),
                                       dense_b->get_size()[1]);

    // Initialization
    exec->run(gmres::make_initialize_1(dense_b, b_norm.get(), residual.get(),
                                       givens_sin.get(), givens_cos.get(),
                                       &stop_status, krylov_dim_));
    // b_norm = norm(b)
    // residual = dense_b
    // givens_sin = givens_cos = 0
    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                          residual.get());
    // residual = residual - Ax

    exec->run(gmres::make_initialize_2(
        residual.get(), residual_norm.get(), residual_norm_collection.get(),
        krylov_bases.get(), &final_iter_nums, krylov_dim_));
    // residual_norm = norm(residual)
    // residual_norm_collection = {residual_norm, 0, ..., 0}
    // krylov_bases(:, 1) = residual / residual_norm
    // final_iter_nums = {0, ..., 0}

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, residual.get());

    int total_iter = -1;
    size_type restart_iter = 0;

    auto before_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(dense_x);
    auto after_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(dense_x);

    while (true) {
        ++total_iter;
        this->template log<log::Logger::iteration_complete>(
            this, total_iter, residual.get(), dense_x, residual_norm.get());
        if (stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual.get())
                .residual_norm(residual_norm.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        if (restart_iter == krylov_dim_) {
            // Restart
            exec->run(gmres::make_step_2(residual_norm_collection.get(),
                                         krylov_bases.get(), hessenberg.get(),
                                         y.get(), before_preconditioner.get(),
                                         &final_iter_nums));
            // Solve upper triangular.
            // y = hessenberg \ residual_norm_collection

            get_preconditioner()->apply(before_preconditioner.get(),
                                        after_preconditioner.get());
            dense_x->add_scaled(one_op.get(), after_preconditioner.get());
            // Solve x
            // x = x + get_preconditioner() * krylov_bases * y
            residual->copy_from(dense_b);
            // residual = dense_b
            system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                                  residual.get());
            // residual = residual - Ax
            exec->run(gmres::make_initialize_2(
                residual.get(), residual_norm.get(),
                residual_norm_collection.get(), krylov_bases.get(),
                &final_iter_nums, krylov_dim_));
            // residual_norm = norm(residual)
            // residual_norm_collection = {residual_norm, 0, ..., 0}
            // krylov_bases(:, 1) = residual / residual_norm
            // final_iter_nums = {0, ..., 0}
            restart_iter = 0;
        }

        apply_preconditioner(get_preconditioner().get(), krylov_bases.get(),
                             preconditioned_vector, restart_iter);
        // preconditioned_vector = get_preconditioner() *
        //                         krylov_bases(:, restart_iter)

        // Do Arnoldi and givens rotation
        auto hessenberg_iter = hessenberg->create_submatrix(
            span{0, restart_iter + 2},
            span{dense_b->get_size()[1] * restart_iter,
                 dense_b->get_size()[1] * (restart_iter + 1)});

        // Start of arnoldi
        system_matrix_->apply(preconditioned_vector.get(),
                              next_krylov_basis.get());
        // next_krylov_basis = A * preconditioned_vector

        exec->run(gmres::make_step_1(
            next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
            residual_norm.get(), residual_norm_collection.get(),
            krylov_bases.get(), hessenberg_iter.get(), b_norm.get(),
            restart_iter, &final_iter_nums, &stop_status));
        // for i in 0:restart_iter
        //     hessenberg(restart_iter, i) = next_krylov_basis' *
        //     krylov_bases(:, i) next_krylov_basis  -= hessenberg(restart_iter,
        //     i) * krylov_bases(:, i)
        // end
        // hessenberg(restart_iter, restart_iter + 1) = norm(next_krylov_basis)
        // next_krylov_basis /= hessenberg(restart_iter, restart_iter + 1)
        // End of arnoldi
        // Start apply givens rotation
        // for j in 0:restart_iter
        //     temp             =  cos(j)*hessenberg(j) +
        //                         sin(j)*hessenberg(j+1)
        //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
        //                         cos(j)*hessenberg(j+1)
        //     hessenberg(j)    =  temp;
        // end
        // Calculate sin and cos
        // hessenberg(restart_iter)   =
        // cos(restart_iter)*hessenberg(restart_iter) +
        //                      sin(restart_iter)*hessenberg(restart_iter)
        // hessenberg(restart_iter+1) = 0
        // End apply givens rotation
        // Calculate residual norm

        restart_iter++;
    }

    // Solve x
    auto krylov_bases_small = krylov_bases->create_submatrix(
        span{0, system_matrix_->get_size()[0]},
        span{0, dense_b->get_size()[1] * (restart_iter + 1)});
    auto hessenberg_small = hessenberg->create_submatrix(
        span{0, restart_iter},
        span{0, dense_b->get_size()[1] * (restart_iter)});

    exec->run(gmres::make_step_2(
        residual_norm_collection.get(), krylov_bases_small.get(),
        hessenberg_small.get(), y.get(), before_preconditioner.get(),
        &final_iter_nums));
    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection

    get_preconditioner()->apply(before_preconditioner.get(),
                                after_preconditioner.get());
    dense_x->add_scaled(one_op.get(), after_preconditioner.get());
    // Solve x
    // x = x + get_preconditioner() * krylov_bases * y
}


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                  const LinOp *residual_norm_collection,
                                  LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(residual_norm_collection);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_GMRES(_type) class Gmres<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES);


}  // namespace solver
}  // namespace gko
