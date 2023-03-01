/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/solver/batch_tridiagonal_solver.hpp>


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/matrix/batch_tridiagonal.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/matrix/batch_tridiagonal_kernels.hpp"
#include "core/solver/batch_tridiagonal_solver_kernels.hpp"


namespace gko {
namespace solver {
namespace batch_tridiagonal_solver {


GKO_REGISTER_OPERATION(apply, batch_tridiagonal_solver::apply);


}  // namespace batch_tridiagonal_solver


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchTridiagonalSolver<ValueType>::transpose() const
{
    return build()
        .with_left_scaling_op(share(
            as<BatchTransposable>(this->get_left_scaling_op())->transpose()))
        .with_right_scaling_op(share(
            as<BatchTransposable>(this->get_right_scaling_op())->transpose()))
        .on(this->get_executor())
        ->generate(share(
            as<BatchTransposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchTridiagonalSolver<ValueType>::conj_transpose()
    const
{
    return build()
        .with_left_scaling_op(
            share(as<BatchTransposable>(this->get_left_scaling_op())
                      ->conj_transpose()))
        .with_right_scaling_op(
            share(as<BatchTransposable>(this->get_right_scaling_op())
                      ->conj_transpose()))
        .on(this->get_executor())
        ->generate(share(as<BatchTransposable>(this->get_system_matrix())
                             ->conj_transpose()));
}


template <typename ValueType>
void BatchTridiagonalSolver<ValueType>::apply_impl(const BatchLinOp* b,
                                                   BatchLinOp* x) const
{
    using BDiag = matrix::BatchDiagonal<ValueType>;
    using Vector = matrix::BatchDense<ValueType>;
    using real_type = remove_complex<ValueType>;

    if (!this->system_matrix_->get_size().stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }  // TODO: Provision for solving non-uniform tridiagonal batch

    auto exec = this->get_executor();

    /* EITHER: (Have the conversion facility)
    std::shared_ptr<gko::matrix::BatchTridiagonal<ValueType>>
        system_matrix_tridiagonal;

    if (auto temp_tridiagonal =
            dynamic_cast<const matrix_type*>(this->system_matrix_.get())) {
        system_matrix_tridiagonal =
            gko::share(gko::clone(exec, temp_tridiagonal));
        // TODO: avoid extra copy if matrix is already sorted
    } else {
        system_matrix_tridiagonal = gko::share(matrix_type::create(exec));
        as<ConvertibleTo<matrix_type>>(this->system_matrix_.get())
            ->convert_to(system_matrix_tridiagonal.get());
    }
    */

    // OR: Restrict the system matrix type

    // That means the only matrix type allowed is batched tridiagonal matrix.
    auto system_matrix_tridiagonal =
        as<const matrix_type>(this->system_matrix_);

    const bool to_scale = std::dynamic_pointer_cast<const BDiag>(
                              this->parameters_.left_scaling_op) &&
                          std::dynamic_pointer_cast<const BDiag>(
                              this->parameters_.right_scaling_op);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);

    auto b_scaled = Vector::create(exec);
    const Vector* b_scaled_ptr{};

    // copies to scale
    if (to_scale) {
        b_scaled->copy_from(dense_b);
        as<const BDiag>(this->left_scaling_)
            ->apply(b_scaled.get(), b_scaled.get());
        b_scaled_ptr = b_scaled.get();
    } else {
        b_scaled_ptr = dense_b;
    }

    exec->run(batch_tridiagonal_solver::make_apply(
        system_matrix_tridiagonal.get(), b_scaled_ptr, dense_x));

    if (to_scale) {
        as<const BDiag>(this->parameters_.right_scaling_op)
            ->apply(dense_x, dense_x);
    }
}


template <typename ValueType>
void BatchTridiagonalSolver<ValueType>::apply_impl(const BatchLinOp* alpha,
                                                   const BatchLinOp* b,
                                                   const BatchLinOp* beta,
                                                   BatchLinOp* x) const
{
    auto dense_x = as<matrix::BatchDense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_BATCH_TRIDIAGONAL_SOLVER(_type) \
    class BatchTridiagonalSolver<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_TRIDIAGONAL_SOLVER);


}  // namespace solver
}  // namespace gko
