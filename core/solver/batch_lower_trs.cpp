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

#include <ginkgo/core/solver/batch_lower_trs.hpp>


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_lower_trs_kernels.hpp"


namespace gko {
namespace solver {
namespace batch_lower_trs {


GKO_REGISTER_OPERATION(apply, batch_lower_trs::apply);


}  // namespace batch_lower_trs


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchLowerTrs<ValueType>::transpose() const
{
    return build()
        .with_skip_sorting(this->parameters_.skip_sorting)
        .with_left_scaling_op(share(
            as<BatchTransposable>(this->get_left_scaling_op())->transpose()))
        .with_right_scaling_op(share(
            as<BatchTransposable>(this->get_right_scaling_op())->transpose()))
        .on(this->get_executor())
        ->generate(share(
            as<BatchTransposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchLowerTrs<ValueType>::conj_transpose() const
{
    return build()
        .with_skip_sorting(this->parameters_.skip_sorting)
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
void BatchLowerTrs<ValueType>::apply_impl(const BatchLinOp* b,
                                          BatchLinOp* x) const
{
    using BCsr = matrix::BatchCsr<ValueType>;
    using BDiag = matrix::BatchDiagonal<ValueType>;
    using Vector = matrix::BatchDense<ValueType>;
    using real_type = remove_complex<ValueType>;

    auto exec = this->get_executor();
    auto system_matrix_new = gko::share(gko::clone(
        exec, this->system_matrix_.get()));  // TODO: avoid extra copy if
                                             // matrix is already sorted

    if (parameters_.skip_sorting != true) {
        if (auto amat = dynamic_cast<matrix::BatchCsr<ValueType>*>(
                system_matrix_new.get())) {
            amat->sort_by_column_index();
        }
    }  // Note: The algorithm assumes that column indices (for each row) of
       // batchell are already sorted.

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

    exec->run(batch_lower_trs::make_apply(system_matrix_new.get(), b_scaled_ptr,
                                          dense_x));

    if (to_scale) {
        as<const BDiag>(this->parameters_.right_scaling_op)
            ->apply(dense_x, dense_x);
    }
}


template <typename ValueType>
void BatchLowerTrs<ValueType>::apply_impl(const BatchLinOp* alpha,
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


#define GKO_DECLARE_BATCH_LOWER_TRS(_type) class BatchLowerTrs<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_LOWER_TRS);


}  // namespace solver
}  // namespace gko
