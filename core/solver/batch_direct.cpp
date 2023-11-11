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

#include <ginkgo/core/solver/batch_direct.hpp>


#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_direct_kernels.hpp"


namespace gko {
namespace solver {
namespace batch_direct {


GKO_REGISTER_OPERATION(vec_scale, batch_dense::batch_scale);
GKO_REGISTER_OPERATION(pre_diag_scale_system_transpose,
                       batch_direct::pre_diag_scale_system_transpose);
GKO_REGISTER_OPERATION(transpose_scale_copy,
                       batch_direct::transpose_scale_copy);
GKO_REGISTER_OPERATION(apply, batch_direct::apply);


}  // namespace batch_direct


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchDirect<ValueType>::transpose() const
{
    return build()
        .on(this->get_executor())
        ->generate(share(
            as<BatchTransposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchDirect<ValueType>::conj_transpose() const
{
    return build()
        .on(this->get_executor())
        ->generate(share(as<BatchTransposable>(this->get_system_matrix())
                             ->conj_transpose()));
}


namespace {


template <typename ValueType>
std::shared_ptr<matrix::BatchDense<ValueType>> convert_and_transpose(
    std::shared_ptr<const Executor> exec,
    const matrix::BatchCsr<ValueType, int>* const a_csr)
{
    auto a_dense_t =
        matrix::BatchDense<ValueType>::create(exec, a_csr->get_size());
    a_csr->convert_to(a_dense_t.get());
    return std::dynamic_pointer_cast<matrix::BatchDense<ValueType>>(
        gko::share(a_dense_t->transpose()));
}


}  // namespace


template <typename ValueType>
void BatchDirect<ValueType>::apply_impl(const BatchLinOp* b,
                                        BatchLinOp* x) const
{
    using Mtx = matrix::BatchCsr<ValueType>;
    using BDense = matrix::BatchDense<ValueType>;
    using BDiag = matrix::BatchDiagonal<ValueType>;
    using Vector = matrix::BatchDense<ValueType>;
    using real_type = remove_complex<ValueType>;

    auto exec = this->get_executor();
    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    const bool to_scale =
        parameters_.left_scaling_op && parameters_.right_scaling_op;
    GKO_ASSERT(to_scale == false);

    log::BatchLogData<ValueType> logdata;  //< Useless

    auto bt =
        std::dynamic_pointer_cast<BDense>(gko::share(dense_b->transpose()));

    exec->copy(dense_system_matrix_->get_num_stored_elements(),
               dense_system_matrix_->get_const_values(),
               workspace_dense_matrix_->get_values());

    exec->run(batch_direct::make_apply(this->workspace_dense_matrix_.get(),
                                       bt.get(), logdata));

    auto btt = std::dynamic_pointer_cast<BDense>(gko::share(bt->transpose()));
    dense_x->copy_from(btt.get());
}


template <typename ValueType>
void BatchDirect<ValueType>::apply_impl(const BatchLinOp* alpha,
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


#define GKO_DECLARE_BATCH_DIRECT(_type) class BatchDirect<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DIRECT);


}  // namespace solver
}  // namespace gko
