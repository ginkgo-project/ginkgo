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

#include <ginkgo/core/matrix/batch_tridiagonal.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_tridiagonal_kernels.hpp"


namespace gko {
namespace matrix {
namespace batch_tridiagonal {}  // namespace batch_tridiagonal


template <typename ValueType>
void BatchTridiagonal<ValueType>::apply_impl(
    const BatchLinOp* b, BatchLinOp* x) const GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchTridiagonal<ValueType>::apply_impl(
    const BatchLinOp* alpha, const BatchLinOp* b, const BatchLinOp* beta,
    BatchLinOp* x) const GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchTridiagonal<ValueType>::convert_to(
    BatchTridiagonal<next_precision<ValueType>>* result) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_tridiagonal): change the code imported from
// matrix/batch_csr if needed
//    result->values_ = this->values_;
//    result->col_idxs_ = this->col_idxs_;
//    result->row_ptrs_ = this->row_ptrs_;
//    result->set_size(this->get_size());
//}


template <typename ValueType>
void BatchTridiagonal<ValueType>::move_to(
    BatchTridiagonal<next_precision<ValueType>>* result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_tridiagonal): change the code imported from
// matrix/batch_csr if needed
//    this->convert_to(result);
//}


template <typename ValueType>
void BatchTridiagonal<ValueType>::convert_to(
    BatchCsr<ValueType, int32>* const result) const GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchTridiagonal<ValueType>::move_to(
    BatchCsr<ValueType, int32>* const result) GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchTridiagonal<ValueType>::read(const std::vector<mat_data>& data)
    GKO_NOT_IMPLEMENTED;

template <typename ValueType>
void BatchTridiagonal<ValueType>::read(const std::vector<mat_data32>& data)
    GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchTridiagonal<ValueType>::write(std::vector<mat_data>& data) const
    GKO_NOT_IMPLEMENTED;

template <typename ValueType>
void BatchTridiagonal<ValueType>::write(std::vector<mat_data32>& data) const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchTridiagonal<ValueType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchTridiagonal<ValueType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchTridiagonal<ValueType>::add_scaled_identity_impl(
    const BatchLinOp* const a, const BatchLinOp* const b) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_tridiagonal): change the code imported from
// matrix/batch_csr if needed
//    bool has_all_diags = false;
//    this->get_executor()->run(
//        batch_tridiagonal::make_check_diagonal_entries_exist(this,
//        has_all_diags));
//    if (!has_all_diags) {
//        // TODO: Replace this with proper exception helper after merging
//        // non-batched add_scaled_identity PR
//        throw std::runtime_error("Matrix does not have all diagonal
//        entries!");
//    }
//    this->get_executor()->run(batch_tridiagonal::make_add_scaled_identity(
//        as<const BatchDense<ValueType>>(a), as<const
//        BatchDense<ValueType>>(b), this));
//}


#define GKO_DECLARE_BATCH_TRIDIAGONAL_MATRIX(ValueType) \
    class BatchTridiagonal<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_TRIDIAGONAL_MATRIX);


}  // namespace matrix
}  // namespace gko
