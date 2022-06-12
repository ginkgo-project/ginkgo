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

#include "core/matrix/fbcsr_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "dpcpp/base/config.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The fixed-size block compressed sparse row matrix format namespace.
 *
 * @ingroup fbcsr
 */
namespace fbcsr {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::Fbcsr<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b,
          matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Fbcsr<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         device_matrix_data<ValueType, IndexType>& data,
                         int block_size, array<IndexType>& row_ptrs,
                         array<IndexType>& col_idxs,
                         array<ValueType>& values) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Fbcsr<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(const std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType>* const source,
                    matrix::Csr<ValueType, IndexType>* const result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::Fbcsr<ValueType, IndexType>* orig,
               matrix::Fbcsr<ValueType, IndexType>* trans) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType>* orig,
                    matrix::Fbcsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType>* to_check,
    bool* is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void sort_by_column_index(const std::shared_ptr<const DpcppExecutor> exec,
                          matrix::Fbcsr<ValueType, IndexType>* const to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Fbcsr<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_EXTRACT_DIAGONAL);


}  // namespace fbcsr
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
