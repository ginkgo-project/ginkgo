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

#include "core/preconditioner/jacobi_kernels.hpp"


#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <vector>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/extended_float.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "dpcpp/components/matrix_operations.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Jacobi preconditioner namespace.
 *
 * @ingroup jacobi
 */
namespace jacobi {


void initialize_precisions(std::shared_ptr<const DpcppExecutor> exec,
                           const Array<precision_reduction> &source,
                           Array<precision_reduction> &precisions)
    GKO_NOT_IMPLEMENTED;


namespace {


template <typename IndexType>
inline bool has_same_nonzero_pattern(
    const IndexType *prev_row_ptr, const IndexType *curr_row_ptr,
    const IndexType *next_row_ptr) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
size_type find_natural_blocks(const matrix::Csr<ValueType, IndexType> *mtx,
                              uint32 max_block_size,
                              IndexType *block_ptrs) GKO_NOT_IMPLEMENTED;


template <typename IndexType>
inline size_type agglomerate_supervariables(
    uint32 max_block_size, size_type num_natural_blocks,
    IndexType *block_ptrs) GKO_NOT_IMPLEMENTED;


}  // namespace


template <typename ValueType, typename IndexType>
void find_blocks(std::shared_ptr<const DpcppExecutor> exec,
                 const matrix::Csr<ValueType, IndexType> *system_matrix,
                 uint32 max_block_size, size_type &num_blocks,
                 Array<IndexType> &block_pointers) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_FIND_BLOCKS_KERNEL);


namespace {


template <typename ValueType, typename IndexType>
inline void extract_block(const matrix::Csr<ValueType, IndexType> *mtx,
                          IndexType block_size, IndexType block_start,
                          ValueType *block,
                          size_type stride) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
inline IndexType choose_pivot(IndexType block_size, const ValueType *block,
                              size_type stride) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
inline void swap_rows(IndexType row1, IndexType row2, IndexType block_size,
                      ValueType *block, size_type stride) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
inline bool apply_gauss_jordan_transform(IndexType row, IndexType col,
                                         IndexType block_size, ValueType *block,
                                         size_type stride) GKO_NOT_IMPLEMENTED;


template <typename SourceValueType, typename ResultValueType,
          typename IndexType,
          typename ValueConverter =
              default_converter<SourceValueType, ResultValueType>>
inline void transpose_block(
    IndexType block_size, const SourceValueType *from, size_type from_stride,
    ResultValueType *to, size_type to_stride,
    ValueConverter converter = {}) noexcept GKO_NOT_IMPLEMENTED;


template <typename SourceValueType, typename ResultValueType,
          typename IndexType,
          typename ValueConverter =
              default_converter<SourceValueType, ResultValueType>>
inline void conj_transpose_block(
    IndexType block_size, const SourceValueType *from, size_type from_stride,
    ResultValueType *to, size_type to_stride,
    ValueConverter converter = {}) noexcept GKO_NOT_IMPLEMENTED;


template <typename SourceValueType, typename ResultValueType,
          typename IndexType,
          typename ValueConverter =
              default_converter<SourceValueType, ResultValueType>>
inline void permute_and_transpose_block(
    IndexType block_size, const IndexType *col_perm,
    const SourceValueType *source, size_type source_stride,
    ResultValueType *result, size_type result_stride,
    ValueConverter converter = {}) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
inline bool invert_block(IndexType block_size, IndexType *perm,
                         ValueType *block,
                         size_type stride) GKO_NOT_IMPLEMENTED;


template <typename ReducedType, typename ValueType, typename IndexType>
inline bool validate_precision_reduction_feasibility(
    std::shared_ptr<const DpcppExecutor> exec, IndexType block_size,
    const ValueType *block, size_type stride) GKO_NOT_IMPLEMENTED;


}  // namespace


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const DpcppExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size,
              remove_complex<ValueType> accuracy,
              const preconditioner::block_interleaved_storage_scheme<IndexType>
                  &storage_scheme,
              Array<remove_complex<ValueType>> &conditioning,
              Array<precision_reduction> &block_precisions,
              const Array<IndexType> &block_pointers,
              Array<ValueType> &blocks) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_GENERATE_KERNEL);


namespace {


template <
    typename ValueType, typename BlockValueType,
    typename ValueConverter = default_converter<BlockValueType, ValueType>>
inline void apply_block(size_type block_size, size_type num_rhs,
                        const BlockValueType *block, size_type stride,
                        ValueType alpha, const ValueType *b, size_type stride_b,
                        ValueType beta, ValueType *x, size_type stride_x,
                        ValueConverter converter = {}) GKO_NOT_IMPLEMENTED;


}  // namespace


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const DpcppExecutor> exec, size_type num_blocks,
           uint32 max_block_size,
           const preconditioner::block_interleaved_storage_scheme<IndexType>
               &storage_scheme,
           const Array<precision_reduction> &block_precisions,
           const Array<IndexType> &block_pointers,
           const Array<ValueType> &blocks,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta,
           matrix::Dense<ValueType> *x) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(
    std::shared_ptr<const DpcppExecutor> exec, size_type num_blocks,
    uint32 max_block_size,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    const Array<precision_reduction> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const matrix::Dense<ValueType> *b,
    matrix::Dense<ValueType> *x) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void transpose_jacobi(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    uint32 max_block_size, const Array<precision_reduction> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    Array<ValueType> &out_blocks) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose_jacobi(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    uint32 max_block_size, const Array<precision_reduction> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    Array<ValueType> &out_blocks) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const DpcppExecutor> exec, size_type num_blocks,
    const Array<precision_reduction> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    ValueType *result_values, size_type result_stride) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_CONVERT_TO_DENSE_KERNEL);


}  // namespace jacobi
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
