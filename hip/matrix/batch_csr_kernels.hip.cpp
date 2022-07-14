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

#include "core/matrix/batch_csr_kernels.hpp"


#include <algorithm>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/merging.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/segment_scan.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_csr
 */
namespace batch_csr {


constexpr int default_block_size = 512;
constexpr int sm_multiplier = 4;


#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const HipExecutor> exec,
          const matrix::BatchCsr<ValueType, IndexType>* const a,
          const matrix::BatchDense<ValueType>* const b,
          matrix::BatchDense<ValueType>* const c)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    hipLaunchKernelGGL(spmv, dim3(num_blocks), dim3(default_block_size), 0, 0,
                       a_ub, b_ub, c_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const HipExecutor> exec,
                   const matrix::BatchDense<ValueType>* const alpha,
                   const matrix::BatchCsr<ValueType, IndexType>* const a,
                   const matrix::BatchDense<ValueType>* const b,
                   const matrix::BatchDense<ValueType>* const beta,
                   matrix::BatchDense<ValueType>* const c)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);
    hipLaunchKernelGGL(advanced_spmv, dim3(num_blocks),
                       dim3(default_block_size), 0, 0, alpha_ub, a_ub, b_ub,
                       beta_ub, c_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const HipExecutor> exec,
                              const IndexType* ptrs, size_type num_rows,
                              IndexType* idxs) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const HipExecutor> exec,
                      const matrix::BatchCsr<ValueType, IndexType>* source,
                      matrix::BatchDense<ValueType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const HipExecutor> exec,
                          const matrix::BatchCsr<ValueType, IndexType>* source,
                          size_type* result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const HipExecutor> exec,
               const matrix::BatchCsr<ValueType, IndexType>* orig,
               matrix::BatchCsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const HipExecutor> exec,
                    const matrix::BatchCsr<ValueType, IndexType>* orig,
                    matrix::BatchCsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* source,
    size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* source,
    array<size_type>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const HipExecutor> exec,
                          matrix::BatchCsr<ValueType, IndexType>* to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* to_check,
    bool* is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void batch_scale(std::shared_ptr<const HipExecutor> exec,
                 const matrix::BatchDiagonal<ValueType>* const left_scale,
                 const matrix::BatchDiagonal<ValueType>* const right_scale,
                 matrix::BatchCsr<ValueType, IndexType>* const mat)
{
    if (!left_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    if (!right_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;

    const auto m_ub = get_batch_struct(mat);

    const int num_blocks = mat->get_num_batch_entries();
    hipLaunchKernelGGL(uniform_batch_scale, dim3(num_blocks),
                       dim3(default_block_size), 0, 0,
                       as_hip_type(left_scale->get_const_values()),
                       as_hip_type(right_scale->get_const_values()), m_ub,
                       mat->get_size().at()[1]);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SCALE);


template <typename ValueType, typename IndexType>
void pre_diag_transform_system(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchDiagonal<ValueType>* const left_op,
    const matrix::BatchDiagonal<ValueType>* const right_op,
    matrix::BatchCsr<ValueType, IndexType>* const a,
    matrix::BatchDense<ValueType>* const b)
{
    const size_type nbatch = a->get_num_batch_entries();
    const int nrows = static_cast<int>(a->get_size().at()[0]);
    const int ncols = static_cast<int>(a->get_size().at()[1]);
    const size_type nnz = a->get_num_stored_elements() / nbatch;
    const int nrhs = static_cast<int>(b->get_size().at()[1]);
    const size_type b_stride = b->get_stride().at();
    hipLaunchKernelGGL(
        pre_diag_scale_system, dim3(nbatch), dim3(default_block_size), 0, 0,
        nbatch, nrows, ncols, nnz, as_hip_type(a->get_values()),
        a->get_const_col_idxs(), a->get_const_row_ptrs(), nrhs, b_stride,
        as_hip_type(b->get_values()), as_hip_type(left_op->get_const_values()),
        as_hip_type(right_op->get_const_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_PRE_DIAG_TRANSFORM_SYSTEM);


template <typename ValueType, typename IndexType>
void convert_to_batch_dense(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const src,
    matrix::BatchDense<ValueType>* const dest)
{
    const size_type nbatches = src->get_num_batch_entries();
    const int nrows = src->get_size().at()[0];
    const int ncols = src->get_size().at()[1];
    const int nnz = static_cast<int>(src->get_num_stored_elements() / nbatches);
    const size_type dstride = dest->get_stride().at();
    hipLaunchKernelGGL(uniform_convert_to_batch_dense, dim3(nbatches),
                       dim3(default_block_size), 0, 0, nbatches, nrows, ncols,
                       nnz, src->get_const_row_ptrs(),
                       src->get_const_col_idxs(),
                       as_hip_type(src->get_const_values()), dstride,
                       as_hip_type(dest->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_BATCH_DENSE);


template <typename ValueType, typename IndexType>
void check_diagonal_entries_exist(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const mtx,
    bool& has_all_diags)
{
    if (!mtx->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const auto nmin = static_cast<int>(
        std::min(mtx->get_size().at(0)[0], mtx->get_size().at(0)[1]));
    array<bool> d_result(exec, 1);
    hipLaunchKernelGGL(check_all_diagonal_locations, 1, default_block_size, 0,
                       0, nmin, mtx->get_const_row_ptrs(),
                       mtx->get_const_col_idxs(), d_result.get_data());
    has_all_diags = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CHECK_DIAGONAL_ENTRIES_EXIST);


template <typename ValueType, typename IndexType>
void add_scaled_identity(std::shared_ptr<const HipExecutor> exec,
                         const matrix::BatchDense<ValueType>* const a,
                         const matrix::BatchDense<ValueType>* const b,
                         matrix::BatchCsr<ValueType, IndexType>* const mtx)
{
    if (!mtx->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const size_type nbatch = mtx->get_num_batch_entries();
    const int nnz = static_cast<int>(mtx->get_num_stored_elements() / nbatch);
    const int nrows = mtx->get_size().at()[0];
    const size_type astride = a->get_stride().at();
    const size_type bstride = b->get_stride().at();
    hipLaunchKernelGGL(add_scaled_identity, nbatch, default_block_size, 0, 0,
                       nbatch, nrows, nnz, mtx->get_const_row_ptrs(),
                       mtx->get_const_col_idxs(),
                       as_hip_type(mtx->get_values()), astride,
                       as_hip_type(a->get_const_values()), bstride,
                       as_hip_type(b->get_const_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_csr
}  // namespace hip
}  // namespace kernels
}  // namespace gko
