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

#include "core/matrix/batch_ell_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/merging.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/segment_scan.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_ell
 */
namespace batch_ell {


constexpr int default_block_size = 512;
constexpr int sm_multiplier = 4;


#include "common/cuda_hip/matrix/batch_ell_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::BatchEll<ValueType, IndexType>* const a,
          const matrix::BatchDense<ValueType>* const b,
          matrix::BatchDense<ValueType>* const c)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    const auto num_rows = a_ub.num_rows;
    const auto nbatches = a_ub.num_batch;
    spmv<<<num_blocks, default_block_size>>>(a_ub, b_ub, c_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::BatchDense<ValueType>* const alpha,
                   const matrix::BatchEll<ValueType, IndexType>* const a,
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
    advanced_spmv<<<num_blocks, default_block_size>>>(alpha_ub, a_ub, b_ub,
                                                      beta_ub, c_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const CudaExecutor> exec,
                              const IndexType* ptrs, size_type num_rows,
                              IndexType* idxs) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::BatchEll<ValueType, IndexType>* source,
                      matrix::BatchDense<ValueType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const CudaExecutor> exec,
                          const matrix::BatchEll<ValueType, IndexType>* source,
                          size_type* result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const CudaExecutor> exec,
               const matrix::BatchEll<ValueType, IndexType>* orig,
               matrix::BatchEll<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::BatchEll<ValueType, IndexType>* orig,
                    matrix::BatchEll<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* source,
    size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* source,
    Array<size_type>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const CudaExecutor> exec,
                          matrix::BatchEll<ValueType, IndexType>* to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* to_check,
    bool* is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void batch_scale(std::shared_ptr<const CudaExecutor> exec,
                 const matrix::BatchDense<ValueType>* const left_scale,
                 const matrix::BatchDense<ValueType>* const right_scale,
                 matrix::BatchEll<ValueType, IndexType>* const mat)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_SCALE);


template <typename ValueType, typename IndexType>
void pre_diag_scale_system(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::BatchDense<ValueType>* const left_scale,
    const matrix::BatchDense<ValueType>* const right_scale,
    matrix::BatchEll<ValueType, IndexType>* const a,
    matrix::BatchDense<ValueType>* const b) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_PRE_DIAG_SCALE_SYSTEM);


template <typename ValueType, typename IndexType>
void convert_to_batch_dense(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* const src,
    matrix::BatchDense<ValueType>* const dest)
{
    const size_type nbatches = src->get_num_batch_entries();
    const int num_rows = src->get_size().at(0)[0];
    const int num_cols = src->get_size().at(0)[1];
    const int nnz = static_cast<int>(src->get_num_stored_elements() / nbatches);
    const size_type dstride = dest->get_stride().at(0);
    const size_type estride = src->get_stride().at(0);
    const auto col_idxs = src->get_const_col_idxs();
    const auto vals = src->get_const_values();

    const dim3 block_size(config::warp_size,
                          config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(num_cols * nbatches, block_size.x),
                             ceildiv(num_rows * nbatches, block_size.y), 1);
    initialize_zero_dense<<<init_grid_dim, block_size>>>(
        nbatches, num_rows, num_cols, dstride,
        as_cuda_type(dest->get_values()));

    const auto grid_dim = ceildiv(num_rows * nbatches, default_block_size);
    fill_in_dense<<<grid_dim, default_block_size>>>(
        nbatches, num_rows, src->get_num_stored_elements_per_row().at(0),
        estride, as_cuda_type(col_idxs), as_cuda_type(vals), dstride,
        as_cuda_type(dest->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONVERT_TO_BATCH_DENSE);


}  // namespace batch_ell
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
