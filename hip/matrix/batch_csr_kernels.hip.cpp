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

#include "core/matrix/batch_csr_kernels.hpp"


#include <algorithm>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/dense_kernels.hpp"
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
// constexpr int warps_in_block = 4;
// constexpr int spmv_block_size = warps_in_block * config::warp_size;
// constexpr int wsize = config::warp_size;
// constexpr int classical_overweight = 32;


/**
 * A compile-time list of the number items per threads for which spmv kernel
 * should be compiled.
 */
using compiled_kernels = syn::value_list<int, 3, 4, 6, 7, 8, 12, 14>;

using classical_kernels =
    syn::value_list<int, config::warp_size, 32, 16, 8, 4, 2, 1>;

using spgeam_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/matrix/batch_csr_kernels.hpp.inc"

template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const HipExecutor> exec,
          const matrix::BatchCsr<ValueType, IndexType> *const a,
          const matrix::BatchDense<ValueType> *const b,
          matrix::BatchDense<ValueType> *const c)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(spmv), dim3(num_blocks),
                       dim3(default_block_size), 0, 0, a_ub, b_ub, c_ub);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const HipExecutor> exec,
                   const matrix::BatchDense<ValueType> *const alpha,
                   const matrix::BatchCsr<ValueType, IndexType> *const a,
                   const matrix::BatchDense<ValueType> *const b,
                   const matrix::BatchDense<ValueType> *const beta,
                   matrix::BatchDense<ValueType> *const c)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(advanced_spmv), dim3(num_blocks),
                       dim3(default_block_size), 0, 0, alpha_ub, a_ub, b_ub,
                       beta_ub, c_ub);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const HipExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const HipExecutor> exec,
                      const matrix::BatchCsr<ValueType, IndexType> *source,
                      matrix::BatchDense<ValueType> *result)
    GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const HipExecutor> exec,
                          const matrix::BatchCsr<ValueType, IndexType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const HipExecutor> exec,
               const matrix::BatchCsr<ValueType, IndexType> *orig,
               matrix::BatchCsr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const HipExecutor> exec,
                    const matrix::BatchCsr<ValueType, IndexType> *orig,
                    matrix::BatchCsr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *source,
    size_type *result) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *source,
    Array<size_type> *result) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const HipExecutor> exec,
                          matrix::BatchCsr<ValueType, IndexType> *to_sort)
    GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *to_check,
    bool *is_sorted) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void batch_scale(std::shared_ptr<const HipExecutor> exec,
                 const matrix::BatchDense<ValueType> *const left_scale,
                 const matrix::BatchDense<ValueType> *const right_scale,
                 matrix::BatchCsr<ValueType, IndexType> *const mat)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SCALE);


}  // namespace batch_csr
}  // namespace hip
}  // namespace kernels
}  // namespace gko
