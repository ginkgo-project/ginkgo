// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/functional.h>
#include <thrust/transform.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "cuda/base/batch_struct.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/kernel_config.hpp"
#include "cuda/base/thrust.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/matrix/batch_struct.hpp"
#include "cuda/solver/batch_bicgstab_kernels.cuh"


namespace gko {
namespace kernels {
namespace cuda {


// NOTE: this default block size is not used for the main solver kernel.
constexpr int default_block_size = 256;
constexpr int sm_oversubscription = 4;


/**
 * @brief The batch Bicgstab solver namespace.
 *
 * @ingroup batch_bicgstab
 */
namespace batch_bicgstab {


#include "common/cuda_hip/base/batch_multi_vector_kernels.hpp.inc"
#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_ell_kernels.hpp.inc"
#include "common/cuda_hip/solver/batch_bicgstab_kernels.hpp.inc"


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_num_threads_per_block(std::shared_ptr<const DefaultExecutor> exec,
                              const int num_rows)
{
    int num_warps = std::max(num_rows / 4, 2);
    constexpr int warp_sz = static_cast<int>(config::warp_size);
    const int min_block_size = 2 * warp_sz;
    const int device_max_threads =
        ((std::max(num_rows, min_block_size)) / warp_sz) * warp_sz;
    cudaFuncAttributes funcattr;
    cudaFuncGetAttributes(&funcattr,
                          apply_kernel<StopType, 9, true, PrecType, LogType,
                                       BatchMatrixType, ValueType>);
    const int num_regs_used = funcattr.numRegs;
    int max_regs_blk = 0;
    cudaDeviceGetAttribute(&max_regs_blk, cudaDevAttrMaxRegistersPerBlock,
                           exec->get_device_id());
    const int max_threads_regs =
        ((max_regs_blk / static_cast<int>(num_regs_used)) / warp_sz) * warp_sz;
    int max_threads = std::min(max_threads_regs, device_max_threads);
    max_threads = max_threads <= 1024 ? max_threads : 1024;
    return std::max(std::min(num_warps * warp_sz, max_threads), min_block_size);
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_max_dynamic_shared_memory(std::shared_ptr<const DefaultExecutor> exec)
{
    int shmem_per_sm = 0;
    cudaDeviceGetAttribute(&shmem_per_sm,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                           exec->get_device_id());
    GKO_ASSERT_NO_CUDA_ERRORS(cudaFuncSetAttribute(
        apply_kernel<StopType, 9, true, PrecType, LogType, BatchMatrixType,
                     ValueType>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 99 /*%*/));
    cudaFuncAttributes funcattr;
    cudaFuncGetAttributes(&funcattr,
                          apply_kernel<StopType, 9, true, PrecType, LogType,
                                       BatchMatrixType, ValueType>);
    return funcattr.maxDynamicSharedSizeBytes;
}


template <typename ValueType, int n_shared, bool prec_shared, typename StopType,
          typename PrecType, typename LogType, typename BatchMatrixType>
void launch_apply_kernel(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::kernels::batch_bicgstab::storage_config& sconf,
    const settings<remove_complex<ValueType>>& settings, LogType& logger,
    PrecType& prec, const BatchMatrixType& mat,
    const cuda_type<ValueType>* const __restrict__ b_values,
    cuda_type<ValueType>* const __restrict__ x_values,
    cuda_type<ValueType>* const __restrict__ workspace_data,
    const int& block_size, const size_t& shared_size);
{
    apply_kernel<StopType, n_shared, prec_shared>
        <<<mat.num_batch_items, block_size, shared_size, exec->get_stream()>>>(
            sconf, settings.max_iterations, as_hip_type(settings.residual_tol),
            logger, prec, mat, b_values, x_values, workspace_data);
}


// begin
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_BICGSTAB_GET_NUM_THREADS_PER_BLOCK);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_BICGSTAB_GET_MAX_DYNAMIC_SHARED_MEMORY);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_0_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_1_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_2_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_3_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_4_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_5_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_6_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_7_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_8_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_9_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_9_TRUE);
// end


}  // namespace batch_bicgstab
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
