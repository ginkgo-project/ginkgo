// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#include "../../batch_cg_settings.hpp"

#if GINKGO_BUILD_CUDA

#include <ginkgo/core/log/batch_logger.hpp>

#include "../../batch_criteria.hpp"
#include "../../batch_identity.hpp"
#include "../../batch_logger.hpp"
#include "../cuda_hip/batch_cg_kernels.hpp"
#include "../cuda_hip/batch_csr_kernels.hpp"
#include "../cuda_hip/batch_multi_vector_kernels.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "cuda/base/config.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_template {
namespace batch_cg {


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_num_threads_per_block(std::shared_ptr<const DefaultExecutor> exec,
                              const int num_rows)
{
    int num_warps = std::max(num_rows / 4, 2);
    constexpr int warp_sz = static_cast<int>(config::warp_size);
    const int min_block_size = 2 * warp_sz;
    const int device_max_threads =
        (std::max(num_rows, min_block_size) / warp_sz) * warp_sz;
    cudaFuncAttributes funcattr;
    cudaFuncGetAttributes(
        &funcattr,
        batch_single_kernels::batch_cg::apply_kernel<
            StopType, 5, true, PrecType, LogType, BatchMatrixType, ValueType>);
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
        batch_single_kernels::batch_cg::apply_kernel<
            StopType, 5, true, PrecType, LogType, BatchMatrixType, ValueType>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 99 /*%*/));
    cudaFuncAttributes funcattr;
    cudaFuncGetAttributes(
        &funcattr,
        batch_single_kernels::batch_cg::apply_kernel<
            StopType, 5, true, PrecType, LogType, BatchMatrixType, ValueType>);
    return funcattr.maxDynamicSharedSizeBytes;
}


template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op mat, batch::multi_vector::uniform_batch<const ValueType> b,
    batch::multi_vector::uniform_batch<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    using real_type = gko::remove_complex<ValueType>;
    using PrecType = batch_preconditioner::Identity<ValueType>;
    using StopType = batch_stop::SimpleAbsResidual<ValueType>;
    using LogType = batch_log::SimpleFinalLogger<real_type>;
    const size_type num_batch_items = mat.num_batch_items;
    constexpr int align_multiple = 8;
    const int padded_num_rows =
        ceildiv(mat.num_rows, align_multiple) * align_multiple;
    const int shmem_per_blk =
        get_max_dynamic_shared_memory<StopType, PrecType, LogType, Op,
                                      ValueType>(exec);
    const int block_size =
        get_num_threads_per_block<StopType, PrecType, LogType, Op, ValueType>(
            exec, mat.num_rows);
    GKO_ASSERT(block_size >= 2 * config::warp_size);

    const size_t prec_size = PrecType::dynamic_work_size(padded_num_rows, -1);
    const auto sconf =
        gko::kernels::batch_cg::compute_shared_storage<PrecType, ValueType>(
            shmem_per_blk, padded_num_rows, -1, b.num_rhs);
    const size_t shared_size =
        sconf.n_shared * padded_num_rows * sizeof(ValueType) +
        (sconf.prec_shared ? prec_size : 0);
    auto workspace = gko::array<ValueType>(
        exec, sconf.gmem_stride_bytes * num_batch_items / sizeof(ValueType));
    GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(ValueType) == 0);

    ValueType* const workspace_data = workspace.get_data();

    auto prec = PrecType();
    auto logger =
        LogType(logdata.res_norms.get_data(), logdata.iter_counts.get_data());
    batch_single_kernels::batch_cg::apply_kernel<StopType, 0, false>
        <<<mat.num_batch_items, block_size, shared_size, exec->get_stream()>>>(
            sconf, options.max_iterations, options.residual_tol, logger, prec,
            mat, b, x, workspace_data);
}


}  // namespace batch_cg
}  // namespace batch_template
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#else

namespace gko {
namespace kernels {
namespace cuda {
namespace batch_template {
namespace batch_cg {


template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op mat, batch::multi_vector::uniform_batch<const ValueType> b,
    batch::multi_vector::uniform_batch<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
    GKO_NOT_IMPLEMENTED;


}  // namespace batch_cg
}  // namespace batch_template
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

#endif
