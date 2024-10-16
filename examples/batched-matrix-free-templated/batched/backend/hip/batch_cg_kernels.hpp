// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/config.hpp>

#if GINKGO_BUILD_HIP

#include <hip/hip_runtime.h>

#include <ginkgo/core/log/batch_logger.hpp>

#include "../../batch_cg_settings.hpp"
#include "../../batch_criteria.hpp"
#include "../../batch_identity.hpp"
#include "../../batch_logger.hpp"
#include "../../batch_multi_vector.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace batch_tempalte {
namespace batch_cg {
namespace config {


constexpr int warp_size = 32;


}


template <typename BatchMatrixType>
int get_num_threads_per_block(std::shared_ptr<const DefaultExecutor> exec,
                              const int num_rows)
{
    int num_warps = std::max(num_rows / 4, 2);
    constexpr int warp_sz = static_cast<int>(config::warp_size);
    const int min_block_size = 2 * warp_sz;
    const int device_max_threads =
        ((std::max(num_rows, min_block_size)) / warp_sz) * warp_sz;
    // This value has been taken from ROCm docs. This is the number of registers
    // that maximizes the occupancy on an AMD GPU (MI200). HIP does not have an
    // API to query the number of registers a function uses.
    const int num_regs_used_per_thread = 64;
    int max_regs_blk = 0;
    GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
        &max_regs_blk, hipDeviceAttributeMaxRegistersPerBlock,
        exec->get_device_id()));
    int max_threads_regs = (max_regs_blk / num_regs_used_per_thread);
    max_threads_regs = (max_threads_regs / warp_sz) * warp_sz;
    int max_threads = std::min(max_threads_regs, device_max_threads);
    max_threads = max_threads <= 1024 ? max_threads : 1024;
    return std::max(std::min(num_warps * warp_sz, max_threads), min_block_size);
}


template <typename StopType, const int n_shared, const bool prec_shared_bool,
          typename PrecType, typename LogType, typename BatchMatrixType,
          typename ValueType>
__global__ void apply_kernel(const kernels::batch_cg::storage_config sconf,
                             const int max_iter,
                             const gko::remove_complex<ValueType> tol,
                             LogType logger, PrecType prec_shared,
                             const BatchMatrixType mat,
                             multi_vector_view<const ValueType> b,
                             multi_vector_view<ValueType> x,
                             ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto num_batch_items =
        static_cast<int32>(mat.get_size().get_num_batch_items());
    const auto num_rows =
        static_cast<int32>(mat.get_size().get_common_size()[0]);
    const auto num_rhs = static_cast<int32>(b.num_rhs);

    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto subgroup = group::tiled_partition<tile_size>(thread_block);

    for (size_type batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        const int gmem_offset =
            batch_id * sconf.gmem_stride_bytes / sizeof(ValueType);
        extern __shared__ char local_mem_sh[];

        const multi_vector_view_item<ValueType> r_sh{
            workspace + gmem_offset, num_rhs, num_rows, num_rhs};
        const multi_vector_view_item<ValueType> z_sh{
            r_sh.values + sconf.padded_vec_len, num_rhs, num_rows, num_rhs};
        const multi_vector_view_item<ValueType> p_sh{
            z_sh.values + sconf.padded_vec_len, num_rhs, num_rows, num_rhs};
        const multi_vector_view_item<ValueType> Ap_sh{
            p_sh.values + sconf.padded_vec_len, num_rhs, num_rows, num_rhs};
        const multi_vector_view_item<ValueType> x_sh{
            Ap_sh.values + sconf.padded_vec_len, num_rhs, num_rows, num_rhs};

        ValueType* prec_work_sh = x_sh.values + sconf.padded_vec_len;

        __shared__ uninitialized_array<ValueType, 1> rho_old_sh;
        __shared__ uninitialized_array<ValueType, 1> rho_new_sh;
        __shared__ uninitialized_array<ValueType, 1> alpha_sh;
        __shared__ real_type norms_rhs_sh[1];
        __shared__ real_type norms_res_sh[1];

        const auto mat_entry = mat.extract_batch_item(batch_id);
        const auto b_global_entry = b.extract_batch_item(batch_id);
        auto x_global_entry = x.extract_batch_item(batch_id);

        // generate preconditioner
        prec_shared.generate(batch_id, mat_entry, prec_work_sh);

        // stopping criterion object
        StopType stop(tol, norms_rhs_sh);

        int iter = 0;
        for (; iter < max_iter; iter++) {
            // Ap = A * p
            apply(mat_entry, p_sh, Ap_sh);
            __syncthreads();
        }

        logger.log_iteration(batch_id, iter, norms_res_sh[0]);
        __syncthreads();
    }
}


template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& settings,
    const Op* mat, multi_vector_view<const ValueType> b,
    multi_vector_view<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    using PrecType = batch_preconditioner::Identity<ValueType>;
    using StopType = batch_stop::SimpleAbsResidual<ValueType>;
    using real_type = gko::remove_complex<ValueType>;
    const size_type num_batch_items = mat->get_size().get_num_batch_items();
    constexpr int align_multiple = 8;
    const int padded_num_rows =
        ceildiv(mat->get_size().get_common_size()[0], align_multiple) *
        align_multiple;
    int shmem_per_blk = 0;
    GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
        &shmem_per_blk, hipDeviceAttributeMaxSharedMemoryPerBlock,
        exec->get_device_id()));
    const int block_size = get_num_threads_per_block<Op>(
        exec, mat->get_size().get_common_size()[0]);
    GKO_ASSERT(block_size >= 2 * config::warp_size);
    GKO_ASSERT(block_size % config::warp_size == 0);

    // Returns amount required in bytes
    const size_t prec_size = PrecType::dynamic_work_size(padded_num_rows, -1);
    const auto sconf = compute_shared_storage<PrecType, ValueType>(
        shmem_per_blk, padded_num_rows, -1, b.num_rhs);
    const size_t shared_size =
        sconf.n_shared * padded_num_rows * sizeof(ValueType) +
        (sconf.prec_shared ? prec_size : 0);
    auto workspace = gko::array<ValueType>(
        exec, sconf.gmem_stride_bytes * num_batch_items / sizeof(ValueType));
    GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(ValueType) == 0);

    ValueType* const workspace_data = workspace.get_data();

    auto prec = PrecType();
    auto logger = batch_log::SimpleFinalLogger<real_type>(
        logdata.res_norms.get_data(), logdata.iter_counts.get_data());

    apply_kernel<StopType, 0, false>
        <<<mat->get_size().get_num_batch_items(), block_size, shared_size,
           exec->get_stream()>>>(sconf, settings.max_iterations,
                                 settings.residual_tol, logger, prec, *mat, b,
                                 x, workspace_data);
}
}  // namespace batch_cg
}  // namespace batch_tempalte
}  // namespace hip
}  // namespace kernels
}  // namespace gko

#else


namespace gko::kernels::hip::batch_tempalte::batch_cg {
template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& settings,
    const Op* mat, multi_vector_view<const ValueType> b,
    multi_vector_view<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
    GKO_NOT_IMPLEMENTED;
}  // namespace gko::kernels::hip::batch_tempalte::batch_cg

#endif
