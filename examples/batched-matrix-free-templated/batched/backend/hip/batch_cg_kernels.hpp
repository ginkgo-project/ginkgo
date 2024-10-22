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
#include "batch_csr_kernels.hpp"
#include "batch_multi_vector_kernels.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "hip/base/config.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace batch_template {
namespace batch_single_kernels {
namespace batch_cg {


template <typename Group, typename BatchMatrixType_entry, typename PrecType,
          typename ValueType>
__device__ __forceinline__ void initialize(
    Group subgroup, const int num_rows, const BatchMatrixType_entry& mat_entry,
    batch::multi_vector::batch_item<const ValueType> b_global_entry,
    batch::multi_vector::batch_item<const ValueType> x_global_entry,
    batch::multi_vector::batch_item<ValueType> x_shared_entry,
    batch::multi_vector::batch_item<ValueType> r_shared_entry,
    const PrecType& prec_shared,
    batch::multi_vector::batch_item<ValueType> z_shared_entry,
    ValueType& rho_old_shared_entry,
    batch::multi_vector::batch_item<ValueType> p_shared_entry,
    typename gko::remove_complex<ValueType>& rhs_norms_sh)
{
    // copy x from global to shared memory
    // r = b
    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        x_shared_entry.values[iz] = x_global_entry.values[iz];
        r_shared_entry.values[iz] = b_global_entry.values[iz];
    }
    __syncthreads();

    // r = b - A*x
    advanced_apply(static_cast<ValueType>(-1.0), mat_entry,
                   batch::to_const(x_shared_entry), static_cast<ValueType>(1.0),
                   r_shared_entry);
    __syncthreads();

    // z = precond * r
    // TODO: add preconditioner again
    single_rhs_copy(num_rows, batch::to_const(r_shared_entry), z_shared_entry);
    __syncthreads();

    if (threadIdx.x / config::warp_size == 0) {
        // Compute norms of rhs
        single_rhs_compute_norm2(subgroup, num_rows,
                                 batch::to_const(b_global_entry), rhs_norms_sh);
    } else if (threadIdx.x / config::warp_size == 1) {
        // rho_old = r' * z
        single_rhs_compute_conj_dot(
            subgroup, num_rows, batch::to_const(r_shared_entry),
            batch::to_const(z_shared_entry), rho_old_shared_entry);
    }

    // p = z
    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        p_shared_entry.values[iz] = z_shared_entry.values[iz];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_p(
    const int num_rows, const ValueType& rho_new_shared_entry,
    const ValueType& rho_old_shared_entry,
    batch::multi_vector::batch_item<const ValueType> z_shared_entry,
    batch::multi_vector::batch_item<ValueType> p_shared_entry)
{
    for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
        const ValueType beta = rho_new_shared_entry / rho_old_shared_entry;
        p_shared_entry.values[li] =
            z_shared_entry.values[li] + beta * p_shared_entry.values[li];
    }
}


template <typename Group, typename ValueType>
__device__ __forceinline__ void update_x_and_r(
    Group subgroup, const int num_rows, const ValueType& rho_old_shared_entry,
    batch::multi_vector::batch_item<const ValueType> p_shared_entry,
    batch::multi_vector::batch_item<const ValueType> Ap_shared_entry,
    ValueType& alpha_shared_entry,
    batch::multi_vector::batch_item<ValueType> x_shared_entry,
    batch::multi_vector::batch_item<ValueType> r_shared_entry)
{
    if (threadIdx.x / config::warp_size == 0) {
        single_rhs_compute_conj_dot(
            subgroup, num_rows, batch::to_const(p_shared_entry),
            batch::to_const(Ap_shared_entry), alpha_shared_entry);
    }
    __syncthreads();

    for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
        const ValueType alpha = rho_old_shared_entry / alpha_shared_entry;
        x_shared_entry.values[li] += alpha * p_shared_entry.values[li];
        r_shared_entry.values[li] -= alpha * Ap_shared_entry.values[li];
    }
}


template <typename StopType, const int n_shared, const bool prec_shared_bool,
          typename PrecType, typename LogType, typename BatchMatrixType,
          typename ValueType>
__global__ void apply_kernel(
    const kernels::batch_cg::storage_config sconf, const int max_iter,
    const gko::remove_complex<ValueType> tol, LogType logger,
    PrecType prec_shared, const BatchMatrixType mat,
    batch::multi_vector::uniform_batch<const ValueType> b,
    batch::multi_vector::uniform_batch<ValueType> x,
    ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto num_batch_items = static_cast<int32>(mat.num_batch_items);
    const auto num_rows = static_cast<int32>(mat.num_rows);
    const auto num_rhs = static_cast<int32>(b.num_rhs);

    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto subgroup = group::tiled_partition<tile_size>(thread_block);

    for (size_type batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        const int gmem_offset =
            batch_id * sconf.gmem_stride_bytes / sizeof(ValueType);
        extern __shared__ char local_mem_sh[];

        const batch::multi_vector::batch_item<ValueType> r_sh{
            workspace + gmem_offset, num_rhs, num_rows, num_rhs};
        const batch::multi_vector::batch_item<ValueType> z_sh{
            r_sh.values + sconf.padded_vec_len, num_rhs, num_rows, num_rhs};
        const batch::multi_vector::batch_item<ValueType> p_sh{
            z_sh.values + sconf.padded_vec_len, num_rhs, num_rows, num_rhs};
        const batch::multi_vector::batch_item<ValueType> Ap_sh{
            p_sh.values + sconf.padded_vec_len, num_rhs, num_rows, num_rhs};
        const batch::multi_vector::batch_item<ValueType> x_sh{
            Ap_sh.values + sconf.padded_vec_len, num_rhs, num_rows, num_rhs};

        ValueType* prec_work_sh = x_sh.values + sconf.padded_vec_len;

        __shared__ uninitialized_array<ValueType, 1> rho_old_sh;
        __shared__ uninitialized_array<ValueType, 1> rho_new_sh;
        __shared__ uninitialized_array<ValueType, 1> alpha_sh;
        __shared__ real_type norms_rhs_sh[1];
        __shared__ real_type norms_res_sh[1];

        const auto mat_entry = batch::extract_batch_item(mat, batch_id);
        const auto b_global_entry = batch::extract_batch_item(b, batch_id);
        auto x_global_entry = batch::extract_batch_item(x, batch_id);

        // generate preconditioner
        // prec_shared.generate(batch_id, mat_entry, prec_work_sh);

        // initialization
        // compute b norms
        // r = b - A*x
        // z = precond*r
        // rho_old = r' * z (' is for hermitian transpose)
        // p = z
        initialize(subgroup, num_rows, mat_entry, b_global_entry,
                   batch::to_const(x_global_entry), x_sh, r_sh, prec_shared,
                   z_sh, rho_old_sh[0], p_sh, norms_rhs_sh[0]);
        __syncthreads();

        // stopping criterion object
        StopType stop(tol, norms_rhs_sh);

        int iter = 0;
        for (; iter < max_iter; iter++) {
            norms_res_sh[0] = sqrt(abs(rho_old_sh[0]));
            __syncthreads();
            if (stop.check_converged(norms_res_sh)) {
                logger.log_iteration(batch_id, iter, norms_res_sh[0]);
                break;
            }

            // Ap = A * p
            simple_apply(mat_entry, batch::to_const(p_sh), Ap_sh);
            __syncthreads();

            // alpha = rho_old / (p' * Ap)
            // x = x + alpha * p
            // r = r - alpha * Ap
            update_x_and_r(subgroup, num_rows, rho_old_sh[0],
                           batch::to_const(p_sh), batch::to_const(Ap_sh),
                           alpha_sh[0], x_sh, r_sh);
            __syncthreads();

            // z = precond * r
            // TODO: add preconditioner again
            single_rhs_copy(num_rows, batch::to_const(r_sh), z_sh);
            __syncthreads();

            if (threadIdx.x / config::warp_size == 0) {
                // rho_new =  (r)' * (z)
                single_rhs_compute_conj_dot(
                    subgroup, num_rows, batch::to_const(r_sh),
                    batch::to_const(z_sh), rho_new_sh[0]);
            }
            __syncthreads();

            // beta = rho_new / rho_old
            // p = z + beta * p
            update_p(num_rows, rho_new_sh[0], rho_old_sh[0],
                     batch::to_const(z_sh), p_sh);
            __syncthreads();

            // rho_old = rho_new
            if (threadIdx.x == 0) {
                rho_old_sh[0] = rho_new_sh[0];
            }
            __syncthreads();
        }

        logger.log_iteration(batch_id, iter, norms_res_sh[0]);

        // copy x back to global memory
        single_rhs_copy(num_rows, batch::to_const(x_sh), x_global_entry);
        __syncthreads();
    }
}
}  // namespace batch_cg
}  // namespace batch_single_kernels

namespace batch_cg {


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

template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& settings,
    const Op mat, batch::multi_vector::uniform_batch<const ValueType> b,
    batch::multi_vector::uniform_batch<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    using PrecType = batch_preconditioner::Identity<ValueType>;
    using StopType = batch_stop::SimpleAbsResidual<ValueType>;
    using real_type = gko::remove_complex<ValueType>;
    const size_type num_batch_items = mat.num_batch_items;
    constexpr int align_multiple = 8;
    const int padded_num_rows =
        ceildiv(mat.num_rows, align_multiple) * align_multiple;
    int shmem_per_blk = 0;
    GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
        &shmem_per_blk, hipDeviceAttributeMaxSharedMemoryPerBlock,
        exec->get_device_id()));
    const int block_size = get_num_threads_per_block<Op>(exec, mat.num_rows);
    GKO_ASSERT(block_size >= 2 * config::warp_size);
    GKO_ASSERT(block_size % config::warp_size == 0);

    // Returns amount required in bytes
    const size_t prec_size = PrecType::dynamic_work_size(padded_num_rows, -1);
    const auto sconf =
        ::gko::kernels::batch_cg::compute_shared_storage<PrecType, ValueType>(
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

    batch_single_kernels::batch_cg::apply_kernel<StopType, 0, false>
        <<<mat.num_batch_items, block_size, shared_size, exec->get_stream()>>>(
            sconf, settings.max_iterations, settings.residual_tol, logger, prec,
            mat, b, x, workspace_data);
}
}  // namespace batch_cg
}  // namespace batch_template
}  // namespace hip
}  // namespace kernels
}  // namespace gko

#else

namespace gko {
namespace kernels {
namespace hip {
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
}  // namespace hip
}  // namespace kernels
}  // namespace gko

#endif
