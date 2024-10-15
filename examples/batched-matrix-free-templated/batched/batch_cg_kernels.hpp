// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_EXAMPLES_BATCHED - MATRIX - FREE - \
    TEMPLATED_BATCHED_BATCH_CG_KERNELS_HPP_
#define GKO_EXAMPLES_BATCHED \
    -MATRIX - FREE - TEMPLATED_BATCHED_BATCH_CG_KERNELS_HPP_


#include <hip/hip_runtime.h>

#include <batch_criteria.hpp>
#include <batch_identity.hpp>
#include <batch_logger.hpp>
#include <batch_multi_vector.hpp>

#include <ginkgo/config.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>

#include "core/base/kernel_declaration.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


// TODO: update when splitting compilation
constexpr bool cg_no_shared_vecs = true;


namespace gko {
namespace kernels {
namespace batch_cg {


template <typename RealType>
struct settings {
    static_assert(std::is_same<RealType, remove_complex<RealType>>::value,
                  "Template parameter must be a real type");
    int max_iterations;
    RealType residual_tol;
    batch::stop::tolerance_type tol_type;
};

template <typename ValueType>
inline int local_memory_requirement(const int num_rows, const int num_rhs)
{
    return (5 * num_rows * num_rhs + 3 * num_rhs) * sizeof(ValueType) +
           2 * num_rhs * sizeof(typename gko::remove_complex<ValueType>);
}


}  // namespace batch_cg


namespace omp {
namespace batch_tempalte {
namespace batch_cg {
template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op* mat, multi_vector_view<const ValueType> b,
    multi_vector_view<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{}
}  // namespace batch_cg
}  // namespace batch_tempalte
}  // namespace omp

namespace cuda {
namespace batch_tempalte {
namespace batch_cg {
template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op* mat, multi_vector_view<const ValueType> b,
    multi_vector_view<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{}
}  // namespace batch_cg
}  // namespace batch_tempalte
}  // namespace cuda


constexpr int max_num_rhs = 1;


namespace reference {
namespace batch_tempalte {
namespace batch_cg {


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
void batch_entry_cg_impl(
    const kernels::batch_cg::settings<remove_complex<ValueType>>& settings,
    LogType logger, PrecType prec, const BatchMatrixType& a,
    multi_vector_view_item<const ValueType> b,
    multi_vector_view_item<ValueType> x, const size_type batch_item_id,
    unsigned char* const local_space)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto num_rows = static_cast<int32>(a.get_size().get_common_size()[0]);
    const auto num_rhs = static_cast<int32>(b.num_rhs);
    GKO_ASSERT(num_rhs <= max_num_rhs);

    unsigned char* const shared_space = local_space;
    ValueType* const r = reinterpret_cast<ValueType*>(shared_space);
    ValueType* const z = r + num_rows * num_rhs;
    ValueType* const p = z + num_rows * num_rhs;
    ValueType* const Ap = p + num_rows * num_rhs;
    ValueType* const prec_work = Ap + num_rows * num_rhs;
    ValueType rho_old[max_num_rhs];
    ValueType rho_new[max_num_rhs];
    ValueType alpha[max_num_rhs];
    ValueType temp[max_num_rhs];
    real_type norms_rhs[max_num_rhs];
    real_type norms_res[max_num_rhs];

    const auto A_entry = a.extract_batch_item(batch_item_id);
    const auto b_entry = b;
    const auto x_entry = x;

    const multi_vector_view_item<ValueType> r_entry{r, num_rhs, num_rows,
                                                    num_rhs};
    const multi_vector_view_item<ValueType> z_entry{z, num_rhs, num_rows,
                                                    num_rhs};
    const multi_vector_view_item<ValueType> p_entry{p, num_rhs, num_rows,
                                                    num_rhs};
    const multi_vector_view_item<ValueType> Ap_entry{Ap, num_rhs, num_rows,
                                                     num_rhs};
    const multi_vector_view_item<ValueType> rho_old_entry{rho_old, num_rhs, 1,
                                                          num_rhs};
    const multi_vector_view_item<ValueType> rho_new_entry{rho_new, num_rhs, 1,
                                                          num_rhs};
    const multi_vector_view_item<ValueType> alpha_entry{alpha, num_rhs, 1,
                                                        num_rhs};
    const multi_vector_view_item<ValueType> rhs_norms_entry{norms_rhs, num_rhs,
                                                            1, num_rhs};
    const multi_vector_view_item<ValueType> res_norms_entry{norms_res, num_rhs,
                                                            1, num_rhs};

    // generate preconditioner
    prec.generate(batch_item_id, A_entry, prec_work);

    // initialization
    // compute b norms
    // r = b - A*x
    // p = z = Ap = 0
    // rho_old = 1, rho_new = 0
    // initialize(A_entry, b_entry, batch::to_const(x_entry), rho_old_entry,
    // rho_new_entry, r_entry, p_entry, z_entry, Ap_entry,
    // rhs_norms_entry);

    // stopping criterion object
    StopType stop(settings.residual_tol, rhs_norms_entry.values);

    int iter = 0;

    while (true) {
        // z = precond * r
        prec.apply(r_entry, z_entry);

        // rho_new =  < r , z > = (r)' * (z)
        // compute_conj_dot_product_kernel<ValueType>(
        // batch::to_const(r_entry), batch::to_const(z_entry),
        // rho_new_entry);
        ++iter;
        // use implicit residual norms
        // res_norms_entry.values[0] = sqrt(abs(rho_new_entry.values[0]));

        // if (iter >= settings.max_iterations ||
        //     stop.check_converged(res_norms_entry.values)) {
        //     logger.log_iteration(batch_item_id, iter,
        //                          res_norms_entry.values[0]);
        //     break;
        // }

        // beta = (rho_new / rho_old)
        // p = z + beta * p
        // update_p(batch::to_const(rho_new_entry),
        // batch::to_const(rho_old_entry),
        // batch::to_const(z_entry), p_entry);

        // Ap = A * p
        apply(A_entry, p_entry, Ap_entry);

        // temp= rho_old / (p' * Ap)
        // x = x + temp * p
        // r = r - temp * Ap
        // update_x_and_r(
        // batch::to_const(rho_new_entry), batch::to_const(p_entry),
        // batch::to_const(Ap_entry), alpha_entry, x_entry, r_entry);

        // rho_old = rho_new
        // copy_kernel(batch::to_const(rho_new_entry), rho_old_entry);
    }

    logger.log_iteration(batch_item_id, iter, res_norms_entry.values[0]);
}


template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op* mat, multi_vector_view<const ValueType> b,
    multi_vector_view<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const size_type num_batch_items = mat->get_size().get_num_batch_items();
    const auto num_rows = mat->get_size().get_common_size()[0];
    const auto num_rhs = b.num_rhs;
    if (num_rhs > 1) {
        GKO_NOT_IMPLEMENTED;
    }

    const size_type local_size_bytes =
        kernels::batch_cg::local_memory_requirement<ValueType>(num_rows,
                                                               num_rhs);
    array<unsigned char> local_space(exec, local_size_bytes);

    batch_log::SimpleFinalLogger<real_type> logger(
        logdata.res_norms.get_data(), logdata.iter_counts.get_data());

    auto prec = batch_preconditioner::Identity<ValueType>();

    for (size_type batch_id = 0; batch_id < num_batch_items; batch_id++) {
        batch_entry_cg_impl<batch_stop::SimpleRelResidual<ValueType>>(
            options, logger, prec, *mat, b.extract_batch_item(batch_id),
            x.extract_batch_item(batch_id), batch_id, local_space.get_data());
    }
}
}  // namespace batch_cg
}  // namespace batch_tempalte
}  // namespace reference

namespace hip {
namespace batch_tempalte {
namespace batch_cg {
namespace config {
constexpr int warp_size = 32;
}


struct storage_config {
    // preconditioner storage
    bool prec_shared;
    // total number of shared vectors
    int n_shared;
    // number of vectors in global memory
    int n_global;
    // global stride from one batch entry to the next
    int gmem_stride_bytes;
    // padded vector length
    int padded_vec_len;
};


template <int align_bytes>
void set_gmem_stride_bytes(storage_config& sconf,
                           const int multi_vector_size_bytes,
                           const int prec_storage_bytes)
{
    int gmem_stride = sconf.n_global * multi_vector_size_bytes;
    if (!sconf.prec_shared) {
        gmem_stride += prec_storage_bytes;
    }
    // align global memory chunks
    sconf.gmem_stride_bytes = ceildiv(gmem_stride, align_bytes) * align_bytes;
}


template <typename Prectype, typename ValueType, int align_bytes = 32>
storage_config compute_shared_storage(const int available_shared_mem,
                                      const int num_rows, const int num_nz,
                                      const int num_rhs)
{
    using real_type = remove_complex<ValueType>;
    const int vec_bytes = num_rows * num_rhs * sizeof(ValueType);
    const int num_main_vecs = 5;
    const int prec_storage = Prectype::dynamic_work_size(num_rows, num_nz);
    int rem_shared = available_shared_mem;
    // Set default values. Initially all vecs are in global memory.
    // {prec_shared, n_shared, n_global, gmem_stride_bytes, padded_vec_len}
    storage_config sconf{false, 0, num_main_vecs, 0, num_rows};
    // If available shared mem is zero, set all vecs to global.
    if (rem_shared <= 0 || true) {
        set_gmem_stride_bytes<align_bytes>(sconf, vec_bytes, prec_storage);
        return sconf;
    }
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
__global__ void apply_kernel(const storage_config sconf, const int max_iter,
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

namespace dpcpp {
namespace batch_tempalte {
namespace batch_cg {
template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op* mat, multi_vector_view<const ValueType> b,
    multi_vector_view<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{}
}  // namespace batch_cg
}  // namespace batch_tempalte
}  // namespace dpcpp


}  // namespace kernels
}  // namespace gko


#endif  // GKO_EXAMPLES_BATCHED-MATRIX-FREE-TEMPLATED_BATCHED_BATCH_CG_KERNELS_HPP_
