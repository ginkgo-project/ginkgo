// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_bicgstab_kernels.hpp"


#include <hip/hip_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "hip/base/batch_struct.hip.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/thrust.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {

/**
 * @brief The batch Bicgstab solver namespace.
 *
 * @ingroup batch_bicgstab
 */
namespace batch_bicgstab {


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


template <typename T>
using settings = gko::kernels::batch_bicgstab::settings<T>;


template <typename ValueType, int n_shared, bool prec_shared, typename StopType,
          typename PrecType, typename LogType, typename BatchMatrixType>
void launch_apply_kernel(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::kernels::batch_bicgstab::storage_config& sconf,
    const settings<remove_complex<ValueType>>& settings, LogType& logger,
    PrecType& prec, const BatchMatrixType& mat,
    const hip_type<ValueType>* const __restrict__ b_values,
    hip_type<ValueType>* const __restrict__ x_values,
    hip_type<ValueType>* const __restrict__ workspace_data,
    const int& block_size, const size_t& shared_size);

#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH(_vtype, _n_shared, _prec_shared, \
                                          mat_t, log_t, pre_t, stop_t)     \
    void launch_apply_kernel<_vtype, _n_shared, _prec_shared,              \
                             stop_t<hip_type<_vtype>>>(                    \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const gko::kernels::batch_bicgstab::storage_config& sconf,         \
        const settings<remove_complex<_vtype>>& settings,                  \
        log_t<hip_type<gko::remove_complex<_vtype>>>& logger,              \
        pre_t<hip_type<_vtype>>& prec,                                     \
        const mat_t<const hip_type<_vtype>>& mat,                          \
        const hip_type<_vtype>* const __restrict__ b_values,               \
        hip_type<_vtype>* const __restrict__ x_values,                     \
        hip_type<_vtype>* const __restrict__ workspace_data,               \
        const int& block_size, const size_t& shared_size)

#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_0_FALSE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 0, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_1_FALSE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 1, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_2_FALSE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 2, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_3_FALSE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 3, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_4_FALSE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 4, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_5_FALSE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 5, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_6_FALSE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 6, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_7_FALSE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 7, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_8_FALSE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 8, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_9_FALSE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 9, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_9_TRUE(_VTYPE) \
    GKO_BATCH_INSTANTIATE(declare_launch, _vtype, 9, true)


template <typename ValueType>
class kernel_caller {
public:
    using hip_value_type = hip_type<ValueType>;

    kernel_caller(std::shared_ptr<const DefaultExecutor> exec,
                  const settings<remove_complex<ValueType>> settings)
        : exec_{exec}, settings_{settings}
    {}

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(
        LogType logger, const BatchMatrixType& mat, PrecType prec,
        const gko::batch::multi_vector::uniform_batch<const hip_value_type>& b,
        const gko::batch::multi_vector::uniform_batch<hip_value_type>& x) const
    {
        using real_type = gko::remove_complex<hip_value_type>;
        const size_type num_batch_items = mat.num_batch_items;
        constexpr int align_multiple = 8;
        const int padded_num_rows =
            ceildiv(mat.num_rows, align_multiple) * align_multiple;
        int shmem_per_blk = 0;
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &shmem_per_blk, hipDeviceAttributeMaxSharedMemoryPerBlock,
            exec_->get_device_id()));
        const int block_size =
            get_num_threads_per_block<BatchMatrixType>(exec_, mat.num_rows);
        GKO_ASSERT(block_size >= 2 * config::warp_size);
        GKO_ASSERT(block_size % config::warp_size == 0);

        // Returns amount required in bytes
        const size_t prec_size = PrecType::dynamic_work_size(
            padded_num_rows, mat.get_single_item_num_nnz());
        const auto sconf = gko::kernels::batch_bicgstab::compute_shared_storage<
            PrecType, hip_value_type>(shmem_per_blk, padded_num_rows,
                                      mat.get_single_item_num_nnz(), b.num_rhs);
        const size_t shared_size =
            sconf.n_shared * padded_num_rows * sizeof(hip_value_type) +
            (sconf.prec_shared ? prec_size : 0);
        auto workspace = gko::array<hip_value_type>(
            exec_,
            sconf.gmem_stride_bytes * num_batch_items / sizeof(hip_value_type));
        GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(hip_value_type) == 0);

        hip_value_type* const workspace_data = workspace.get_data();

        // Template parameters launch_apply_kernel<StopType, n_shared,
        // prec_shared)
        if (sconf.prec_shared) {
            launch_apply_kernel<ValueType, 9, true, StopType>(
                exec_, sconf, settings_, logger, prec, mat, b.values, x.values,
                workspace_data, block_size, shared_size);
        } else {
            switch (sconf.n_shared) {
            case 0:
                launch_apply_kernel<ValueType, 0, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 1:
                launch_apply_kernel<ValueType, 1, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 2:
                launch_apply_kernel<ValueType, 2, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 3:
                launch_apply_kernel<ValueType, 3, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 4:
                launch_apply_kernel<ValueType, 4, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 5:
                launch_apply_kernel<ValueType, 5, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 6:
                launch_apply_kernel<ValueType, 6, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 7:
                launch_apply_kernel<ValueType, 7, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 8:
                launch_apply_kernel<ValueType, 8, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 9:
                launch_apply_kernel<ValueType, 9, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            default:
                GKO_NOT_IMPLEMENTED;
            }
        }
    }

private:
    std::shared_ptr<const DefaultExecutor> exec_;
    const settings<remove_complex<ValueType>> settings_;
};


}  // namespace batch_bicgstab
}  // namespace hip
}  // namespace kernels
}  // namespace gko
