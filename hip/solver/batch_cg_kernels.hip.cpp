// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_cg_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/batch_multi_vector_kernels.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "common/cuda_hip/solver/batch_cg_kernels.hpp"
#include "core/base/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "hip/solver/batch_cg_launch.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
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
        const auto sconf =
            gko::kernels::batch_cg::compute_shared_storage<PrecType,
                                                           hip_value_type>(
                shmem_per_blk, padded_num_rows, mat.get_single_item_num_nnz(),
                b.num_rhs);
        const size_t shared_size =
            sconf.n_shared * padded_num_rows * sizeof(hip_value_type) +
            (sconf.prec_shared ? prec_size : 0);
        auto workspace = gko::array<hip_value_type>(
            exec_,
            sconf.gmem_stride_bytes * num_batch_items / sizeof(hip_value_type));
        GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(hip_value_type) == 0);

        hip_value_type* const workspace_data = workspace.get_data();

        // Template parameters launch_apply_kernel<ValueType, n_shared,
        // prec_shared, StopType>
        if (sconf.prec_shared) {
            launch_apply_kernel<ValueType, 5, true, StopType>(
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
            default:
                GKO_NOT_IMPLEMENTED;
            }
        }
    }

private:
    std::shared_ptr<const DefaultExecutor> exec_;
    const settings<remove_complex<ValueType>> settings_;
};


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const settings<remove_complex<ValueType>>& settings,
           const batch::BatchLinOp* const mat,
           const batch::BatchLinOp* const precon,
           const batch::MultiVector<ValueType>* const b,
           batch::MultiVector<ValueType>* const x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    using hip_value_type = hip_type<ValueType>;
    auto dispatcher = batch::solver::create_dispatcher<ValueType>(
        kernel_caller<hip_value_type>(exec, settings), settings, mat, precon);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG_APPLY_KERNEL);


}  // namespace batch_cg
}  // namespace hip
}  // namespace kernels
}  // namespace gko
