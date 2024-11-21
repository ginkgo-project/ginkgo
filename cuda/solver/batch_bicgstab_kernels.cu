// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_bicgstab_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/batch_multi_vector_kernels.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "common/cuda_hip/solver/batch_bicgstab_kernels.hpp"
#include "core/base/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "cuda/solver/batch_bicgstab_launch.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_bicgstab {


template <typename ValueType>
class kernel_caller {
public:
    using cuda_value_type = cuda_type<ValueType>;

    kernel_caller(std::shared_ptr<const DefaultExecutor> exec,
                  const settings<remove_complex<ValueType>> settings)
        : exec_{std::move(exec)}, settings_{settings}
    {}

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(
        LogType logger, const BatchMatrixType& mat, PrecType prec,
        const gko::batch::multi_vector::uniform_batch<const cuda_value_type>& b,
        const gko::batch::multi_vector::uniform_batch<cuda_value_type>& x) const
    {
        using real_type = gko::remove_complex<cuda_value_type>;
        const size_type num_batch_items = mat.num_batch_items;
        constexpr int align_multiple = 8;
        const int padded_num_rows =
            ceildiv(mat.num_rows, align_multiple) * align_multiple;
        const int shmem_per_blk =
            get_max_dynamic_shared_memory<StopType, PrecType, LogType,
                                          BatchMatrixType, cuda_value_type>(
                exec_);
        const int block_size =
            get_num_threads_per_block<StopType, PrecType, LogType,
                                      BatchMatrixType, cuda_value_type>(
                exec_, mat.num_rows);
        GKO_ASSERT(block_size >= 2 * config::warp_size);

        const size_t prec_size = PrecType::dynamic_work_size(
            padded_num_rows, mat.get_single_item_num_nnz());
        const auto sconf = gko::kernels::batch_bicgstab::compute_shared_storage<
            PrecType, cuda_value_type>(shmem_per_blk, padded_num_rows,
                                       mat.get_single_item_num_nnz(),
                                       b.num_rhs);
        const size_t shared_size =
            sconf.n_shared * padded_num_rows * sizeof(cuda_value_type) +
            (sconf.prec_shared ? prec_size : 0);
        auto workspace = gko::array<cuda_value_type>(
            exec_, sconf.gmem_stride_bytes * num_batch_items /
                       sizeof(cuda_value_type));
        GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(cuda_value_type) == 0);

        cuda_value_type* const workspace_data = workspace.get_data();

        // Template parameters launch_apply_kernel<StopType, n_shared,
        // prec_shared>
        if (sconf.prec_shared) {
            launch_apply_kernel<cuda_value_type, 9, true, StopType>(
                exec_, sconf, settings_, logger, prec, mat, b.values, x.values,
                workspace_data, block_size, shared_size);
        } else {
            switch (sconf.n_shared) {
            case 0:
                launch_apply_kernel<cuda_value_type, 0, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 1:
                launch_apply_kernel<cuda_value_type, 1, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 2:
                launch_apply_kernel<cuda_value_type, 2, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 3:
                launch_apply_kernel<cuda_value_type, 3, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 4:
                launch_apply_kernel<cuda_value_type, 4, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 5:
                launch_apply_kernel<cuda_value_type, 5, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 6:
                launch_apply_kernel<cuda_value_type, 6, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 7:
                launch_apply_kernel<cuda_value_type, 7, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 8:
                launch_apply_kernel<cuda_value_type, 8, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 9:
                launch_apply_kernel<cuda_value_type, 9, false, StopType>(
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
           const batch::BatchLinOp* mat, const batch::BatchLinOp* precon,
           const batch::MultiVector<ValueType>* b,
           batch::MultiVector<ValueType>* x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    auto dispatcher = batch::solver::create_dispatcher<ValueType>(
        kernel_caller<ValueType>(exec, settings), settings, mat, precon);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL);


}  // namespace batch_bicgstab
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
