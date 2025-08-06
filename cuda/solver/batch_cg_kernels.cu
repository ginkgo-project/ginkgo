// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_cg_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/batch_multi_vector_kernels.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "common/cuda_hip/solver/batch_cg_kernels.hpp"
#include "core/base/batch_instantiation.hpp"
#include "core/base/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "cuda/solver/batch_cg_launch.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_cg {


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
        const auto sconf =
            gko::kernels::batch_cg::compute_shared_storage<PrecType,
                                                           cuda_value_type>(
                shmem_per_blk, padded_num_rows, mat.get_single_item_num_nnz(),
                b.num_rhs);
        const size_t shared_size =
            sconf.n_shared * padded_num_rows * sizeof(cuda_value_type) +
            (sconf.prec_shared ? prec_size : 0);
        auto workspace = gko::array<cuda_value_type>(
            exec_, sconf.gmem_stride_bytes * num_batch_items /
                       sizeof(cuda_value_type));
        GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(cuda_value_type) == 0);

        cuda_value_type* const workspace_data = workspace.get_data();

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


template <typename ValueType, typename BatchMatrixType, typename PrecType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const settings<remove_complex<ValueType>>& settings,
           const BatchMatrixType* mat, const PrecType* precond,
           const batch::MultiVector<ValueType>* b,
           batch::MultiVector<ValueType>* x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    auto dispatcher = batch::solver::create_dispatcher<ValueType>(
        kernel_caller<ValueType>(exec, settings), settings, mat, precond);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_BATCH_VALUE_MATRIX_PRECONDITIONER_BASE(
    GKO_DECLARE_BATCH_CG_APPLY_KERNEL_WRAPPER);


}  // namespace batch_cg
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
