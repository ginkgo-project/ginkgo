// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_cg_kernels.hpp"

#include <sycl/sycl.hpp>

#include <ginkgo/core/solver/batch_cg.hpp>

#include "core/base/batch_instantiation.hpp"
#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/base/math.hpp"
#include "dpcpp/base/types.hpp"
#include "dpcpp/matrix/batch_struct.hpp"
#include "dpcpp/solver/batch_cg_kernels.hpp"
#include "dpcpp/solver/batch_cg_launch.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_cg {


template <typename T>
using settings = gko::kernels::batch_cg::settings<T>;


int get_group_size(int value, int subgroup_size = config::warp_size)
{
    int num_sg = ceildiv(value, subgroup_size);
    return num_sg * subgroup_size;
}


template <typename ValueType>
class kernel_caller {
public:
    using sycl_value_type = sycl_type<ValueType>;
    kernel_caller(std::shared_ptr<const DefaultExecutor> exec,
                  const settings<remove_complex<ValueType>> settings)
        : exec_{std::move(exec)}, settings_{settings}
    {}

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(
        LogType logger, const BatchMatrixType& mat, PrecType prec,
        const gko::batch::multi_vector::uniform_batch<const sycl_value_type>& b,
        const gko::batch::multi_vector::uniform_batch<sycl_value_type>& x) const
    {
        using real_type = typename gko::remove_complex<ValueType>;
        const size_type num_batch_items = mat.num_batch_items;
        const auto num_rows = mat.num_rows;
        const auto num_rhs = b.num_rhs;
        GKO_ASSERT(num_rhs == 1);

        auto device = exec_->get_queue()->get_device();
        auto max_group_size =
            device.get_info<sycl::info::device::max_work_group_size>();
        int group_size = get_group_size(num_rows);
        group_size = std::min(
            std::max(group_size, static_cast<int>(2 * config::warp_size)),
            static_cast<int>(max_group_size));

        // reserve 3 for intermediate rho,
        // alpha and two norms
        // If the value available is negative, then set it to 0
        const int static_var_mem =
            3 * sizeof(sycl_value_type) + 2 * sizeof(real_type);
        int shmem_per_blk = std::max(
            static_cast<int>(
                device.get_info<sycl::info::device::local_mem_size>()) -
                static_var_mem,
            0);
        const int padded_num_rows = num_rows;
        const size_type prec_size = PrecType::dynamic_work_size(
            padded_num_rows, mat.get_single_item_num_nnz());
        const auto sconf =
            gko::kernels::batch_cg::compute_shared_storage<PrecType,
                                                           sycl_value_type>(
                shmem_per_blk, padded_num_rows, mat.get_single_item_num_nnz(),
                b.num_rhs);
        const size_t shared_size = sconf.n_shared * padded_num_rows +
                                   (sconf.prec_shared ? prec_size : 0);
        auto workspace = gko::array<sycl_value_type>(
            exec_, sconf.gmem_stride_bytes * num_batch_items /
                       sizeof(sycl_value_type));
        GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(sycl_value_type) == 0);

        sycl_value_type* const workspace_data = workspace.get_data();
        int n_shared_total = sconf.n_shared + int(sconf.prec_shared);

        // template
        // launch_apply_kernel<StopType, subgroup_size, n_shared_total>
        if (num_rows <= 32 && n_shared_total == 6) {
            launch_apply_kernel<ValueType, StopType, 16, 6>(
                exec_, sconf, settings_, logger, prec, mat, b.values, x.values,
                workspace_data, group_size, shared_size);
        } else {
            switch (n_shared_total) {
            case 0:
                launch_apply_kernel<ValueType, StopType, 32, 0>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, group_size, shared_size);
                break;
            case 1:
                launch_apply_kernel<ValueType, StopType, 32, 1>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, group_size, shared_size);
                break;
            case 2:
                launch_apply_kernel<ValueType, StopType, 32, 2>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, group_size, shared_size);
                break;
            case 3:
                launch_apply_kernel<ValueType, StopType, 32, 3>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, group_size, shared_size);
                break;
            case 4:
                launch_apply_kernel<ValueType, StopType, 32, 4>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, group_size, shared_size);
                break;
            case 5:
                launch_apply_kernel<ValueType, StopType, 32, 5>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, group_size, shared_size);
                break;
            case 6:
                launch_apply_kernel<ValueType, StopType, 32, 6>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, group_size, shared_size);
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

GKO_INSTANTIATE_FOR_BATCH_VALUE_MATRIX_PRECONDITIONER(
    GKO_DECLARE_BATCH_CG_APPLY_KERNEL_WRAPPER);


}  // namespace batch_cg
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
