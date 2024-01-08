// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_bicgstab_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/solver/batch_bicgstab.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The batch Bicgstab solver namespace.
 *
 * @ingroup batch_bicgstab
 */
namespace batch_bicgstab {


#include "dpcpp/base/batch_multi_vector_kernels.hpp.inc"
#include "dpcpp/matrix/batch_csr_kernels.hpp.inc"
#include "dpcpp/matrix/batch_dense_kernels.hpp.inc"
#include "dpcpp/matrix/batch_ell_kernels.hpp.inc"
#include "dpcpp/solver/batch_bicgstab_kernels.hpp.inc"


template <typename T>
using settings = gko::kernels::batch_bicgstab::settings<T>;


__dpct_inline__ int get_group_size(int value,
                                   int subgroup_size = config::warp_size)
{
    int num_sg = ceildiv(value, subgroup_size);
    return num_sg * subgroup_size;
}


template <typename ValueType>
class KernelCaller {
public:
    KernelCaller(std::shared_ptr<const DefaultExecutor> exec,
                 const settings<remove_complex<ValueType>> settings)
        : exec_{std::move(exec)}, settings_{settings}
    {}

    template <typename StopType, const int subgroup_size,
              const int n_shared_total, typename PrecType, typename LogType,
              typename BatchMatrixType>
    __dpct_inline__ void launch_apply_kernel(
        const gko::kernels::batch_bicgstab::storage_config& sconf,
        LogType& logger, PrecType& prec, const BatchMatrixType mat,
        const ValueType* const __restrict__ b_values,
        ValueType* const __restrict__ x_values,
        ValueType* const __restrict__ workspace, const int& group_size,
        const int& shared_size) const
    {
        auto num_rows = mat.num_rows;

        const dim3 block(group_size);
        const dim3 grid(mat.num_batch_items);

        auto max_iters = settings_.max_iterations;
        auto res_tol = settings_.residual_tol;

        exec_->get_queue()->submit([&](sycl::handler& cgh) {
            sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                slm_values(sycl::range<1>(shared_size), cgh);

            cgh.parallel_for(
                sycl_nd_range(grid, block),
                [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                    subgroup_size)]] [[intel::kernel_args_restrict]] {
                    auto batch_id = item_ct1.get_group_linear_id();
                    const auto mat_global_entry =
                        gko::batch::matrix::extract_batch_item(mat, batch_id);
                    const ValueType* const b_global_entry =
                        gko::batch::multi_vector::batch_item_ptr(
                            b_values, 1, num_rows, batch_id);
                    ValueType* const x_global_entry =
                        gko::batch::multi_vector::batch_item_ptr(
                            x_values, 1, num_rows, batch_id);
                    apply_kernel<StopType, n_shared_total>(
                        sconf, max_iters, res_tol, logger, prec,
                        mat_global_entry, b_global_entry, x_global_entry,
                        num_rows, mat.get_single_item_num_nnz(),
                        static_cast<ValueType*>(slm_values.get_pointer()),
                        item_ct1, workspace);
                });
        });
    }

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(
        LogType logger, const BatchMatrixType& mat, PrecType prec,
        const gko::batch::multi_vector::uniform_batch<const ValueType>& b,
        const gko::batch::multi_vector::uniform_batch<ValueType>& x) const
    {
        using real_type = gko::remove_complex<ValueType>;
        const size_type num_batch_items = mat.num_batch_items;
        const auto num_rows = mat.num_rows;
        const auto num_rhs = b.num_rhs;
        GKO_ASSERT(num_rhs == 1);

        auto device = exec_->get_queue()->get_device();
        auto max_group_size =
            device.get_info<sycl::info::device::max_work_group_size>();
        int group_size =
            device.get_info<sycl::info::device::max_work_group_size>();
        if (group_size > num_rows) {
            group_size = get_group_size(num_rows);
        };
        group_size = std::min(
            std::max(group_size, static_cast<int>(2 * config::warp_size)),
            static_cast<int>(max_group_size));

        // reserve 5 for intermediate rho-s, norms,
        // alpha, omega, temp and for reduce_over_group
        // If the value available is negative, then set it to 0
        const int static_var_mem =
            (group_size + 5) * sizeof(ValueType) + 2 * sizeof(real_type);
        int shmem_per_blk = std::max(
            static_cast<int>(
                device.get_info<sycl::info::device::local_mem_size>()) -
                static_var_mem,
            0);
        const int padded_num_rows = num_rows;
        const size_type prec_size = PrecType::dynamic_work_size(
            padded_num_rows, mat.get_single_item_num_nnz());
        const auto sconf =
            gko::kernels::batch_bicgstab::compute_shared_storage<PrecType,
                                                                 ValueType>(
                shmem_per_blk, padded_num_rows, mat.get_single_item_num_nnz(),
                b.num_rhs);
        const size_t shared_size = sconf.n_shared * padded_num_rows +
                                   (sconf.prec_shared ? prec_size : 0);
        auto workspace = gko::array<ValueType>(
            exec_,
            sconf.gmem_stride_bytes * num_batch_items / sizeof(ValueType));
        GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(ValueType) == 0);

        ValueType* const workspace_data = workspace.get_data();
        int n_shared_total = sconf.n_shared + int(sconf.prec_shared);

        // template
        // launch_apply_kernel<StopType, subgroup_size, n_shared_total,
        // sg_kernel_all>
        if (num_rows <= 32 && n_shared_total == 10) {
            launch_apply_kernel<StopType, 32, 10>(
                sconf, logger, prec, mat, b.values, x.values, workspace_data,
                group_size, shared_size);
        } else if (num_rows <= 256 && n_shared_total == 10) {
            launch_apply_kernel<StopType, 32, 10>(
                sconf, logger, prec, mat, b.values, x.values, workspace_data,
                group_size, shared_size);
        } else {
            switch (n_shared_total) {
            case 0:
                launch_apply_kernel<StopType, 32, 0>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
                break;
            case 1:
                launch_apply_kernel<StopType, 32, 1>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
                break;
            case 2:
                launch_apply_kernel<StopType, 32, 2>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
                break;
            case 3:
                launch_apply_kernel<StopType, 32, 3>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
                break;
            case 4:
                launch_apply_kernel<StopType, 32, 4>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
                break;
            case 5:
                launch_apply_kernel<StopType, 32, 5>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
                break;
            case 6:
                launch_apply_kernel<StopType, 32, 6>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
                break;
            case 7:
                launch_apply_kernel<StopType, 32, 7>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
                break;
            case 8:
                launch_apply_kernel<StopType, 32, 8>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
                break;
            case 9:
                launch_apply_kernel<StopType, 32, 9>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
                break;
            case 10:
                launch_apply_kernel<StopType, 32, 10>(
                    sconf, logger, prec, mat, b.values, x.values,
                    workspace_data, group_size, shared_size);
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
           const batch::BatchLinOp* const precond,
           const batch::MultiVector<ValueType>* const b,
           batch::MultiVector<ValueType>* const x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    auto dispatcher = batch::solver::create_dispatcher<ValueType>(
        KernelCaller<ValueType>(exec, settings), settings, mat, precond);
    dispatcher.apply(b, x, logdata);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL);


}  // namespace batch_bicgstab
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
