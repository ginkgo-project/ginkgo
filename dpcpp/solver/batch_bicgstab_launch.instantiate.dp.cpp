// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "dpcpp/solver/batch_bicgstab_launch.hpp"

#include <sycl/sycl.hpp>

#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "dpcpp/base/batch_multi_vector_kernels.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/base/math.hpp"
#include "dpcpp/base/types.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/solver/batch_bicgstab_kernels.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_bicgstab {


template <typename ValueType, typename StopType, const int subgroup_size,
          const int n_shared_total, typename PrecType, typename LogType,
          typename BatchMatrixType>
void launch_apply_kernel(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::kernels::batch_bicgstab::storage_config& sconf,
    const settings<remove_complex<ValueType>>& settings, LogType& logger,
    PrecType& prec, const BatchMatrixType& mat,
    const device_type<ValueType>* const __restrict__ b_values,
    device_type<ValueType>* const __restrict__ x_values,
    device_type<ValueType>* const __restrict__ workspace, const int& group_size,
    const int& shared_size)
{
    auto num_rows = mat.num_rows;

    const dim3 block(group_size);
    const dim3 grid(mat.num_batch_items);

    auto max_iters = settings.max_iterations;
    auto res_tol = as_device_type(settings.residual_tol);

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<device_type<ValueType>, 1> slm_values(
            sycl::range<1>(shared_size), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                subgroup_size)]] [[intel::kernel_args_restrict]] {
                auto batch_id = item_ct1.get_group_linear_id();
                const auto mat_global_entry =
                    gko::batch::matrix::extract_batch_item(mat, batch_id);
                const device_type<ValueType>* const b_global_entry =
                    gko::batch::multi_vector::batch_item_ptr(
                        b_values, 1, num_rows, batch_id);
                device_type<ValueType>* const x_global_entry =
                    gko::batch::multi_vector::batch_item_ptr(
                        x_values, 1, num_rows, batch_id);
                batch_single_kernels::apply_kernel<StopType, n_shared_total>(
                    sconf, max_iters, res_tol, logger, prec, mat_global_entry,
                    b_global_entry, x_global_entry, num_rows,
                    mat.get_single_item_num_nnz(),
                    static_cast<device_type<ValueType>*>(
                        slm_values.get_pointer()),
                    item_ct1, workspace);
            });
    });
}


// begin
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_0;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_1;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_2;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_3;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_4;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_5;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_6;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_7;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_8;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_9;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_10;
// split
GKO_INSTANTIATE_BATCH_BICGSTAB_LAUNCH_10_16;
// end


}  // namespace batch_bicgstab
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
