// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_dense_kernels.hpp"


#include <algorithm>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/base/batch_struct.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/batch_struct.hpp"
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
 * @brief The Dense matrix format namespace.
 *
 * @ingroup batch_dense
 */
namespace batch_dense {


#include "dpcpp/matrix/batch_dense_kernels.hpp.inc"


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const batch::matrix::Dense<ValueType>* mat,
                  const batch::MultiVector<ValueType>* b,
                  batch::MultiVector<ValueType>* x)
{
    const size_type num_rows = mat->get_common_size()[0];
    const size_type num_cols = mat->get_common_size()[1];

    const auto num_batch_items = mat->get_num_batch_items();
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batch_items);
    const auto x_ub = get_batch_struct(x);
    const auto b_ub = get_batch_struct(b);
    const auto mat_ub = get_batch_struct(mat);
    if (b_ub.num_rhs > 1) {
        GKO_NOT_IMPLEMENTED;
    }

    // Launch a kernel that has nbatches blocks, each block has max group size
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto mat_b =
                        batch::matrix::extract_batch_item(mat_ub, group_id);
                    const auto b_b = batch::extract_batch_item(b_ub, group_id);
                    const auto x_b = batch::extract_batch_item(x_ub, group_id);
                    simple_apply_kernel(mat_b, b_b.values, x_b.values,
                                        item_ct1);
                });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void advanced_apply(std::shared_ptr<const DefaultExecutor> exec,
                    const batch::MultiVector<ValueType>* alpha,
                    const batch::matrix::Dense<ValueType>* mat,
                    const batch::MultiVector<ValueType>* b,
                    const batch::MultiVector<ValueType>* beta,
                    batch::MultiVector<ValueType>* x)
{
    const auto mat_ub = get_batch_struct(mat);
    const auto b_ub = get_batch_struct(b);
    const auto x_ub = get_batch_struct(x);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);

    if (b_ub.num_rhs > 1) {
        GKO_NOT_IMPLEMENTED;
    }

    const auto num_batch_items = mat_ub.num_batch_items;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batch_items);

    // Launch a kernel that has nbatches blocks, each block has max group size
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto mat_b =
                        batch::matrix::extract_batch_item(mat_ub, group_id);
                    const auto b_b = batch::extract_batch_item(b_ub, group_id);
                    const auto x_b = batch::extract_batch_item(x_ub, group_id);
                    const auto alpha_b =
                        batch::extract_batch_item(alpha_ub, group_id);
                    const auto beta_b =
                        batch::extract_batch_item(beta_ub, group_id);
                    advanced_apply_kernel(alpha_b.values[0], mat_b, b_b.values,
                                          beta_b.values[0], x_b.values,
                                          item_ct1);
                });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADVANCED_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const array<ValueType>* col_scale, const array<ValueType>* row_scale,
           batch::matrix::Dense<ValueType>* input)
{
    const auto col_scale_vals = col_scale->get_const_data();
    const auto row_scale_vals = row_scale->get_const_data();
    const auto num_rows = static_cast<int>(input->get_common_size()[0]);
    const auto num_cols = static_cast<int>(input->get_common_size()[1]);
    const auto stride = input->get_common_size()[1];
    auto mat_ub = get_batch_struct(input);

    const auto num_batch_items = mat_ub.num_batch_items;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batch_items);

    // Launch a kernel that has nbatches blocks, each block has max group size
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto col_scale_b =
                        col_scale_vals + num_cols * group_id;
                    const auto row_scale_b =
                        row_scale_vals + num_rows * group_id;
                    auto input_mat =
                        batch::matrix::extract_batch_item(mat_ub, group_id);
                    scale_kernel(col_scale_b, row_scale_b, input_mat, item_ct1);
                });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);


template <typename ValueType>
void scale_add(std::shared_ptr<const DefaultExecutor> exec,
               const batch::MultiVector<ValueType>* alpha,
               const batch::matrix::Dense<ValueType>* mat,
               batch::matrix::Dense<ValueType>* in_out)
{
    const auto alpha_ub = get_batch_struct(alpha);
    const auto mat_ub = get_batch_struct(mat);
    const auto in_out_ub = get_batch_struct(in_out);

    const auto num_batch_items = mat_ub.num_batch_items;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batch_items);

    // Launch a kernel that has nbatches blocks, each block has max group size
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                config::warp_size)]] {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto alpha_b =
                    gko::batch::extract_batch_item(alpha_ub, group_id);
                const auto mat_b =
                    gko::batch::matrix::extract_batch_item(mat_ub, group_id);
                const auto in_out_b =
                    gko::batch::matrix::extract_batch_item(in_out_ub, group_id);
                scale_add_kernel(alpha_b.values[0], mat_b, in_out_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_ADD_KERNEL);


template <typename ValueType>
void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec,
                         const batch::MultiVector<ValueType>* alpha,
                         const batch::MultiVector<ValueType>* beta,
                         batch::matrix::Dense<ValueType>* mat)
{
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);
    const auto mat_ub = get_batch_struct(mat);

    const auto num_batch_items = mat_ub.num_batch_items;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batch_items);

    // Launch a kernel that has nbatches blocks, each block has max group size
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto alpha_b =
                        gko::batch::extract_batch_item(alpha_ub, group_id);
                    const auto beta_b =
                        gko::batch::extract_batch_item(beta_ub, group_id);
                    const auto mat_b = gko::batch::matrix::extract_batch_item(
                        mat_ub, group_id);
                    add_scaled_identity_kernel(
                        alpha_b.values[0], beta_b.values[0], mat_b, item_ct1);
                });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
