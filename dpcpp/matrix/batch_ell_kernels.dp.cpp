// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_ell_kernels.hpp"


#include <algorithm>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "dpcpp/base/batch_struct.hpp"
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
 * @brief The Ell matrix format namespace.
 * @ref Ell
 * @ingroup batch_ell
 */
namespace batch_ell {


#include "dpcpp/matrix/batch_ell_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const batch::matrix::Ell<ValueType, IndexType>* mat,
                  const batch::MultiVector<ValueType>* b,
                  batch::MultiVector<ValueType>* x)
{
    const size_type num_rows = mat->get_common_size()[0];
    const size_type num_cols = mat->get_common_size()[1];

    const auto num_batch_items = mat->get_num_batch_items();
    auto device = exec->get_queue()->get_device();
    // TODO: use runtime selection of group size based on num_rows.
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

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_ELL_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_apply(std::shared_ptr<const DefaultExecutor> exec,
                    const batch::MultiVector<ValueType>* alpha,
                    const batch::matrix::Ell<ValueType, IndexType>* mat,
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
    // TODO: use runtime selection of group size based on num_rows.
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

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_ELL_ADVANCED_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const array<ValueType>* col_scale, const array<ValueType>* row_scale,
           batch::matrix::Ell<ValueType, IndexType>* input)
{
    const auto col_scale_vals = col_scale->get_const_data();
    const auto row_scale_vals = row_scale->get_const_data();
    const auto num_rows = static_cast<int>(input->get_common_size()[0]);
    const auto num_cols = static_cast<int>(input->get_common_size()[1]);
    auto mat_ub = get_batch_struct(input);

    const auto num_batch_items = mat_ub.num_batch_items;
    auto device = exec->get_queue()->get_device();
    // TODO: use runtime selection of group size based on num_rows.
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batch_items);

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
                    auto mat_item =
                        batch::matrix::extract_batch_item(mat_ub, group_id);
                    scale_kernel(col_scale_b, row_scale_b, mat_item, item_ct1);
                });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_ELL_SCALE_KERNEL);


template <typename ValueType, typename IndexType>
void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec,
                         const batch::MultiVector<ValueType>* alpha,
                         const batch::MultiVector<ValueType>* beta,
                         batch::matrix::Ell<ValueType, IndexType>* mat)
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

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_ELL_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_ell
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
