// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/batch_multi_vector_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The MultiVector matrix format namespace.
 * @ref MultiVector
 * @ingroup batch_multi_vector
 */
namespace batch_multi_vector {


#include "dpcpp/base/batch_multi_vector_kernels.hpp.inc"


template <typename ValueType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const batch::MultiVector<ValueType>* const alpha,
           batch::MultiVector<ValueType>* const x)
{
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);

    const auto num_batches = x_ub.num_batch_items;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    // Launch a kernel that has nbatches blocks, each block has max group size
    if (alpha->get_common_size()[1] == 1) {
        exec->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto alpha_b =
                        batch::extract_batch_item(alpha_ub, group_id);
                    const auto x_b = batch::extract_batch_item(x_ub, group_id);
                    scale_kernel(alpha_b, x_b, item_ct1,
                                 [](int col) { return 0; });
                });
        });
    } else {
        exec->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto alpha_b =
                        batch::extract_batch_item(alpha_ub, group_id);
                    const auto x_b = batch::extract_batch_item(x_ub, group_id);
                    scale_kernel(alpha_b, x_b, item_ct1,
                                 [](int col) { return col; });
                });
        });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const DefaultExecutor> exec,
                const batch::MultiVector<ValueType>* const alpha,
                const batch::MultiVector<ValueType>* const x,
                batch::MultiVector<ValueType>* const y)
{
    const size_type num_rows = x->get_common_size()[0];
    const size_type num_cols = x->get_common_size()[1];

    const auto num_batches = x->get_num_batch_items();
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    if (alpha->get_common_size()[1] == 1) {
        exec->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto alpha_b =
                        batch::extract_batch_item(alpha_ub, group_id);
                    const auto x_b = batch::extract_batch_item(x_ub, group_id);
                    const auto y_b = batch::extract_batch_item(y_ub, group_id);
                    add_scaled_kernel(alpha_b, x_b, y_b, item_ct1,
                                      [](auto col) { return 0; });
                });
        });
    } else {
        exec->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto alpha_b =
                        batch::extract_batch_item(alpha_ub, group_id);
                    const auto x_b = batch::extract_batch_item(x_ub, group_id);
                    const auto y_b = batch::extract_batch_item(y_ub, group_id);
                    add_scaled_kernel(alpha_b, x_b, y_b, item_ct1,
                                      [](auto col) { return col; });
                });
        });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_ADD_SCALED_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const DefaultExecutor> exec,
                 const batch::MultiVector<ValueType>* const x,
                 const batch::MultiVector<ValueType>* const y,
                 batch::MultiVector<ValueType>* const result)
{
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    const auto res_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch_items;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    // TODO: Remove reqd_sub_group size and use sycl::reduce_over_group
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                config::warp_size)]] {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto x_b = batch::extract_batch_item(x_ub, group_id);
                const auto y_b = batch::extract_batch_item(y_ub, group_id);
                const auto res_b = batch::extract_batch_item(res_ub, group_id);
                compute_gen_dot_product_kernel(x_b, y_b, res_b, item_ct1,
                                               [](auto val) { return val; });
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const DefaultExecutor> exec,
                      const batch::MultiVector<ValueType>* const x,
                      const batch::MultiVector<ValueType>* const y,
                      batch::MultiVector<ValueType>* const result)
{
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    const auto res_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch_items;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto x_b = batch::extract_batch_item(x_ub, group_id);
                    const auto y_b = batch::extract_batch_item(y_ub, group_id);
                    const auto res_b =
                        batch::extract_batch_item(res_ub, group_id);
                    compute_gen_dot_product_kernel(
                        x_b, y_b, res_b, item_ct1,
                        [](auto val) { return conj(val); });
                });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_CONJ_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const DefaultExecutor> exec,
                   const batch::MultiVector<ValueType>* const x,
                   batch::MultiVector<remove_complex<ValueType>>* const result)
{
    const auto x_ub = get_batch_struct(x);
    const auto res_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch_items;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(config::warp_size)]] {
                                 auto group = item_ct1.get_group();
                                 auto group_id = group.get_group_linear_id();
                                 const auto x_b =
                                     batch::extract_batch_item(x_ub, group_id);
                                 const auto res_b = batch::extract_batch_item(
                                     res_ub, group_id);
                                 compute_norm2_kernel(x_b, res_b, item_ct1);
                             });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const batch::MultiVector<ValueType>* x,
          batch::MultiVector<ValueType>* result)
{
    const auto x_ub = get_batch_struct(x);
    const auto result_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch_items;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto x_b = batch::extract_batch_item(x_ub, group_id);
                const auto result_b =
                    batch::extract_batch_item(result_ub, group_id);
                copy_kernel(x_b, result_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL);


}  // namespace batch_multi_vector
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
