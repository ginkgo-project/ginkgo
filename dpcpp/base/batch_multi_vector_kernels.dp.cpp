/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
 * @brief The BatchMultiVector matrix format namespace.
 * @ref BatchMultiVector
 * @ingroup batch_multi_vector
 */
namespace batch_multi_vector {


#include "dpcpp/base/batch_multi_vector_kernels.hpp.inc"


template <typename ValueType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const BatchMultiVector<ValueType>* const alpha,
           BatchMultiVector<ValueType>* const x)
{
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);

    const auto num_batches = x_ub.num_batch_entries;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    // Launch a kernel that has nbatches blocks, each block has max group size
    if (alpha->get_common_size()[1] == 1) {
        (exec->get_queue())->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto alpha_b = batch::batch_entry(alpha_ub, group_id);
                    const auto x_b = batch::batch_entry(x_ub, group_id);
                    scale_kernel(alpha_b, x_b, item_ct1,
                                 [](int col) { return 0; });
                });
        });
    } else {
        (exec->get_queue())->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto alpha_b = batch::batch_entry(alpha_ub, group_id);
                    const auto x_b = batch::batch_entry(x_ub, group_id);
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
                const BatchMultiVector<ValueType>* const alpha,
                const BatchMultiVector<ValueType>* const x,
                BatchMultiVector<ValueType>* const y)
{
    const size_type num_rows = x->get_common_size()[0];
    const size_type num_cols = x->get_common_size()[1];

    const auto num_batches = x->get_num_batch_entries();
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    if (alpha->get_common_size()[1] == 1) {
        (exec->get_queue())->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto alpha_b = batch::batch_entry(alpha_ub, group_id);
                    const auto x_b = batch::batch_entry(x_ub, group_id);
                    const auto y_b = batch::batch_entry(y_ub, group_id);
                    add_scaled_kernel(alpha_b, x_b, y_b, item_ct1,
                                      [](auto col) { return 0; });
                });
        });
    } else {
        (exec->get_queue())->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    const auto alpha_b = batch::batch_entry(alpha_ub, group_id);
                    const auto x_b = batch::batch_entry(x_ub, group_id);
                    const auto y_b = batch::batch_entry(y_ub, group_id);
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
                 const BatchMultiVector<ValueType>* const x,
                 const BatchMultiVector<ValueType>* const y,
                 BatchMultiVector<ValueType>* const result)
{
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    const auto res_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch_entries;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    // TODO: Remove reqd_sub_group size and use sycl::reduce_over_group
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=
        ](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                                            config::warp_size)]] {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto x_b = batch::batch_entry(x_ub, group_id);
                const auto y_b = batch::batch_entry(y_ub, group_id);
                const auto res_b = batch::batch_entry(res_ub, group_id);
                compute_gen_dot_product_kernel(x_b, y_b, res_b, item_ct1,
                                               [](auto val) { return val; });
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const DefaultExecutor> exec,
                      const BatchMultiVector<ValueType>* const x,
                      const BatchMultiVector<ValueType>* const y,
                      BatchMultiVector<ValueType>* const result)
{
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    const auto res_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch_entries;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=
        ](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                                            config::warp_size)]] {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto x_b = batch::batch_entry(x_ub, group_id);
                const auto y_b = batch::batch_entry(y_ub, group_id);
                const auto res_b = batch::batch_entry(res_ub, group_id);
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
                   const BatchMultiVector<ValueType>* const x,
                   BatchMultiVector<remove_complex<ValueType>>* const result)
{
    const auto x_ub = get_batch_struct(x);
    const auto res_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch_entries;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=
        ](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                                            config::warp_size)]] {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto x_b = batch::batch_entry(x_ub, group_id);
                const auto res_b = batch::batch_entry(res_ub, group_id);
                compute_norm2_kernel(x_b, res_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const BatchMultiVector<ValueType>* x,
          BatchMultiVector<ValueType>* result)
{
    const auto x_ub = get_batch_struct(x);
    const auto result_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch_entries;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto x_b = batch::batch_entry(x_ub, group_id);
                const auto result_b = batch::batch_entry(result_ub, group_id);
                copy_kernel(x_b, result_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL);


}  // namespace batch_multi_vector
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
