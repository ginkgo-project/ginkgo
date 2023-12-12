// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_diagonal_kernels.hpp"


// Copyright (c) 2017-2023, the Ginkgo authors
#include <algorithm>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>


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
 * @brief The Diagonal matrix format namespace.
 * @ref Diagonal
 * @ingroup batch_diagonal
 */
namespace batch_diagonal {


#include "dpcpp/matrix/batch_diagonal_kernels.hpp.inc"


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const batch::matrix::Diagonal<ValueType>* mat,
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

    // Launch a kernel that has nbatches blocks, each block has max group size
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    auto group = item_ct1.get_group();
                    auto group_id = group.get_group_linear_id();
                    int offset = group_id * num_rows;
                    simple_apply_kernel(
                        num_rows, mat->get_const_values() + offset,
                        b->get_common_size()[1], b->get_common_size()[1],
                        b->get_const_values() +
                            offset * b->get_common_size()[1],
                        x->get_common_size()[1],
                        x->get_values() + offset * x->get_common_size()[1],
                        item_ct1);
                });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIAGONAL_SIMPLE_APPLY_KERNEL);


}  // namespace batch_diagonal
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
