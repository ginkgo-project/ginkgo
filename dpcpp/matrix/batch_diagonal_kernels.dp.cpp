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

#include "core/matrix/batch_diagonal_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>


#include "core/matrix/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The BatchDense matrix format namespace.
 * @ref BatchDense
 * @ingroup batch_diagonal
 */
namespace batch_diagonal {


#include "dpcpp/matrix/batch_diagonal_kernels.hpp.inc"


template <typename ValueType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::BatchDiagonal<ValueType>* const diag,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x)
{
    if (!b->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const auto b_stride = b->get_stride().at();
    const auto x_stride = x->get_stride().at();
    const auto num_rows = static_cast<int>(diag->get_size().at()[0]);
    const auto num_cols = static_cast<int>(diag->get_size().at()[1]);
    const auto num_rhs = static_cast<int>(x->get_size().at()[1]);
    const int mindim = min(num_rows, num_cols);

    const auto diag_values = diag->get_const_values();
    const auto b_values = b->get_const_values();
    auto x_values = x->get_values();

    const auto num_batches = b->get_num_batch_entries();
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_group_linear_id();
                const auto b_ptr = gko::batch::batch_entry_ptr(
                    b_values, b_stride, num_cols, batch_id);
                const auto x_ptr = gko::batch::batch_entry_ptr(
                    x_values, x_stride, num_rows, batch_id);
                const auto d_ptr = gko::batch::batch_entry_ptr(
                    diag_values, 1, mindim, batch_id);
                apply_kernel(num_rows, num_cols, d_ptr, num_rhs, b_stride,
                             b_ptr, x_stride, x_ptr, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DIAGONAL_APPLY_KERNEL);


template <typename ValueType>
void apply_in_place(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::BatchDiagonal<ValueType>* const diag,
                    matrix::BatchDense<ValueType>* const b)
{
    if (!diag->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;

    const auto stride = b->get_stride().at();
    const auto num_rows = b->get_size().at()[0];
    const auto num_rhs = b->get_size().at()[1];

    const auto diag_values = diag->get_const_values();
    auto b_values = b->get_values();

    const auto num_batches = b->get_num_batch_entries();
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             auto group = item_ct1.get_group();
                             auto batch_id = group.get_group_linear_id();
                             const auto b_ptr = gko::batch::batch_entry_ptr(
                                 b_values, stride, num_rows, batch_id);
                             const auto d_ptr = gko::batch::batch_entry_ptr(
                                 diag_values, 1, num_rows, batch_id);
                             apply_in_place_kernel(num_rows, stride, num_rhs,
                                                   d_ptr, b_ptr, item_ct1);
                         });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIAGONAL_APPLY_IN_PLACE_KERNEL);


}  // namespace batch_diagonal
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
