/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/matrix/sellp_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The SELL-P matrix format namespace.
 *
 * @ingroup sellp
 */
namespace sellp {


constexpr int default_block_size = 256;


namespace {


template <typename ValueType, typename IndexType>
void spmv_kernel(size_type num_rows, size_type num_right_hand_sides,
                 size_type b_stride, size_type c_stride,
                 const size_type* __restrict__ slice_lengths,
                 const size_type* __restrict__ slice_sets,
                 const ValueType* __restrict__ a,
                 const IndexType* __restrict__ col,
                 const ValueType* __restrict__ b, ValueType* __restrict__ c,
                 sycl::nd_item<3> item_ct1)
{
    const auto slice_id = item_ct1.get_group(2);
    const auto slice_size = item_ct1.get_local_range().get(2);
    const auto row_in_slice = item_ct1.get_local_id(2);
    const auto global_row =
        static_cast<size_type>(slice_size) * slice_id + row_in_slice;
    const auto column_id = item_ct1.get_group(1);
    ValueType val = 0;
    IndexType ind = 0;
    if (global_row < num_rows && column_id < num_right_hand_sides) {
        for (size_type i = 0; i < slice_lengths[slice_id]; i++) {
            ind = row_in_slice + (slice_sets[slice_id] + i) * slice_size;
            val += a[ind] * b[col[ind] * b_stride + column_id];
        }
        c[global_row * c_stride + column_id] = val;
    }
}

GKO_ENABLE_DEFAULT_HOST(spmv_kernel, spmv_kernel);


template <typename ValueType, typename IndexType>
void advanced_spmv_kernel(size_type num_rows, size_type num_right_hand_sides,
                          size_type b_stride, size_type c_stride,
                          const size_type* __restrict__ slice_lengths,
                          const size_type* __restrict__ slice_sets,
                          const ValueType* __restrict__ alpha,
                          const ValueType* __restrict__ a,
                          const IndexType* __restrict__ col,
                          const ValueType* __restrict__ b,
                          const ValueType* __restrict__ beta,
                          ValueType* __restrict__ c, sycl::nd_item<3> item_ct1)
{
    const auto slice_id = item_ct1.get_group(2);
    const auto slice_size = item_ct1.get_local_range().get(2);
    const auto row_in_slice = item_ct1.get_local_id(2);
    const auto global_row =
        static_cast<size_type>(slice_size) * slice_id + row_in_slice;
    const auto column_id = item_ct1.get_group(1);
    ValueType val = 0;
    IndexType ind = 0;
    if (global_row < num_rows && column_id < num_right_hand_sides) {
        for (size_type i = 0; i < slice_lengths[slice_id]; i++) {
            ind = row_in_slice + (slice_sets[slice_id] + i) * slice_size;
            val += alpha[0] * a[ind] * b[col[ind] * b_stride + column_id];
        }
        c[global_row * c_stride + column_id] =
            beta[0] * c[global_row * c_stride + column_id] + val;
    }
}

GKO_ENABLE_DEFAULT_HOST(advanced_spmv_kernel, advanced_spmv_kernel);


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::Sellp<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    const dim3 blockSize(matrix::default_slice_size);
    const dim3 gridSize(ceildiv(a->get_size()[0], matrix::default_slice_size),
                        b->get_size()[1]);

    spmv_kernel(gridSize, blockSize, 0, exec->get_queue(), a->get_size()[0],
                b->get_size()[1], b->get_stride(), c->get_stride(),
                a->get_const_slice_lengths(), a->get_const_slice_sets(),
                a->get_const_values(), a->get_const_col_idxs(),
                b->get_const_values(), c->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Sellp<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    const dim3 blockSize(matrix::default_slice_size);
    const dim3 gridSize(ceildiv(a->get_size()[0], matrix::default_slice_size),
                        b->get_size()[1]);

    advanced_spmv_kernel(gridSize, blockSize, 0, exec->get_queue(),
                         a->get_size()[0], b->get_size()[1], b->get_stride(),
                         c->get_stride(), a->get_const_slice_lengths(),
                         a->get_const_slice_sets(), alpha->get_const_values(),
                         a->get_const_values(), a->get_const_col_idxs(),
                         b->get_const_values(), beta->get_const_values(),
                         c->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


}  // namespace sellp
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
