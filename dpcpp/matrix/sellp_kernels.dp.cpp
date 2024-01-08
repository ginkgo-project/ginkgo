// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
                 size_type b_stride, size_type c_stride, size_type slice_size,
                 const size_type* __restrict__ slice_sets,
                 const ValueType* __restrict__ a,
                 const IndexType* __restrict__ cols,
                 const ValueType* __restrict__ b, ValueType* __restrict__ c,
                 sycl::nd_item<3> item_ct1)
{
    const auto row = thread::get_thread_id_flat(item_ct1);
    const auto slice_id = row / slice_size;
    const auto row_in_slice = row % slice_size;
    const auto column_id = item_ct1.get_group(1);
    auto val = zero<ValueType>();
    if (row < num_rows && column_id < num_right_hand_sides) {
        for (auto i = slice_sets[slice_id]; i < slice_sets[slice_id + 1]; i++) {
            const auto ind = row_in_slice + i * slice_size;
            const auto col = cols[ind];
            if (col != invalid_index<IndexType>()) {
                val += a[ind] * b[col * b_stride + column_id];
            }
        }
        c[row * c_stride + column_id] = val;
    }
}

GKO_ENABLE_DEFAULT_HOST(spmv_kernel, spmv_kernel);


template <typename ValueType, typename IndexType>
void advanced_spmv_kernel(size_type num_rows, size_type num_right_hand_sides,
                          size_type b_stride, size_type c_stride,
                          size_type slice_size,
                          const size_type* __restrict__ slice_sets,
                          const ValueType* __restrict__ alpha,
                          const ValueType* __restrict__ a,
                          const IndexType* __restrict__ cols,
                          const ValueType* __restrict__ b,
                          const ValueType* __restrict__ beta,
                          ValueType* __restrict__ c, sycl::nd_item<3> item_ct1)
{
    const auto row = thread::get_thread_id_flat(item_ct1);
    const auto slice_id = row / slice_size;
    const auto row_in_slice = row % slice_size;
    const auto column_id = item_ct1.get_group(1);
    auto val = zero<ValueType>();
    if (row < num_rows && column_id < num_right_hand_sides) {
        for (auto i = slice_sets[slice_id]; i < slice_sets[slice_id + 1]; i++) {
            const auto ind = row_in_slice + i * slice_size;
            const auto col = cols[ind];
            if (col != invalid_index<IndexType>()) {
                val += a[ind] * b[col * b_stride + column_id];
            }
        }
        c[row * c_stride + column_id] =
            beta[0] * c[row * c_stride + column_id] + alpha[0] * val;
    }
}

GKO_ENABLE_DEFAULT_HOST(advanced_spmv_kernel, advanced_spmv_kernel);


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::Sellp<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    const dim3 blockSize(default_block_size);
    const dim3 gridSize(ceildiv(a->get_size()[0], default_block_size),
                        b->get_size()[1]);

    spmv_kernel(gridSize, blockSize, 0, exec->get_queue(), a->get_size()[0],
                b->get_size()[1], b->get_stride(), c->get_stride(),
                a->get_slice_size(), a->get_const_slice_sets(),
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
    const dim3 blockSize(default_block_size);
    const dim3 gridSize(ceildiv(a->get_size()[0], default_block_size),
                        b->get_size()[1]);

    advanced_spmv_kernel(
        gridSize, blockSize, 0, exec->get_queue(), a->get_size()[0],
        b->get_size()[1], b->get_stride(), c->get_stride(), a->get_slice_size(),
        a->get_const_slice_sets(), alpha->get_const_values(),
        a->get_const_values(), a->get_const_col_idxs(), b->get_const_values(),
        beta->get_const_values(), c->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


}  // namespace sellp
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
