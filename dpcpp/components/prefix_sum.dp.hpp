// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_PREFIX_SUM_DP_HPP_
#define GKO_DPCPP_COMPONENTS_PREFIX_SUM_DP_HPP_


#include <type_traits>


#include <CL/sycl.hpp>


#include "core/base/types.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


// TODO: porting - some functions name still use subwarp


/**
 * @internal
 * Computes the prefix sum and total sum of `element` over a subgroup.
 *
 * @param element     the element over which we compute the prefix sum.
 * @param prefix_sum  will be set to the sum of all `element`s from lower
 *                    lanes, plus the local `element` if `inclusive` is `true`.
 * @param total_sum   will be set to the total sum of `element` in this
 * subgroup.
 * @param subgroup     the cooperative group representing the subgroup.
 *
 * @tparam inclusive  if this is true, the computed prefix sum will be
 *                    inclusive, otherwise it will be exclusive.
 *
 * @note For this function to work on architectures with independent thread
 * scheduling, all threads of the subgroup have to execute it.
 */
template <bool inclusive, typename ValueType, typename Group>
__dpct_inline__ void subwarp_prefix_sum(ValueType element,
                                        ValueType& prefix_sum,
                                        ValueType& total_sum, Group subgroup)
{
    prefix_sum = inclusive ? element : zero<ValueType>();
    total_sum = element;
#pragma unroll
    // hypercube prefix sum
    for (int step = 1; step < subgroup.size(); step *= 2) {
        auto neighbor = subgroup.shfl_xor(total_sum, step);
        total_sum += neighbor;
        prefix_sum += bool(subgroup.thread_rank() & step) ? neighbor : 0;
    }
}

/**
 * @internal
 * Computes the prefix sum of `element` over a subgroup.
 *
 * @param element     the element over which we compute the prefix sum.
 * @param prefix_sum  will be set to the sum of all `element`s from lower
 *                    lanes, plus the local `element` if `inclusive` is `true`.
 * @param subgroup     the cooperative group representing the subgroup.
 *
 * @tparam inclusive  if this is true, the computed prefix sum will be
 *                    inclusive, otherwise it will be exclusive.
 *
 * @note All threads of the subgroup have to execute this function for it to
 * work (and not dead-lock on newer architectures).
 */
template <bool inclusive, typename ValueType, typename Group>
__dpct_inline__ void subwarp_prefix_sum(ValueType element,
                                        ValueType& prefix_sum, Group subgroup)
{
    ValueType tmp{};
    subwarp_prefix_sum<inclusive>(element, prefix_sum, tmp, subgroup);
}


/**
 * @internal
 * First step of the calculation of a prefix sum. Calculates the prefix sum
 * in-place on parts of the array `elements`.
 *
 * @param elements  array on which the prefix sum is to be calculated
 * @param block_sum  array which stores the total sum of each block, requires at
 *                   least `ceildiv(num_elements, block_size) - 1` elements
 * @param num_elements  total number of entries in `elements`
 *
 * @tparam block_size  thread block size for this kernel, also size of blocks on
 *                     which this kernel calculates the prefix sum in-place
 *
 * @note To calculate the prefix sum over an array of size bigger than
 *       `block_size`, `finalize_prefix_sum` has to be used as well.
 */
template <typename DeviceConfig, typename ValueType>
void start_prefix_sum(
    size_type num_elements, ValueType* __restrict__ elements,
    ValueType* __restrict__ block_sum, sycl::nd_item<3> item_ct1,
    uninitialized_array<ValueType, DeviceConfig::block_size>& prefix_helper)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    const auto element_id = item_ct1.get_local_id(2);
    constexpr auto block_size = DeviceConfig::block_size;

    // do not need to access the last element when exclusive prefix sum
    prefix_helper[element_id] =
        (tidx + 1 < num_elements) ? elements[tidx] : zero<ValueType>();
    auto this_block = group::this_thread_block(item_ct1);
    this_block.sync();

    // Do a normal reduction
#pragma unroll
    for (int i = 1; i < block_size; i <<= 1) {
        const auto ai = i * (2 * element_id + 1) - 1;
        const auto bi = i * (2 * element_id + 2) - 1;
        if (bi < block_size) {
            prefix_helper[bi] += prefix_helper[ai];
        }
        this_block.sync();
    }

    if (element_id == 0) {
        // Store the total sum except the last block
        if (item_ct1.get_group(2) + 1 < item_ct1.get_group_range(2)) {
            block_sum[item_ct1.get_group(2)] = prefix_helper[block_size - 1];
        }
        prefix_helper[block_size - 1] = zero<ValueType>();
    }

    this_block.sync();

    // Perform the down-sweep phase to get the true prefix sum
#pragma unroll
    for (int i = block_size >> 1; i > 0; i >>= 1) {
        const auto ai = i * (2 * element_id + 1) - 1;
        const auto bi = i * (2 * element_id + 2) - 1;
        if (bi < block_size) {
            auto tmp = prefix_helper[ai];
            prefix_helper[ai] = prefix_helper[bi];
            prefix_helper[bi] += tmp;
        }
        this_block.sync();
    }
    if (tidx < num_elements) {
        elements[tidx] = prefix_helper[element_id];
    }
}

template <typename DeviceConfig, typename ValueType>
void start_prefix_sum(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                      sycl::queue* queue, size_type num_elements,
                      ValueType* elements, ValueType* block_sum)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<uninitialized_array<ValueType, DeviceConfig::block_size>,
                       0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            prefix_helper_acc_ct1(cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             start_prefix_sum<DeviceConfig>(
                                 num_elements, elements, block_sum, item_ct1,
                                 *prefix_helper_acc_ct1.get_pointer());
                         });
    });
}


/**
 * @internal
 * Second step of the calculation of a prefix sum. Increases the value of each
 * entry of `elements` by the total sum of all preceding blocks.
 *
 * @param elements  array on which the prefix sum is to be calculated
 * @param block_sum  array storing the total sum of each block
 * @param num_elements  total number of entries in `elements`
 *
 * @tparam block_size  thread block size for this kernel, has to be the same as
 *                    for `start_prefix_sum`
 *
 * @note To calculate a prefix sum, first `start_prefix_sum` has to be called.
 */
template <typename DeviceConfig, typename ValueType>
void finalize_prefix_sum(size_type num_elements,
                         ValueType* __restrict__ elements,
                         const ValueType* __restrict__ block_sum,
                         sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx < num_elements) {
        ValueType prefix_block_sum = zero<ValueType>();
        for (size_type i = 0; i < item_ct1.get_group(2); i++) {
            prefix_block_sum += block_sum[i];
        }
        elements[tidx] += prefix_block_sum;
    }
}

template <typename DeviceConfig, typename ValueType>
void finalize_prefix_sum(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                         sycl::queue* queue, size_type num_elements,
                         ValueType* elements, const ValueType* block_sum)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             finalize_prefix_sum<DeviceConfig>(
                                 num_elements, elements, block_sum, item_ct1);
                         });
    });
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_PREFIX_SUM_DP_HPP_
