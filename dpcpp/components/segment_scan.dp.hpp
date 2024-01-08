// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_SEGMENT_SCAN_DP_HPP_
#define GKO_DPCPP_COMPONENTS_SEGMENT_SCAN_DP_HPP_


#include <CL/sycl.hpp>


#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


/**
 * @internal
 *
 * Compute a segment scan using add operation (+) of a subgroup_size. Each
 * segment performs suffix sum. Works on the source array and returns whether
 * the thread is the first element of its segment with same `ind`.
 */
template <unsigned subgroup_size, typename ValueType, typename IndexType>
__dpct_inline__ bool segment_scan(
    const group::thread_block_tile<subgroup_size>& group, const IndexType ind,
    ValueType* __restrict__ val)
{
    bool head = true;
#pragma unroll
    for (int i = 1; i < subgroup_size; i <<= 1) {
        const IndexType add_ind = group.shfl_up(ind, i);
        ValueType add_val = zero<ValueType>();
        if (add_ind == ind && group.thread_rank() >= i) {
            add_val = *val;
            if (i == 1) {
                head = false;
            }
        }
        add_val = group.shfl_down(add_val, i);
        if (group.thread_rank() < subgroup_size - i) {
            *val += add_val;
        }
    }
    return head;
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_SEGMENT_SCAN_DP_HPP_
