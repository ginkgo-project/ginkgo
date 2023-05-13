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
 * Compute a segement scan using add operation (+) of a subgroup_size. Each
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


/**
 * @internal
 *
 * Compute a segement scan using add operation (+) of a subgroup_size. Each
 * segment performs suffix sum. Works on the source array and returns whether
 * the thread is the first element of its segment with same `ind`.
 */
template <unsigned subgroup_size, typename ValueType, typename IndexType,
          typename Operator>
__dpct_inline__ bool segment_scan(
    const group::thread_block_tile<subgroup_size>& group, const IndexType ind,
    ValueType* __restrict__ val, Operator op)
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
            *val = op(*val, add_val);
        }
    }
    return head;
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_SEGMENT_SCAN_DP_HPP_
