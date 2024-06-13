// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @internal
 *
 * Compute a segment scan using add operation (+) of a subwarp. Each segment
 * performs suffix sum. Works on the source array and returns whether the thread
 * is the first element of its segment with same `ind`.
 */
template <unsigned subwarp_size, typename ValueType, typename IndexType,
          typename Operator>
__device__ __forceinline__ bool segment_scan(
    const group::thread_block_tile<subwarp_size>& group, const IndexType ind,
    ValueType& val, Operator op)
{
    bool head = true;
#pragma unroll
    for (int i = 1; i < subwarp_size; i <<= 1) {
        const IndexType add_ind = group.shfl_up(ind, i);
        ValueType add_val{};
        if (add_ind == ind && group.thread_rank() >= i) {
            add_val = val;
            if (i == 1) {
                head = false;
            }
        }
        add_val = group.shfl_down(add_val, i);
        if (group.thread_rank() < subwarp_size - i) {
            val = op(val, add_val);
        }
    }
    return head;
}
