// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_SEARCHING_DP_HPP_
#define GKO_DPCPP_COMPONENTS_SEARCHING_DP_HPP_


#include <CL/sycl.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


/**
 * @internal
 * Generic binary search that finds the first index where a predicate is true.
 * It assumes that the predicate partitions the range [offset, offset + length)
 * into two subranges [offset, middle), [middle, offset + length) such that
 * the predicate is `false` for all elements in the first range and `true` for
 * all elements in the second range. `middle` is called the partition point.
 * If the predicate is `false` everywhere, `middle` equals `offset + length`.
 * The implementation is based on Stepanov & McJones, "Elements of Programming".
 *
 * @param offset  the starting index of the partitioned range
 * @param length  the length of the partitioned range
 * @param p  the predicate to be evaluated on the range - it should not have
 *           side-effects and map from `IndexType` to `bool`
 * @returns  the index of `middle`, i.e., the partition point
 */
template <typename IndexType, typename Predicate>
__dpct_inline__ IndexType binary_search(IndexType offset, IndexType length,
                                        Predicate p)
{
    while (length > 0) {
        auto half_length = length / 2;
        auto mid = offset + half_length;
        auto pred = p(mid);
        length = pred ? half_length : length - (half_length + 1);
        offset = pred ? offset : mid + 1;
    }
    return offset;
}


/**
 * @internal
 * Generic implementation of a fixed-size binary search.
 * The implementation makes sure that the number of predicate evaluations only
 * depends on `size` and not on the actual position of the partition point.
 * It assumes that the predicate partitions the range [0, size) into two
 * subranges [0, middle), [middle, size) such that the predicate is `false` for
 * all elements in the first range and `true` for all elements in the second
 * range. `middle` is called the partition point.
 * If the predicate is `false` everywhere, `middle` equals `size`.
 *
 * @tparam size  the length of the partitioned range - must be a power of two
 * @param p  the predicate to be evaluated on the range - it should not have
 *           side-effects and map from `int` to `bool`
 * @returns  the index of `middle`, i.e., the partition point
 */
template <int size, typename Predicate>
__dpct_inline__ int synchronous_fixed_binary_search(Predicate p)
{
    if (size == 0) {
        return 0;
    }
    int begin{};
    static_assert(size > 0, "size must be positive");
    static_assert(!(size & (size - 1)), "size must be a power of two");
#pragma unroll
    for (auto cur_size = size; cur_size > 1; cur_size /= 2) {
        auto half_size = cur_size / 2;
        auto mid = begin + half_size;
        // invariant: [begin, begin + cur_size] contains partition point
        begin = p(mid) ? begin : mid;
    }
    // cur_size is now 1, so the partition point is either begin or begin + 1
    return p(begin) ? begin : begin + 1;
}


/**
 * @internal
 * Generic implementation of a synchronous binary search.
 * The implementation makes sure that the number of predicate evaluations only
 * depends on `size` and not on the actual position of the partition point.
 * It assumes that the predicate partitions the range [0, size) into two
 * subranges [0, middle), [middle, size) such that the predicate is `false` for
 * all elements in the first range and `true` for all elements in the second
 * range. `middle` is called the partition point.
 * If the predicate is `false` everywhere, `middle` equals `size`.
 *
 * @param size  the length of the partitioned range - must be a power of two
 * @param p  the predicate to be evaluated on the range - it should not have
 *           side-effects and map from `int` to `bool`
 * @returns  the index of `middle`, i.e., the partition point
 */
template <typename Predicate>
__dpct_inline__ int synchronous_binary_search(int size, Predicate p)
{
    if (size == 0) {
        return 0;
    }
    int begin{};
    for (auto cur_size = size; cur_size > 1; cur_size /= 2) {
        auto half_size = cur_size / 2;
        auto mid = begin + half_size;
        // invariant: [begin, begin + cur_size] contains partition point
        begin = p(mid) ? begin : mid;
    }
    // cur_size is now 1, so the partition point is either begin or begin + 1
    return p(begin) ? begin : begin + 1;
}


/**
 * @internal
 * Generic search that finds the first index where a predicate is true.
 * It assumes that the predicate partitions the range [offset, offset + length)
 * into two subranges [offset, middle), [middle, offset + length) such that
 * the predicate is `false` for all elements in the first range and `true` for
 * all elements in the second range. `middle` is called the partition point.
 * If the predicate is `false` everywhere, `middle` equals `offset + length`.
 *
 * It executes `log2(length / group.size())` coalescing calls to `p`.
 *
 * This implementation is based on the w-wide search mentioned in
 * Green et al., "GPU merge path: a GPU merging algorithm"
 *
 * @param offset  the starting index of the partitioned range
 * @param length  the length of the partitioned range
 * @param group   the coalescing group executing the search
 * @param p  the predicate to be evaluated on the range - it should not have
 *           side-effects and map from `IndexType` to `bool`
 * @returns  the index of `middle`, i.e., the partition point
 */
template <typename IndexType, typename Group, typename Predicate>
__dpct_inline__ IndexType group_wide_search(IndexType offset, IndexType length,
                                            Group group, Predicate p)
{
    // binary search on the group-sized blocks
    IndexType num_blocks = (length + group.size() - 1) / group.size();
    auto group_pos = binary_search(IndexType{}, num_blocks, [&](IndexType i) {
        auto idx = i * group.size();
        return p(offset + idx);
    });
    // case 1: p is true everywhere: middle is at the beginning
    if (group_pos == 0) {
        return offset;
    }
    /*
     * case 2: p is false somewhere:
     *
     * p(group_pos * g.size()) is true, so either this is the partition point,
     * or the partition point is one of the g.size() - 1 previous indices.
     *   |block group_pos-1|
     * 0 | 0 * * * * * * * | 1
     *       ^               ^
     *       we load this range, with the 1 acting as a sentinel for ffs(...)
     *
     * additionally, this means that we can't call p out-of-bounds
     */
    auto base_idx = (group_pos - 1) * group.size() + 1;
    auto idx = base_idx + group.thread_rank();
    auto pos = ffs(group.ballot(idx >= length || p(offset + idx))) - 1;
    return offset + base_idx + pos;
}


/**
 * @internal
 * Generic search that finds the first index where a predicate is true.
 * It assumes that the predicate partitions the range [offset, offset + length)
 * into two subranges [offset, middle), [middle, offset + length) such that
 * the predicate is `false` for all elements in the first range and `true` for
 * all elements in the second range. `middle` is called the partition point.
 * If the predicate is `false` everywhere, `middle` equals `offset + length`.
 *
 * It executes `log2(length) / log2(group.size())` calls to `p` that effectively
 * follow a random-access pattern.
 *
 * This implementation is based on the w-partition search mentioned in
 * Green et al., "GPU merge path: a GPU merging algorithm"
 *
 * @param offset  the starting index of the partitioned range
 * @param length  the length of the partitioned range
 * @param group   the coalescing group executing the search
 * @param p  the predicate to be evaluated on the range - it should not have
 *           side-effects and map from `IndexType` to `bool`
 * @returns  the index of `middle`, i.e., the partition point
 */
template <typename IndexType, typename Group, typename Predicate>
__dpct_inline__ IndexType group_ary_search(IndexType offset, IndexType length,
                                           Group group, Predicate p)
{
    IndexType end = offset + length;
    // invariant: [offset, offset + length] contains middle
    while (length > group.size()) {
        auto stride = length / group.size();
        auto idx = offset + group.thread_rank() * stride;
        auto mask = group.ballot(p(idx));
        // if the mask is 0, the partition point is in the last block
        // if the mask is ~0, the partition point is in the first block
        // otherwise, we go to the last block that returned a 0.
        auto pos = mask == 0 ? group.size() - 1 : ffs(mask >> 1) - 1;
        auto last_length = length - stride * (group.size() - 1);
        length = pos == group.size() - 1 ? last_length : stride;
        offset += stride * pos;
    }
    auto idx = offset + group.thread_rank();
    // if the mask is 0, the partition point is at the end
    // otherwise it is the first set bit
    auto mask = group.ballot(idx >= end || p(idx));
    auto pos = mask == 0 ? group.size() : ffs(mask) - 1;
    return offset + pos;
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_SEARCHING_DP_HPP_
