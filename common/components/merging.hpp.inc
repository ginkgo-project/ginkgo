/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

namespace detail {


/**
 * @internal
 * The result from the @ref parallel_merge_step function.
 */
template <typename ValueType>
struct merge_result {
    /** The element of a being merged in the current thread. */
    ValueType a_val;
    /** The element of b being merged in the current thread. */
    ValueType b_val;
    /** The index from a that is being merged in the current thread. */
    int a_idx;
    /** The index from b that is being merged in the current thread. */
    int b_idx;
    /** The number of elements from a that have been merged in total. */
    int a_advance;
    /** The number of elements from b that have been merged in total. */
    int b_advance;
};


template <typename ValueType, typename IndexType>
__device__ ValueType
checked_load(const ValueType *p, IndexType i, IndexType size,
             ValueType sentinel = device_numeric_limits<ValueType>::max)
{
    return i < size ? p[i] : sentinel;
}


}  // namespace detail


/**
 * @internal
 * Warp-parallel merge algorithm that merges the first `warp_size` elements from
 * two ranges, where each warp stores a single element from each range.
 * It assumes that the elements are sorted in ascending order, i.e. for i < j,
 * the value of `a` at thread i is smaller or equal the value at thread j, and
 * the same holds for `b`.
 *
 * @param a      the element from the first range
 * @param b      the element from the second range
 * @param size   the number of elements in the output range
 * @param group  the cooperative group that executes the merge
 * @return  a structure containing the merge result distributed over the group.
 */
template <typename ValueType, typename Group>
__device__ detail::merge_result<ValueType> group_merge_step(ValueType a,
                                                            ValueType b,
                                                            int size,
                                                            Group group)
{
    // round up to the next power of two
    auto size_pow2 = 1 << (32 - clz(uint32(size - 1)));
    // thread i takes care of ith element of the merged sequence
    auto i = int(group.thread_rank());

    // we want to find the smallest index `x` such that a[x] >= b[i - x - 1]
    // or `i` if no such index exists
    //
    // if x = i then c[0...i - 1] = a[0...i - 1]
    //     => merge a[i] with b[0]
    // if x = 0 then c[0...i - 1] = b[0...i - 1]
    //     => merge a[0] with b[i]
    // otherwise c[0...i - 1] contains a[0...x - 1] and b[0...i - x - 1]
    //   because the minimality of `x` implies
    //   b[i - x] >= a[x - 1]
    //   and a[x] >= a[0...x - 1], b[0...i - x - 1]
    //     => merge a[x] with b[i - x]
    auto minx = synchronous_binary_search(size_pow2, [&](int x) {
        // potentially faster (because of unrolling):
        // auto minx = synchronous_fixed_binary_search<group.size()>([&](int x){
        auto a_remote = group.shfl(a, x);
        auto b_remote = group.shfl(b, max(i - x - 1, 0));
        return a_remote >= b_remote || x >= i;
    });

    auto a_idx = minx;
    auto b_idx = max(i - minx, 0);
    auto a_val = group.shfl(a, a_idx);
    auto b_val = group.shfl(b, b_idx);
    auto cmp = a_val < b_val;
    auto a_advance = int(popcnt(group.ballot(cmp)));
    auto b_advance = int(group.size()) - a_advance;

    return {a_val, b_val, a_idx, b_idx, a_advance, b_advance};
}


/**
 * @internal
 * Warp-parallel merge algorithm that merges two sorted ranges of arbitrary
 * size. `merge_fun` will be called for each merged element.
 *
 * @param a       the first range
 * @param a_size  the size of the first range
 * @param b       the second range
 * @param b_size  the size of the second range
 * @param group   the group that executes the merge
 * @param merge_fn  the callback that is being called for each merged element.
 *                  It takes five parameters:
 *                  `IndexType a_idx, ValueType a_val,
 *                   IndexType b_idx, ValueType b_val, IndexType c_index`.
 *                   `*_val` and `*_idx` are the values resp. the indices of the
 *                   values from a/b being compared at output index `c_index`.
 */
template <typename IndexType, typename ValueType, typename Group,
          typename Callback>
__device__ void group_merge(const ValueType *a, IndexType a_size,
                            const ValueType *b, IndexType b_size, Group group,
                            Callback merge_fn)
{
    auto c_size = a_size + b_size;
    IndexType a_begin{};
    IndexType b_begin{};
    IndexType c_begin{};
    auto lane = IndexType(group.thread_rank());
    auto a_cur = detail::checked_load(a, a_begin + lane, a_size);
    auto b_cur = detail::checked_load(b, b_begin + lane, a_size);
    while (c_begin < c_size) {
        auto merge_size = min<int>(group.size(), c_size - c_begin);
        auto merge_result = group_merge_step(a_cur, b_cur, merge_size, group);
        if (c_begin + lane < c_size) {
            merge_fn(merge_result.a_idx, merge_result.a_val, merge_result.b_idx,
                     merge_result.b_val, c_begin + lane);
        }
        auto a_advance = merge_result.a_advance;
        auto b_advance = merge_result.b_advance;
        a_begin += a_advance;
        b_begin += b_advance;
        c_begin += group.size();

        // shuffle the unmerged elements to the front
        a_cur = group.shfl_down(a_cur, a_advance);
        b_cur = group.shfl_down(b_cur, b_advance);
        /*
         * To optimize memory access, we load the new elements for `a` and `b`
         * with a single load instruction:
         * the lower part of the group loads new elements for `a`
         * the upper part of the group loads new elements for `b`
         * `load_lane` is the part-local lane idx
         * The elements for `a` have to be shuffled up afterwards.
         */
        auto load_a = lane < a_advance;
        auto load_lane = load_a ? lane : lane - a_advance;
        auto load_source = load_a ? a : b;
        auto load_begin = load_a ? a_begin + b_advance : b_begin + a_advance;
        auto load_size = load_a ? a_size : b_size;

        auto load_idx = load_begin + load_lane;
        auto loaded = detail::checked_load(load_source, load_idx, load_size);
        // shuffle the `a` values to the end of the warp
        auto lower_loaded = group.shfl_up(loaded, b_advance);
        a_cur = lane < b_advance ? a_cur : lower_loaded;
        b_cur = lane < a_advance ? b_cur : loaded;
    }
}


/**
 * @internal
 * Warp-parallel merge algorithm that merges two sorted ranges of arbitrary
 * size, where the first range is stored in two halves.
 * `merge_fun` will be called for each merged element.
 *
 * @param a1  the first half of the first range
 * @param a1_begin  the beginning offset of the first half of the first range
 * @param a1_size the size of the first half of the first range.
 * @param a2  the second half of the first range
 * @param a2_begin  the beginning offset of the second half of the first range
 * @param a2_size the size of the second half of the first range.
 * @param b  the second range
 * @param b_size the size of the second range
 * @param group   the group that executes the merge
 * @param merge_fn  the callback that is being called for each merged element.
 *                  It takes five parameters:
 *                  `IndexType a_idx, ValueType a_val,
 *                   IndexType b_idx, ValueType b_val, IndexType c_index`.
 *                   `*_val` and `*_idx` are the values resp. the indices of the
 *                   values from a/b being compared at output index `c_index`.
 */
template <typename IndexType, typename ValueType, typename Group,
          typename Callback>
__device__ void group_merge_3way(const ValueType *a1, IndexType a1_size,
                                 const ValueType *a2, IndexType a2_size,
                                 const ValueType *b, IndexType b_size,
                                 Group group, Callback merge_fn)
{
    auto a_size = a1_size + a2_size;
    auto c_size = a_size + b_size;
    IndexType a_begin{};
    IndexType b_begin{};
    IndexType c_begin{};
    auto lane = IndexType(group.thread_rank());
    auto a = lane < a1_size ? a1 : a2 - a1_size;
    auto a_cur = detail::checked_load(a, a_begin + lane, a_size);
    auto b_cur = detail::checked_load(b, b_begin + lane, a_size);
    while (c_begin < c_size) {
        auto merge_size = min<int>(group.size(), c_size - c_begin);
        auto merge_result = group_merge_step(a_cur, b_cur, merge_size, group);
        if (c_begin + lane < c_size) {
            merge_fn(merge_result.a_idx, merge_result.a_val, merge_result.b_idx,
                     merge_result.b_val, c_begin + lane);
        }
        auto a_advance = merge_result.a_advance;
        auto b_advance = merge_result.b_advance;
        a_begin += a_advance;
        b_begin += b_advance;
        c_begin += group.size();
        a = a_begin + lane < a1_size ? a1 : a2 - a1_size;

        // shuffle the unmerged elements to the front
        a_cur = group.shfl_down(a_cur, a_advance);
        b_cur = group.shfl_down(b_cur, b_advance);
        /*
         * To optimize memory access, we load the new elements for `a` and `b`
         * with a single load instruction:
         * the lower part of the group loads new elements for `a`
         * the upper part of the group loads new elements for `b`
         * `load_lane` is the part-local lane idx
         * The elements for `a` have to be shuffled up afterwards.
         */
        auto load_a = lane < a_advance;
        auto load_lane = load_a ? lane : lane - a_advance;
        auto load_source = load_a ? a : b;
        auto load_begin = load_a ? a_begin + b_advance : b_begin + a_advance;
        auto load_size = load_a ? a_size : b_size;

        auto load_idx = load_begin + load_lane;
        auto loaded = detail::checked_load(load_source, load_idx, load_size);
        // shuffle the `a` values to the end of the warp
        auto lower_loaded = group.shfl_up(loaded, b_advance);
        a_cur = lane < b_advance ? a_cur : lower_loaded;
        b_cur = lane < a_advance ? b_cur : loaded;
    }
}


/**
 * @internal
 * Sequential merge algorithm that merges two sorted ranges of arbitrary
 * size. `merge_fun` will be called for each merged element.
 *
 * @param a  the first range
 * @param a_size the size of the first range
 * @param b  the second range
 * @param b_size the size of the second range
 * @param merge_fn  the callback that is being called for each merged element.
 *                  It takes five parameters:
 *                  `IndexType a_idx, ValueType a_val,
 *                   IndexType b_idx, ValueType b_val, IndexType c_index`.
 *                   `*_val` and `*_idx` are the values resp. the indices of the
 *                   values from a/b being compared at output index `c_index`.
 */
template <typename IndexType, typename ValueType, typename Callback>
__device__ void sequential_merge(const ValueType *a, IndexType a_size,
                                 const ValueType *b, IndexType b_size,
                                 Callback merge_fn)
{
    auto c_size = a_size + b_size;
    IndexType a_begin{};
    IndexType b_begin{};
    IndexType c_begin{};
    auto a_cur = detail::checked_load(a, a_begin, a_size);
    auto b_cur = detail::checked_load(b, b_begin, b_size);
    while (c_begin < c_size) {
        merge_fn(a_begin, a_cur, b_begin, b_cur, c_begin);
        auto a_advance = a_cur < b_cur;
        auto b_advance = !a_advance;
        a_begin += a_advance;
        b_begin += b_advance;
        c_begin++;

        auto load = a_advance ? a : b;
        auto load_size = a_advance ? a_size : b_size;
        auto load_idx = a_advance ? a_begin : b_begin;
        auto loaded = detail::checked_load(load, load_idx, load_size);
        a_cur = a_advance ? loaded : a_cur;
        b_cur = b_advance ? loaded : b_cur;
    }
}


/**
 * @internal
 * Sequential merge algorithm that merges two sorted ranges of arbitrary
 * size, where the first range is stored in two halves.
 * `merge_fun` will be called for each merged element.
 *
 * @param a1  the first half of the first range
 * @param a1_begin  the beginning offset of the first half of the first range
 * @param a1_size the size of the first half of the first range.
 * @param a2  the second half of the first range
 * @param a2_begin  the beginning offset of the second half of the first range
 * @param a2_size the size of the second half of the first range.
 * @param b  the second range
 * @param b_size the size of the second range
 * @param merge_fn  the callback that is being called for each merged element.
 *                  It takes five parameters:
 *                  `IndexType a_idx, ValueType a_val,
 *                   IndexType b_idx, ValueType b_val, IndexType c_index`.
 *                   `*_val` and `*_idx` are the values resp. the indices of the
 *                   values from a/b being compared at output index `c_index`.
 */
template <typename IndexType, typename ValueType, typename Callback>
__device__ void sequential_merge_3way(const ValueType *a1, IndexType a1_size,
                                      const ValueType *a2, IndexType a2_size,
                                      const ValueType *b, IndexType b_size,
                                      Callback merge_fn)
{
    auto a = a1_size > 0 ? a1 : a2;
    auto a_size = a1_size + a2_size;
    auto c_size = a_size + b_size;
    IndexType a_begin{};
    IndexType b_begin{};
    IndexType c_begin{};
    auto a_cur = detail::checked_load(a, a_begin, a_size);
    auto b_cur = detail::checked_load(b, b_begin, b_size);
    while (c_begin < c_size) {
        merge_fn(a_begin, a_cur, b_begin, b_cur, c_begin);
        auto a_advance = a_cur < b_cur;
        auto b_advance = !a_advance;
        a_begin += a_advance;
        b_begin += b_advance;
        c_begin++;
        if (a_begin == a1_size) {
            a = a2 - a1_size;
        }

        auto load = a_advance ? a : b;
        auto load_size = a_advance ? a_size : b_size;
        auto load_idx = a_advance ? a_begin : b_begin;
        auto loaded = detail::checked_load(load, load_idx, load_size);
        a_cur = a_advance ? loaded : a_cur;
        b_cur = b_advance ? loaded : b_cur;
    }
}