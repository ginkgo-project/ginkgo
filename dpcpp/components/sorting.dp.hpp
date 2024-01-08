// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_SORTING_DP_HPP_
#define GKO_DPCPP_COMPONENTS_SORTING_DP_HPP_


#include <CL/sycl.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


namespace detail {


/**
 * @internal
 * Bitonic sorting operation for two elements.
 *
 * @param reverse  sorts in ascending order if `false` and
 *                 descending order if `true`.
 */
template <typename ValueType>
__dpct_inline__ void bitonic_cas(ValueType& a, ValueType& b, bool reverse)
{
    auto tmp = a;
    bool cmp = (a < b) != reverse;
    a = cmp ? a : b;
    b = cmp ? b : tmp;
}


/**
 * @internal
 * This is a recursive implementation of a bitonic sorting network,
 * executed sequentially on locally stored data.
 *
 * Based on Batcher, "Sorting Networks and Their Applications", 1968.
 */
template <typename ValueType, int num_elements>
struct bitonic_local {
    using half = bitonic_local<ValueType, num_elements / 2>;
    static_assert(num_elements > 0, "number of elements must be positive");
    static_assert((num_elements & (num_elements - 1)) == 0,
                  "number of elements must be a power of two");

    // merges two bitonic sequences els[0, n / 2), els[n / 2, n)
    __dpct_inline__ static void merge(ValueType* els, bool reverse)
    {
        auto els_mid = els + (num_elements / 2);
        for (int i = 0; i < num_elements / 2; ++i) {
            bitonic_cas(els[i], els_mid[i], reverse);
        }
        half::merge(els, reverse);
        half::merge(els_mid, reverse);
    }

    // sorts an unsorted sequence els [0, n)
    __dpct_inline__ static void sort(ValueType* els, bool reverse)
    {
        auto els_mid = els + (num_elements / 2);
        // sort first half normally
        half::sort(els, reverse);
        // sort second half reversed
        half::sort(els_mid, !reverse);
        // merge two halves
        merge(els, reverse);
    }
};

template <typename ValueType>
struct bitonic_local<ValueType, 1> {
    // nothing to do for a single element
    __dpct_inline__ static void merge(ValueType*, bool) {}
    __dpct_inline__ static void sort(ValueType*, bool) {}
};


/**
 * @internal
 * This is a recursive implementation of a bitonic sorting network,
 * executed in parallel within a warp using lane shuffle instructions.
 *
 * Based on Hou et al., "Fast Segmented Sort on GPUs", 2017.
 */
template <typename ValueType, int num_local, int num_threads>
struct bitonic_warp {
    constexpr static auto num_elements = num_local * num_threads;
    using half = bitonic_warp<ValueType, num_local, num_threads / 2>;
    static_assert(num_threads > 0, "number of threads must be positive");
    static_assert(num_local > 0, "number of local elements must be positive");
    static_assert(
        config::warp_size % num_threads == 0 &&
            num_threads <= config::warp_size,
        "number of threads must be a power of two smaller than warp_size");

    // check if we are in the upper half of all threads in this group
    // this is important as
    // 1. for sorting, we have to reverse the sort order in the upper half
    // 2. for merging, we have to determine for the XOR shuffle if we are
    //    the "smaller" thread, as this thread gets the "smaller" element.
    __dpct_inline__ static bool upper_half(sycl::nd_item<3> item_ct1)
    {
        return bool(item_ct1.get_local_id(2) & (num_threads / 2));
    }

    __dpct_inline__ static void merge(ValueType* els, bool reverse,
                                      sycl::nd_item<3> item_ct1)
    {
        auto tile = group::tiled_partition<config::warp_size>(
            group::this_thread_block(item_ct1));
        auto new_reverse = reverse != upper_half(item_ct1);
        for (int i = 0; i < num_local; ++i) {
            auto other = tile.shfl_xor(els[i], num_threads / 2);
            bitonic_cas(els[i], other, new_reverse);
        }
        half::merge(els, reverse, item_ct1);
    }

    __dpct_inline__ static void sort(ValueType* els, bool reverse,
                                     sycl::nd_item<3> item_ct1)
    {
        auto new_reverse = reverse != upper_half(item_ct1);
        half::sort(els, new_reverse, item_ct1);
        merge(els, reverse, item_ct1);
    }
};

template <typename ValueType, int NumLocalElements>
struct bitonic_warp<ValueType, NumLocalElements, 1> {
    using local = bitonic_local<ValueType, NumLocalElements>;
    __dpct_inline__ static void merge(ValueType* els, bool reverse,
                                      sycl::nd_item<3> item_ct1)
    {
        local::merge(els, reverse);
    }
    __dpct_inline__ static void sort(ValueType* els, bool reverse,
                                     sycl::nd_item<3> item_ct1)
    {
        local::sort(els, reverse);
    }
};


/**
 * @internal
 * This is a recursive implementation of a bitonic sorting network,
 * executed in parallel in a thread block using shared memory.
 *
 * We use a tiled storage pattern to avoid memory bank collisions on shared
 * memory accesses, see @ref shared_idx.
 */
template <typename ValueType, int num_local, int num_threads, int num_groups,
          int num_total_threads>
struct bitonic_global {
    constexpr static auto num_elements = num_local * num_threads * num_groups;
    using half = bitonic_global<ValueType, num_local, num_threads,
                                num_groups / 2, num_total_threads>;
    static_assert(num_groups > 0, "number of groups must be positive");
    static_assert(num_threads > 0,
                  "number of threads per group must be positive");
    static_assert(num_local > 0, "number of local elements must be positive");
    static_assert(num_total_threads > 0, "number of threads must be positive");
    static_assert(32 % num_groups == 0,
                  "num_groups must be a power of two <= 32");

    __dpct_inline__ static int shared_idx(int local, sycl::nd_item<3> item_ct1)
    {
        auto rank = group::this_thread_block(item_ct1).thread_rank();
        // use the same memory-bank to avoid bank conflicts
        return rank + local * num_total_threads;
    }

    // check if we are in the upper half of all groups in this block
    // this is important as for sorting, we have to reverse the sort order in
    // the upper half
    __dpct_inline__ static bool upper_half(sycl::nd_item<3> item_ct1)
    {
        auto rank = group::this_thread_block(item_ct1).thread_rank();
        return bool(rank & (num_groups * num_threads / 2));
    }

    __dpct_inline__ static void merge(ValueType* local_els,
                                      ValueType* shared_els, bool reverse,
                                      sycl::nd_item<3> item_ct1)
    {
        group::this_thread_block(item_ct1).sync();
        auto upper_shared_els = shared_els + (num_groups * num_threads / 2);
        // only the lower group executes the CAS
        if (!upper_half(item_ct1)) {
            for (int i = 0; i < num_local; ++i) {
                auto j = shared_idx(i, item_ct1);
                bitonic_cas(shared_els[j], upper_shared_els[j], reverse);
            }
        }
        half::merge(local_els, shared_els, reverse, item_ct1);
    }

    __dpct_inline__ static void sort(ValueType* local_els,
                                     ValueType* shared_els, bool reverse,
                                     sycl::nd_item<3> item_ct1)
    {
        auto new_reverse = reverse != upper_half(item_ct1);
        half::sort(local_els, shared_els, new_reverse, item_ct1);
        merge(local_els, shared_els, reverse, item_ct1);
    }
};

template <typename ValueType, int num_local, int num_threads,
          int num_total_threads>
struct bitonic_global<ValueType, num_local, num_threads, 1, num_total_threads> {
    using warp = bitonic_warp<ValueType, num_local, num_threads>;

    __dpct_inline__ static int shared_idx(int local, sycl::nd_item<3> item_ct1)
    {
        // use the indexing from the general struct
        return bitonic_global<ValueType, num_local, num_threads, 2,
                              num_total_threads>::shared_idx(local, item_ct1);
    }

    __dpct_inline__ static void merge(ValueType* local_els,
                                      ValueType* shared_els, bool reverse,
                                      sycl::nd_item<3> item_ct1)
    {
        group::this_thread_block(item_ct1).sync();
        for (int i = 0; i < num_local; ++i) {
            local_els[i] = shared_els[shared_idx(i, item_ct1)];
        }
        warp::merge(local_els, reverse, item_ct1);
        for (int i = 0; i < num_local; ++i) {
            shared_els[shared_idx(i, item_ct1)] = local_els[i];
        }
    }

    __dpct_inline__ static void sort(ValueType* local_els,
                                     ValueType* shared_els, bool reverse,
                                     sycl::nd_item<3> item_ct1)
    {
        auto rank = group::this_thread_block(item_ct1).thread_rank();
        // This is the first step, so we don't need to load from shared memory
        warp::sort(local_els, reverse, item_ct1);
        // store the sorted elements in shared memory
        for (int i = 0; i < num_local; ++i) {
            shared_els[shared_idx(i, item_ct1)] = local_els[i];
        }
    }
};


}  // namespace detail


/**
 * @internal
 *
 * This function sorts elements within a thread block.
 *
 * It takes a local array of elements and the pointer to a shared buffer of size
 * `num_elements` as input. After the execution, the thread with rank `i` in the
 * thread block (determined by `group::this_thread_block().thread_rank()`) has
 * the elements at index `num_local * i` up to `num_local * i + (num_local - 1)`
 * in the sorted sequence stored in its `local_elements` at index 0 up to
 * `num_local - 1`.
 *
 * @note The shared-memory buffer uses a striped layout to limit bank
 *       collisions, so it should not directly be used to access elements from
 *       the sorted sequence. If `num_elements <= num_local * warp_size`, the
 *       algorithm doesn't use/need the shared-memory buffer, so it can be null.
 *
 * @param local_elements  the `num_local` input/output elements from this
 *                        thread.
 * @param shared_elements  the shared-memory buffer of size `num_elements`
 * @tparam num_elements  the number of elements - it must be a power of two!
 * @tparam num_local  the number of elements stored per thread - it must be a
 *                    power of two!
 * @tparam ValueType  the type of the elements to be sorted - it must implement
 *                    the less-than operator!
 */
template <int num_elements, int num_local, typename ValueType>
__dpct_inline__ void bitonic_sort(ValueType* local_elements,
                                  ValueType* shared_elements,
                                  sycl::nd_item<3> item_ct1)
{
    constexpr auto num_threads = num_elements / num_local;
    constexpr auto num_warps = num_threads / config::warp_size;
    static_assert(num_threads <= config::max_block_size,
                  "bitonic_sort exceeds thread block");
    if (num_warps > 1) {
        // these checks are necessary since the `if` is not evaluated at
        // compile-time so even though the branch is never taken, it still gets
        // instantiated and must thus compile.
        constexpr auto _num_warps = num_warps <= 1 ? 1 : num_warps;
        constexpr auto _num_threads =
            num_threads <= config::warp_size ? config::warp_size : num_threads;
        detail::bitonic_global<ValueType, num_local, config::warp_size,
                               _num_warps, _num_threads>::sort(local_elements,
                                                               shared_elements,
                                                               false, item_ct1);
    } else {
        constexpr auto _num_threads = num_warps > 1 ? 1 : num_threads;
        detail::bitonic_warp<ValueType, num_local, _num_threads>::sort(
            local_elements, false, item_ct1);
    }
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_SORTING_DP_HPP_
