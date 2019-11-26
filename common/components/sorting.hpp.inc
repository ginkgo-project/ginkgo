/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


template <typename ValueType>
__device__ void bitonic_cas(ValueType &a, ValueType &b, bool reverse)
{
    auto tmp = a;
    bool cmp = (a < b) != reverse;
    a = cmp ? a : b;
    b = cmp ? b : tmp;
}

template <typename ValueType, int num_elements>
struct bitonic_local {
    // # local elements
    using half = bitonic_local<ValueType, num_elements / 2>;

    // merges two bitonic sequences els[0, n / 2), els[n / 2, n)
    __host__ __device__ static void merge(ValueType *els, bool odd)
    {
        auto els_mid = els + (num_elements / 2);
        for (auto i = 0; i < num_elements / 2; ++i) {
            bitonic_cas(els[i], els_mid[i], odd);
        }
        half::merge(els, odd);
        half::merge(els_mid, odd);
    }

    // sorts an unsorted sequence els [0, n)
    __device__ static void sort(ValueType *els, bool reverse)
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
    __device__ static void merge(ValueType *, bool) {}
    __device__ static void sort(ValueType *, bool) {}
};


template <typename ValueType, int num_local, int num_threads>
struct bitonic_warp {
    constexpr static auto num_elements = num_local * num_threads;
    using half = bitonic_warp<ValueType, num_local, num_threads / 2>;
    static_assert(
        config::warp_size % num_threads == 0 &&
            num_threads <= config::warp_size,
        "number of threads must be a power of two smaller than warp_size");

    // check if we are in the upper half of all threads in this group
    // this is important as
    // 1. for sorting, we have to reverse the sort order in the upper half
    // 2. for merging, we have to determine for the XOR shuffle if we are
    //    the "smaller" thread, as this thread gets the "smaller" element.
    __device__ static bool upper_half()
    {
        return bool(threadIdx.x & (num_threads / 2));
    }

    __device__ static void merge(ValueType *els, bool reverse)
    {
        auto tile = group::thread_block_tile<num_threads>{};
        auto new_reverse = reverse != upper_half();
        for (auto i = 0; i < num_local; ++i) {
            auto other = tile.shfl_xor(els[i], num_threads / 2);
            bitonic_cas(els[i], other, new_reverse);
        }
        half::merge(els, reverse);
    }

    __device__ static void sort(ValueType *els, bool reverse)
    {
        auto new_reverse = reverse != upper_half();
        half::sort(els, new_reverse);
        merge(els, reverse);
    }
};

template <typename ValueType, int NumLocalElements>
struct bitonic_warp<ValueType, NumLocalElements, 1> {
    using local = bitonic_local<ValueType, NumLocalElements>;
    __device__ static void merge(ValueType *els, bool reverse)
    {
        local::merge(els, reverse);
    }
    __device__ static void sort(ValueType *els, bool reverse)
    {
        local::sort(els, reverse);
    }
};


template <typename ValueType, int num_local, int num_threads, int num_groups,
          int num_total_threads>
struct bitonic_global {
    constexpr static auto num_elements = num_local * num_threads * num_groups;
    using half = bitonic_global<ValueType, num_local, num_threads,
                                num_groups / 2, num_total_threads>;
    static_assert(32 % num_groups == 0,
                  "num_groups must be a power of two <= 32");

    __device__ static int shared_idx(int local)
    {
        auto rank = group::this_thread_block().thread_rank();
        // use the same memory-bank to avoid bank conflicts
        return rank + local * num_total_threads;
    }

    // check if we are in the upper half of all groups in this block
    // this is important as for sorting, we have to reverse the sort order in
    // the upper half
    __device__ static bool upper_half()
    {
        auto rank = group::this_thread_block().thread_rank();
        return bool(rank & (num_groups * num_threads / 2));
    }

    __device__ static void merge(ValueType *local_els, ValueType *shared_els,
                                 bool reverse)
    {
        group::this_thread_block().sync();
        auto upper_shared_els = shared_els + (num_groups * num_threads / 2);
        // only the lower group executes the CAS
        if (!upper_half()) {
            for (auto i = 0; i < num_local; ++i) {
                auto j = shared_idx(i);
                bitonic_cas(shared_els[j], upper_shared_els[j], reverse);
            }
        }
        half::merge(local_els, shared_els, reverse);
    }

    __device__ static void sort(ValueType *local_els, ValueType *shared_els,
                                bool reverse)
    {
        auto new_reverse = reverse != upper_half();
        half::sort(local_els, shared_els, new_reverse);
        merge(local_els, shared_els, reverse);
    }
};

template <typename ValueType, int num_local, int num_threads,
          int num_total_threads>
struct bitonic_global<ValueType, num_local, num_threads, 1, num_total_threads> {
    using warp = bitonic_warp<ValueType, num_local, num_threads>;

    __device__ static int shared_idx(int local)
    {
        // use the indexing from the general struct
        return bitonic_global<ValueType, num_local, num_threads, 2,
                              num_total_threads>::shared_idx(local);
    }

    __device__ static void merge(ValueType *local_els, ValueType *shared_els,
                                 bool reverse)
    {
        group::this_thread_block().sync();
        for (auto i = 0; i < num_local; ++i) {
            local_els[i] = shared_els[shared_idx(i)];
        }
        warp::merge(local_els, reverse);
        for (auto i = 0; i < num_local; ++i) {
            shared_els[shared_idx(i)] = local_els[i];
        }
    }

    __device__ static void sort(ValueType *local_els, ValueType *shared_els,
                                bool reverse)
    {
        auto rank = group::this_thread_block().thread_rank();
        // This is the first step, so we don't need to load from shared memory
        warp::sort(local_els, reverse);
        // store the sorted elements in shared memory
        for (auto i = 0; i < num_local; ++i) {
            shared_els[shared_idx(i)] = local_els[i];
        }
    }
};


}  // namespace detail


template <int num_elements, int num_local, typename ValueType>
__device__ void bitonic_sort(ValueType *local_elements,
                             ValueType *shared_elements)
{
    constexpr auto num_threads = num_elements / num_local;
    constexpr auto num_warps = num_threads / config::warp_size;
    static_assert(num_threads <= config::max_block_size,
                  "bitonic_sort exceeds thread block");
    if (num_warps > 1) {
        detail::bitonic_global<ValueType, num_local, config::warp_size,
                               num_warps, num_threads>::sort(local_elements,
                                                             shared_elements,
                                                             false);
    } else {
        constexpr auto _num_threads =
            num_warps > 1 ? 1 : num_threads;  // sentinel value to make sure we
                                              // don't break anything
        detail::bitonic_warp<ValueType, num_local, _num_threads>::sort(
            local_elements, false);
    }
}
