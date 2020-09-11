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

#ifndef GKO_OMP_COMPONENTS_SORT_SMALL_HPP_
#define GKO_OMP_COMPONENTS_SORT_SMALL_HPP_


#include <algorithm>


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace omp {


#ifdef __GNUG__
#define GKO_GNU_EXPECT_LIKELY(v) __builtin_expect(v, true)
#else
#define GKO_GNU_EXPECT_LIKELY(v) v
#endif  // defined __GNUG__


namespace detail {


#ifdef _MSC_VER
template <typename TargetType, typename Functor>
inline void comparator(TargetType *arr, size_type x, size_type y, Functor comp)
{
    const auto should_swap = !comp(arr[x], arr[y]);
    const auto tmp = arr[x];
    arr[x] = should_swap ? arr[y] : arr[x];
    arr[y] = should_swap ? tmp : arr[y];
}
#else
template <typename TargetType, typename Functor>
inline void comparator(TargetType *arr, size_type x, size_type y, Functor comp)
{
    const auto tx = arr[x];
    const auto ty = arr[y];
    const auto should_swap = !comp(tx, ty);
    arr[x] = ((-should_swap) & ty) | ((should_swap - 1) & tx);
    arr[y] = ((-should_swap) & tx) | ((should_swap - 1) & ty);
}
#endif  // defined _MSC_VER


}  // namespace detail


/**
 * Sorts elements, such that comp(arr[i]), arr[i + 1]) is true for all
 * i up to n - 1.
 * Features a fast path for n < 128, especially performant for n < 8.
 * For larger sizes delegates to std::sort.
 */
template <typename TargetType, typename Functor>
void sort_small(TargetType *arr, size_type n, Functor comp)
{
    const auto sort2 = [&](size_type x, size_type y) {
        detail::comparator(arr, x, y, comp);
    };

    if (n < 2) {
        return;
    }
    if (n == 2) {
        sort2(0, 1);
        return;
    }
    if (n == 3) {
        sort2(1, 2);
        sort2(0, 2);
        sort2(0, 1);
        return;
    }
    if (n == 4) {
        // clang-format off
        sort2(0,1);sort2(2,3);
        sort2(0,2);sort2(1,3);
        sort2(1,2);
        // clang-format on
        return;
    }
    if (n == 5) {
        // clang-format off
        sort2(0,4);sort2(1,3);
        sort2(0,2);
        sort2(2,4);sort2(0,1);
        sort2(2,3);sort2(1,4);
        sort2(1,2);sort2(3,4);
        // clang-format on
        return;
    }
    if (n == 6) {
        // clang-format off
        sort2(0,4);sort2(1,5);
        sort2(0,2);sort2(1,3);
        sort2(2,4);sort2(3,5);sort2(0,1);
        sort2(2,3);sort2(4,5);
        sort2(1,4);
        sort2(1,2);sort2(3,4);
        // clang-format on
        return;
    }
    if (n == 7) {
        // clang-format off
        sort2(0,4);sort2(1,5);sort2(2,6);
        sort2(0,2);sort2(1,3);sort2(4,6);
        sort2(2,4);sort2(3,5);sort2(0,1);
        sort2(2,3);sort2(4,5);
        sort2(1,4);sort2(3,6);
        sort2(1,2);sort2(3,4);sort2(5,6);
        // clang-format on
        return;
    }
    if (n < 128) {
        // clang-format off
        sort2(0,4);sort2(1,5);sort2(2,6);sort2(3,7);
        sort2(0,2);sort2(1,3);sort2(4,6);sort2(5,7);
        sort2(2,4);sort2(3,5);sort2(0,1);sort2(6,7);
        sort2(2,3);sort2(4,5);
        sort2(1,4);sort2(3,6);
        sort2(1,2);sort2(3,4);sort2(5,6);
        // clang-format on

        const int8 m = n < 48 ? n : 48;
        for (int8 i = 8; i < m; ++i) {
            auto j = i;
            const auto tmp = arr[j];
            for (; GKO_GNU_EXPECT_LIKELY(j >= 2 && comp(tmp, arr[j - 2]));
                 j -= 2) {
                std::swap(arr[j - 1], arr[j]);
                std::swap(arr[j - 2], arr[j - 1]);
            }

            if (j == 0) {
                continue;
            }
            sort2(j - 1, j);
        }
        for (int8 i = 48; i < n; ++i) {
            auto j = i;
            const auto tmp = arr[j];
            for (; GKO_GNU_EXPECT_LIKELY(j >= 4 && comp(tmp, arr[j - 4]));
                 j -= 4) {
                std::swap(arr[j - 1], arr[j]);
                std::swap(arr[j - 2], arr[j - 1]);
                std::swap(arr[j - 3], arr[j - 2]);
                std::swap(arr[j - 4], arr[j - 3]);
            }

            if (j == 0) {
                continue;
            }
            sort2(j - 1, j);
            if (j == 1) {
                continue;
            }
            sort2(j - 2, j - 1);
            if (j == 2) {
                continue;
            }
            sort2(j - 3, j - 2);
        }
        return;
    }
    std::sort(arr, arr + n, comp);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_OMP_COMPONENTS_SORT_SMALL_HPP_
