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

#include <algorithm>


namespace gko {
namespace kernels {
namespace omp {


#ifdef __GNUG__
#define GKO_GCC_EXPECT_LIKELY(v) __builtin_expect(v, true)
#else
#define GKO_GCC_EXPECT_LIKELY(v) v
#endif

#ifdef _MSC_VER
#define GKO_COMPARATOR(x, y)                            \
    {                                                   \
        const auto should_swap = !comp(arr[x], arr[y]); \
        const auto tmp = arr[x];                        \
        arr[x] = should_swap ? arr[y] : arr[x];         \
        arr[y] = should_swap ? tmp : arr[y];            \
    }
#else
#define GKO_COMPARATOR(x, y)                                         \
    {                                                                \
        const auto should_swap = !comp(arr[x], arr[y]);              \
        const auto tx = arr[x];                                      \
        const auto ty = arr[y];                                      \
        arr[x] = ((-(should_swap)) & ty) | ((should_swap - 1) & tx); \
        arr[y] = ((-(should_swap)) & tx) | ((should_swap - 1) & ty); \
    }
#endif


/**
 * Sorts elements, such that comp(arr[i]), arr[i + 1]) is true for all
 * i up to n - 1.
 * Features a fast path for n < 128, especially performant for n < 8.
 * For larger sizes delegates to std::sort.
 */
template <typename TargetType, typename Functor>
void sort_small(TargetType *arr, size_type n, Functor comp)
{
    if (n < 2) {
        return;
    }
    if (n == 2) {
        GKO_COMPARATOR(0, 1);
        return;
    }
    if (n == 3) {
        GKO_COMPARATOR(1, 2);
        GKO_COMPARATOR(0, 2);
        GKO_COMPARATOR(0, 1);
        return;
    }
    if (n == 4) {
        // clang-format off
        GKO_COMPARATOR(0, 1)GKO_COMPARATOR(2, 3)
        GKO_COMPARATOR(0, 2)GKO_COMPARATOR(1, 3)
        GKO_COMPARATOR(1, 2);
        // clang-format on
        return;
    }
    if (n == 5) {
        // clang-format off
        GKO_COMPARATOR(0, 4)GKO_COMPARATOR(1, 3)
        GKO_COMPARATOR(0, 2)
        GKO_COMPARATOR(2, 4)GKO_COMPARATOR(0, 1)
        GKO_COMPARATOR(2, 3)GKO_COMPARATOR(1, 4)
        GKO_COMPARATOR(1, 2)GKO_COMPARATOR(3, 4);
        // clang-format on
        return;
    }
    if (n == 6) {
        // clang-format off
        GKO_COMPARATOR(0, 4)GKO_COMPARATOR(1, 5)
        GKO_COMPARATOR(0, 2)GKO_COMPARATOR(1, 3)
        GKO_COMPARATOR(2, 4)GKO_COMPARATOR(3, 5)GKO_COMPARATOR(0, 1)
        GKO_COMPARATOR(2, 3)GKO_COMPARATOR(4, 5)
        GKO_COMPARATOR(1, 4)
        GKO_COMPARATOR(1, 2)GKO_COMPARATOR(3, 4);
        // clang-format on
        return;
    }
    if (n == 7) {
        // clang-format off
        GKO_COMPARATOR(0,4)GKO_COMPARATOR(1,5)GKO_COMPARATOR(2,6)
        GKO_COMPARATOR(0,2)GKO_COMPARATOR(1,3)GKO_COMPARATOR(4,6)
        GKO_COMPARATOR(2,4)GKO_COMPARATOR(3,5)GKO_COMPARATOR(0,1)
        GKO_COMPARATOR(2,3)GKO_COMPARATOR(4,5)
        GKO_COMPARATOR(1,4)GKO_COMPARATOR(3,6)
        GKO_COMPARATOR(1,2)GKO_COMPARATOR(3,4)GKO_COMPARATOR(5,6);
        // clang-format on
        return;
    }
    if (n < 128) {
        // clang-format off
        GKO_COMPARATOR(0,4)GKO_COMPARATOR(1,5)GKO_COMPARATOR(2,6)GKO_COMPARATOR(3,7)
        GKO_COMPARATOR(0,2)GKO_COMPARATOR(1,3)GKO_COMPARATOR(4,6)GKO_COMPARATOR(5,7)
        GKO_COMPARATOR(2,4)GKO_COMPARATOR(3,5)GKO_COMPARATOR(0,1)GKO_COMPARATOR(6,7)
        GKO_COMPARATOR(2,3)GKO_COMPARATOR(4,5)
        GKO_COMPARATOR(1,4)GKO_COMPARATOR(3,6)
        GKO_COMPARATOR(1,2)GKO_COMPARATOR(3,4)GKO_COMPARATOR(5,6);
        // clang-format on

        const int8 m = n < 48 ? n : 48;
        for (int8 i = 8; i < m; ++i) {
            auto j = i;
            const auto tmp = arr[j];
            for (; GKO_GCC_EXPECT_LIKELY(j >= 2 && comp(tmp, arr[j - 2]));
                 j -= 2) {
                std::swap(arr[j - 1], arr[j]);
                std::swap(arr[j - 2], arr[j - 1]);
            }

            if (j == 0) {
                continue;
            }
            GKO_COMPARATOR(j - 1, j);
        }
        for (int8 i = 48; i < n; ++i) {
            auto j = i;
            const auto tmp = arr[j];
            for (; GKO_GCC_EXPECT_LIKELY(j >= 4 && comp(tmp, arr[j - 4]));
                 j -= 4) {
                std::swap(arr[j - 1], arr[j]);
                std::swap(arr[j - 2], arr[j - 1]);
                std::swap(arr[j - 3], arr[j - 2]);
                std::swap(arr[j - 4], arr[j - 3]);
            }

            if (j == 0) {
                continue;
            }
            GKO_COMPARATOR(j - 1, j);
            if (j == 1) {
                continue;
            }
            GKO_COMPARATOR(j - 2, j - 1);
            if (j == 2) {
                continue;
            }
            GKO_COMPARATOR(j - 3, j - 2);
        }
        return;
    }
    std::sort(arr, arr + n, comp);
}


#undef GKO_GCC_EXPECT_LIKELY
#undef GKO_COMPARATOR


}  // namespace omp
}  // namespace kernels
}  // namespace gko
