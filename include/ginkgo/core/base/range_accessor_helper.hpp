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

#ifndef GKO_CORE_BASE_RANGE_ACCESSOR_HELPER_HPP_
#define GKO_CORE_BASE_RANGE_ACCESSOR_HELPER_HPP_

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace accessor {
/**
 * This namespace contains helper functionality for the accessors.
 *
 * @note This namespace is not part of the public interface and can change
 * without notice.
 */
namespace helper {


/**
 * This helper runs from first to last dimension in order to compute the index.
 * The index is computed like this:
 * indices: x1, x2, x3, ...
 * compute(stride, x1, x2, x3) -> x1 * stride[0] + x2 * stride[1] + x3
 */
template <typename IndexType, int32 total_dim, int32 current_iter = total_dim>
struct row_major_index {
    static constexpr int32 current_dim{total_dim - current_iter};
    template <typename FirstType, typename... Indices>
    static constexpr GKO_ATTRIBUTES GKO_INLINE IndexType
    compute(const dim<total_dim> &size,
            const std::array<IndexType, (total_dim > 1 ? total_dim - 1 : 0)>
                &stride,
            FirstType &&first, Indices &&... idxs)
    {
        return GKO_ASSERT(first < size[current_dim]),
               first * stride[current_dim] +
                   row_major_index<IndexType, total_dim, current_iter - 1>::
                       compute(size, stride, std::forward<Indices>(idxs)...);
    }
};

template <typename IndexType, int32 total_dim>
struct row_major_index<IndexType, total_dim, 1> {
    static constexpr int32 current_dim{total_dim - 1};
    template <typename FirstType>
    static constexpr GKO_ATTRIBUTES GKO_INLINE IndexType
    compute(const dim<total_dim> &size,
            const std::array<IndexType, (total_dim > 1 ? total_dim - 1 : 0)>,
            FirstType &&first)
    {
        return GKO_ASSERT(first < size[current_dim]), first;
    }
};


/**
 * This helper must be walked through in reverse (highest to lowest dimension)
 * in order to properly use the stride for the scalar indices. The mask
 * indicates which indices are actually used. The least significant bit set
 * means using the last index, second bit corresponds to the second last
 * dimension, and so on.
 *
 * Basically, this computes indices in a similar fashion as `row_major_index`
 * while having a mask signaling which indices to skip.
 *
 * Example: Mask = 0b01101
 * compute(stride, tuple(x1, x2, x3, x4))
 * -> x4 + x2 * stride[2] + x1 * stride[1]
 */
template <typename IndexType, int32 total_dim, int32 mask,
          int32 current_dim = total_dim - 1,
          int32 current_scalar = total_dim - 1>
struct row_major_mask_index {
    static constexpr int32 msb_mask{static_cast<int32>(1)};
    static constexpr int32 stride_size = total_dim > 1 ? total_dim - 1 : 0;
    static constexpr int32 use_current_stride{mask & static_cast<int32>(1)};
    static constexpr bool is_last_stride_dim{stride_size == 0 ||
                                             current_scalar == total_dim - 1};

    template <typename... Args>
    static constexpr GKO_ATTRIBUTES GKO_INLINE IndexType
    compute(const dim<total_dim> &size,
            const std::array<IndexType, stride_size> &stride,
            std::tuple<Args...> indices)
    {
        return GKO_ASSERT(std::get<current_dim>(indices) < size[current_dim]),
               (use_current_stride
                    ? std::get<current_dim>(indices) *
                          (is_last_stride_dim ? 1 : stride[current_scalar])
                    : 0) +
                   row_major_mask_index<
                       IndexType, total_dim, (mask >> 1), current_dim - 1,
                       current_scalar - use_current_stride>::compute(size,
                                                                     stride,
                                                                     indices);
    }
};

template <typename IndexType, int32 total_dim, int32 mask, int32 current_scalar>
struct row_major_mask_index<IndexType, total_dim, mask, 0, current_scalar> {
    static constexpr int32 stride_size = total_dim > 1 ? total_dim - 1 : 0;
    static constexpr int32 use_current_stride{mask & static_cast<int32>(1)};
    static constexpr int32 current_dim{0};
    static constexpr bool is_last_stride_dim{stride_size == 0 ||
                                             current_scalar == total_dim - 1};

    template <typename... Args>
    static constexpr GKO_ATTRIBUTES GKO_INLINE IndexType
    compute(const dim<total_dim> &size,
            const std::array<IndexType, stride_size> &stride,
            std::tuple<Args...> indices)
    {
        return GKO_ASSERT(std::get<current_dim>(indices) < size[current_dim]),
               (use_current_stride
                    ? std::get<current_dim>(indices) *
                          (is_last_stride_dim ? 1 : stride[current_scalar])
                    : 0);
    }
};


/**
 * Evaluates if all Args fulfill std::is_integral
 */
template <typename... Args>
struct are_all_integral : public std::true_type {
};

template <typename First, typename... Args>
struct are_all_integral<First, Args...>
    : public std::conditional<std::is_integral<std::decay_t<First>>::value,
                              are_all_integral<Args...>,
                              std::false_type>::type {
};


namespace detail {


template <bool has_span, typename... Args>
struct are_span_compatible_impl
    : public std::integral_constant<bool, has_span> {
};

template <bool has_span, typename First, typename... Args>
struct are_span_compatible_impl<has_span, First, Args...>
    : public std::conditional<
          std::is_integral<std::decay_t<First>>::value ||
              std::is_same<std::decay_t<First>, span>::value,
          are_span_compatible_impl<
              has_span || std::is_same<std::decay_t<First>, span>::value,
              Args...>,
          std::false_type>::type {
};


}  // namespace detail


/**
 * Evaluates if at least one type of Args is a gko::span and the others either
 * also gko::span or fulfill std::is_integral
 */
template <typename... Args>
using are_span_compatible = detail::are_span_compatible_impl<false, Args...>;


namespace detail {


template <typename ValueType, size_type Iter, size_type N>
constexpr GKO_ATTRIBUTES GKO_INLINE std::enable_if_t<Iter == N, ValueType>
mult_array(const dim<N> &size)
{
    return 1;
}

template <typename ValueType, size_type Iter, size_type N>
constexpr GKO_ATTRIBUTES GKO_INLINE std::enable_if_t<Iter + 1 <= N, ValueType>
mult_array(const dim<N> &size)
{
    return size[Iter] * mult_array<ValueType, Iter + 1>(size);
}


template <typename ValueType, size_type Iter = 1, size_type N, typename... Args>
constexpr GKO_ATTRIBUTES GKO_INLINE
    std::enable_if_t<N == 0 || (Iter == N && Iter == sizeof...(Args) + 1),
                     std::array<ValueType, N == 0 ? 0 : N - 1>>
    extract_factorization(const dim<N> &size, Args... args)
{
    return {{args...}};
}


template <typename ValueType, size_type Iter = 1, size_type N, typename... Args>
constexpr GKO_ATTRIBUTES GKO_INLINE std::enable_if_t<
    (Iter < N) && (Iter == sizeof...(Args) + 1), std::array<ValueType, N - 1>>
extract_factorization(const dim<N> &size, Args... args)
{
    return extract_factorization<ValueType, Iter + 1>(
        size, args..., mult_array<ValueType, Iter>(size));
}


}  // namespace detail


/**
 * Computes the default stride array from a size, assuming there is no padding
 */
template <typename ValueType, size_type N>
constexpr GKO_ATTRIBUTES GKO_INLINE auto compute_stride_array(
    const dim<N> &size)
{
    return detail::extract_factorization<ValueType>(size);
}


namespace detail {


template <size_type Iter, size_type N, typename Callable, typename... Indices>
GKO_ATTRIBUTES std::enable_if_t<Iter == N> multidim_for_each_impl(
    const dim<N> &size, Callable &&callable, Indices... indices)
{
    static_assert(Iter == sizeof...(Indices),
                  "Number arguments must match current iteration!");
    callable(indices...);
}

template <size_type Iter, size_type N, typename Callable, typename... Indices>
GKO_ATTRIBUTES std::enable_if_t<(Iter < N)> multidim_for_each_impl(
    const dim<N> &size, Callable &&callable, Indices... indices)
{
    static_assert(Iter == sizeof...(Indices),
                  "Number arguments must match current iteration!");
    for (size_type i = 0; i < size[Iter]; ++i) {
        multidim_for_each_impl<Iter + 1>(size, std::forward<Callable>(callable),
                                         indices..., i);
    }
}


}  // namespace detail


/**
 * Creates a recursive for-loop for each dimension and calls dest(indices...) =
 * source(indices...)
 */
template <size_type N, typename Callable>
GKO_ATTRIBUTES void multidim_for_each(const dim<N> &size, Callable &&callable)
{
    detail::multidim_for_each_impl<0>(size, std::forward<Callable>(callable));
}


namespace detail {


template <size_type Iter, size_type N>
GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<Iter == N, int>
spans_in_size(const dim<N> &size)
{
    return 0;
}

template <size_type Iter, size_type N, typename First, typename... Remaining>
GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<(Iter < N), int>
spans_in_size(const dim<N> &size, First &&first, Remaining &&... remaining)
{
    static_assert(sizeof...(Remaining) + 1 == N - Iter,
                  "Number of remaining spans must be equal to N - Iter");
    return GKO_ASSERT(span{first}.is_valid()),
           GKO_ASSERT(span{first} <= span{size[Iter]}),
           spans_in_size<Iter + 1>(size, std::forward<Remaining>(remaining)...);
}


}  // namespace detail


template <size_type N, typename... Spans>
GKO_ATTRIBUTES GKO_INLINE constexpr int validate_spans(const dim<N> &size,
                                                       Spans... spans)
{
    return detail::spans_in_size<0>(size, std::forward<Spans>(spans)...);
}


}  // namespace helper
}  // namespace accessor
}  // namespace gko


#endif  // GKO_CORE_BASE_RANGE_ACCESSOR_HELPER_HPP_
