/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_ACCESSOR_HELPER_HPP_
#define GKO_CORE_BASE_ACCESSOR_HELPER_HPP_


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
namespace detail {


/**
 * This helper runs from first to last dimension in order to compute the index.
 * The index is computed like this:
 * indices: x1, x2, x3, ...
 * compute(stride, x1, x2, x3) -> x1 * stride[0] + x2 * stride[1] + x3
 */
template <typename ValueType, size_type total_dim, size_type current_iter = 1>
struct row_major_helper_s {
    static_assert(total_dim >= 1, "Dimensionality must be >= 1");
    static_assert(current_iter < total_dim, "Iteration must be < total_dim!");

    static constexpr size_type dim_idx{current_iter - 1};

    template <typename FirstType, typename... Indices>
    static constexpr GKO_ATTRIBUTES ValueType
    compute(const dim<total_dim> &size,
            const std::array<ValueType, (total_dim > 1 ? total_dim - 1 : 0)>
                &stride,
            FirstType first, Indices &&... idxs)
    {
        // The ASSERT size check must NOT be indexed with `dim_idx` directy,
        // otherwise, it leads to a linker error. The reason is likely that
        // `dim<N>::operator[](const size_type &)` uses a reference. Since
        // `dim_idx` is constexpr (and not defined in a namespace scope), it
        // can't be odr-used.
        return GKO_ASSERT(first < size[static_cast<size_type>(dim_idx)]),
               first * stride[dim_idx] +
                   row_major_helper_s<ValueType, total_dim, current_iter + 1>::
                       compute(size, stride, std::forward<Indices>(idxs)...);
    }
};

template <typename ValueType, size_type total_dim>
struct row_major_helper_s<ValueType, total_dim, total_dim> {
    template <typename FirstType>
    static constexpr GKO_ATTRIBUTES ValueType
    compute(const dim<total_dim> &size,
            const std::array<ValueType, (total_dim > 1 ? total_dim - 1 : 0)>,
            FirstType first)
    {
        return GKO_ASSERT(first < size[total_dim - 1]), first;
    }
};


}  // namespace detail


/**
 * Computes the storage index for the given indices with respect to the given
 * stride array
 */
template <typename ValueType, size_type total_dim, typename... Indices>
constexpr GKO_ATTRIBUTES ValueType compute_storage_index(
    const dim<total_dim> &size,
    const std::array<ValueType, (total_dim > 1 ? total_dim - 1 : 0)> &stride,
    Indices &&... idxs)
{
    return detail::row_major_helper_s<ValueType, total_dim>::compute(
        size, stride, std::forward<Indices>(idxs)...);
}


namespace detail {


/**
 * This helper walks through the index arguments from left to right (lowest to
 * highest dimension) in order to properly use the stride for the scalar
 * indices. The mask indicates which indices are actually used. The least
 * significant bit set means using the last index, second bit corresponds to the
 * second last dimension, and so on.
 *
 * Basically, this computes indices in a similar fashion as `row_major_helper_s`
 * while having a mask signaling which indices to skip.
 *
 * Example: Mask = 0b1101
 * compute(stride, tuple(x1, x2, x3, x4))
 * -> x1 * stride[1] + x2 * stride[2] + x4    (x3 skipped since bit not set)
 */
template <typename ValueType, size_type mask, size_type set_bits_processed,
          size_type stride_size, size_type dim_idx, size_type total_dim,
          // Determine if the bit in the mask is set for dim_idx
          // If the end is reached (dim_idx == total_dim), set it to false in
          // order to only need one specialization
          bool mask_set = (dim_idx < total_dim)
                              ? static_cast<bool>(mask &(
                                    size_type{1} << (total_dim - 1 - dim_idx)))
                              : false>
struct row_major_masked_helper_s {};


// bit for current dimensionality is set
template <typename ValueType, size_type mask, size_type set_bits_processed,
          size_type stride_size, size_type dim_idx, size_type total_dim>
struct row_major_masked_helper_s<ValueType, mask, set_bits_processed,
                                 stride_size, dim_idx, total_dim, true> {
    static_assert(mask &
                      (size_type{1}
                       << (dim_idx < total_dim ? total_dim - 1 - dim_idx : 0)),
                  "Do not touch the `mask_set` template parameter!");
    static_assert(dim_idx < total_dim,
                  "The iteration count must be smaller than total_dim here!!!");
    static_assert(set_bits_processed <= stride_size,
                  "The processed bits must be < total number of set bits!");

    template <typename... Args>
    static constexpr GKO_ATTRIBUTES std::array<ValueType, stride_size>
    build_stride(const dim<total_dim> &size, Args &&... args)
    {
        return row_major_masked_helper_s<
            ValueType, mask, set_bits_processed + 1, stride_size, dim_idx + 1,
            total_dim>::build_stride(size, std::forward<Args>(args)...,
                                     mult_size_upwards(size));
    }

    static constexpr GKO_ATTRIBUTES ValueType
    mult_size_upwards(const dim<total_dim> &size)
    {
        return size[dim_idx] *
               row_major_masked_helper_s<
                   ValueType, mask, set_bits_processed + 1, stride_size,
                   dim_idx + 1, total_dim>::mult_size_upwards(size);
    }

    template <typename First, typename... Indices>
    static constexpr GKO_ATTRIBUTES ValueType
    compute_mask_idx(const dim<total_dim> &size,
                     const std::array<ValueType, stride_size> &stride,
                     First first, Indices &&... idxs)
    {
        static_assert(sizeof...(Indices) + 1 == total_dim - dim_idx,
                      "Mismatching number of Idxs!");
        // If it is the last set dimension, there is no need for a stride
        return GKO_ASSERT(first < size[dim_idx]),
               first * (set_bits_processed == stride_size
                            ? 1
                            : stride[set_bits_processed]) +
                   row_major_masked_helper_s<
                       ValueType, mask, set_bits_processed + 1, stride_size,
                       dim_idx + 1,
                       total_dim>::compute_mask_idx(size, stride,
                                                    std::forward<Indices>(
                                                        idxs)...);
    }

    template <typename First, typename... Indices>
    static constexpr GKO_ATTRIBUTES ValueType
    compute_direct_idx(const dim<total_dim> &size,
                       const std::array<ValueType, stride_size> &stride,
                       First first, Indices &&... idxs)
    {
        static_assert(sizeof...(Indices) == stride_size - set_bits_processed,
                      "Mismatching number of Idxs!");
        // If it is the last set dimension, there is no need for a stride
        return GKO_ASSERT(first < size[dim_idx]),
               first * (set_bits_processed == stride_size
                            ? 1
                            : stride[set_bits_processed]) +
                   row_major_masked_helper_s<
                       ValueType, mask, set_bits_processed + 1, stride_size,
                       dim_idx + 1,
                       total_dim>::compute_direct_idx(size, stride,
                                                      std::forward<Indices>(
                                                          idxs)...);
    }
};

// first set bit (from the left) for current dimensionality encountered:
//     Do not calculate stride value for it (since no lower dimension needs it)!
template <typename ValueType, size_type mask, size_type stride_size,
          size_type dim_idx, size_type total_dim>
struct row_major_masked_helper_s<ValueType, mask, 0, stride_size, dim_idx,
                                 total_dim, true> {
    static constexpr size_type set_bits_processed{0};
    static_assert(mask &
                      (size_type{1}
                       << (dim_idx < total_dim ? total_dim - 1 - dim_idx : 0)),
                  "Do not touch the `mask_set` template parameter!");
    static_assert(dim_idx < total_dim,
                  "The iteration count must be smaller than total_dim here!!!");

    template <typename... Args>
    static constexpr GKO_ATTRIBUTES std::array<ValueType, stride_size>
    build_stride(const dim<total_dim> &size, Args &&... args)
    {
        return row_major_masked_helper_s<
            ValueType, mask, set_bits_processed + 1, stride_size, dim_idx + 1,
            total_dim>::build_stride(size, std::forward<Args>(args)...);
    }

    static constexpr GKO_ATTRIBUTES ValueType
    mult_size_upwards(const dim<total_dim> &size)
    {
        return row_major_masked_helper_s<
            ValueType, mask, set_bits_processed + 1, stride_size, dim_idx + 1,
            total_dim>::mult_size_upwards(size);
    }

    template <typename First, typename... Indices>
    static constexpr GKO_ATTRIBUTES ValueType
    compute_mask_idx(const dim<total_dim> &size,
                     const std::array<ValueType, stride_size> &stride,
                     First first, Indices &&... idxs)
    {
        static_assert(sizeof...(Indices) + 1 == total_dim - dim_idx,
                      "Mismatching number of Idxs!");
        // If it is the last set dimension, there is no need for a stride
        return GKO_ASSERT(first < size[dim_idx]),
               first * (set_bits_processed == stride_size
                            ? 1
                            : stride[set_bits_processed]) +
                   row_major_masked_helper_s<
                       ValueType, mask, set_bits_processed + 1, stride_size,
                       dim_idx + 1,
                       total_dim>::compute_mask_idx(size, stride,
                                                    std::forward<Indices>(
                                                        idxs)...);
    }

    template <typename First, typename... Indices>
    static constexpr GKO_ATTRIBUTES ValueType
    compute_direct_idx(const dim<total_dim> &size,
                       const std::array<ValueType, stride_size> &stride,
                       First first, Indices &&... idxs)
    {
        static_assert(sizeof...(Indices) == stride_size - set_bits_processed,
                      "Mismatching number of Idxs!");
        // If it is the last set dimension, there is no need for a stride
        return GKO_ASSERT(first < size[dim_idx]),
               first * (set_bits_processed == stride_size
                            ? 1
                            : stride[set_bits_processed]) +
                   row_major_masked_helper_s<
                       ValueType, mask, set_bits_processed + 1, stride_size,
                       dim_idx + 1,
                       total_dim>::compute_direct_idx(size, stride,
                                                      std::forward<Indices>(
                                                          idxs)...);
    }
};

// bit for current dimensionality is not set
template <typename ValueType, size_type mask, size_type set_bits_processed,
          size_type stride_size, size_type dim_idx, size_type total_dim>
struct row_major_masked_helper_s<ValueType, mask, set_bits_processed,
                                 stride_size, dim_idx, total_dim, false> {
    static_assert((mask & (size_type{1}
                           << (dim_idx < total_dim ? total_dim - 1 - dim_idx
                                                   : 0))) == 0,
                  "Do not touch the `mask_set` template parameter!");
    static_assert(dim_idx < total_dim,
                  "The iteration count must be smaller than total_dim here!!!");
    static_assert(set_bits_processed <= stride_size + 1,
                  "The processed bits must be < total number of set bits!");
    template <typename... Args>
    static constexpr GKO_ATTRIBUTES std::array<ValueType, stride_size>
    build_stride(const dim<total_dim> &size, Args &&... args)
    {
        return row_major_masked_helper_s<
            ValueType, mask, set_bits_processed, stride_size, dim_idx + 1,
            total_dim>::build_stride(size, std::forward<Args>(args)...);
    }

    static constexpr GKO_ATTRIBUTES ValueType
    mult_size_upwards(const dim<total_dim> &size)
    {
        return row_major_masked_helper_s<ValueType, mask, set_bits_processed,
                                         stride_size, dim_idx + 1,
                                         total_dim>::mult_size_upwards(size);
    }

    template <typename First, typename... Indices>
    static constexpr GKO_ATTRIBUTES ValueType
    compute_mask_idx(const dim<total_dim> &size,
                     const std::array<ValueType, stride_size> &stride, First,
                     Indices &&... idxs)
    {
        static_assert(sizeof...(Indices) + 1 == total_dim - dim_idx,
                      "Mismatching number of Idxs!");
        return row_major_masked_helper_s<
            ValueType, mask, set_bits_processed, stride_size, dim_idx + 1,
            total_dim>::compute_mask_idx(size, stride,
                                         std::forward<Indices>(idxs)...);
    }

    template <typename... Indices>
    static constexpr GKO_ATTRIBUTES ValueType compute_direct_idx(
        const dim<total_dim> &size,
        const std::array<ValueType, stride_size> &stride, Indices &&... idxs)
    {
        return row_major_masked_helper_s<
            ValueType, mask, set_bits_processed, stride_size, dim_idx + 1,
            total_dim>::compute_direct_idx(size, stride,
                                           std::forward<Indices>(idxs)...);
    }
};

// Specialization for the end of recursion: build_stride array from created
// arguments
template <typename ValueType, size_type mask, size_type set_bits_processed,
          size_type stride_size, size_type total_dim>
struct row_major_masked_helper_s<ValueType, mask, set_bits_processed,
                                 stride_size, total_dim, total_dim, false> {
    static_assert(set_bits_processed <= stride_size + 1,
                  "The processed bits must be smaller than the total number of "
                  "set bits!");
    template <typename... Args>
    static constexpr GKO_ATTRIBUTES std::array<ValueType, stride_size>
    build_stride(const dim<total_dim> &, Args &&... args)
    {
        return {{std::forward<Args>(args)...}};
    }
    static constexpr GKO_ATTRIBUTES ValueType
    mult_size_upwards(const dim<total_dim> &)
    {
        return 1;
    }

    static constexpr GKO_ATTRIBUTES ValueType compute_mask_idx(
        const dim<total_dim> &, const std::array<ValueType, stride_size> &)
    {
        return 0;
    }
    static constexpr GKO_ATTRIBUTES ValueType compute_direct_idx(
        const dim<total_dim> &, const std::array<ValueType, stride_size> &)
    {
        return 0;
    }
};


}  // namespace detail


/**
 * Computes the memory index for the given indices considering the stride.
 * Only indices are considered where the corresponding mask bit is set.
 */
template <typename ValueType, size_type mask, size_type stride_size,
          size_type total_dim, typename... Indices>
constexpr GKO_ATTRIBUTES auto compute_masked_index(
    const dim<total_dim> &size,
    const std::array<ValueType, stride_size> &stride, Indices &&... idxs)
{
    return detail::row_major_masked_helper_s<
        ValueType, mask, 0, stride_size, 0,
        total_dim>::compute_mask_idx(size, stride,
                                     std::forward<Indices>(idxs)...);
}


/**
 * Computes the memory index for the given indices considering the stride.
 */
template <typename ValueType, size_type mask, size_type stride_size,
          size_type total_dim, typename... Indices>
constexpr GKO_ATTRIBUTES auto compute_masked_index_direct(
    const dim<total_dim> &size,
    const std::array<ValueType, stride_size> &stride, Indices &&... idxs)
{
    return detail::row_major_masked_helper_s<
        ValueType, mask, 0, stride_size, 0,
        total_dim>::compute_direct_idx(size, stride,
                                       std::forward<Indices>(idxs)...);
}


/**
 * Computes the default stride array from a size and a given mask which
 * indicates which array indices to consider. It is assumed that there is no
 * padding
 */
template <typename ValueType, size_type mask, size_type stride_size,
          size_type total_dim>
constexpr GKO_ATTRIBUTES auto compute_default_masked_stride_array(
    const dim<total_dim> &size)
{
    return detail::row_major_masked_helper_s<ValueType, mask, 0, stride_size, 0,
                                             total_dim>::build_stride(size);
}


namespace detail {


template <bool has_span, typename... Args>
struct are_span_compatible_impl
    : public std::integral_constant<bool, has_span> {};

template <bool has_span, typename First, typename... Args>
struct are_span_compatible_impl<has_span, First, Args...>
    : public std::conditional<
          std::is_integral<std::decay_t<First>>::value ||
              std::is_same<std::decay_t<First>, span>::value,
          are_span_compatible_impl<
              has_span || std::is_same<std::decay_t<First>, span>::value,
              Args...>,
          std::false_type>::type {};


}  // namespace detail


/**
 * Evaluates if at least one type of Args is a gko::span and the others either
 * also gko::span or fulfill std::is_integral
 */
template <typename... Args>
using are_span_compatible = detail::are_span_compatible_impl<false, Args...>;


namespace detail {


template <size_type iter, size_type N, typename Callable, typename... Indices>
GKO_ATTRIBUTES std::enable_if_t<iter == N> multidim_for_each_impl(
    const dim<N> &, Callable callable, Indices &&... indices)
{
    static_assert(iter == sizeof...(Indices),
                  "Number arguments must match current iteration!");
    callable(std::forward<Indices>(indices)...);
}

template <size_type iter, size_type N, typename Callable, typename... Indices>
GKO_ATTRIBUTES std::enable_if_t<(iter < N)> multidim_for_each_impl(
    const dim<N> &size, Callable &&callable, Indices &&... indices)
{
    static_assert(iter == sizeof...(Indices),
                  "Number arguments must match current iteration!");
    for (size_type i = 0; i < size[iter]; ++i) {
        multidim_for_each_impl<iter + 1>(size, std::forward<Callable>(callable),
                                         std::forward<Indices>(indices)..., i);
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


template <size_type iter, size_type N>
constexpr GKO_ATTRIBUTES std::enable_if_t<iter == N, int> spans_in_size(
    const dim<N> &)
{
    return 0;
}

template <size_type iter, size_type N, typename First, typename... Remaining>
constexpr GKO_ATTRIBUTES std::enable_if_t<(iter < N), int> spans_in_size(
    const dim<N> &size, First first, Remaining &&... remaining)
{
    static_assert(sizeof...(Remaining) + 1 == N - iter,
                  "Number of remaining spans must be equal to N - iter");
    return GKO_ASSERT(span{first}.is_valid()),
           GKO_ASSERT(span{first} <= span{size[iter]}),
           spans_in_size<iter + 1>(size, std::forward<Remaining>(remaining)...);
}


}  // namespace detail


template <size_type N, typename... Spans>
constexpr GKO_ATTRIBUTES int validate_spans(const dim<N> &size,
                                            Spans &&... spans)
{
    return detail::spans_in_size<0>(size, std::forward<Spans>(spans)...);
}


namespace detail {


template <size_type, size_type N, size_type iter = 0>
constexpr std::enable_if_t<iter == N, size_type>
count_mask_dimensionality_impl()
{
    return 0;
}

template <size_type mask, size_type N, size_type iter = 0>
constexpr std::enable_if_t<(iter < N), size_type>
count_mask_dimensionality_impl()
{
    return (mask & size_type{1}) +
           count_mask_dimensionality_impl<(mask >> 1), N, iter + 1>();
}


}  // namespace detail


template <size_type mask, size_type N>
constexpr size_type count_mask_dimensionality()
{
    return detail::count_mask_dimensionality_impl<mask, N>();
}


}  // namespace helper
}  // namespace accessor
}  // namespace gko


#endif  // GKO_CORE_BASE_ACCESSOR_HELPER_HPP_
