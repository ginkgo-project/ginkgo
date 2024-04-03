// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_ACCESSOR_HELPER_HPP_
#define GKO_ACCESSOR_ACCESSOR_HELPER_HPP_


#include <array>
#include <cinttypes>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>


#include "index_span.hpp"
#include "utils.hpp"


namespace gko {
namespace acc {
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
 * The index is expected to be of type `IndexType`, and will be computed in
 * `IndexType`.
 * The index is computed like this: indices: x1, x2, x3, ...
 * compute(stride, x1, x2, x3) -> x1 * stride[0] + x2 * stride[1] + x3
 */
template <typename IndexType, size_type total_dim, size_type current_iter = 1>
struct row_major_helper_s {
    static_assert(total_dim >= 1, "Dimensionality must be >= 1");
    static_assert(current_iter < total_dim, "Iteration must be < total_dim!");

    static constexpr size_type dim_idx{current_iter - 1};

    template <typename SizeType, typename... Indices>
    static constexpr GKO_ACC_ATTRIBUTES IndexType compute(
        const std::array<SizeType, total_dim>& size,
        const std::array<SizeType, (total_dim > 1 ? total_dim - 1 : 0)>& stride,
        IndexType first, Indices&&... idxs)
    {
        // The ASSERT size check must NOT be indexed with `dim_idx` directly,
        // otherwise, it leads to a linker error. The reason is likely that
        // `std::array<size_type, N>::operator[](const size_type &)` uses a
        // reference. Since `dim_idx` is constexpr (and not defined in a
        // namespace scope), it can't be odr-used.
        return GKO_ACC_ASSERT(first < static_cast<IndexType>(size[dim_idx])),
               first * static_cast<IndexType>(stride[dim_idx]) +
                   row_major_helper_s<IndexType, total_dim, current_iter + 1>::
                       compute(size, stride, std::forward<Indices>(idxs)...);
    }
};

template <typename IndexType, size_type total_dim>
struct row_major_helper_s<IndexType, total_dim, total_dim> {
    template <typename SizeType>
    static constexpr GKO_ACC_ATTRIBUTES IndexType
    compute(const std::array<SizeType, total_dim>& size,
            const std::array<SizeType, (total_dim > 1 ? total_dim - 1 : 0)>,
            IndexType first)
    {
        return GKO_ACC_ASSERT(first <
                              static_cast<IndexType>(size[total_dim - 1])),
               first;
    }
};


template <typename ValueType, std::size_t iter, std::size_t N,
          typename DimensionType>
constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<iter == N, ValueType>
mult_dim_upwards(const std::array<DimensionType, N>&)
{
    return 1;
}

template <typename ValueType, std::size_t iter, std::size_t N,
          typename DimensionType>
constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<(iter < N), ValueType>
mult_dim_upwards(const std::array<DimensionType, N>& size)
{
    return size[iter] * mult_dim_upwards<ValueType, iter + 1>(size);
}


template <typename ValueType, std::size_t iter = 1, std::size_t N,
          typename DimensionType, typename... Args>
constexpr GKO_ACC_ATTRIBUTES
    std::enable_if_t<N == 0 || (iter == N && iter == sizeof...(Args) + 1),
                     std::array<ValueType, N == 0 ? 0 : N - 1>>
    compute_default_row_major_stride_array(const std::array<DimensionType, N>&,
                                           Args&&... args)
{
    return {{std::forward<Args>(args)...}};
}

template <typename ValueType, std::size_t iter = 1, std::size_t N,
          typename DimensionType, typename... Args>
constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<
    (iter < N) && (iter == sizeof...(Args) + 1), std::array<ValueType, N - 1>>
compute_default_row_major_stride_array(const std::array<DimensionType, N>& size,
                                       Args&&... args)
{
    return compute_default_row_major_stride_array<ValueType, iter + 1>(
        size, std::forward<Args>(args)...,
        mult_dim_upwards<ValueType, iter>(size));
}


}  // namespace detail


/**
 * Computes the storage index for the given indices with respect to the given
 * stride array
 */
template <typename IndexType, std::size_t total_dim, typename SizeType,
          typename... Indices>
constexpr GKO_ACC_ATTRIBUTES IndexType compute_row_major_index(
    const std::array<SizeType, total_dim>& size,
    const std::array<SizeType, (total_dim > 1 ? total_dim - 1 : 0)>& stride,
    Indices&&... idxs)
{
    return detail::row_major_helper_s<IndexType, total_dim>::compute(
        size, stride, std::forward<Indices>(idxs)...);
}


/**
 * Computes the default stride array from a given size, assuming there is no
 * padding.
 *
 * Example: std::array<ValueType, 4> size={2, 3, 5, 7} results in a return value
 * of: std::array<ValueType, 3> = {3*5*7, 5*7, 7}
 *
 * @tparam ValueType  value type of the values in the returned array
 *
 * @tparam dimensions  number of dimensions in the given size
 *
 * @tparam DimensionType  value type of the stored size
 *
 * @returns an std::array<ValueType, dimensions - 1> with the stride
 *          information.
 */
template <std::size_t dimensions, typename SizeType>
constexpr GKO_ACC_ATTRIBUTES
    std::array<SizeType, (dimensions > 0 ? dimensions - 1 : 0)>
    compute_default_row_major_stride_array(
        const std::array<SizeType, dimensions>& size)
{
    return detail::compute_default_row_major_stride_array<SizeType>(size);
}


namespace detail {


/**
 * This helper walks through the index arguments from left to right (lowest to
 * highest dimension) in order to properly use the stride for the scalar
 * indices. The mask indicates which indices are actually used. The least
 * significant bit set means using the last index, second bit corresponds to the
 * second last dimension, and so on.
 *
 * The index is expected to be of type `IndexType`, and will be computed in
 * `IndexType`.
 *
 * Basically, this computes indices in a similar fashion as `row_major_helper_s`
 * while having a mask signaling which indices to skip.
 *
 * Example: Mask = 0b1101
 * compute(stride, tuple(x1, x2, x3, x4))
 * -> x1 * stride[1] + x2 * stride[2] + x4    (x3 skipped since bit not set)
 */
template <
    typename IndexType, std::uint64_t mask, std::size_t set_bits_processed,
    std::size_t stride_size, std::size_t dim_idx, std::size_t total_dim,
    // Determine if the bit in the mask is set for dim_idx
    // If the end is reached (dim_idx == total_dim), set it to false in
    // order to only need one specialization
    bool mask_set = (dim_idx < total_dim)
                        ? static_cast<bool>(mask&(std::uint64_t{1}
                                                  << (total_dim - 1 - dim_idx)))
                        : false>
struct row_major_masked_helper_s {};


// bit for current dimensionality is set
template <typename IndexType, std::uint64_t mask,
          std::size_t set_bits_processed, std::size_t stride_size,
          std::size_t dim_idx, std::size_t total_dim>
struct row_major_masked_helper_s<IndexType, mask, set_bits_processed,
                                 stride_size, dim_idx, total_dim, true> {
    static_assert(mask &
                      (std::uint64_t{1}
                       << (dim_idx < total_dim ? total_dim - 1 - dim_idx : 0)),
                  "Do not touch the `mask_set` template parameter!");
    static_assert(dim_idx < total_dim,
                  "The iteration count must be smaller than total_dim here!!!");
    static_assert(set_bits_processed <= stride_size,
                  "The processed bits must be < total number of set bits!");

    template <typename SizeType, typename... Args>
    static constexpr GKO_ACC_ATTRIBUTES std::array<SizeType, stride_size>
    build_stride(const std::array<SizeType, total_dim>& size, Args&&... args)
    {
        return row_major_masked_helper_s<
            IndexType, mask, set_bits_processed + 1, stride_size, dim_idx + 1,
            total_dim>::build_stride(size, std::forward<Args>(args)...,
                                     mult_size_upwards(size));
    }

    template <typename SizeType>
    static constexpr GKO_ACC_ATTRIBUTES SizeType
    mult_size_upwards(const std::array<SizeType, total_dim>& size)
    {
        return size[dim_idx] *
               row_major_masked_helper_s<
                   IndexType, mask, set_bits_processed + 1, stride_size,
                   dim_idx + 1, total_dim>::mult_size_upwards(size);
    }

    template <typename SizeType, typename... Indices>
    static constexpr GKO_ACC_ATTRIBUTES IndexType
    compute_mask_idx(const std::array<SizeType, total_dim>& size,
                     const std::array<SizeType, stride_size>& stride,
                     IndexType first, Indices&&... idxs)
    {
        static_assert(sizeof...(Indices) + 1 == total_dim - dim_idx,
                      "Mismatching number of Idxs!");
        // If it is the last set dimension, there is no need for a stride
        return GKO_ACC_ASSERT(first < static_cast<IndexType>(size[dim_idx])),
               first * (set_bits_processed == stride_size
                            ? 1
                            : stride[set_bits_processed]) +
                   row_major_masked_helper_s<
                       IndexType, mask, set_bits_processed + 1, stride_size,
                       dim_idx + 1,
                       total_dim>::compute_mask_idx(size, stride,
                                                    std::forward<Indices>(
                                                        idxs)...);
    }

    template <typename SizeType, typename... Indices>
    static constexpr GKO_ACC_ATTRIBUTES IndexType
    compute_direct_idx(const std::array<SizeType, total_dim>& size,
                       const std::array<SizeType, stride_size>& stride,
                       IndexType first, Indices&&... idxs)
    {
        static_assert(sizeof...(Indices) == stride_size - set_bits_processed,
                      "Mismatching number of Idxs!");
        // If it is the last set dimension, there is no need for a stride
        return GKO_ACC_ASSERT(first < static_cast<IndexType>(size[dim_idx])),
               first * (set_bits_processed == stride_size
                            ? 1
                            : stride[set_bits_processed]) +
                   row_major_masked_helper_s<
                       IndexType, mask, set_bits_processed + 1, stride_size,
                       dim_idx + 1,
                       total_dim>::compute_direct_idx(size, stride,
                                                      std::forward<Indices>(
                                                          idxs)...);
    }
};

// first set bit (from the left) for current dimensionality encountered:
//     Do not calculate stride value for it (since no lower dimension needs it)!
template <typename IndexType, std::uint64_t mask, std::size_t stride_size,
          std::size_t dim_idx, std::size_t total_dim>
struct row_major_masked_helper_s<IndexType, mask, 0, stride_size, dim_idx,
                                 total_dim, true> {
    static constexpr std::size_t set_bits_processed{0};
    static_assert(mask &
                      (std::uint64_t{1}
                       << (dim_idx < total_dim ? total_dim - 1 - dim_idx : 0)),
                  "Do not touch the `mask_set` template parameter!");
    static_assert(dim_idx < total_dim,
                  "The iteration count must be smaller than total_dim here!!!");

    template <typename SizeType, typename... Args>
    static constexpr GKO_ACC_ATTRIBUTES std::array<SizeType, stride_size>
    build_stride(const std::array<SizeType, total_dim>& size, Args&&... args)
    {
        return row_major_masked_helper_s<
            IndexType, mask, set_bits_processed + 1, stride_size, dim_idx + 1,
            total_dim>::build_stride(size, std::forward<Args>(args)...);
    }

    template <typename SizeType>
    static constexpr GKO_ACC_ATTRIBUTES SizeType
    mult_size_upwards(const std::array<SizeType, total_dim>& size)
    {
        return row_major_masked_helper_s<
            IndexType, mask, set_bits_processed + 1, stride_size, dim_idx + 1,
            total_dim>::mult_size_upwards(size);
    }

    template <typename SizeType, typename... Indices>
    static constexpr GKO_ACC_ATTRIBUTES IndexType
    compute_mask_idx(const std::array<SizeType, total_dim>& size,
                     const std::array<SizeType, stride_size>& stride,
                     IndexType first, Indices&&... idxs)
    {
        static_assert(sizeof...(Indices) + 1 == total_dim - dim_idx,
                      "Mismatching number of Idxs!");
        // If it is the last set dimension, there is no need for a stride
        return GKO_ACC_ASSERT(first < size[dim_idx]),
               first * (set_bits_processed == stride_size
                            ? 1
                            : stride[set_bits_processed]) +
                   row_major_masked_helper_s<
                       IndexType, mask, set_bits_processed + 1, stride_size,
                       dim_idx + 1,
                       total_dim>::compute_mask_idx(size, stride,
                                                    std::forward<Indices>(
                                                        idxs)...);
    }

    template <typename SizeType, typename... Indices>
    static constexpr GKO_ACC_ATTRIBUTES IndexType
    compute_direct_idx(const std::array<SizeType, total_dim>& size,
                       const std::array<SizeType, stride_size>& stride,
                       IndexType first, Indices&&... idxs)
    {
        static_assert(sizeof...(Indices) == stride_size - set_bits_processed,
                      "Mismatching number of Idxs!");
        // If it is the last set dimension, there is no need for a stride
        return GKO_ACC_ASSERT(first < static_cast<IndexType>(size[dim_idx])),
               first * (set_bits_processed == stride_size
                            ? 1
                            : stride[set_bits_processed]) +
                   row_major_masked_helper_s<
                       IndexType, mask, set_bits_processed + 1, stride_size,
                       dim_idx + 1,
                       total_dim>::compute_direct_idx(size, stride,
                                                      std::forward<Indices>(
                                                          idxs)...);
    }
};

// bit for current dimensionality is not set
template <typename IndexType, std::uint64_t mask,
          std::size_t set_bits_processed, std::size_t stride_size,
          std::size_t dim_idx, std::size_t total_dim>
struct row_major_masked_helper_s<IndexType, mask, set_bits_processed,
                                 stride_size, dim_idx, total_dim, false> {
    static_assert((mask & (std::uint64_t{1}
                           << (dim_idx < total_dim ? total_dim - 1 - dim_idx
                                                   : 0))) == 0,
                  "Do not touch the `mask_set` template parameter!");
    static_assert(dim_idx < total_dim,
                  "The iteration count must be smaller than total_dim here!!!");
    static_assert(set_bits_processed <= stride_size + 1,
                  "The processed bits must be < total number of set bits!");
    template <typename SizeType, typename... Args>
    static constexpr GKO_ACC_ATTRIBUTES std::array<SizeType, stride_size>
    build_stride(const std::array<SizeType, total_dim>& size, Args&&... args)
    {
        return row_major_masked_helper_s<
            IndexType, mask, set_bits_processed, stride_size, dim_idx + 1,
            total_dim>::build_stride(size, std::forward<Args>(args)...);
    }

    template <typename SizeType>
    static constexpr GKO_ACC_ATTRIBUTES SizeType
    mult_size_upwards(const std::array<SizeType, total_dim>& size)
    {
        return row_major_masked_helper_s<IndexType, mask, set_bits_processed,
                                         stride_size, dim_idx + 1,
                                         total_dim>::mult_size_upwards(size);
    }

    template <typename SizeType, typename... Indices>
    static constexpr GKO_ACC_ATTRIBUTES IndexType
    compute_mask_idx(const std::array<SizeType, total_dim>& size,
                     const std::array<SizeType, stride_size>& stride, IndexType,
                     Indices&&... idxs)
    {
        static_assert(sizeof...(Indices) + 1 == total_dim - dim_idx,
                      "Mismatching number of Idxs!");
        return row_major_masked_helper_s<
            IndexType, mask, set_bits_processed, stride_size, dim_idx + 1,
            total_dim>::compute_mask_idx(size, stride,
                                         std::forward<Indices>(idxs)...);
    }

    template <typename SizeType, typename... Indices>
    static constexpr GKO_ACC_ATTRIBUTES IndexType compute_direct_idx(
        const std::array<SizeType, total_dim>& size,
        const std::array<SizeType, stride_size>& stride, Indices&&... idxs)
    {
        return row_major_masked_helper_s<
            IndexType, mask, set_bits_processed, stride_size, dim_idx + 1,
            total_dim>::compute_direct_idx(size, stride,
                                           std::forward<Indices>(idxs)...);
    }
};

// Specialization for the end of recursion: build_stride array from created
// arguments
template <typename IndexType, std::uint64_t mask,
          std::size_t set_bits_processed, std::size_t stride_size,
          std::size_t total_dim>
struct row_major_masked_helper_s<IndexType, mask, set_bits_processed,
                                 stride_size, total_dim, total_dim, false> {
    static_assert(set_bits_processed <= stride_size + 1,
                  "The processed bits must be smaller than the total number of "
                  "set bits!");
    template <typename SizeType, typename... Args>
    static constexpr GKO_ACC_ATTRIBUTES std::array<SizeType, stride_size>
    build_stride(const std::array<SizeType, total_dim>&, Args&&... args)
    {
        return {{std::forward<Args>(args)...}};
    }

    template <typename SizeType>
    static constexpr GKO_ACC_ATTRIBUTES SizeType
    mult_size_upwards(const std::array<SizeType, total_dim>&)
    {
        return 1;
    }

    template <typename SizeType>
    static constexpr GKO_ACC_ATTRIBUTES IndexType
    compute_mask_idx(const std::array<SizeType, total_dim>&,
                     const std::array<SizeType, stride_size>&)
    {
        return 0;
    }

    template <typename SizeType>
    static constexpr GKO_ACC_ATTRIBUTES IndexType
    compute_direct_idx(const std::array<SizeType, total_dim>&,
                       const std::array<SizeType, stride_size>&)
    {
        return 0;
    }
};


}  // namespace detail


/**
 * Computes the memory index for the given indices considering the stride.
 * Only indices are considered where the corresponding mask bit is set.
 */
template <typename IndexType, std::uint64_t mask, std::size_t stride_size,
          std::size_t total_dim, typename SizeType, typename... Indices>
constexpr GKO_ACC_ATTRIBUTES IndexType compute_masked_index(
    const std::array<SizeType, total_dim>& size,
    const std::array<SizeType, stride_size>& stride, Indices&&... idxs)
{
    return detail::row_major_masked_helper_s<
        IndexType, mask, 0, stride_size, 0,
        total_dim>::compute_mask_idx(size, stride,
                                     std::forward<Indices>(idxs)...);
}


/**
 * Computes the memory index for the given indices considering the stride.
 */
template <typename IndexType, std::uint64_t mask, std::size_t stride_size,
          std::size_t total_dim, typename SizeType, typename... Indices>
constexpr GKO_ACC_ATTRIBUTES auto compute_masked_index_direct(
    const std::array<SizeType, total_dim>& size,
    const std::array<SizeType, stride_size>& stride, Indices&&... idxs)
{
    return detail::row_major_masked_helper_s<
        IndexType, mask, 0, stride_size, 0,
        total_dim>::compute_direct_idx(size, stride,
                                       std::forward<Indices>(idxs)...);
}


/**
 * Computes the default stride array from a size and a given mask which
 * indicates which array indices to consider. It is assumed that there is no
 * padding
 */
template <std::uint64_t mask, std::size_t stride_size, std::size_t total_dim,
          typename SizeType>
constexpr GKO_ACC_ATTRIBUTES auto compute_default_masked_row_major_stride_array(
    const std::array<SizeType, total_dim>& size)
{
    return detail::row_major_masked_helper_s<SizeType, mask, 0, stride_size, 0,
                                             total_dim>::build_stride(size);
}


namespace detail {


template <bool has_span, typename... Args>
struct are_index_span_compatible_impl
    : public std::integral_constant<bool, has_span> {};

template <bool has_span, typename First, typename... Args>
struct are_index_span_compatible_impl<has_span, First, Args...>
    : public std::conditional_t<
          std::is_integral<std::decay_t<First>>::value ||
              std::is_same<std::decay_t<First>, index_span>::value,
          are_index_span_compatible_impl<
              has_span || std::is_same<std::decay_t<First>, index_span>::value,
              Args...>,
          std::false_type> {};


}  // namespace detail


/**
 * Evaluates if at least one type of Args is a gko::acc::index_span and the
 * others either also gko::acc::index_span or fulfill std::is_integral
 */
template <typename... Args>
using are_index_span_compatible =
    detail::are_index_span_compatible_impl<false, Args...>;


namespace detail {


template <std::size_t iter, std::size_t N, typename DimensionType,
          typename Callable, typename... Indices>
GKO_ACC_ATTRIBUTES std::enable_if_t<iter == N> multidim_for_each_impl(
    const std::array<DimensionType, N>&, Callable callable,
    Indices&&... indices)
{
    static_assert(iter == sizeof...(Indices),
                  "Number arguments must match current iteration!");
    callable(std::forward<Indices>(indices)...);
}

template <std::size_t iter, std::size_t N, typename DimensionType,
          typename Callable, typename... Indices>
GKO_ACC_ATTRIBUTES std::enable_if_t<(iter < N)> multidim_for_each_impl(
    const std::array<DimensionType, N>& size, Callable callable,
    Indices... indices)
{
    static_assert(iter == sizeof...(Indices),
                  "Number arguments must match current iteration!");
    for (DimensionType i = 0; i < size[iter]; ++i) {
        multidim_for_each_impl<iter + 1>(size, callable, indices..., i);
    }
}


}  // namespace detail


/**
 * Creates a recursive for-loop for each dimension and calls dest(indices...) =
 * source(indices...)
 */
template <std::size_t N, typename DimensionType, typename Callable>
GKO_ACC_ATTRIBUTES void multidim_for_each(
    const std::array<DimensionType, N>& size, Callable&& callable)
{
    detail::multidim_for_each_impl<0>(size, std::forward<Callable>(callable));
}


namespace detail {


template <std::size_t iter, typename DimensionType, std::size_t N>
constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<iter == N, int>
index_spans_in_size(const std::array<DimensionType, N>&)
{
    return 0;
}

template <std::size_t iter, typename DimensionType, std::size_t N,
          typename First, typename... Remaining>
constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<(iter < N), int>
index_spans_in_size(const std::array<DimensionType, N>& size, First first,
                    Remaining&&... remaining)
{
    static_assert(sizeof...(Remaining) + 1 == N - iter,
                  "Number of remaining spans must be equal to N - iter");
    return GKO_ACC_ASSERT(index_span{first}.is_valid()),
           GKO_ACC_ASSERT(index_span{first} <= index_span{size[iter]}),
           index_spans_in_size<iter + 1>(size,
                                         std::forward<Remaining>(remaining)...);
}


}  // namespace detail


template <typename DimensionType, std::size_t N, typename... Spans>
constexpr GKO_ACC_ATTRIBUTES int validate_index_spans(
    const std::array<DimensionType, N>& size, Spans&&... spans)
{
    return detail::index_spans_in_size<0>(size, std::forward<Spans>(spans)...);
}


namespace detail {


template <std::uint64_t, std::size_t N, std::size_t iter = 0>
constexpr std::enable_if_t<iter == N, std::size_t>
count_mask_dimensionality_impl()
{
    return 0;
}

template <std::uint64_t mask, std::size_t N, std::size_t iter = 0>
constexpr std::enable_if_t<(iter < N), std::size_t>
count_mask_dimensionality_impl()
{
    return (mask & std::uint64_t{1}) +
           count_mask_dimensionality_impl<(mask >> 1), N, iter + 1>();
}


}  // namespace detail


template <std::uint64_t mask, std::size_t N>
constexpr std::size_t count_mask_dimensionality()
{
    return detail::count_mask_dimensionality_impl<mask, N>();
}


/**
 * Namespace for helper functions and structs for
 * the block column major accessor.
 */
namespace blk_col_major {


/**
 * Runs from first to last dimension in order to compute the index. The index is
 * expected to be of type `IndexType`, and all computations will be performed in
 * that type.
 *
 * The index is computed like this:
 * indices: x1, x2, x3, ..., xn
 * compute(stride, x1, x2, x3, ..., x(n-1), xn) ->
 *  x1 * stride[0] + x2 * stride[1] + ...
 *    + x(n-2) * stride[n-3] + x(n-1) + xn * stride[n-2]
 * Note that swap of the last two strides, making this 'block column major'.
 */
template <typename IndexType, std::size_t total_dim,
          std::size_t current_iter = 1>
struct index_helper_s {
    static_assert(total_dim >= 1, "Dimensionality must be >= 1");
    static_assert(current_iter <= total_dim, "Iteration must be <= total_dim!");

    static constexpr std::size_t dim_idx{current_iter - 1};

    template <typename SizeType, typename... Indices>
    static constexpr GKO_ACC_ATTRIBUTES IndexType compute(
        const std::array<SizeType, total_dim>& size,
        const std::array<SizeType, (total_dim > 0 ? total_dim - 1 : 0)>& stride,
        IndexType first, Indices&&... idxs)
    {
        if (current_iter == total_dim - 1) {
            return GKO_ACC_ASSERT(first <
                                  static_cast<IndexType>(size[dim_idx])),
                   first +
                       index_helper_s<IndexType, total_dim, current_iter + 1>::
                           compute(size, stride,
                                   std::forward<Indices>(idxs)...);
        }

        return GKO_ACC_ASSERT(first < static_cast<IndexType>(size[dim_idx])),
               first * static_cast<IndexType>(stride[dim_idx]) +
                   index_helper_s<IndexType, total_dim, current_iter + 1>::
                       compute(size, stride, std::forward<Indices>(idxs)...);
    }
};

template <typename IndexType, std::size_t total_dim>
struct index_helper_s<IndexType, total_dim, total_dim> {
    static_assert(total_dim >= 2, "Dimensionality must be >= 2");

    static constexpr std::size_t dim_idx{total_dim - 1};

    template <typename SizeType>
    static constexpr GKO_ACC_ATTRIBUTES IndexType compute(
        const std::array<SizeType, total_dim>& size,
        const std::array<SizeType, (total_dim > 1 ? total_dim - 1 : 0)>& stride,
        IndexType first)
    {
        return GKO_ACC_ASSERT(first <
                              static_cast<IndexType>(size[total_dim - 1])),
               first * static_cast<IndexType>(stride[dim_idx - 1]);
    }
};

/**
 * Computes the flat storage index for block-column-major access.
 *
 * @param size  the multi-dimensional sizes of the range of values
 * @param stride  the stride array
 * @param idxs  the multi-dimensional indices of the desired entry
 */
template <typename IndexType, typename SizeType, std::size_t total_dim,
          typename... Indices>
constexpr GKO_ACC_ATTRIBUTES IndexType compute_index(
    const std::array<SizeType, total_dim>& size,
    const std::array<SizeType, (total_dim > 0 ? total_dim - 1 : 0)>& stride,
    Indices&&... idxs)
{
    return index_helper_s<IndexType, total_dim>::compute(
        size, stride, std::forward<Indices>(idxs)...);
}


template <std::size_t iter = 1, typename ValueType, std::size_t N,
          typename... Args>
constexpr GKO_ACC_ATTRIBUTES
    std::enable_if_t<(iter == N - 1) && (iter == sizeof...(Args) + 1),
                     std::array<ValueType, N - 1>>
    default_stride_array_impl(const std::array<ValueType, N>& size,
                              Args&&... args)
{
    return {{std::forward<Args>(args)..., size[N - 2]}};
}

template <std::size_t iter = 1, typename ValueType, std::size_t N,
          typename... Args>
constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<(iter < N - 1 || iter == N) &&
                                                  (iter == sizeof...(Args) + 1),
                                              std::array<ValueType, N - 1>>
default_stride_array_impl(const std::array<ValueType, N>& size, Args&&... args)
{
    return default_stride_array_impl<iter + 1>(
        size, std::forward<Args>(args)...,
        detail::mult_dim_upwards<size_type, iter>(size));
}

template <typename ValueType, std::size_t dimensions>
constexpr GKO_ACC_ATTRIBUTES
    std::array<ValueType, (dimensions > 0 ? dimensions - 1 : 0)>
    default_stride_array(const std::array<ValueType, dimensions>& size)
{
    return default_stride_array_impl(size);
}


}  // namespace blk_col_major


}  // namespace helper
}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_ACCESSOR_HELPER_HPP_
