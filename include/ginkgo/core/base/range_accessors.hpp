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

#ifndef GKO_PUBLIC_CORE_BASE_RANGE_ACCESSORS_HPP_
#define GKO_PUBLIC_CORE_BASE_RANGE_ACCESSORS_HPP_


#include <array>
#include <type_traits>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
/**
 * @brief The accessor namespace.
 *
 * @ingroup accessor
 */
namespace accessor {


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
    compute(const std::array<ValueType, total_dim> &size,
            const std::array<ValueType, (total_dim > 1 ? total_dim - 1 : 0)>
                &stride,
            FirstType first, Indices &&... idxs)
    {
        return GKO_ASSERT(first < size[dim_idx]),
               first * stride[dim_idx] +
                   row_major_helper_s<ValueType, total_dim, current_iter + 1>::
                       compute(size, stride, std::forward<Indices>(idxs)...);
    }
};

template <typename ValueType, size_type total_dim>
struct row_major_helper_s<ValueType, total_dim, total_dim> {
    template <typename FirstType>
    static constexpr GKO_ATTRIBUTES ValueType
    compute(const std::array<ValueType, total_dim> &size,
            const std::array<ValueType, (total_dim > 1 ? total_dim - 1 : 0)>,
            FirstType first)
    {
        return GKO_ASSERT(first < size[total_dim - 1]), first;
    }
};


/**
 * Computes the storage index for the given indices with respect to the given
 * stride array for row-major access
 *
 * @param size  the multi-dimensional sizes of the range of values
 * @param stride  the stride array
 * @param idxs  the multi-dimensional indices of the desired entry
 */
template <typename ValueType, size_type total_dim, typename... Indices>
constexpr GKO_ATTRIBUTES ValueType compute_storage_index(
    const std::array<ValueType, total_dim> &size,
    const std::array<ValueType, (total_dim > 1 ? total_dim - 1 : 0)> &stride,
    Indices &&... idxs)
{
    return row_major_helper_s<ValueType, total_dim>::compute(
        size, stride, std::forward<Indices>(idxs)...);
}


template <size_type iter, typename ValueType, size_type N>
constexpr GKO_ATTRIBUTES std::enable_if_t<iter == N, int> spans_in_size(
    const std::array<ValueType, N> &)
{
    return 0;
}

template <size_type iter, typename ValueType, size_type N, typename First,
          typename... Remaining>
constexpr GKO_ATTRIBUTES std::enable_if_t<(iter < N), int> spans_in_size(
    const std::array<ValueType, N> &size, First first,
    Remaining &&... remaining)
{
    static_assert(sizeof...(Remaining) + 1 == N - iter,
                  "Number of remaining spans must be equal to N - iter");
    return GKO_ASSERT(span{first}.is_valid()),
           GKO_ASSERT(span{first} <= span{size[iter]}),
           spans_in_size<iter + 1>(size, std::forward<Remaining>(remaining)...);
}


template <typename ValueType, size_type N, typename... Spans>
constexpr GKO_ATTRIBUTES int validate_spans(
    const std::array<ValueType, N> &size, Spans &&... spans)
{
    return detail::spans_in_size<0>(size, std::forward<Spans>(spans)...);
}


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


/**
 * Evaluates if at least one type of Args is a gko::span and the others either
 * also gko::span or fulfill std::is_integral
 */
template <typename... Args>
using are_span_compatible = are_span_compatible_impl<false, Args...>;


template <typename ValueType, size_type N, size_type current, typename... Dims>
constexpr GKO_ATTRIBUTES
    std::enable_if_t<(current == N), std::array<ValueType, N>>
    to_array_impl(const dim<N> &, Dims &&... dims)
{
    static_assert(sizeof...(Dims) == N,
                  "Number of arguments must match dimensionality!");
    return {{std::forward<Dims>(dims)...}};
}


template <typename ValueType, size_type N, size_type current, typename... Dims>
constexpr GKO_ATTRIBUTES
    std::enable_if_t<(current < N), std::array<ValueType, N>>
    to_array_impl(const dim<N> &size, Dims &&... dims)
{
    return to_array_impl<ValueType, N, current + 1>(
        size, std::forward<Dims>(dims)..., size[current]);
}


template <typename ValueType, size_type N>
constexpr GKO_ATTRIBUTES std::array<ValueType, N> to_array(const dim<N> &size)
{
    return to_array_impl<ValueType, N, 0>(size);
}


template <size_type iter, typename ValueType, size_type N>
constexpr GKO_ATTRIBUTES std::enable_if_t<iter == N, ValueType>
mult_dim_upwards_impl(const std::array<ValueType, N> &)
{
    return 1;
}

template <size_type iter, typename ValueType, size_type N>
constexpr GKO_ATTRIBUTES std::enable_if_t<(iter < N), ValueType>
mult_dim_upwards_impl(const std::array<ValueType, N> &size)
{
    return size[iter] * mult_dim_upwards_impl<iter + 1>(size);
}


template <size_type iter = 1, typename ValueType, size_type N, typename... Args>
constexpr GKO_ATTRIBUTES
    std::enable_if_t<N == 0 || (iter == N && iter == sizeof...(Args) + 1),
                     std::array<ValueType, N == 0 ? 0 : N - 1>>
    compute_default_stride_array_impl(const std::array<ValueType, N> &,
                                      Args &&... args)
{
    return {{std::forward<Args>(args)...}};
}

template <size_type iter = 1, typename ValueType, size_type N, typename... Args>
constexpr GKO_ATTRIBUTES std::enable_if_t<
    (iter < N) && (iter == sizeof...(Args) + 1), std::array<ValueType, N - 1>>
compute_default_stride_array_impl(const std::array<ValueType, N> &size,
                                  Args &&... args)
{
    return compute_default_stride_array_impl<iter + 1>(
        size, std::forward<Args>(args)..., mult_dim_upwards_impl<iter>(size));
}


template <typename ValueType, size_type dimensions>
constexpr GKO_ATTRIBUTES
    std::array<ValueType, (dimensions > 0 ? dimensions - 1 : 0)>
    compute_default_stride_array(const std::array<ValueType, dimensions> &size)
{
    return compute_default_stride_array_impl(size);
}


template <size_type iter, typename ValueType, size_type N, typename Callable,
          typename... Indices>
GKO_ATTRIBUTES std::enable_if_t<iter == N> multidim_for_each_impl(
    const std::array<ValueType, N> &, Callable callable, Indices &&... indices)
{
    static_assert(iter == sizeof...(Indices),
                  "Number arguments must match current iteration!");
    callable(std::forward<Indices>(indices)...);
}

template <size_type iter, typename ValueType, size_type N, typename Callable,
          typename... Indices>
GKO_ATTRIBUTES std::enable_if_t<(iter < N)> multidim_for_each_impl(
    const std::array<ValueType, N> &size, Callable &&callable,
    Indices &&... indices)
{
    static_assert(iter == sizeof...(Indices),
                  "Number arguments must match current iteration!");
    for (size_type i = 0; i < size[iter]; ++i) {
        multidim_for_each_impl<iter + 1>(size, std::forward<Callable>(callable),
                                         std::forward<Indices>(indices)..., i);
    }
}


/**
 * Creates a recursive for-loop for each dimension and calls dest(indices...) =
 * source(indices...)
 */
template <typename ValueType, size_type N, typename Callable>
GKO_ATTRIBUTES void multidim_for_each(const std::array<ValueType, N> &size,
                                      Callable &&callable)
{
    multidim_for_each_impl<0>(size, std::forward<Callable>(callable));
}


}  // namespace detail


/**
 * A row_major accessor is a bridge between a range and the row-major memory
 * layout.
 *
 * You should never try to explicitly create an instance of this accessor.
 * Instead, supply it as a template parameter to a range, and pass the
 * constructor parameters for this class to the range (it will forward it to
 * this class).
 *
 * @warning For backward compatability reasons, a specialization is provided
 *          for dimensionality == 2.
 *
 * @tparam ValueType  type of values this accessor returns
 * @tparam Dimensionality  number of dimensions of this accessor
 */
template <typename ValueType, size_type Dimensionality>
class row_major {
public:
    friend class range<row_major>;

    static_assert(Dimensionality != 0,
                  "This accessor does not support a dimensionality of 0!");
    static constexpr size_type dimensionality = Dimensionality;

    /**
     * Type of values returned by the accessor.
     */
    using value_type = ValueType;

    /**
     * Type of underlying data storage.
     */
    using data_type = value_type *;

    /**
     * Number of dimensions of the accessor.
     */

    using const_accessor = row_major<const ValueType, Dimensionality>;
    using stride_type = std::array<const size_type, dimensionality - 1>;
    using length_type = std::array<const size_type, dimensionality>;

protected:
    /**
     * Creates a row_major accessor.
     *
     * @param data  pointer to the block of memory containing the data
     * @param lengths size / length of the accesses of each dimension
     * @param stride  distance (in elements) between starting positions of
     *                the dimensions (i.e.
     *                `x_1 * stride_1 + x_2 * stride_2 * ... + x_n`
     *                points to the element at (x_1, x_2, ..., x_n))
     */
    constexpr GKO_ATTRIBUTES explicit row_major(data_type data,
                                                dim<dimensionality> size,
                                                stride_type stride)
        : data{data},
          lengths(detail::to_array<const size_type>(size)),
          stride(stride)
    {}

    /**
     * Creates a row_major accessor with a default stride (assumes no padding)
     *
     * @param data  pointer to the block of memory containing the data
     * @param lengths size / length of the accesses of each dimension
     */
    constexpr GKO_ATTRIBUTES explicit row_major(data_type data,
                                                dim<dimensionality> size)
        : data{data},
          lengths(detail::to_array<const size_type>(size)),
          stride(detail::compute_default_stride_array(lengths))
    {}

public:
    /**
     * Creates a row_major range which contains a read-only version of the
     * current accessor.
     *
     * @returns  a row major range which is read-only.
     */
    constexpr GKO_ATTRIBUTES range<const_accessor> to_const() const
    {
        // TODO Remove this functionality all together (if requested)
        return range<const_accessor>(data, lengths, stride);
    }

    /**
     * Returns the data element at the specified indices
     *
     * @param row  row index
     * @param col  column index
     *
     * @return data element at (indices...)
     */
    template <typename... Indices>
    constexpr GKO_ATTRIBUTES
        std::enable_if_t<are_all_integral<Indices...>::value, value_type &>
        operator()(Indices &&... indices) const
    {
        return data[detail::compute_storage_index(
            lengths, stride, std::forward<Indices>(indices)...)];
    }

    /**
     * Returns the sub-range spanning the range (x1_span, x2_span, ...)
     *
     * @param rows  row span
     * @param cols  column span
     *
     * @return sub-range spanning the given spans
     */
    template <typename... SpanTypes>
    constexpr GKO_ATTRIBUTES std::enable_if_t<
        detail::are_span_compatible<SpanTypes...>::value, range<row_major>>
    operator()(SpanTypes... spans) const
    {
        return detail::validate_spans(lengths, spans...),
               range<row_major>{
                   data + detail::compute_storage_index(lengths, stride,
                                                        (span{spans}.begin)...),
                   dim<dimensionality>{
                       (span{spans}.end - span{spans}.begin)...},
                   stride};
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @return length in dimension `dimension`
     */
    constexpr GKO_ATTRIBUTES size_type length(size_type dimension) const
    {
        return lengths[dimension];
    }

    /**
     * Copies data from another accessor
     *
     * @warning Do not use this function since it is not optimized for a
     *          specific executor. It will always be performed sequentially.
     *          Please write an optimized version (adjusted to the architecture)
     *          by iterating through the values yourself.
     *
     * @tparam OtherAccessor  type of the other accessor
     *
     * @param other  other accessor
     */
    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        detail::multidim_for_each(lengths, [this, &other](auto... indices) {
            (*this)(indices...) = other(indices...);
        });
    }

    /**
     * Reference to the underlying data.
     */
    const data_type data;

    /**
     * An array of dimension sizes.
     */
    const length_type lengths;

    /**
     * Distance between consecutive rows for each dimension (except the first).
     */
    const stride_type stride;
};


/**
 * A row_major accessor is a bridge between a range and the row-major memory
 * layout.
 *
 * You should never try to explicitly create an instance of this accessor.
 * Instead, supply it as a template parameter to a range, and pass the
 * constructor parameters for this class to the range (it will forward it to
 * this class).
 *
 * @note  This is the original implementation, which is now the specialization
 *        for Dimensionality = 2.
 *
 * @tparam ValueType  type of values this accessor returns
 */
template <typename ValueType>
class row_major<ValueType, 2> {
public:
    friend class range<row_major>;

    /**
     * Number of dimensions of the accessor.
     */
    static constexpr size_type dimensionality = 2;

    /**
     * Type of values returned by the accessor.
     */
    using value_type = ValueType;

    /**
     * Type of underlying data storage.
     */
    using data_type = value_type *;

    using const_accessor = row_major<const ValueType, dimensionality>;

protected:
    /**
     * Creates a row_major accessor.
     *
     * @param data  pointer to the block of memory containing the data
     * @param num_row  number of rows of the accessor
     * @param num_cols  number of columns of the accessor
     * @param stride  distance (in elements) between starting positions of
     *                consecutive rows (i.e. `data + i * stride` points to
     *                the `i`-th row)
     */
    constexpr GKO_ATTRIBUTES explicit row_major(data_type data,
                                                size_type num_rows,
                                                size_type num_cols,
                                                size_type stride)
        : data{data}, lengths{num_rows, num_cols}, stride{stride}
    {}

    /**
     * Creates a row_major accessor.
     *
     * @param data  pointer to the block of memory containing the data
     * @param lengths size / length of the accesses of each dimension
     * @param stride  distance (in elements) between starting positions of
     *                consecutive rows (i.e. `data + i * stride` points to
     *                the `i`-th row)
     */
    constexpr GKO_ATTRIBUTES explicit row_major(data_type data,
                                                dim<dimensionality> size,
                                                size_type stride)
        : data{data},
          lengths(detail::to_array<const size_type>(size)),
          stride(stride)
    {}

    /**
     * Creates a row_major accessor with a default stride (assumes no padding)
     *
     * @param data  pointer to the block of memory containing the data
     * @param lengths size / length of the accesses of each dimension
     */
    constexpr GKO_ATTRIBUTES explicit row_major(data_type data,
                                                dim<dimensionality> size)
        : data{data},
          lengths(detail::to_array<const size_type>(size)),
          stride{size[1]}
    {}

public:
    /**
     * Creates a row_major range which contains a read-only version of the
     * current accessor.
     *
     * @returns  a row major range which is read-only.
     */
    constexpr GKO_ATTRIBUTES range<const_accessor> to_const() const
    {
        return range<const_accessor>{data, lengths[0], lengths[1], stride};
    }

    /**
     * Returns the data element at position (row, col)
     *
     * @param row  row index
     * @param col  column index
     *
     * @return data element at (row, col)
     */
    constexpr GKO_ATTRIBUTES value_type &operator()(size_type row,
                                                    size_type col) const
    {
        return GKO_ASSERT(row < lengths[0]), GKO_ASSERT(col < lengths[1]),
               data[row * stride + col];
    }

    /**
     * Returns the sub-range spanning the range (rows, cols)
     *
     * @param rows  row span
     * @param cols  column span
     *
     * @return sub-range spanning the range (rows, cols)
     */
    constexpr GKO_ATTRIBUTES range<row_major> operator()(const span &rows,
                                                         const span &cols) const
    {
        return GKO_ASSERT(rows.is_valid()), GKO_ASSERT(cols.is_valid()),
               GKO_ASSERT(rows <= span{lengths[0]}),
               GKO_ASSERT(cols <= span{lengths[1]}),
               range<row_major>(data + rows.begin * stride + cols.begin,
                                rows.end - rows.begin, cols.end - cols.begin,
                                stride);
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @return length in dimension `dimension`
     */
    constexpr GKO_ATTRIBUTES size_type length(size_type dimension) const
    {
        return dimension < 2 ? lengths[dimension] : 1;
    }

    /**
     * Copies data from another accessor
     *
     * @warning Do not use this function since it is not optimized for a
     *          specific executor. It will always be performed sequentially.
     *          Please write an optimized version (adjusted to the architecture)
     *          by iterating through the values yourself.
     *
     * @tparam OtherAccessor  type of the other accessor
     *
     * @param other  other accessor
     */
    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        for (size_type i = 0; i < lengths[0]; ++i) {
            for (size_type j = 0; j < lengths[1]; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
    }

    /**
     * Reference to the underlying data.
     */
    const data_type data;

    /**
     * An array of dimension sizes.
     */
    const std::array<const size_type, dimensionality> lengths;

    /**
     * Distance between consecutive rows.
     */
    const size_type stride;
};


namespace detail {
/**
 * Namespace for helper functions and structs for
 * the block column major accessor.
 */
namespace blk_col_major {


/**
 * Runs from first to last dimension in order to compute the index.
 *
 * The index is computed like this:
 * indices: x1, x2, x3, ..., xn
 * compute(stride, x1, x2, x3, ..., x(n-1), xn) ->
 *  x1 * stride[0] + x2 * stride[1] + ...
 *    + x(n-2) * stride[n-3] + x(n-1) + xn * stride[n-2]
 * Note that swap of the last two strides, making this 'block column major'.
 */
template <typename ValueType, size_type total_dim, size_type current_iter = 1>
struct index_helper_s {
    static_assert(total_dim >= 1, "Dimensionality must be >= 1");
    static_assert(current_iter <= total_dim, "Iteration must be <= total_dim!");

    static constexpr size_type dim_idx{current_iter - 1};

    template <typename FirstType, typename... Indices>
    static constexpr GKO_ATTRIBUTES ValueType
    compute(const std::array<ValueType, total_dim> &size,
            const std::array<ValueType, (total_dim > 0 ? total_dim - 1 : 0)>
                &stride,
            FirstType first, Indices &&... idxs)
    {
        if (current_iter == total_dim - 1) {
            return GKO_ASSERT(first < size[dim_idx]),
                   first +
                       index_helper_s<ValueType, total_dim, current_iter + 1>::
                           compute(size, stride,
                                   std::forward<Indices>(idxs)...);
        }

        return GKO_ASSERT(first < size[dim_idx]),
               first * stride[dim_idx] +
                   index_helper_s<ValueType, total_dim, current_iter + 1>::
                       compute(size, stride, std::forward<Indices>(idxs)...);
    }
};

template <typename ValueType, size_type total_dim>
struct index_helper_s<ValueType, total_dim, total_dim> {
    static_assert(total_dim >= 2, "Dimensionality must be >= 2");

    static constexpr size_type dim_idx{total_dim - 1};

    template <typename FirstType>
    static constexpr GKO_ATTRIBUTES ValueType
    compute(const std::array<ValueType, total_dim> &size,
            const std::array<ValueType, (total_dim > 1 ? total_dim - 1 : 0)>
                &stride,
            FirstType first)
    {
        return GKO_ASSERT(first < size[total_dim - 1]),
               first * stride[dim_idx - 1];
    }
};

/**
 * Computes the flat storage index for block-column-major access.
 *
 * @param size  the multi-dimensional sizes of the range of values
 * @param stride  the stride array
 * @param idxs  the multi-dimensional indices of the desired entry
 */
template <typename ValueType, size_type total_dim, typename... Indices>
constexpr GKO_ATTRIBUTES ValueType compute_index(
    const std::array<ValueType, total_dim> &size,
    const std::array<ValueType, (total_dim > 0 ? total_dim - 1 : 0)> &stride,
    Indices &&... idxs)
{
    return index_helper_s<ValueType, total_dim>::compute(
        size, stride, std::forward<Indices>(idxs)...);
}


template <size_type iter = 1, typename ValueType, size_type N, typename... Args>
constexpr GKO_ATTRIBUTES
    std::enable_if_t<(iter == N - 1) && (iter == sizeof...(Args) + 1),
                     std::array<ValueType, N - 1>>
    default_stride_array_impl(const std::array<ValueType, N> &size,
                              Args &&... args)
{
    return {{std::forward<Args>(args)..., size[N - 2]}};
}

template <size_type iter = 1, typename ValueType, size_type N, typename... Args>
constexpr GKO_ATTRIBUTES std::enable_if_t<(iter < N - 1 || iter == N) &&
                                              (iter == sizeof...(Args) + 1),
                                          std::array<ValueType, N - 1>>
default_stride_array_impl(const std::array<ValueType, N> &size, Args &&... args)
{
    return default_stride_array_impl<iter + 1>(
        size, std::forward<Args>(args)...,
        detail::mult_dim_upwards_impl<iter>(size));
}

template <typename ValueType, size_type dimensions>
constexpr GKO_ATTRIBUTES
    std::array<ValueType, (dimensions > 0 ? dimensions - 1 : 0)>
    default_stride_array(const std::array<ValueType, dimensions> &size)
{
    return default_stride_array_impl(size);
}


}  // namespace blk_col_major
}  // namespace detail


/**
 * A bridge between a range and a block-column-major memory layout.
 *
 * Only the innermost two dimensions are regarded as defining
 * a column-major matrix, and the rest of the dimensions are treated
 * identically to \ref row_major.
 *
 * You should not try to explicitly create an instance of this accessor.
 * Instead, supply it as a template parameter to a range, and pass the
 * constructor parameters for this class to the range (it will forward it to
 * this class).
 *
 * @tparam ValueType  type of values this accessor returns
 * @tparam Dimensionality  number of dimensions of this accessor
 */
template <typename ValueType, size_type Dimensionality>
class block_col_major {
public:
    friend class range<block_col_major>;

    static_assert(Dimensionality != 0,
                  "This accessor does not support a dimensionality of 0!");
    static_assert(Dimensionality != 1,
                  "Please use row_major accessor for 1D ranges.");

    /**
     * Number of dimensions of the accessor.
     */
    static constexpr size_type dimensionality = Dimensionality;

    /**
     * Type of values returned by the accessor.
     */
    using value_type = ValueType;

    /**
     * Type of underlying data storage.
     */
    using data_type = value_type *;

    using const_accessor = block_col_major<const ValueType, Dimensionality>;
    using stride_type = std::array<const size_type, dimensionality - 1>;
    using length_type = std::array<const size_type, dimensionality>;

protected:
    /**
     * Creates a block_col_major accessor.
     *
     * @param data  pointer to the block of memory containing the data
     * @param lengths size / length of the accesses of each dimension
     * @param stride  distance (in elements) between starting positions of
     *                the dimensions (i.e.
     *   `x_1 * stride_1 + x_2 * stride_2 * ... + x_(n-1) + x_n * stride_(n-1)`
     *                points to the element at (x_1, x_2, ..., x_n))
     */
    constexpr GKO_ATTRIBUTES explicit block_col_major(data_type data,
                                                      dim<dimensionality> size,
                                                      stride_type stride)
        : data{data},
          lengths(detail::to_array<const size_type>(size)),
          stride(stride)
    {}

    /**
     * Creates a block_col_major accessor with a default stride
     * (assumes no padding)
     *
     * @param data  pointer to the block of memory containing the data
     * @param lengths size / length of the accesses of each dimension
     */
    constexpr GKO_ATTRIBUTES explicit block_col_major(data_type data,
                                                      dim<dimensionality> size)
        : data{data},
          lengths(detail::to_array<const size_type>(size)),
          stride(detail::blk_col_major::default_stride_array(lengths))
    {}

public:
    /**
     * Creates a block_col_major range which contains a read-only version of
     * the current accessor.
     *
     * @returns  a block column major range which is read-only.
     */
    constexpr GKO_ATTRIBUTES range<const_accessor> to_const() const
    {
        // TODO Remove this functionality all together (if requested)
        return range<const_accessor>(data, lengths, stride);
    }

    /**
     * Returns the data element at the specified indices
     *
     * @param row  row index
     * @param col  column index
     *
     * @return data element at (indices...)
     */
    template <typename... Indices>
    constexpr GKO_ATTRIBUTES
        std::enable_if_t<are_all_integral<Indices...>::value, value_type &>
        operator()(Indices &&... indices) const
    {
        return data[detail::blk_col_major::compute_index(
            lengths, stride, std::forward<Indices>(indices)...)];
    }

    /**
     * Returns the sub-range spanning the range (x1_span, x2_span, ...)
     *
     * @param rows  row span
     * @param cols  column span
     *
     * @return sub-range spanning the given spans
     */
    template <typename... SpanTypes>
    constexpr GKO_ATTRIBUTES
        std::enable_if_t<detail::are_span_compatible<SpanTypes...>::value,
                         range<block_col_major>>
        operator()(SpanTypes... spans) const
    {
        return detail::validate_spans(lengths, spans...),
               range<block_col_major>{
                   data + detail::blk_col_major::compute_index(
                              lengths, stride, (span{spans}.begin)...),
                   dim<dimensionality>{
                       (span{spans}.end - span{spans}.begin)...},
                   stride};
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @return length in dimension `dimension`
     */
    constexpr GKO_ATTRIBUTES size_type length(size_type dimension) const
    {
        return lengths[dimension];
    }

    /**
     * Copies data from another accessor
     *
     * @warning Do not use this function since it is not optimized for a
     *          specific executor. It will always be performed sequentially.
     *          Please write an optimized version (adjusted to the architecture)
     *          by iterating through the values yourself.
     *
     * @tparam OtherAccessor  type of the other accessor
     *
     * @param other  other accessor
     */
    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        detail::multidim_for_each(lengths, [this, &other](auto... indices) {
            (*this)(indices...) = other(indices...);
        });
    }

    /**
     * Reference to the underlying data.
     */
    const data_type data;

    /**
     * An array of dimension sizes.
     */
    const length_type lengths;

    /**
     * Distance between consecutive 'layers' for each dimension
     * (except the second, for which it is 1).
     */
    const stride_type stride;
};


}  // namespace accessor
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_RANGE_ACCESSORS_HPP_
