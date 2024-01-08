// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_ROW_MAJOR_HPP_
#define GKO_ACCESSOR_ROW_MAJOR_HPP_

#include <array>
#include <cinttypes>


#include "accessor_helper.hpp"
#include "range.hpp"
#include "utils.hpp"


namespace gko {
namespace acc {


/**
 * A row_major accessor is a bridge between a range and the row-major memory
 * layout.
 *
 * You should never try to explicitly create an instance of this accessor.
 * Instead, supply it as a template parameter to a range, and pass the
 * constructor parameters for this class to the range (it will forward it to
 * this class).
 *
 * @warning For backward compatibility reasons, a specialization is provided
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
    using data_type = value_type*;

    using const_accessor = row_major<const ValueType, Dimensionality>;
    using length_type = std::array<size_type, dimensionality>;
    using stride_type = std::array<size_type, dimensionality - 1>;

private:
    using index_type = std::int64_t;

protected:
    /**
     * Creates a row_major accessor.
     *
     * @param lengths size / length of the accesses of each dimension
     * @param data  pointer to the block of memory containing the data
     * @param stride  distance (in elements) between starting positions of
     *                the dimensions (i.e.
     *                `x_1 * stride_1 + x_2 * stride_2 * ... + x_n`
     *                points to the element at (x_1, x_2, ..., x_n))
     */
    constexpr GKO_ACC_ATTRIBUTES explicit row_major(length_type size,
                                                    data_type data,
                                                    stride_type stride)
        : lengths(size), data{data}, stride(stride)
    {}

    /**
     * Creates a row_major accessor with a default stride (assumes no
     * padding)
     *
     * @param lengths size / length of the accesses of each dimension
     * @param data  pointer to the block of memory containing the data
     */
    constexpr GKO_ACC_ATTRIBUTES explicit row_major(length_type size,
                                                    data_type data)
        : row_major{size, data,
                    helper::compute_default_row_major_stride_array(size)}
    {}

public:
    /**
     * Creates a row_major range which contains a read-only version of the
     * current accessor.
     *
     * @returns  a row major range which is read-only.
     */
    constexpr GKO_ACC_ATTRIBUTES range<const_accessor> to_const() const
    {
        // TODO Remove this functionality all together (if requested)
        return range<const_accessor>(lengths, data, stride);
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
    constexpr GKO_ACC_ATTRIBUTES
        std::enable_if_t<are_all_integral<Indices...>::value, value_type&>
        operator()(Indices&&... indices) const
    {
        return data[helper::compute_row_major_index<index_type>(
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
    constexpr GKO_ACC_ATTRIBUTES
        std::enable_if_t<helper::are_index_span_compatible<SpanTypes...>::value,
                         range<row_major>>
        operator()(SpanTypes... spans) const
    {
        return helper::validate_index_spans(lengths, spans...),
               range<row_major>{
                   length_type{
                       (index_span{spans}.end - index_span{spans}.begin)...},
                   data + helper::compute_row_major_index<index_type>(
                              lengths, stride, (index_span{spans}.begin)...),
                   stride};
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @return length in dimension `dimension`
     */
    constexpr GKO_ACC_ATTRIBUTES size_type length(size_type dimension) const
    {
        return lengths[dimension];
    }

    /**
     * An array of dimension sizes.
     */
    const length_type lengths;

    /**
     * Reference to the underlying data.
     */
    const data_type data;

    /**
     * Distance between consecutive rows for each dimension (except the
     * first).
     */
    const stride_type stride;
};


}  // namespace acc
}  // namespace gko

#endif  // GKO_ACCESSOR_ROW_MAJOR_HPP_
