// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_BLOCK_COL_MAJOR_HPP_
#define GKO_ACCESSOR_BLOCK_COL_MAJOR_HPP_

#include <array>
#include <cinttypes>


#include "accessor_helper.hpp"
#include "range.hpp"
#include "utils.hpp"


namespace gko {
namespace acc {


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
template <typename ValueType, std::size_t Dimensionality>
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
    static constexpr auto dimensionality = Dimensionality;

    /**
     * Type of values returned by the accessor.
     */
    using value_type = ValueType;

    /**
     * Type of underlying data storage.
     */
    using data_type = value_type*;

    using const_accessor = block_col_major<const ValueType, Dimensionality>;
    using stride_type = std::array<size_type, dimensionality - 1>;
    using length_type = std::array<size_type, dimensionality>;

private:
    using index_type = std::int64_t;

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
    constexpr GKO_ACC_ATTRIBUTES explicit block_col_major(length_type size,
                                                          data_type data,
                                                          stride_type stride)
        : lengths(size), data{data}, stride(stride)
    {}

    /**
     * Creates a block_col_major accessor with a default stride
     * (assumes no padding)
     *
     * @param data  pointer to the block of memory containing the data
     * @param lengths size / length of the accesses of each dimension
     */
    constexpr GKO_ACC_ATTRIBUTES explicit block_col_major(length_type size,
                                                          data_type data)
        : lengths(size),
          data{data},
          stride(helper::blk_col_major::default_stride_array(lengths))
    {}

public:
    /**
     * Creates a block_col_major range which contains a read-only version of
     * the current accessor.
     *
     * @returns  a block column major range which is read-only.
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
        return data[helper::blk_col_major::compute_index<index_type>(
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
                         range<block_col_major>>
        operator()(SpanTypes... spans) const
    {
        return helper::validate_index_spans(lengths, spans...),
               range<block_col_major>{
                   length_type{
                       (index_span{spans}.end - index_span{spans}.begin)...},
                   data + helper::blk_col_major::compute_index<index_type>(
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
     * Distance between consecutive 'layers' for each dimension
     * (except the second, for which it is 1).
     */
    const stride_type stride;
};


}  // namespace acc
}  // namespace gko

#endif  // GKO_ACCESSOR_BLOCK_COL_MAJOR_HPP_
