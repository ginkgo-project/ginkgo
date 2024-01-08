// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_RANGE_ACCESSORS_HPP_
#define GKO_PUBLIC_CORE_BASE_RANGE_ACCESSORS_HPP_


#include <array>


#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
/**
 * @brief The accessor namespace.
 *
 * @ingroup accessor
 */
namespace accessor {


/**
 * A row_major accessor is a bridge between a range and the row-major memory
 * layout.
 *
 * You should never try to explicitly create an instance of this accessor.
 * Instead, supply it as a template parameter to a range, and pass the
 * constructor parameters for this class to the range (it will forward it to
 * this class).
 *
 * @warning The current implementation is incomplete, and only allows for
 *          2-dimensional ranges.
 *
 * @tparam ValueType  type of values this accessor returns
 * @tparam Dimensionality  number of dimensions of this accessor (has to be 2)
 */
template <typename ValueType, size_type Dimensionality>
class row_major {
public:
    friend class range<row_major>;

    static_assert(Dimensionality == 2,
                  "This accessor is only implemented for matrices");

    /**
     * Type of values returned by the accessor.
     */
    using value_type = ValueType;

    /**
     * Type of underlying data storage.
     */
    using data_type = value_type*;

    /**
     * Number of dimensions of the accessor.
     */
    static constexpr size_type dimensionality = 2;

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

public:
    /**
     * Returns the data element at position (row, col)
     *
     * @param row  row index
     * @param col  column index
     *
     * @return data element at (row, col)
     */
    constexpr GKO_ATTRIBUTES value_type& operator()(size_type row,
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
    constexpr GKO_ATTRIBUTES range<row_major> operator()(const span& rows,
                                                         const span& cols) const
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
    GKO_ATTRIBUTES void copy_from(const OtherAccessor& other) const
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


}  // namespace accessor
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_RANGE_ACCESSORS_HPP_
