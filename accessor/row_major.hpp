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

#ifndef GKO_ACCESSOR_ROW_MAJOR_HPP_
#define GKO_ACCESSOR_ROW_MAJOR_HPP_

#include <array>

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
    constexpr GKO_ACC_ATTRIBUTES explicit row_major(data_type data,
                                                    length_type size,
                                                    stride_type stride)
        : data{data}, lengths(size), stride(stride)
    {}

    /**
     * Creates a row_major accessor with a default stride (assumes no
     * padding)
     *
     * @param data  pointer to the block of memory containing the data
     * @param lengths size / length of the accesses of each dimension
     */
    constexpr GKO_ACC_ATTRIBUTES explicit row_major(data_type data,
                                                    length_type size)
        : row_major{
              data, size,
              helper::compute_default_row_major_stride_array<const size_type>(
                  size)}
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
    constexpr GKO_ACC_ATTRIBUTES
        std::enable_if_t<are_all_integral<Indices...>::value, value_type &>
        operator()(Indices &&... indices) const
    {
        return data[helper::compute_row_major_index(
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
                   data + helper::compute_row_major_index(
                              lengths, stride, (index_span{spans}.begin)...),
                   length_type{
                       (index_span{spans}.end - index_span{spans}.begin)...},
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
     * Copies data from another accessor
     *
     * @warning Do not use this function since it is not optimized for a
     *          specific executor. It will always be performed sequentially.
     *          Please write an optimized version (adjusted to the
     * architecture) by iterating through the values yourself.
     *
     * @tparam OtherAccessor  type of the other accessor
     *
     * @param other  other accessor
     */
    template <typename OtherAccessor>
    GKO_ACC_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        helper::multidim_for_each(lengths, [this, &other](auto... indices) {
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
     * Distance between consecutive rows for each dimension (except the
     * first).
     */
    const stride_type stride;
};


}  // namespace acc
}  // namespace gko

#endif  // GKO_ACCESSOR_ROW_MAJOR_HPP_
