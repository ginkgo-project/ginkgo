/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_RANGE_ACCESSORS_HPP_
#define GKO_CORE_BASE_RANGE_ACCESSORS_HPP_


#include "core/base/range.hpp"


namespace gko {
namespace accessor {


template <typename ValueType, size_type Dimensionality>
class row_major {};  // TODO: implement accessor for other dimensionalities

template <typename ValueType>
class row_major<ValueType, 2> {
public:
    using value_type = ValueType;
    static constexpr size_type dimensionality = 2;
    using data_type = value_type *;

    GKO_ATTRIBUTES explicit row_major(data_type data, size_type num_rows,
                                      size_type num_cols, size_type stride)
        : data{data}, lengths{num_rows, num_cols}, stride{stride}
    {}

    GKO_ATTRIBUTES value_type &operator()(size_type row, size_type col) const
    {
        GKO_ASSERT(row < lengths[0]);
        GKO_ASSERT(col < lengths[1]);
        return data[row * stride + col];
    }

    GKO_ATTRIBUTES range<row_major> operator()(const span &rows,
                                               const span &cols) const
    {
        GKO_ASSERT(rows.begin <= rows.end);
        GKO_ASSERT(cols.begin <= cols.end);
        GKO_ASSERT(rows <= span::empty(lengths[0]));
        GKO_ASSERT(cols <= span::empty(lengths[1]));
        return range<row_major>(data + rows.begin * stride + cols.begin,
                                rows.end - rows.begin, cols.end - cols.begin,
                                stride);
    }

    GKO_ATTRIBUTES size_type length(size_type dimension) const
    {
        return dimension < 2 ? lengths[dimension] : 1;
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        for (size_type i = 0; i < lengths[0]; ++i) {
            for (size_type j = 0; j < lengths[1]; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
    };

    const data_type data;
    const std::array<const size_type, dimensionality> lengths;
    const size_type stride;
};


}  // namespace accessor
}  // namespace gko

#endif  // GKO_CORE_BASE_RANGE_ACCESSORS_HPP_
