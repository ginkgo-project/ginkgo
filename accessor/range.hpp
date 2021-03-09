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

#ifndef GKO_ACCESSOR_RANGE_HPP_
#define GKO_ACCESSOR_RANGE_HPP_

#include <utility>


#include "utils.hpp"


namespace gko {
namespace acc {


template <typename Accessor>
class range {
public:
    /**
     * The type of the underlying accessor.
     */
    using accessor = Accessor;

    /**
     * The number of dimensions of the range.
     */
    static constexpr size_type dimensionality = accessor::dimensionality;

    /**
     * Use the default destructor.
     */
    ~range() = default;

    /**
     * Creates a new range.
     *
     * @tparam AccessorParam  types of parameters forwarded to the accessor
     *                        constructor
     *
     * @param params  parameters forwarded to Accessor constructor.
     */
    template <typename... AccessorParams>
    GKO_ACC_ATTRIBUTES constexpr explicit range(AccessorParams &&... params)
        : accessor_{std::forward<AccessorParams>(params)...}
    {}

    /**
     * Returns a value (or a sub-range) with the specified indexes.
     *
     * @tparam DimensionTypes  The types of indexes. Supported types depend on
     *                         the underlying accessor, but are usually either
     *                         integer types or index_spans. If at least one
     *                         index is a span, the returned value will be a
     *                         sub-range (if that is supported by the accessor).
     *
     * @param dimensions  the indexes of the values or index_spans for the new
     *                    range.
     *
     * @return a value on position `(dimensions...)` or a sub-range with the
     *         given index_spans.
     */
    template <typename... DimensionTypes>
    GKO_ACC_ATTRIBUTES constexpr auto operator()(
        DimensionTypes &&... dimensions) const
        -> decltype(std::declval<accessor>()(
            std::forward<DimensionTypes>(dimensions)...))
    {
        static_assert(sizeof...(dimensions) <= dimensionality,
                      "Too many dimensions in range call");
        return accessor_(std::forward<DimensionTypes>(dimensions)...);
    }

    range(const range &other) = default;

    /**
     * Returns the length of the specified dimension of the range.
     *
     * @param dimension  the dimensions whose length is returned
     *
     * @return  the length of the `dimension`-th dimension of the range
     */
    GKO_ACC_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return accessor_.length(dimension);
    }

    /**
     * Returns a pointer to the accessor.
     *
     * Can be used to access data and functions of a specific accessor.
     *
     * @return pointer to the accessor
     */
    GKO_ACC_ATTRIBUTES constexpr const accessor *operator->() const noexcept
    {
        return &accessor_;
    }

    /**
     * `Returns a reference to the accessor.
     *
     * @return reference to the accessor
     */
    GKO_ACC_ATTRIBUTES constexpr const accessor &get_accessor() const noexcept
    {
        return accessor_;
    }

private:
    accessor accessor_;
};


}  // namespace acc
}  // namespace gko

#endif  // GKO_ACCESSOR_RANGE_HPP_
