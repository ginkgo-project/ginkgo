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

#ifndef GKO_REFERENCE_COMPONENTS_FIXED_BLOCK_HPP_
#define GKO_REFERENCE_COMPONENTS_FIXED_BLOCK_HPP_


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace blockutils {


/**
 * @brief A dense block of values with compile-time constant dimensions
 *
 * The blocks are interpreted as row-major. However, in the future,
 *  a layout template parameter can be added if needed.
 *
 * The primary use is to reinterpret subsets of entries in a big array as
 *  small dense blocks.
 *
 * @tparam ValueType  The numeric type of entries of the block
 * @tparam nrows  Number of rows
 * @tparam ncols  Number of columns
 */
template <typename ValueType, int nrows, int ncols>
class FixedBlock final {
    static_assert(nrows > 0, "Requires positive number of rows!");
    static_assert(ncols > 0, "Requires positive number of columns!");

public:
    using value_type = ValueType;

    value_type &at(const int row, const int col)
    {
        return vals[row * ncols + col];
    }

    const value_type &at(const int row, const int col) const
    {
        return vals[row * ncols + col];
    }

    value_type &operator()(const int row, const int col)
    {
        return at(row, col);
    }

    const value_type &operator()(const int row, const int col) const
    {
        return at(row, col);
    }

private:
    ValueType vals[nrows * ncols];
};


}  // namespace blockutils
}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_FIXED_BLOCK_HPP_
