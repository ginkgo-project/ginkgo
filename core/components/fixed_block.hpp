/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_COMPONENTS_FIXED_BLOCK_HPP_
#define GKO_CORE_COMPONENTS_FIXED_BLOCK_HPP_


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/math.hpp>

#include <type_traits>

namespace gko {
namespace blockutils {


/// A dense block of values with compile-time constant dimensions
/** The blocks are stored row-major. However, in future,
 *  a layout template parameter can be added if needed.
 *
 * The primary use is to reinterpret subsets of entries in a big array as
 *  small dense blocks.
 *
 * @tparam ValueType The numeric type of entries of the block
 * @tparam nrows Number of rows
 * @tparam ncols Number of columns
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


/**
 * A lightweight dynamic block type for the host space
 *
 * @tparam ValueType The numeric type of entries of the block
 */
template <typename ValueType>
class DenseBlock final {
public:
    using value_type = ValueType;

    DenseBlock() : nrows_{0}, ncols_{0}, vals_{nullptr} {}

    DenseBlock(const int num_rows, const int num_cols)
        : nrows_{num_rows},
          ncols_{num_cols},
          vals_{new value_type[num_rows * num_cols]}
    {}

    ~DenseBlock() { delete[] vals_; }

    value_type &at(const int row, const int col)
    {
        return vals_[row * ncols_ + col];
    }

    const value_type &at(const int row, const int col) const
    {
        return vals_[row * ncols_ + col];
    }

    value_type &operator()(const int row, const int col)
    {
        return at(row, col);
    }

    const value_type &operator()(const int row, const int col) const
    {
        return at(row, col);
    }

    int size() const { return nrows_ * ncols_; }

    void resize(const int nrows, const int ncols)
    {
        nrows_ = nrows;
        ncols_ = ncols;
        delete[] vals_;
        vals_ = new value_type[nrows_ * ncols_];
    }

    void zero()
    {
        for (int i = 0; i < nrows_ * ncols_; i++)
            vals_[i] = static_cast<value_type>(0);
    }

private:
    int nrows_;
    int ncols_;
    value_type *vals_;
};

/// A view into a an array of dense blocks of some runtime-defined size
/** Note that accessing BSR values using this type of view abstracts away the
 * storage layout within the individual blocks, as long as all blocks use the
 * same layout. For now, row-major blocks are assumed.
 *
 * @tparam ValueType The numeric type of entries of the block
 * @tparam IndexType The type of integer used to identify the different blocks
 */
template <typename ValueType, typename IndexType = int32>
class DenseBlocksView final {
public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * @param buffer Segment of memory to be interpreted as an array of 2D
     * blocks
     * @param num_rows Number of rows in each block
     * @param num_cols Number of columns in each block
     */
    DenseBlocksView(ValueType *const buffer, const int num_rows,
                    const int num_cols)
        : nrows_{num_rows}, ncols_{num_cols}, vals_{buffer}
    {}

    value_type &at(const index_type block, const int row, const int col)
    {
        return vals_[block * nrows_ * ncols_ + row * ncols_ + col];
    }

    const typename std::remove_const<value_type>::type &at(
        const index_type block, const int row, const int col) const
    {
        return vals_[block * nrows_ * ncols_ + row * ncols_ + col];
    }

    value_type &operator()(const index_type block, const int row, const int col)
    {
        return at(block, row, col);
    }

    const typename std::remove_const<value_type>::type &operator()(
        const index_type block, const int row, const int col) const
    {
        return at(block, row, col);
    }

private:
    int nrows_;  ///< Number of rows in each block
    int ncols_;  ///< Number of columns in each block
    value_type *vals_;
};


}  // namespace blockutils
}  // namespace gko

#endif
