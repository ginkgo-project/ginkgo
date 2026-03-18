// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_DEVICE_DENSE_HPP_
#define GKO_PUBLIC_CORE_MATRIX_DEVICE_DENSE_HPP_

#include <cassert>

#include <ginkgo/core/base/dim.hpp>


namespace gko {
namespace matrix {
namespace view {


/**
 * Non-owning view of a matrix::Dense to be used inside device kernels.
 * This type is used to provide a simple and stable ABI for passing data between
 * libraries.
 * The data is stored in row-major order.
 *
 * @tparam ValueType  the value type used to store matrix entries.
 */
template <typename ValueType>
struct dense {
    dim<2> size;
    size_type stride;
    ValueType* values;

    /** Constructs a dense view from size, stride and values. */
    constexpr dense(dim<2> size, size_type stride, ValueType* values)
        : size{size}, stride{stride}, values{values}
    {
        assert(stride >= size[1]);
    }

    /** Returns a const view of the same values */
    constexpr dense<const ValueType> as_const() const
    {
        return dense<const ValueType>{size, stride, values};
    }

    /** Subscript operator accessing the given row and column */
    constexpr ValueType& operator()(size_type row, size_type col) const
    {
        assert(row < size[0] && col < size[1]);
        return values[row * stride + col];
    }
};


/**
 * Non-owning view of a matrix::Ell to be used inside device kernels.
 * This type is used to provide a simple and stable ABI for passing data between
 * libraries.
 *
 * @tparam ValueType  the value type used to store matrix entries.
 */
template <typename ValueType, typename IndexType>
struct ell {
    dim<2> size;
    size_type num_stored_elements_per_row;
    size_type stride;
    ValueType* values;
    IndexType* col_idxs;

    /** Constructs a ell view */
    constexpr ell(dim<2> size, size_type num_stored_elements_per_row,
                  size_type stride, ValueType* values, IndexType* col_idxs)
        : size{size},
          num_stored_elements_per_row{num_stored_elements_per_row},
          stride{stride},
          values{values},
          col_idxs(col_idxs)
    {
        assert(stride >= size[0]);
    }

    /** Returns a const view of the same values */
    constexpr ell<const ValueType, const IndexType> as_const() const
    {
        return ell<const ValueType, const IndexType>{
            size, num_stored_elements_per_row, stride, values, col_idxs};
    }

    /** Return the index of Ell storage */
    constexpr size_type linearize_index(size_type row,
                                        size_type idx) const noexcept
    {
        assert(idx < num_stored_elements_per_row && row < size[0]);
        return row + stride * idx;
    }

    /** accessing the value of the given row and idx-th stored element of the
     * row */
    constexpr ValueType& val_at(size_type row, size_type idx) const
    {
        return values[this->linearize_index(row, idx)];
    }

    /** accessing the column index of the given row and idx-th stored element of
     * the row */
    constexpr IndexType& col_at(size_type row, size_type idx) const
    {
        return col_idxs[this->linearize_index(row, idx)];
    }
};


}  // namespace view
}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_DEVICE_DENSE_HPP_
