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
 * Non-owning view of a matrix::MultiVector to be used inside device kernels.
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
 * Non-owning view of a matrix::Coo to be used inside device kernels.
 * This type is used to provide a simple and stable ABI for passing data between
 * libraries.
 *
 * @tparam ValueType  the value type used to store matrix entries.
 * @tparam IndexType  the index type used to store row and column indices.
 */
template <typename ValueType, typename IndexType>
struct coo {
    dim<2> size;
    size_type num_stored_elements;
    ValueType* values;
    IndexType* row_idxs;
    IndexType* col_idxs;

    /** Constructs a coo view from size, nnz, values, row and column indices. */
    constexpr coo(dim<2> size, size_type num_stored_elements, ValueType* values,
                  IndexType* row_idxs, IndexType* col_idxs)
        : size{size},
          num_stored_elements{num_stored_elements},
          values{values},
          row_idxs{row_idxs},
          col_idxs{col_idxs}
    {}

    /** Returns a const view of the same data */
    constexpr coo<const ValueType, const IndexType> as_const() const
    {
        return coo<const ValueType, const IndexType>{
            size, num_stored_elements, values, row_idxs, col_idxs};
    }
};


/**
 * Non-owning view of a matrix::Csr to be used inside device kernels.
 * This type is used to provide a simple and stable ABI for passing data between
 * libraries.
 *
 * @tparam ValueType  the value type used to store matrix entries.
 * @tparam IndexType  the index type used to store row and column indices.
 */
template <typename ValueType, typename IndexType>
struct csr {
    dim<2> size;
    size_type num_stored_elements;
    ValueType* values;
    IndexType* row_ptrs;
    IndexType* col_idxs;

    /** Constructs a coo view from size, nnz, values, row pointers, and column
     * indices. */
    constexpr csr(dim<2> size, size_type num_stored_elements, ValueType* values,
                  IndexType* row_ptrs, IndexType* col_idxs)
        : size{size},
          num_stored_elements{num_stored_elements},
          values{values},
          row_ptrs{row_ptrs},
          col_idxs{col_idxs}
    {}

    /** Returns a const view of the same data */
    constexpr csr<const ValueType, const IndexType> as_const() const
    {
        return csr<const ValueType, const IndexType>{
            size, num_stored_elements, values, row_ptrs, col_idxs};
    }
};


/**
 * Non-owning view of a matrix::Ell to be used inside device kernels.
 * This type is used to provide a simple and stable ABI for passing data between
 * libraries.
 *
 * @tparam ValueType  the value type used to store matrix values.
 * @tparam IndexType  the index type used to store matrix columns.
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

    /** accessing the value of the idx-th element within the given row */
    constexpr ValueType& val_at(size_type row, size_type idx) const
    {
        return values[this->linearize_index(row, idx)];
    }

    /** accessing the column index of the idx-th element within the given row */
    constexpr IndexType& col_at(size_type row, size_type idx) const
    {
        return col_idxs[this->linearize_index(row, idx)];
    }

private:
    /** Return the index of Ell storage */
    constexpr size_type linearize_index(size_type row,
                                        size_type idx) const noexcept
    {
        assert(idx < num_stored_elements_per_row && row < size[0]);
        return row + stride * idx;
    }
};


/**
 * Non-owning view of a matrix::Sellp to be used inside device kernels.
 * This type is used to provide a simple and stable ABI for passing data between
 * libraries.
 *
 * @tparam ValueType  the value type used to store matrix values.
 * @tparam IndexType  the index type used to store matrix columns.
 */
template <typename ValueType, typename IndexType>
struct sellp {
    dim<2> size;
    size_type slice_size;
    size_type stride_factor;
    size_type total_cols;
    ValueType* values;
    IndexType* col_idxs;
    static_assert(std::is_const_v<ValueType> == std::is_const_v<IndexType>,
                  "ValueType and IndexType must share the same constness");
    using adapt_size_type = std::conditional_t<std::is_const_v<ValueType>,
                                               const size_type, size_type>;
    adapt_size_type* slice_lengths;
    adapt_size_type* slice_sets;

    /** Constructs a sellp view */
    constexpr sellp(dim<2> size, size_type slice_size, size_type stride_factor,
                    size_type total_cols, ValueType* values,
                    IndexType* col_idxs, adapt_size_type* slice_lengths,
                    adapt_size_type* slice_sets)
        : size{size},
          slice_size{slice_size},
          stride_factor{stride_factor},
          total_cols{total_cols},
          values{values},
          col_idxs{col_idxs},
          slice_lengths{slice_lengths},
          slice_sets{slice_sets}
    {}

    /** Returns a const view of the same values */
    constexpr sellp<const ValueType, const IndexType> as_const() const
    {
        return sellp<const ValueType, const IndexType>{
            size,   slice_size, stride_factor, total_cols,
            values, col_idxs,   slice_lengths, slice_sets};
    }

    /** accessing the value of the idx-th element within the given row */
    constexpr ValueType& val_at(size_type row, size_type slice_set,
                                size_type idx) const
    {
        return values[this->linearize_index(row, slice_set, idx)];
    }

    /** accessing the column index of the idx-th element within the given row */
    constexpr IndexType& col_at(size_type row, size_type slice_set,
                                size_type idx) const
    {
        return col_idxs[this->linearize_index(row, slice_set, idx)];
    }

private:
    /** Return the index of Sellp storage */
    constexpr size_type linearize_index(size_type row, size_type slice_set,
                                        size_type idx) const noexcept
    {
        assert(row < slice_size);
        // note the following does not catch all idx out of bound access.
        assert(idx < total_cols);
        return (slice_set + idx) * slice_size + row;
    }
};


/**
 * Non-owning view of a matrix::Hybrid to be used inside device kernels.
 * This type is used to provide a simple and stable ABI for passing data between
 * libraries.
 *
 * @tparam ValueType  the value type used to store matrix values.
 * @tparam IndexType  the index type used to store matrix columns.
 */
template <typename ValueType, typename IndexType>
struct hybrid {
    static_assert(std::is_const_v<ValueType> == std::is_const_v<IndexType>,
                  "ValueType and IndexType must share the same constness");
    dim<2> size;
    ell<ValueType, IndexType> ell_part;
    coo<ValueType, IndexType> coo_part;

    /** Constructs a hybrid view */
    constexpr hybrid(ell<ValueType, IndexType> ell_,
                     coo<ValueType, IndexType> coo_)
        : size(ell_.size), ell_part(ell_), coo_part(coo_)
    {
        assert(ell_part.size == coo_part.size);
    }

    /** Returns a const view of the same values */
    constexpr hybrid<const ValueType, const IndexType> as_const() const
    {
        return {ell_part.as_const(), coo_part.as_const()};
    }
};


}  // namespace view
}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_DEVICE_DENSE_HPP_
