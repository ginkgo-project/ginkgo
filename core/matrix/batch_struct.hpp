// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_BATCH_STRUCT_HPP_
#define GKO_CORE_MATRIX_BATCH_STRUCT_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


namespace gko {
namespace batch {
namespace matrix {
namespace csr {


/**
 * Encapsulates one matrix from a batch of csr matrices.
 */
template <typename ValueType, typename IndexType>
struct batch_item {
    using value_type = ValueType;
    using index_type = IndexType;

    ValueType* values;
    const index_type* col_idxs;
    const index_type* row_ptrs;
    index_type num_rows;
    index_type num_cols;
};


/**
 * A 'simple' structure to store a global uniform batch of csr matrices.
 */
template <typename ValueType, typename IndexType>
struct uniform_batch {
    using value_type = ValueType;
    using index_type = IndexType;
    using entry_type = batch_item<value_type, index_type>;

    ValueType* values;
    const index_type* col_idxs;
    const index_type* row_ptrs;
    size_type num_batch_items;
    index_type num_rows;
    index_type num_cols;
    index_type num_nnz_per_item;

    inline size_type get_single_item_num_nnz() const
    {
        return static_cast<size_type>(num_nnz_per_item);
    }
};


}  // namespace csr


namespace dense {


/**
 * Encapsulates one matrix from a batch of dense matrices.
 */
template <typename ValueType>
struct batch_item {
    using value_type = ValueType;
    value_type* values;
    int32 stride;
    int32 num_rows;
    int32 num_cols;
};


/**
 * A 'simple' structure to store a global uniform batch of dense matrices.
 */
template <typename ValueType>
struct uniform_batch {
    using value_type = ValueType;
    using entry_type = batch_item<ValueType>;

    ValueType* values;
    size_type num_batch_items;
    int32 stride;
    int32 num_rows;
    int32 num_cols;

    inline size_type get_single_item_num_nnz() const
    {
        return static_cast<size_type>(stride * num_rows);
    }
};


}  // namespace dense


namespace ell {


/**
 * Encapsulates one matrix from a batch of ell matrices.
 */
template <typename ValueType, typename IndexType>
struct batch_item {
    using value_type = ValueType;
    using index_type = IndexType;

    ValueType* values;
    const index_type* col_idxs;
    index_type stride;
    index_type num_rows;
    index_type num_cols;
    index_type num_stored_elems_per_row;
};


/**
 * A 'simple' structure to store a global uniform batch of ell matrices.
 */
template <typename ValueType, typename IndexType>
struct uniform_batch {
    using value_type = ValueType;
    using index_type = IndexType;
    using entry_type = batch_item<value_type, index_type>;

    ValueType* values;
    const index_type* col_idxs;
    size_type num_batch_items;
    index_type stride;
    index_type num_rows;
    index_type num_cols;
    index_type num_stored_elems_per_row;

    inline size_type get_single_item_num_nnz() const
    {
        return static_cast<size_type>(stride * num_stored_elems_per_row);
    }
};


}  // namespace ell


template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE csr::batch_item<const ValueType, const IndexType>
to_const(const csr::batch_item<ValueType, IndexType>& b)
{
    return {b.values, b.col_idxs, b.row_ptrs, b.num_rows, b.num_cols};
}


template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE csr::uniform_batch<const ValueType, const IndexType>
to_const(const csr::uniform_batch<ValueType, IndexType>& ub)
{
    return {ub.values,   ub.col_idxs, ub.row_ptrs,        ub.num_batch_items,
            ub.num_rows, ub.num_cols, ub.num_nnz_per_item};
}


template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE csr::batch_item<ValueType, IndexType>
extract_batch_item(const csr::uniform_batch<ValueType, IndexType>& batch,
                   const size_type batch_idx)
{
    return {batch.values + batch_idx * batch.num_nnz_per_item, batch.col_idxs,
            batch.row_ptrs, batch.num_rows, batch.num_cols};
}

template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE csr::batch_item<ValueType, IndexType>
extract_batch_item(ValueType* const batch_values,
                   IndexType* const batch_col_idxs,
                   IndexType* const batch_row_ptrs, const int num_rows,
                   const int num_cols, int num_nnz_per_item,
                   const size_type batch_idx)
{
    return {batch_values + batch_idx * num_nnz_per_item, batch_col_idxs,
            batch_row_ptrs, num_rows, num_cols};
}


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE dense::batch_item<const ValueType> to_const(
    const dense::batch_item<ValueType>& b)
{
    return {b.values, b.stride, b.num_rows, b.num_cols};
}


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE dense::uniform_batch<const ValueType> to_const(
    const dense::uniform_batch<ValueType>& ub)
{
    return {ub.values, ub.num_batch_items, ub.stride, ub.num_rows, ub.num_cols};
}


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE dense::batch_item<ValueType> extract_batch_item(
    const dense::uniform_batch<ValueType>& batch, const size_type batch_idx)
{
    return {batch.values + batch_idx * batch.stride * batch.num_rows,
            batch.stride, batch.num_rows, batch.num_cols};
}

template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE dense::batch_item<ValueType> extract_batch_item(
    ValueType* const batch_values, const int32 stride, const int32 num_rows,
    const int32 num_cols, const size_type batch_idx)
{
    return {batch_values + batch_idx * stride * num_rows, stride, num_rows,
            num_cols};
}


template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE ell::batch_item<const ValueType, const IndexType>
to_const(const ell::batch_item<ValueType, IndexType>& b)
{
    return {b.values,   b.col_idxs, b.stride,
            b.num_rows, b.num_cols, b.num_stored_elems_per_row};
}


template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE ell::uniform_batch<const ValueType, const IndexType>
to_const(const ell::uniform_batch<ValueType, IndexType>& ub)
{
    return {ub.values,   ub.col_idxs, ub.num_batch_items,         ub.stride,
            ub.num_rows, ub.num_cols, ub.num_stored_elems_per_row};
}


template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE ell::batch_item<ValueType, IndexType>
extract_batch_item(const ell::uniform_batch<ValueType, IndexType>& batch,
                   const size_type batch_idx)
{
    return {batch.values +
                batch_idx * batch.num_stored_elems_per_row * batch.num_rows,
            batch.col_idxs,
            batch.stride,
            batch.num_rows,
            batch.num_cols,
            batch.num_stored_elems_per_row};
}

template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE ell::batch_item<ValueType, IndexType>
extract_batch_item(ValueType* const batch_values,
                   IndexType* const batch_col_idxs, const int stride,
                   const int num_rows, const int num_cols,
                   int num_elems_per_row, const size_type batch_idx)
{
    return {batch_values + batch_idx * num_elems_per_row * num_rows,
            batch_col_idxs,
            stride,
            num_rows,
            num_cols,
            num_elems_per_row};
}


}  // namespace matrix
}  // namespace batch
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BATCH_STRUCT_HPP_
