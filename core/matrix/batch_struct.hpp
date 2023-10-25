/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_CORE_MATRIX_BATCH_STRUCT_HPP_
#define GKO_CORE_MATRIX_BATCH_STRUCT_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


namespace gko {
namespace batch {
namespace matrix {
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

    inline size_type get_num_nnz() const
    {
        return static_cast<size_type>(stride * num_rows);
    }

    inline size_type get_storage_size() const
    {
        return get_num_nnz() * sizeof(value_type);
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

    inline size_type get_num_nnz() const
    {
        return static_cast<size_type>(stride * num_stored_elems_per_row);
    }

    inline size_type get_storage_size() const
    {
        return get_num_nnz() * sizeof(value_type);
    }
};


}  // namespace ell


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
