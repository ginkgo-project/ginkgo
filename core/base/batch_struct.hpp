// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_BATCH_STRUCT_HPP_
#define GKO_CORE_BASE_BATCH_STRUCT_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace batch {
namespace multi_vector {


/**
 * Encapsulates one matrix from a batch of multi-vectors.
 */
template <typename ValueType>
struct batch_item {
    using value_type = ValueType;
    ValueType* values;
    int stride;
    int num_rows;
    int num_rhs;
};


/**
 * A 'simple' structure to store a global uniform batch of multi-vectors.
 */
template <typename ValueType>
struct uniform_batch {
    using value_type = ValueType;
    using entry_type = batch_item<ValueType>;

    ValueType* values;
    size_type num_batch_items;
    int stride;
    int num_rows;
    int num_rhs;

    size_type get_entry_storage() const
    {
        return num_rows * stride * sizeof(value_type);
    }
};


}  // namespace multi_vector


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE multi_vector::batch_item<const ValueType> to_const(
    const multi_vector::batch_item<ValueType>& b)
{
    return {b.values, b.stride, b.num_rows, b.num_rhs};
}


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE multi_vector::uniform_batch<const ValueType> to_const(
    const multi_vector::uniform_batch<ValueType>& ub)
{
    return {ub.values, ub.num_batch_items, ub.stride, ub.num_rows, ub.num_rhs};
}


/**
 * Extract one object (matrix, vector etc.) from a batch of objects
 *
 * This overload is for batch multi-vectors.
 * These overloads are intended to be called from within a kernel.
 *
 * @param batch  The batch of objects to extract from
 * @param batch_idx  The position of the desired object in the batch
 */
template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE multi_vector::batch_item<ValueType>
extract_batch_item(const multi_vector::uniform_batch<ValueType>& batch,
                   const size_type batch_idx)
{
    return {batch.values + batch_idx * batch.stride * batch.num_rows,
            batch.stride, batch.num_rows, batch.num_rhs};
}

template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE multi_vector::batch_item<ValueType>
extract_batch_item(ValueType* const batch_values, const int stride,
                   const int num_rows, const int num_rhs,
                   const size_type batch_idx)
{
    return {batch_values + batch_idx * stride * num_rows, stride, num_rows,
            num_rhs};
}


}  // namespace batch
}  // namespace gko


#endif  // GKO_CORE_BASE_BATCH_STRUCT_HPP_
