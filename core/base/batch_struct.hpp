// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_BATCH_STRUCT_HPP_
#define GKO_CORE_BASE_BATCH_STRUCT_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>

#include "accessor/index_span.hpp"
#include "accessor/range.hpp"


namespace gko {
namespace batch {
namespace multi_vector {


/**
 * Encapsulates one matrix from a batch of multi-vectors.
 *
 * @tparam ValueType  the type of the stored values
 * @tparam Values  a pointer or a range over the values
 */
template <typename ValueType, typename Values = ValueType*>
struct batch_item {
    using value_type = ValueType;
    Values values;
    int32 stride;
    int32 num_rows;
    int32 num_rhs;
};


/**
 * A 'simple' structure to store a global uniform batch of multi-vectors.
 *
 * @tparam ValueType  the type of the stored values
 * @tparam Values  the type that accesses them, see batch_item
 */
template <typename ValueType, typename Values = ValueType*>
struct uniform_batch {
    using value_type = ValueType;
    using entry_type = batch_item<ValueType, Values>;

    Values values;
    size_type num_batch_items;
    int32 stride;
    int32 num_rows;
    int32 num_rhs;

    inline size_type get_single_item_num_nnz() const
    {
        return static_cast<size_type>(stride * num_rows);
    }
};


/**
 * Returns the start of one item of a batch.
 */
template <typename Values>
GKO_ATTRIBUTES GKO_INLINE Values batch_item_ptr(const Values batch_start,
                                                const size_type stride,
                                                const int num_rows,
                                                const size_type batch_idx)
{
    return batch_start + batch_idx * stride * num_rows;
}


/**
 * Returns the values of one item of a batch as a sub-range.
 */
template <typename Accessor>
GKO_ATTRIBUTES GKO_INLINE acc::range<Accessor> batch_item_ptr(
    const acc::range<Accessor>& batch_start, const size_type stride,
    const int num_rows, const size_type batch_idx)
{
    const auto begin = batch_idx * stride * num_rows;
    return batch_start(acc::index_span(begin, begin + stride * num_rows));
}


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
template <typename ValueType, typename Values>
GKO_ATTRIBUTES GKO_INLINE multi_vector::batch_item<ValueType, Values>
extract_batch_item(const multi_vector::uniform_batch<ValueType, Values>& batch,
                   const size_type batch_idx)
{
    return {multi_vector::batch_item_ptr(batch.values, batch.stride,
                                         batch.num_rows, batch_idx),
            batch.stride, batch.num_rows, batch.num_rhs};
}

template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE multi_vector::batch_item<ValueType>
extract_batch_item(ValueType* const batch_values, const int32 stride,
                   const int32 num_rows, const int32 num_rhs,
                   const size_type batch_idx)
{
    return {batch_values + batch_idx * stride * num_rows, stride, num_rows,
            num_rhs};
}


}  // namespace batch
}  // namespace gko


#endif  // GKO_CORE_BASE_BATCH_STRUCT_HPP_
