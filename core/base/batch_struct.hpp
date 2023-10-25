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
    int32 stride;
    int32 num_rows;
    int32 num_rhs;
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
    int32 stride;
    int32 num_rows;
    int32 num_rhs;

    size_type get_storage_size() const
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
