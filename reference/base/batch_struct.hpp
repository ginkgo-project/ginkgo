// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_BASE_BATCH_STRUCT_HPP_
#define GKO_REFERENCE_BASE_BATCH_STRUCT_HPP_


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/base/batch_struct.hpp"


namespace gko {
namespace kernels {
/**
 * @brief A namespace for shared functionality between omp and reference
 *  executors.
 */
namespace host {


/** @file batch_struct.hpp
 *
 * Helper functions to generate a batch struct from a batch LinOp.
 *
 * A specialization is needed for every format of every kind of linear algebra
 * object. These are intended to be called on the host.
 */


/**
 * Generates an immutable uniform batch struct from a batch of multi-vectors.
 */
template <typename ValueType>
inline batch::multi_vector::uniform_batch<const ValueType> get_batch_struct(
    const batch::MultiVector<ValueType>* const op)
{
    return {op->get_const_values(), op->get_num_batch_items(),
            static_cast<int32>(op->get_common_size()[1]),
            static_cast<int32>(op->get_common_size()[0]),
            static_cast<int32>(op->get_common_size()[1])};
}


/**
 * Generates a uniform batch struct from a batch of multi-vectors.
 */
template <typename ValueType>
inline batch::multi_vector::uniform_batch<ValueType> get_batch_struct(
    batch::MultiVector<ValueType>* const op)
{
    return {op->get_values(), op->get_num_batch_items(),
            static_cast<int32>(op->get_common_size()[1]),
            static_cast<int32>(op->get_common_size()[0]),
            static_cast<int32>(op->get_common_size()[1])};
}


}  // namespace host
}  // namespace kernels
}  // namespace gko


#endif  // GKO_REFERENCE_BASE_BATCH_STRUCT_HPP_
