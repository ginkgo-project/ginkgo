// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_BATCH_STRUCT_HPP_
#define GKO_DPCPP_BASE_BATCH_STRUCT_HPP_


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


/** @file batch_struct.hpp
 *
 * Helper functions to generate a batch struct from a batch LinOp,
 * while also shallow-casting to the required DPCPP scalar type.
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
            static_cast<int>(op->get_common_size()[1]),
            static_cast<int>(op->get_common_size()[0]),
            static_cast<int>(op->get_common_size()[1])};
}


/**
 * Generates a uniform batch struct from a batch of multi-vectors.
 */
template <typename ValueType>
inline batch::multi_vector::uniform_batch<ValueType> get_batch_struct(
    batch::MultiVector<ValueType>* const op)
{
    return {op->get_values(), op->get_num_batch_items(),
            static_cast<int>(op->get_common_size()[1]),
            static_cast<int>(op->get_common_size()[0]),
            static_cast<int>(op->get_common_size()[1])};
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_BATCH_STRUCT_HPP_
