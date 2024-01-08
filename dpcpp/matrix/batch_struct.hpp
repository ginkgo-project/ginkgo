// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_MATRIX_BATCH_STRUCT_HPP_
#define GKO_DPCPP_MATRIX_BATCH_STRUCT_HPP_


#include "core/matrix/batch_struct.hpp"


#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/base/batch_struct.hpp"


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
 * Generates an immutable uniform batch struct from a batch of csr matrices.
 */
template <typename ValueType, typename IndexType>
inline batch::matrix::csr::uniform_batch<const ValueType, const IndexType>
get_batch_struct(const batch::matrix::Csr<ValueType, IndexType>* const op)
{
    return {op->get_const_values(),
            op->get_const_col_idxs(),
            op->get_const_row_ptrs(),
            op->get_num_batch_items(),
            static_cast<IndexType>(op->get_common_size()[0]),
            static_cast<IndexType>(op->get_common_size()[1]),
            static_cast<IndexType>(op->get_num_elements_per_item())};
}


/**
 * Generates a uniform batch struct from a batch of csr matrices.
 */
template <typename ValueType, typename IndexType>
inline batch::matrix::csr::uniform_batch<ValueType, IndexType> get_batch_struct(
    batch::matrix::Csr<ValueType, IndexType>* const op)
{
    return {op->get_values(),
            op->get_col_idxs(),
            op->get_row_ptrs(),
            op->get_num_batch_items(),
            static_cast<IndexType>(op->get_common_size()[0]),
            static_cast<IndexType>(op->get_common_size()[1]),
            static_cast<IndexType>(op->get_num_elements_per_item())};
}


/**
 * Generates an immutable uniform batch struct from a batch of dense matrices.
 */
template <typename ValueType>
inline batch::matrix::dense::uniform_batch<const ValueType> get_batch_struct(
    const batch::matrix::Dense<ValueType>* const op)
{
    return {op->get_const_values(), op->get_num_batch_items(),
            static_cast<int32>(op->get_common_size()[1]),
            static_cast<int32>(op->get_common_size()[0]),
            static_cast<int32>(op->get_common_size()[1])};
}


/**
 * Generates a uniform batch struct from a batch of dense matrices.
 */
template <typename ValueType>
inline batch::matrix::dense::uniform_batch<ValueType> get_batch_struct(
    batch::matrix::Dense<ValueType>* const op)
{
    return {op->get_values(), op->get_num_batch_items(),
            static_cast<int32>(op->get_common_size()[1]),
            static_cast<int32>(op->get_common_size()[0]),
            static_cast<int32>(op->get_common_size()[1])};
}


/**
 * Generates an immutable uniform batch struct from a batch of ell matrices.
 */
template <typename ValueType, typename IndexType>
inline batch::matrix::ell::uniform_batch<const ValueType, const IndexType>
get_batch_struct(const batch::matrix::Ell<ValueType, IndexType>* const op)
{
    return {op->get_const_values(),
            op->get_const_col_idxs(),
            op->get_num_batch_items(),
            static_cast<IndexType>(op->get_common_size()[0]),
            static_cast<IndexType>(op->get_common_size()[0]),
            static_cast<IndexType>(op->get_common_size()[1]),
            static_cast<IndexType>(op->get_num_stored_elements_per_row())};
}


/**
 * Generates a uniform batch struct from a batch of ell matrices.
 */
template <typename ValueType, typename IndexType>
inline batch::matrix::ell::uniform_batch<ValueType, IndexType> get_batch_struct(
    batch::matrix::Ell<ValueType, IndexType>* const op)
{
    return {op->get_values(),
            op->get_col_idxs(),
            op->get_num_batch_items(),
            static_cast<IndexType>(op->get_common_size()[0]),
            static_cast<IndexType>(op->get_common_size()[0]),
            static_cast<IndexType>(op->get_common_size()[1]),
            static_cast<IndexType>(op->get_num_stored_elements_per_row())};
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_MATRIX_BATCH_STRUCT_HPP_
