// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_MATRIX_BATCH_STRUCT_HPP_
#define GKO_COMMON_CUDA_HIP_MATRIX_BATCH_STRUCT_HPP_


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/unified/base/kernel_launch.hpp"
#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/** @file batch_struct.hpp
 *
 * Helper functions to generate a batch struct from a batch LinOp,
 * while also shallow-casting to the required GKO_DEVICE_NAMESPACE scalar
 * type.
 *
 * A specialization is needed for every format of every kind of linear algebra
 * object. These are intended to be called on the host.
 */


/**
 * Generates an immutable uniform batch struct from a batch of csr matrices.
 */
template <typename ValueType, typename IndexType>
inline batch::matrix::csr::uniform_batch<const device_type<ValueType>,
                                         const IndexType>
get_batch_struct(const batch::matrix::Csr<ValueType, IndexType>* const op)
{
    return {as_device_type(op->get_const_values()),
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
inline batch::matrix::csr::uniform_batch<device_type<ValueType>, IndexType>
get_batch_struct(batch::matrix::Csr<ValueType, IndexType>* const op)
{
    return {as_device_type(op->get_values()),
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
inline batch::matrix::dense::uniform_batch<const device_type<ValueType>>
get_batch_struct(const batch::matrix::Dense<ValueType>* const op)
{
    return {as_device_type(op->get_const_values()), op->get_num_batch_items(),
            static_cast<int32>(op->get_common_size()[1]),
            static_cast<int32>(op->get_common_size()[0]),
            static_cast<int32>(op->get_common_size()[1])};
}


/**
 * Generates a uniform batch struct from a batch of dense matrices.
 */
template <typename ValueType>
inline batch::matrix::dense::uniform_batch<device_type<ValueType>>
get_batch_struct(batch::matrix::Dense<ValueType>* const op)
{
    return {as_device_type(op->get_values()), op->get_num_batch_items(),
            static_cast<int32>(op->get_common_size()[1]),
            static_cast<int32>(op->get_common_size()[0]),
            static_cast<int32>(op->get_common_size()[1])};
}


/**
 * Generates an immutable uniform batch struct from a batch of ell matrices.
 */
template <typename ValueType, typename IndexType>
inline batch::matrix::ell::uniform_batch<const device_type<ValueType>,
                                         const IndexType>
get_batch_struct(const batch::matrix::Ell<ValueType, IndexType>* const op)
{
    return {as_device_type(op->get_const_values()),
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
inline batch::matrix::ell::uniform_batch<device_type<ValueType>, IndexType>
get_batch_struct(batch::matrix::Ell<ValueType, IndexType>* const op)
{
    return {as_device_type(op->get_values()),
            op->get_col_idxs(),
            op->get_num_batch_items(),
            static_cast<IndexType>(op->get_common_size()[0]),
            static_cast<IndexType>(op->get_common_size()[0]),
            static_cast<IndexType>(op->get_common_size()[1]),
            static_cast<IndexType>(op->get_num_stored_elements_per_row())};
}


/**
 * Passes a batch of csr matrices to a kernel on values of type KernelValueType
 * with its values as an arithmetic() range.
 */
template <typename KernelValueType, typename ValueType, typename IndexType>
inline auto arithmetic(
    const batch::matrix::csr::uniform_batch<ValueType, IndexType>& batch)
{
    const auto values = arithmetic<KernelValueType>(
        batch.values, batch.num_batch_items * batch.num_nnz_per_item);
    return batch::matrix::csr::uniform_batch<ValueType, IndexType,
                                             decltype(values)>{
        values,
        batch.col_idxs,
        batch.row_ptrs,
        batch.num_batch_items,
        batch.num_rows,
        batch.num_cols,
        batch.num_nnz_per_item};
}


/**
 * @copydoc arithmetic(const batch::matrix::csr::uniform_batch<ValueType,
 * IndexType>&)
 */
template <typename KernelValueType, typename ValueType>
inline auto arithmetic(
    const batch::matrix::dense::uniform_batch<ValueType>& batch)
{
    const auto values = arithmetic<KernelValueType>(
        batch.values, batch.num_batch_items * batch.num_rows * batch.stride);
    return batch::matrix::dense::uniform_batch<ValueType, decltype(values)>{
        values, batch.num_batch_items, batch.stride, batch.num_rows,
        batch.num_cols};
}


/**
 * @copydoc arithmetic(const batch::matrix::csr::uniform_batch<ValueType,
 * IndexType>&)
 */
template <typename KernelValueType, typename ValueType, typename IndexType>
inline auto arithmetic(
    const batch::matrix::ell::uniform_batch<ValueType, IndexType>& batch)
{
    const auto values = arithmetic<KernelValueType>(
        batch.values, batch.num_batch_items * batch.num_rows *
                          batch.num_stored_elems_per_row);
    return batch::matrix::ell::uniform_batch<ValueType, IndexType,
                                             decltype(values)>{
        values,
        batch.col_idxs,
        batch.num_batch_items,
        batch.stride,
        batch.num_rows,
        batch.num_cols,
        batch.num_stored_elems_per_row};
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_MATRIX_BATCH_STRUCT_HPP_
