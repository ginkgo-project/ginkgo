// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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
 * Generates an immutable uniform batch struct from a batch of external
 * operators.
 */
template <typename ValueType>
inline batch::matrix::external::uniform_batch<const cuda_type<ValueType>>
get_batch_struct(const batch::matrix::External<ValueType>* const op)
{
    printf("s=%p\n", op->get_simple_apply_functions().cuda_apply);
    printf("a=%p\n", op->get_advanced_apply_functions().cuda_apply);
    assert(op->get_simple_apply_functions().cuda_apply);
    assert(op->get_advanced_apply_functions().cuda_apply);
    return {op->get_num_batch_items(),
            static_cast<int32>(op->get_common_size()[0]),
            static_cast<int32>(op->get_common_size()[1]),
            op->get_simple_apply_functions().cuda_apply,
            op->get_advanced_apply_functions().cuda_apply,
            op->get_payload()};
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_MATRIX_BATCH_STRUCT_HPP_
