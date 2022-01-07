/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_HIP_MATRIX_BATCH_STRUCT_HIP_HPP_
#define GKO_HIP_MATRIX_BATCH_STRUCT_HIP_HPP_


#include "core/matrix/batch_struct.hpp"


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "hip/base/config.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


/** @file batch_struct.hpp
 *
 * Helper functions to generate a batch struct from a batch LinOp,
 * while also shallow-casting to the requried Hip scalar type.
 *
 * A specialization is needed for every format of every kind of linear algebra
 * object. These are intended to be called on the host.
 */


/**
 * Generates an immutable uniform batch struct from a batch of dense matrices.
 */
template <typename ValueType>
inline gko::batch_dense::UniformBatch<const hip_type<ValueType>>
get_batch_struct(const matrix::BatchDense<ValueType>* const op)
{
    return {
        as_hip_type(op->get_const_values()),
        op->get_num_batch_entries(),
        op->get_stride().at(0),
        static_cast<int>(op->get_size().at(0)[0]),
        static_cast<int>(op->get_size().at(0)[1]),
        static_cast<int>(op->get_size().at(0)[0] * op->get_size().at(0)[1])};
}

/**
 * Generates a uniform batch struct from a batch of dense matrices.
 */
template <typename ValueType>
inline gko::batch_dense::UniformBatch<hip_type<ValueType>> get_batch_struct(
    matrix::BatchDense<ValueType>* const op)
{
    return {
        as_hip_type(op->get_values()),
        op->get_num_batch_entries(),
        op->get_stride().at(0),
        static_cast<int>(op->get_size().at(0)[0]),
        static_cast<int>(op->get_size().at(0)[1]),
        static_cast<int>(op->get_size().at(0)[0] * op->get_size().at(0)[1])};
}

/**
 * Generates an immutable uniform batch struct from a batch of CSR matrices.
 */
template <typename ValueType>
inline gko::batch_csr::UniformBatch<const hip_type<ValueType>> get_batch_struct(
    const matrix::BatchCsr<ValueType, int32>* const op)
{
    return {as_hip_type(op->get_const_values()),
            op->get_const_col_idxs(),
            op->get_const_row_ptrs(),
            op->get_num_batch_entries(),
            static_cast<int>(op->get_size().at(0)[0]),
            static_cast<int>(op->get_num_stored_elements() /
                             op->get_num_batch_entries())};
}


/**
 * Generates an mutable uniform batch struct from a batch of CSR matrices.
 */
template <typename ValueType>
inline gko::batch_csr::UniformBatch<hip_type<ValueType>> get_batch_struct(
    matrix::BatchCsr<ValueType>* const op)
{
    return {as_hip_type(op->get_values()),
            op->get_const_col_idxs(),
            op->get_const_row_ptrs(),
            op->get_num_batch_entries(),
            static_cast<int>(op->get_size().at(0)[0]),
            static_cast<int>(op->get_num_stored_elements() /
                             op->get_num_batch_entries())};
}


/**
 * Generates an immutable uniform batch struct from a batch of CSR matrices.
 */
template <typename ValueType>
inline gko::batch_ell::UniformBatch<const hip_type<ValueType>> get_batch_struct(
    const matrix::BatchEll<ValueType, int32>* const op)
{
    return {as_hip_type(op->get_const_values()),
            op->get_const_col_idxs(),
            op->get_num_batch_entries(),
            op->get_num_stored_elements_per_row().at(0),
            op->get_stride().at(0),
            static_cast<int>(op->get_size().at(0)[0]),
            static_cast<int>(op->get_num_stored_elements() /
                             op->get_num_batch_entries())};
}


/**
 * Generates an mutable uniform batch struct from a batch of CSR matrices.
 */
template <typename ValueType>
inline gko::batch_ell::UniformBatch<hip_type<ValueType>> get_batch_struct(
    matrix::BatchEll<ValueType>* const op)
{
    return {as_hip_type(op->get_values()),
            op->get_const_col_idxs(),
            op->get_num_batch_entries(),
            op->get_num_stored_elements_per_row().at(0),
            op->get_stride().at(0),
            static_cast<int>(op->get_size().at(0)[0]),
            static_cast<int>(op->get_num_stored_elements() /
                             op->get_num_batch_entries())};
}


/**
 * Generates an immutable uniform batch struct from a batch of dense matrices
 * that may be null.
 */
template <typename ValueType>
inline gko::batch_dense::UniformBatch<const hip_type<ValueType>>
maybe_null_batch_struct(const matrix::BatchDense<ValueType>* const op)
{
    if (op) {
        return {as_hip_type(op->get_const_values()),
                op->get_num_batch_entries(), op->get_stride().at(0),
                static_cast<int>(op->get_size().at(0)[0]),
                static_cast<int>(op->get_size().at(0)[1])};
    } else {
        return {nullptr, 0, 0, 0, 0};
    }
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko
#endif  // GKO_HIP_MATRIX_BATCH_STRUCT_HIP_HPP_
