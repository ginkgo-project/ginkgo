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

#ifndef GKO_CORE_MATRIX_BATCH_STRUCT_HPP_
#define GKO_CORE_MATRIX_BATCH_STRUCT_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>

namespace gko {


namespace batch_ell {


/**
 * Encapsulates (refers to) one matrix from a batch of ELL matrices
 */
template <typename ValueType>
struct BatchEntry {
    using value_type = ValueType;
    using index_type = int;
    ValueType* values;
    const int* col_idxs;
    size_type num_stored_elems_per_row;
    size_type stride;
    int num_rows;
    int num_nnz;
};

/**
 * A 'simple' structure to store a global uniform batch of ELL matrices.
 *
 * It is uniform in the sense that all matrices in the batch have a common
 * sparsity pattern.
 */
template <typename ValueType>
struct UniformBatch {
    using value_type = ValueType;
    using index_type = int;
    using entry_type = BatchEntry<ValueType>;

    ValueType* values;    ///< Concatenated values array of all matrices
    const int* col_idxs;  ///< (common) column indices
    size_type num_batch;  ///< Number of matrices in the batch
    size_type num_stored_elems_per_row;  ///< Number of matrices in the batch
    size_type stride;                    ///< Number of matrices in the batch
    int num_rows;  ///< (common) number of rows in each matrix
    int num_nnz;   ///< (common) number of nonzeros in each matrix

    size_type get_entry_storage() const
    {
        return num_nnz * (sizeof(value_type) + sizeof(index_type));
    }
};


}  // namespace batch_ell


namespace batch_csr {


/**
 * Encapsulates (refers to) one matrix from a batch of CSR matrices
 */
template <typename ValueType>
struct BatchEntry {
    using value_type = ValueType;
    using index_type = int;
    ValueType* values;
    const int* col_idxs;
    const int* row_ptrs;
    int num_rows;
    int num_nnz;
};

/**
 * A 'simple' structure to store a global uniform batch of CSR matrices.
 *
 * It is uniform in the sense that all matrices in the batch have a common
 * sparsity pattern.
 */
template <typename ValueType>
struct UniformBatch {
    using value_type = ValueType;
    using index_type = int;
    using entry_type = BatchEntry<ValueType>;

    ValueType* values;    ///< Concatenated values array of all matrices
    const int* col_idxs;  ///< (common) column indices
    const int* row_ptrs;  ///< (common) row pointers
    size_type num_batch;  ///< Number of matrices in the batch
    int num_rows;         ///< (common) number of rows in each matrix
    int num_nnz;          ///< (common) number of nonzeros in each matrix

    size_type get_entry_storage() const
    {
        return num_nnz * (sizeof(value_type) + sizeof(index_type)) +
               (num_rows + 1) * sizeof(index_type);
    }
};


}  // namespace batch_csr


namespace batch_dense {


/**
 * Encapsulates one matrix from a batch of dense matrices (vectors).
 */
template <typename ValueType>
struct BatchEntry {
    using value_type = ValueType;
    ValueType* values;
    size_type stride;
    int num_rows;
    int num_rhs;
};

/**
 * A 'simple' structure to store a global uniform batch of dense matrices.
 *
 * It is uniform in the sense that all matrices in the batch have common sizes.
 */
template <typename ValueType>
struct UniformBatch {
    using value_type = ValueType;
    using entry_type = BatchEntry<ValueType>;

    ValueType* values;    ///< Concatenated values of all matrices in the batch
    size_type num_batch;  ///< Number of matrices in the batch
    size_type stride;     ///< Common stride of each dense matrix
    int num_rows;         ///< Common number of rows in each matrix
    int num_rhs;          ///< Common number of columns of each matrix
    int num_nnz;          ///< Common number of non-zeros of each matrix, ie.,
                          ///< the number or rows times the number of columns

    size_type get_entry_storage() const { return num_nnz * sizeof(value_type); }
};


}  // namespace batch_dense


namespace batch {


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE gko::batch_dense::BatchEntry<const ValueType>
to_const(const gko::batch_dense::BatchEntry<ValueType>& b)
{
    return {b.values, b.stride, b.num_rows, b.num_rhs};
}


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE gko::batch_dense::UniformBatch<const ValueType>
to_const(const gko::batch_dense::UniformBatch<ValueType>& ub)
{
    return {ub.values, ub.num_batch, ub.stride, ub.num_rows, ub.num_rhs};
}


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE gko::batch_csr::BatchEntry<const ValueType> to_const(
    const gko::batch_csr::BatchEntry<ValueType>& b)
{
    return {b.values, b.col_idxs, b.row_ptrs, b.num_rows, b.num_nnz};
}


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE gko::batch_csr::UniformBatch<const ValueType>
to_const(const gko::batch_csr::UniformBatch<ValueType>& ub)
{
    return {ub.values,    ub.col_idxs, ub.row_ptrs,
            ub.num_batch, ub.num_rows, ub.num_nnz};
}


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE gko::batch_ell::BatchEntry<const ValueType> to_const(
    const gko::batch_ell::BatchEntry<ValueType>& b)
{
    return {b.values, b.col_idxs, b.num_stored_elems_per_row,
            b.stride, b.num_rows, b.num_nnz};
}


template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE gko::batch_ell::UniformBatch<const ValueType>
to_const(const gko::batch_ell::UniformBatch<ValueType>& ub)
{
    return {ub.values, ub.col_idxs, ub.num_batch, ub.num_stored_elems_per_row,
            ub.stride, ub.num_rows, ub.num_nnz};
}


/**
 * Extract one object (matrix, vector etc.) from a batch of objects
 *
 * This overload is for batch dense matrices.
 * These overloads are intended to be called from within a kernel.
 *
 * @param batch  The batch of objects to extract from
 * @param batch_idx  The position of the desired object in the batch
 */
template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE batch_dense::BatchEntry<ValueType> batch_entry(
    const batch_dense::UniformBatch<ValueType>& batch,
    const size_type batch_idx)
{
    return {batch.values + batch_idx * batch.stride * batch.num_rows,
            batch.stride, batch.num_rows, batch.num_rhs};
}

template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE batch_dense::BatchEntry<ValueType> batch_entry(
    ValueType* const batch_values, const size_type stride, const int num_rows,
    const int num_rhs, const size_type batch_idx)
{
    return {batch_values + batch_idx * stride * num_rows, stride, num_rows,
            num_rhs};
}

template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE ValueType* batch_entry_ptr(
    ValueType* const batch_start, const size_type stride, const int num_rows,
    const size_type batch_idx)
{
    return batch_start + batch_idx * stride * num_rows;
}


/**
 * Extract one object (matrix, vector etc.) from a batch of objects
 *
 * This overload is for batch CSR matrices.
 * These overloads are intended to be called from within a kernel.
 *
 * @param batch  The batch of objects to extract from
 * @param batch_idx  The position of the desired object in the batch
 */
template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE batch_csr::BatchEntry<ValueType> batch_entry(
    const batch_csr::UniformBatch<ValueType>& batch, const size_type batch_idx)
{
    return {batch.values + batch_idx * batch.num_nnz, batch.col_idxs,
            batch.row_ptrs, batch.num_rows, batch.num_nnz};
}


/**
 * Extract one object (matrix, vector etc.) from a batch of objects
 *
 * This overload is for batch ELL matrices.
 * These overloads are intended to be called from within a kernel.
 *
 * @param batch  The batch of objects to extract from
 * @param batch_idx  The position of the desired object in the batch
 */
template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE batch_ell::BatchEntry<ValueType> batch_entry(
    const batch_ell::UniformBatch<ValueType>& batch, const size_type batch_idx)
{
    return {batch.values + batch_idx * batch.num_nnz,
            batch.col_idxs,
            batch.num_stored_elems_per_row,
            batch.stride,
            batch.num_rows,
            batch.num_nnz};
}


}  // namespace batch


}  // namespace gko

#endif  // GKO_CORE_MATRIX_BATCH_STRUCT_HPP_
