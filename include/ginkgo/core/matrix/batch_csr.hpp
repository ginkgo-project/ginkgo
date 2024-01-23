// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_CSR_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_CSR_HPP_


#include <initializer_list>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace batch {
namespace matrix {


/**
 * Csr is a general sparse matrix format that stores the column indices for each
 * nonzero entry and a cumulative sum of the number of nonzeros in each row. It
 * is one of the most popular sparse matrix formats due to its versatility and
 * ability to store a wide range of sparsity patterns in an efficient fashion.
 *
 * @note It is assumed that the sparsity pattern of all the items in the
 * batch is the same and therefore only a single copy of the sparsity pattern is
 * stored.
 *
 * @note Currently only IndexType of int32 is supported.
 *
 * @tparam ValueType  value precision of matrix elements
 * @tparam IndexType  index precision of matrix elements
 *
 * @ingroup batch_csr
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Csr final
    : public EnableBatchLinOp<Csr<ValueType, IndexType>>,
      public EnableCreateMethod<Csr<ValueType, IndexType>>,
      public ConvertibleTo<Csr<next_precision<ValueType>, IndexType>> {
    friend class EnableCreateMethod<Csr>;
    friend class EnablePolymorphicObject<Csr, BatchLinOp>;
    friend class Csr<to_complex<ValueType>, IndexType>;
    friend class Csr<next_precision<ValueType>, IndexType>;
    static_assert(std::is_same<IndexType, int32>::value,
                  "IndexType must be a 32 bit integer");

public:
    using EnableBatchLinOp<Csr>::convert_to;
    using EnableBatchLinOp<Csr>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using unbatch_type = gko::matrix::Csr<value_type, index_type>;
    using absolute_type = remove_complex<Csr>;
    using complex_type = to_complex<Csr>;

    void convert_to(
        Csr<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(Csr<next_precision<ValueType>, IndexType>* result) override;

    /**
     * Creates a mutable view (of matrix::Csr type) of one item of the
     * batch::matrix::Csr<value_type> object. Does not perform any deep
     * copies, but only returns a view of the data.
     *
     * @param item_id  The index of the batch item
     *
     * @return  a batch::matrix::Csr object with the data from the batch item
     * at the given index.
     */
    std::unique_ptr<unbatch_type> create_view_for_item(size_type item_id);

    /**
     * @copydoc create_view_for_item(size_type)
     */
    std::unique_ptr<const unbatch_type> create_const_view_for_item(
        size_type item_id) const;

    /**
     * Returns a pointer to the array of values of the matrix
     *
     * @return the pointer to the array of values
     */
    value_type* get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    /**
     * Returns a pointer to the array of column indices of the matrix
     *
     * @return the pointer to the array of column indices
     */
    index_type* get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc get_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    /**
     * Returns a pointer to the array of row pointers of the matrix
     *
     * @return the pointer to the array of row pointers
     */
    index_type* get_row_ptrs() noexcept { return row_ptrs_.get_data(); }

    /**
     * @copydoc get_row_ptrs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_row_ptrs() const noexcept
    {
        return row_ptrs_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the batch matrix,
     * cumulative across all the batch items.
     *
     * @return the number of elements explicitly stored in the vector,
     *         cumulative across all the batch items
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_size();
    }

    /**
     * Returns the number of stored elements in each batch item.
     *
     * @return the number of stored elements per batch item.
     */
    size_type get_num_elements_per_item() const noexcept
    {
        return this->get_num_stored_elements() / this->get_num_batch_items();
    }

    /**
     * Returns a pointer to the array of values of the matrix for a
     * specific batch item.
     *
     * @param batch_id  the id of the batch item.
     *
     * @return the pointer to the array of values
     */
    value_type* get_values_for_item(size_type batch_id) noexcept
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        GKO_ASSERT(values_.get_size() >=
                   batch_id * this->get_num_elements_per_item());
        return values_.get_data() +
               batch_id * this->get_num_elements_per_item();
    }

    /**
     * @copydoc get_values_for_item(size_type)
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_values_for_item(
        size_type batch_id) const noexcept
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        GKO_ASSERT(values_.get_size() >=
                   batch_id * this->get_num_elements_per_item());
        return values_.get_const_data() +
               batch_id * this->get_num_elements_per_item();
    }

    /**
     * Creates a constant (immutable) batch csr matrix from a constant
     * array. Only a single sparsity pattern (column indices and row pointers)
     * is stored and hence the user needs to ensure that each batch item has the
     * same sparsity pattern.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param col_idxs the col_idxs array of a single batch item of the matrix.
     * @param row_ptrs  the row_ptrs array of a single batch item of the matrix.
     *
     * @return A smart pointer to the constant matrix wrapping the input
     * array (if it resides on the same executor as the matrix) or a copy of the
     * array on the correct executor.
     */
    static std::unique_ptr<const Csr> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        gko::detail::const_array_view<value_type>&& values,
        gko::detail::const_array_view<index_type>&& col_idxs,
        gko::detail::const_array_view<index_type>&& row_ptrs);

    /**
     * Apply the matrix to a multi-vector. Represents the matrix vector
     * multiplication, x = A * b, where x and b are both multi-vectors.
     *
     * @param b  the multi-vector to be applied to
     * @param x  the output multi-vector
     */
    Csr* apply(ptr_param<const MultiVector<value_type>> b,
               ptr_param<MultiVector<value_type>> x);

    /**
     * Apply the matrix to a multi-vector with a linear combination of the given
     * input vector. Represents the matrix vector multiplication, x = alpha * A
     * * b + beta * x, where x and b are both multi-vectors.
     *
     * @param alpha  the scalar to scale the matrix-vector product with
     * @param b      the multi-vector to be applied to
     * @param beta   the scalar to scale the x vector with
     * @param x      the output multi-vector
     */
    Csr* apply(ptr_param<const MultiVector<value_type>> alpha,
               ptr_param<const MultiVector<value_type>> b,
               ptr_param<const MultiVector<value_type>> beta,
               ptr_param<MultiVector<value_type>> x);

    /**
     * @copydoc apply(const MultiVector<value_type>*, MultiVector<value_type>*)
     */
    const Csr* apply(ptr_param<const MultiVector<value_type>> b,
                     ptr_param<MultiVector<value_type>> x) const;

    /**
     * @copydoc apply(const MultiVector<value_type>*, const
     * MultiVector<value_type>*, const MultiVector<value_type>*,
     * MultiVector<value_type>*)
     */
    const Csr* apply(ptr_param<const MultiVector<value_type>> alpha,
                     ptr_param<const MultiVector<value_type>> b,
                     ptr_param<const MultiVector<value_type>> beta,
                     ptr_param<MultiVector<value_type>> x) const;

    /**
     * Performs in-place row and column scaling for this matrix.
     *
     * @param row_scale  the row scalars
     * @param col_scale  the column scalars
     */
    void scale(const array<value_type>& row_scale,
               const array<value_type>& col_scale);

    /**
     * Performs the operation this = alpha*I + beta*this.
     *
     * @param alpha the scalar for identity
     * @param beta  the scalar to multiply this matrix
     *
     * @note Performs the operation in-place for this batch matrix
     * @note This operation fails in case this matrix does not have all its
     *       diagonal entries.
     */
    void add_scaled_identity(ptr_param<const MultiVector<value_type>> alpha,
                             ptr_param<const MultiVector<value_type>> beta);

private:
    /**
     * Creates an uninitialized Csr matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros_per_item  number of nonzeros in each item of the
     * batch matrix
     *
     * @internal It is necessary to pass in the correct nnz_per_item to ensure
     * that the arrays are allocated correctly. An incorrect value will result
     * in a runtime failure when the user tries to use any batch matrix
     * utilities such as create_view_from_item etc.
     */
    Csr(std::shared_ptr<const Executor> exec,
        const batch_dim<2>& size = batch_dim<2>{},
        size_type num_nonzeros_per_item = {});

    /**
     * Creates a Csr matrix from an already allocated (and initialized)
     * array. The column indices array needs to be the same for all batch items.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param col_idxs  the col_idxs array of a single batch item of the matrix.
     * @param row_ptrs  the row_ptrs array of a single batch item of the matrix.
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray,
              typename RowPtrsArray>
    Csr(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
        ValuesArray&& values, ColIdxsArray&& col_idxs, RowPtrsArray&& row_ptrs)
        : EnableBatchLinOp<Csr>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          row_ptrs_{exec, std::forward<RowPtrsArray>(row_ptrs)}
    {
        // Ensure that the value and col_idxs arrays have the correct size
        auto max_num_elems = this->get_common_size()[0] *
                             this->get_common_size()[1] *
                             this->get_num_batch_items();
        GKO_ASSERT(values_.get_size() <= max_num_elems);
        GKO_ASSERT_EQ(row_ptrs_.get_size(), this->get_common_size()[0] + 1);
        GKO_ASSERT_EQ(this->get_num_elements_per_item(), col_idxs_.get_size());
    }

    void apply_impl(const MultiVector<value_type>* b,
                    MultiVector<value_type>* x) const;

    void apply_impl(const MultiVector<value_type>* alpha,
                    const MultiVector<value_type>* b,
                    const MultiVector<value_type>* beta,
                    MultiVector<value_type>* x) const;

    array<value_type> values_;
    array<index_type> col_idxs_;
    array<index_type> row_ptrs_;
};


}  // namespace matrix
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_CSR_HPP_
