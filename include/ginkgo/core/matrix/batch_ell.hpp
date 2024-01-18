// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_ELL_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_ELL_HPP_


#include <initializer_list>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/matrix/ell.hpp>


namespace gko {
namespace batch {
namespace matrix {


/**
 * Ell is a sparse matrix format that stores the same number of nonzeros in each
 * row, enabling coalesced accesses. It is suitable for sparsity patterns that
 * have a similar number of nonzeros in every row. The values are stored in a
 * column-major fashion similar to the monolithic gko::matrix::Ell class.
 *
 * Similar to the monolithic gko::matrix::Ell class, invalid_index<IndexType> is
 * used as the column index for padded zero entries.
 *
 * @note It is also assumed that the sparsity pattern of all the items in the
 * batch is the same and therefore only a single copy of the sparsity pattern is
 * stored.
 *
 * @note Currently only IndexType of int32 is supported.
 *
 * @tparam ValueType  value precision of matrix elements
 * @tparam IndexType  index precision of matrix elements
 *
 * @ingroup batch_ell
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Ell final
    : public EnableBatchLinOp<Ell<ValueType, IndexType>>,
      public EnableCreateMethod<Ell<ValueType, IndexType>>,
      public ConvertibleTo<Ell<next_precision<ValueType>, IndexType>> {
    friend class EnableCreateMethod<Ell>;
    friend class EnablePolymorphicObject<Ell, BatchLinOp>;
    friend class Ell<to_complex<ValueType>, IndexType>;
    friend class Ell<next_precision<ValueType>, IndexType>;
    static_assert(std::is_same<IndexType, int32>::value,
                  "IndexType must be a 32 bit integer");

public:
    using EnableBatchLinOp<Ell>::convert_to;
    using EnableBatchLinOp<Ell>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using unbatch_type = gko::matrix::Ell<value_type, index_type>;
    using absolute_type = remove_complex<Ell>;
    using complex_type = to_complex<Ell>;

    void convert_to(
        Ell<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(Ell<next_precision<ValueType>, IndexType>* result) override;

    /**
     * Creates a mutable view (of matrix::Ell type) of one item of the
     * batch::matrix::Ell<value_type> object. Does not perform any deep
     * copies, but only returns a view of the data.
     *
     * @param item_id  The index of the batch item
     *
     * @return  a batch::matrix::Ell object with the data from the batch item
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
     * Returns the number of elements per row explicitly stored.
     *
     * @return the number of elements stored in each row of the ELL matrix. Same
     * for each batch item
     */
    index_type get_num_stored_elements_per_row() const noexcept
    {
        return num_elems_per_row_;
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
     * Returns a pointer to the array of col_idxs of the matrix. This is shared
     * across all batch items.
     *
     * @param batch_id  the id of the batch item.
     *
     * @return the pointer to the array of col_idxs
     */
    index_type* get_col_idxs_for_item(size_type batch_id) noexcept
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return col_idxs_.get_data();
    }

    /**
     * @copydoc get_col_idxs_for_item(size_type)
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_col_idxs_for_item(
        size_type batch_id) const noexcept
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return col_idxs_.get_const_data();
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
        return values_.get_const_data() +
               batch_id * this->get_num_elements_per_item();
    }

    /**
     * Creates a constant (immutable) batch ell matrix from a constant
     * array. The column indices array needs to be the same for all batch items.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param num_elems_per_row  the number of elements to be stored in each row
     * @param values  the value array of the matrix
     * @param col_idxs the col_idxs array of a single batch item of the matrix.
     *
     * @return A smart pointer to the constant matrix wrapping the input
     * array (if it resides on the same executor as the matrix) or a copy of the
     * array on the correct executor.
     */
    static std::unique_ptr<const Ell> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        const index_type num_elems_per_row,
        gko::detail::const_array_view<value_type>&& values,
        gko::detail::const_array_view<index_type>&& col_idxs);

    /**
     * Apply the matrix to a multi-vector. Represents the matrix vector
     * multiplication, x = A * b, where x and b are both multi-vectors.
     *
     * @param b  the multi-vector to be applied to
     * @param x  the output multi-vector
     */
    Ell* apply(ptr_param<const MultiVector<value_type>> b,
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
    Ell* apply(ptr_param<const MultiVector<value_type>> alpha,
               ptr_param<const MultiVector<value_type>> b,
               ptr_param<const MultiVector<value_type>> beta,
               ptr_param<MultiVector<value_type>> x);

    /**
     * @copydoc apply(const MultiVector<value_type>*, MultiVector<value_type>*)
     */
    const Ell* apply(ptr_param<const MultiVector<value_type>> b,
                     ptr_param<MultiVector<value_type>> x) const;

    /**
     * @copydoc apply(const MultiVector<value_type>*, const
     * MultiVector<value_type>*, const MultiVector<value_type>*,
     * MultiVector<value_type>*)
     */
    const Ell* apply(ptr_param<const MultiVector<value_type>> alpha,
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
     * @note Performs the operation in-place for this batch matrix.
     * @note This operation fails in case this matrix does not have all its
     *       diagonal entries.
     */
    void add_scaled_identity(ptr_param<const MultiVector<value_type>> alpha,
                             ptr_param<const MultiVector<value_type>> beta);

private:
    size_type compute_num_elems(const batch_dim<2>& size,
                                IndexType num_elems_per_row)
    {
        return size.get_num_batch_items() * size.get_common_size()[0] *
               num_elems_per_row;
    }

    /**
     * Creates an uninitialized Ell matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_elems_per_row  the number of elements to be stored in each row
     */
    Ell(std::shared_ptr<const Executor> exec,
        const batch_dim<2>& size = batch_dim<2>{},
        const IndexType num_elems_per_row = 0);

    /**
     * Creates a Ell matrix from an already allocated (and initialized)
     * array. The column indices array needs to be the same for all batch items.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_elems_per_row  the number of elements to be stored in each row
     * @param values  array of matrix values
     * @param col_idxs the col_idxs array of a single batch item of the matrix.
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename ValuesArray, typename IndicesArray>
    Ell(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
        const IndexType num_elems_per_row, ValuesArray&& values,
        IndicesArray&& col_idxs)
        : EnableBatchLinOp<Ell>(exec, size),
          num_elems_per_row_{num_elems_per_row},
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<IndicesArray>(col_idxs)}
    {
        // Ensure that the value and col_idxs arrays have the correct size
        auto num_elems = this->get_common_size()[0] * num_elems_per_row *
                         this->get_num_batch_items();
        GKO_ASSERT_EQ(num_elems, values_.get_size());
        GKO_ASSERT_EQ(this->get_num_elements_per_item(), col_idxs_.get_size());
    }

    void apply_impl(const MultiVector<value_type>* b,
                    MultiVector<value_type>* x) const;

    void apply_impl(const MultiVector<value_type>* alpha,
                    const MultiVector<value_type>* b,
                    const MultiVector<value_type>* beta,
                    MultiVector<value_type>* x) const;

    index_type num_elems_per_row_;
    array<value_type> values_;
    array<index_type> col_idxs_;
};


}  // namespace matrix
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_ELL_HPP_
