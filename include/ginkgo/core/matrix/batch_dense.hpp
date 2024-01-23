// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_


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
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace batch {
namespace matrix {


/**
 * Dense is a batch matrix format which explicitly stores all values of the
 * matrix in each of the batches.
 *
 * The values in each of the batches are stored in row-major format (values
 * belonging to the same row appear consecutive in the memory and the values of
 * each batch item are also stored consecutively in memory).
 *
 * @note Though the storage layout is the same as the multi-vector object, the
 * class semantics and the operations it aims to provide are different. Hence it
 * is recommended to create multi-vector objects if the user means to view the
 * data as a set of vectors.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup batch_dense
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class Dense final : public EnableBatchLinOp<Dense<ValueType>>,
                    public EnableCreateMethod<Dense<ValueType>>,
                    public ConvertibleTo<Dense<next_precision<ValueType>>> {
    friend class EnableCreateMethod<Dense>;
    friend class EnablePolymorphicObject<Dense, BatchLinOp>;
    friend class Dense<to_complex<ValueType>>;
    friend class Dense<next_precision<ValueType>>;

public:
    using EnableBatchLinOp<Dense>::convert_to;
    using EnableBatchLinOp<Dense>::move_to;

    using value_type = ValueType;
    using index_type = int32;
    using transposed_type = Dense<ValueType>;
    using unbatch_type = gko::matrix::Dense<ValueType>;
    using absolute_type = remove_complex<Dense>;
    using complex_type = to_complex<Dense>;

    void convert_to(Dense<next_precision<ValueType>>* result) const override;

    void move_to(Dense<next_precision<ValueType>>* result) override;

    /**
     * Creates a mutable view (of gko::matrix::Dense type) of one item of the
     * batch::matrix::Dense<value_type> object. Does not perform any deep
     * copies, but only returns a view of the data.
     *
     * @param item_id  The index of the batch item
     *
     * @return  a gko::matrix::Dense object with the data from the batch item
     * at the given index.
     */
    std::unique_ptr<unbatch_type> create_view_for_item(size_type item_id);

    /**
     * @copydoc create_view_for_item(size_type)
     */
    std::unique_ptr<const unbatch_type> create_const_view_for_item(
        size_type item_id) const;

    /**
     * Get the cumulative storage size offset
     *
     * @param batch_id the batch id
     *
     * @return the cumulative offset
     */
    size_type get_cumulative_offset(size_type batch_id) const
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return batch_id * this->get_common_size()[0] *
               this->get_common_size()[1];
    }

    /**
     * Returns a pointer to the array of values of the multi-vector
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
     * Returns a single element for a particular batch item.
     *
     * @param batch_id  the batch item index to be queried
     * @param row  the row of the requested element
     * @param col  the column of the requested element
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU Dense object
     *        from the OMP may result in incorrect behaviour)
     */
    value_type& at(size_type batch_id, size_type row, size_type col)
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_data()[linearize_index(batch_id, row, col)];
    }

    /**
     * @copydoc Dense::at(size_type, size_type, size_type)
     */
    value_type at(size_type batch_id, size_type row, size_type col) const
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_const_data()[linearize_index(batch_id, row, col)];
    }

    /**
     * Returns a single element for a particular batch item.
     *
     * Useful for iterating across all elements of the matrix.
     * However, it is less efficient than the two-parameter variant of this
     * method.
     *
     * @param batch_id  the batch item index to be queried
     * @param idx  a linear index of the requested element
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU Dense object
     *        from the OMP may result in incorrect behaviour)
     */
    ValueType& at(size_type batch_id, size_type idx) noexcept
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_data()[linearize_index(batch_id, idx)];
    }

    /**
     * @copydoc Dense::at(size_type, size_type, size_type)
     */
    ValueType at(size_type batch_id, size_type idx) const noexcept
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_const_data()[linearize_index(batch_id, idx)];
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
        return values_.get_data() + this->get_cumulative_offset(batch_id);
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
        return values_.get_const_data() + this->get_cumulative_offset(batch_id);
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
     * Creates a constant (immutable) batch dense matrix from a constant
     * array.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     *
     * @return A smart pointer to the constant matrix wrapping the input
     * array (if it resides on the same executor as the matrix) or a copy of the
     * array on the correct executor.
     */
    static std::unique_ptr<const Dense<value_type>> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        gko::detail::const_array_view<ValueType>&& values);

    /**
     * Apply the matrix to a multi-vector. Represents the matrix vector
     * multiplication, x = A * b, where x and b are both multi-vectors.
     *
     * @param b  the multi-vector to be applied to
     * @param x  the output multi-vector
     */
    Dense* apply(ptr_param<const MultiVector<value_type>> b,
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
    Dense* apply(ptr_param<const MultiVector<value_type>> alpha,
                 ptr_param<const MultiVector<value_type>> b,
                 ptr_param<const MultiVector<value_type>> beta,
                 ptr_param<MultiVector<value_type>> x);

    /**
     * @copydoc apply(const MultiVector<value_type>*, MultiVector<value_type>*)
     */
    const Dense* apply(ptr_param<const MultiVector<value_type>> b,
                       ptr_param<MultiVector<value_type>> x) const;

    /**
     * @copydoc apply(const MultiVector<value_type>*, const
     * MultiVector<value_type>*, const MultiVector<value_type>*,
     * MultiVector<value_type>*)
     */
    const Dense* apply(ptr_param<const MultiVector<value_type>> alpha,
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
     * Performs the operation this = alpha*this + b.
     *
     * @param alpha the scalar to multiply this matrix
     * @param b  the matrix to add
     *
     * @note Performs the operation in-place for this batch matrix
     */
    void scale_add(ptr_param<const MultiVector<value_type>> alpha,
                   ptr_param<const batch::matrix::Dense<value_type>> b);

    /**
     * Performs the operation this = alpha*I + beta*this.
     *
     * @param alpha the scalar for identity
     * @param beta  the scalar to multiply this matrix
     *
     * @note Performs the operation in-place for this batch matrix
     */
    void add_scaled_identity(ptr_param<const MultiVector<value_type>> alpha,
                             ptr_param<const MultiVector<value_type>> beta);

private:
    inline size_type compute_num_elems(const batch_dim<2>& size)
    {
        return size.get_num_batch_items() * size.get_common_size()[0] *
               size.get_common_size()[1];
    }

    /**
     * Creates an uninitialized Dense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Dense(std::shared_ptr<const Executor> exec,
          const batch_dim<2>& size = batch_dim<2>{});

    /**
     * Creates a Dense matrix from an already allocated (and initialized)
     * array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  sizes of the batch matrices in a batch_dim object
     * @param values  array of matrix values
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename ValuesArray>
    Dense(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
          ValuesArray&& values)
        : EnableBatchLinOp<Dense>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)}
    {
        // Ensure that the values array has the correct size
        auto num_elems = compute_num_elems(size);
        GKO_ENSURE_IN_BOUNDS(num_elems, values_.get_size() + 1);
    }

    void apply_impl(const MultiVector<value_type>* b,
                    MultiVector<value_type>* x) const;

    void apply_impl(const MultiVector<value_type>* alpha,
                    const MultiVector<value_type>* b,
                    const MultiVector<value_type>* beta,
                    MultiVector<value_type>* x) const;

    size_type linearize_index(size_type batch, size_type row,
                              size_type col) const noexcept
    {
        return this->get_cumulative_offset(batch) +
               row * this->get_size().get_common_size()[1] + col;
    }

    size_type linearize_index(size_type batch, size_type idx) const noexcept
    {
        return linearize_index(batch, idx / this->get_common_size()[1],
                               idx % this->get_common_size()[1]);
    }

    array<value_type> values_;
};


}  // namespace matrix
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_
