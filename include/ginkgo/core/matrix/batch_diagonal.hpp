// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_DIAGONAL_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_DIAGONAL_HPP_


// Copyright (c) 2017-2023, the Ginkgo authors
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


namespace gko {
namespace batch {
namespace matrix {


/**
 * The batch Diagonal matrix, which represents a batch of Diagonal matrices.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup batch_diagonal
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class Diagonal final
    : public EnableBatchLinOp<Diagonal<ValueType>>,
      public EnableCreateMethod<Diagonal<ValueType>>,
      public ConvertibleTo<Diagonal<next_precision<ValueType>>> {
    friend class EnableCreateMethod<Diagonal>;
    friend class EnablePolymorphicObject<Diagonal, BatchLinOp>;
    friend class Diagonal<next_precision<ValueType>>;

public:
    using EnableBatchLinOp<Diagonal>::convert_to;
    using EnableBatchLinOp<Diagonal>::move_to;

    using value_type = ValueType;
    using index_type = int32;
    using unbatch_type = gko::matrix::Diagonal<ValueType>;
    using absolute_type = remove_complex<Diagonal>;
    using complex_type = to_complex<Diagonal>;

    void convert_to(Diagonal<next_precision<ValueType>>* result) const override;

    void move_to(Diagonal<next_precision<ValueType>>* result) override;

    /**
     * Creates a mutable view (of matrix::Diagonal type) of one item of the
     * batch::matrix::Diagonal<value_type> object. Does not perform any deep
     * copies, but only returns a view of the data.
     *
     * @param item_id  The index of the batch item
     *
     * @return  a batch::matrix::Diagonal object with the data from the batch
     * item at the given index.
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
     * Returns the number of elements explicitly stored in the batch matrix,
     * cumulative across all the batch items.
     *
     * @return the number of elements explicitly stored in the vector,
     *         cumulative across all the batch items
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_num_elems();
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
        GKO_ASSERT(values_.get_num_elems() >=
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
        GKO_ASSERT(values_.get_num_elems() >=
                   batch_id * this->get_num_elements_per_item());
        return values_.get_const_data() +
               batch_id * this->get_num_elements_per_item();
    }

    /**
     * Creates a constant (immutable) batch diagonal matrix from a constant
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
    static std::unique_ptr<const Diagonal> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        gko::detail::const_array_view<value_type>&& values);

    /**
     * Apply the matrix to a multi-vector. Represents the matrix vector
     * multiplication, x = I * b, where x and b are both multi-vectors.
     *
     * @param b  the multi-vector to be applied to
     * @param x  the output multi-vector
     */
    Diagonal* apply(ptr_param<const MultiVector<value_type>> b,
                    ptr_param<MultiVector<value_type>> x);

    /**
     * Apply the matrix to a multi-vector with a linear combination of the given
     * input vector. Represents the matrix vector multiplication, x = alpha * I
     * * b + beta * x, where x and b are both multi-vectors.
     *
     * @param alpha  the scalar to scale the matrix-vector product with
     * @param b      the multi-vector to be applied to
     * @param beta   the scalar to scale the x vector with
     * @param x      the output multi-vector
     */
    Diagonal* apply(ptr_param<const MultiVector<value_type>> alpha,
                    ptr_param<const MultiVector<value_type>> b,
                    ptr_param<const MultiVector<value_type>> beta,
                    ptr_param<MultiVector<value_type>> x);

    /**
     * @copydoc apply(const MultiVector<value_type>*, MultiVector<value_type>*)
     */
    const Diagonal* apply(ptr_param<const MultiVector<value_type>> b,
                          ptr_param<MultiVector<value_type>> x) const;

    /**
     * @copydoc apply(const MultiVector<value_type>*, const
     * MultiVector<value_type>*, const MultiVector<value_type>*,
     * MultiVector<value_type>*)
     */
    const Diagonal* apply(ptr_param<const MultiVector<value_type>> alpha,
                          ptr_param<const MultiVector<value_type>> b,
                          ptr_param<const MultiVector<value_type>> beta,
                          ptr_param<MultiVector<value_type>> x) const;

private:
    /**
     * Creates an Diagonal matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the batch matrices in a batch_dim object
     */
    Diagonal(std::shared_ptr<const Executor> exec,
             const batch_dim<2>& size = batch_dim<2>{});

    /**
     * Creates a Diagonal matrix from an already allocated (and initialized)
     * array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename ValuesArray>
    Diagonal(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
             ValuesArray&& values)
        : EnableBatchLinOp<Diagonal>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)}
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_DIMENSIONS(this->get_size());
        GKO_ASSERT(values_.get_num_elems() ==
                   size.get_num_batch_items() * size.get_common_size()[0]);
    }

    void apply_impl(const MultiVector<value_type>* b,
                    MultiVector<value_type>* x) const;

    void apply_impl(const MultiVector<value_type>* alpha,
                    const MultiVector<value_type>* b,
                    const MultiVector<value_type>* beta,
                    MultiVector<value_type>* x) const;

    array<value_type> values_;
};


}  // namespace matrix
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_DIAGONAL_HPP_
