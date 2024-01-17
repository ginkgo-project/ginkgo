// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_BATCH_MULTI_VECTOR_HPP_
#define GKO_PUBLIC_CORE_BASE_BATCH_MULTI_VECTOR_HPP_


#include <initializer_list>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_dim.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace batch {


/**
 * MultiVector stores multiple vectors in a batched fashion and is useful
 * for batched operations. For example, if you want to store two batch items
 * with multi-vectors of size (3 x 2) given below:
 *
 * [1 2 ; 3 4
 *  1 2 ; 3 4
 *  1 2 ; 3 4]
 *
 * In memory, they would be stored as a single array:
 * [1 2 1 2 1 2 3 4 3 4 3 4].
 *
 * Access functions @at can help access individual
 * item if necessary.
 *
 * The values of the different batch items are stored consecutively and in each
 * batch item, the multi-vectors are stored in a row-major fashion.
 *
 * @tparam ValueType  precision of multi-vector elements
 *
 * @ingroup batch_multi_vector
 * @ingroup batched
 */
template <typename ValueType = default_precision>
class MultiVector
    : public EnablePolymorphicObject<MultiVector<ValueType>>,
      public EnablePolymorphicAssignment<MultiVector<ValueType>>,
      public EnableCreateMethod<MultiVector<ValueType>>,
      public ConvertibleTo<MultiVector<next_precision<ValueType>>> {
    friend class EnableCreateMethod<MultiVector>;
    friend class EnablePolymorphicObject<MultiVector>;
    friend class MultiVector<to_complex<ValueType>>;
    friend class MultiVector<next_precision<ValueType>>;

public:
    using EnablePolymorphicAssignment<MultiVector>::convert_to;
    using EnablePolymorphicAssignment<MultiVector>::move_to;
    using ConvertibleTo<MultiVector<next_precision<ValueType>>>::convert_to;
    using ConvertibleTo<MultiVector<next_precision<ValueType>>>::move_to;

    using value_type = ValueType;
    using index_type = int32;
    using unbatch_type = gko::matrix::Dense<ValueType>;
    using absolute_type = remove_complex<MultiVector<ValueType>>;
    using complex_type = to_complex<MultiVector<ValueType>>;

    /**
     * Creates a MultiVector with the configuration of another
     * MultiVector.
     *
     * @param other  The other multi-vector whose configuration needs to copied.
     */
    static std::unique_ptr<MultiVector> create_with_config_of(
        ptr_param<const MultiVector> other);

    void convert_to(
        MultiVector<next_precision<ValueType>>* result) const override;

    void move_to(MultiVector<next_precision<ValueType>>* result) override;

    /**
     * Creates a mutable view (of matrix::Dense type) of one item of the Batch
     * MultiVector object. Does not perform any deep copies, but only returns a
     * view of the data.
     *
     * @param item_id  The index of the batch item
     *
     * @return  a matrix::Dense object with the data from the batch item at the
     *          given index.
     */
    std::unique_ptr<unbatch_type> create_view_for_item(size_type item_id);

    /**
     * @copydoc create_view_for_item(size_type)
     */
    std::unique_ptr<const unbatch_type> create_const_view_for_item(
        size_type item_id) const;

    /**
     * Returns the batch size.
     *
     * @return the batch size
     */
    batch_dim<2> get_size() const { return batch_size_; }

    /**
     * Returns the number of batch items.
     *
     * @return the number of batch items
     */
    size_type get_num_batch_items() const
    {
        return batch_size_.get_num_batch_items();
    }

    /**
     * Returns the common size of the batch items.
     *
     * @return the common size stored
     */
    dim<2> get_common_size() const { return batch_size_.get_common_size(); }

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
     * Returns a pointer to the array of values of the multi-vector for a
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
     * Get the cumulative storage size offset
     *
     * @param batch_id the batch id
     *
     * @return the cumulative offset
     */
    size_type get_cumulative_offset(size_type batch_id) const
    {
        return batch_id * this->get_common_size()[0] *
               this->get_common_size()[1];
    }

    /**
     * Returns a single element for a particular batch item.
     *
     * @param batch_id  the batch item index to be queried
     * @param row  the row of the requested element
     * @param col  the column of the requested element
     *
     * @note  the method has to be called on the same Executor the vector is
     *        stored at (e.g. trying to call this method on a GPU multi-vector
     *        from the OMP results in a runtime error)
     */
    value_type& at(size_type batch_id, size_type row, size_type col)
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_data()[linearize_index(batch_id, row, col)];
    }

    /**
     * @copydoc MultiVector::at(size_type, size_type, size_type)
     */
    value_type at(size_type batch_id, size_type row, size_type col) const
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_const_data()[linearize_index(batch_id, row, col)];
    }

    /**
     * Returns a single element for a particular batch item.
     *
     * Useful for iterating across all elements of the vector.
     * However, it is less efficient than the two-parameter variant of this
     * method.
     *
     * @param batch_id  the batch item index to be queried
     * @param idx  a linear index of the requested element
     *
     * @note  the method has to be called on the same Executor the vector is
     *        stored at (e.g. trying to call this method on a GPU multi-vector
     *        from the OMP results in a runtime error)
     */
    ValueType& at(size_type batch_id, size_type idx) noexcept
    {
        return values_.get_data()[linearize_index(batch_id, idx)];
    }

    /**
     * @copydoc MultiVector::at(size_type, size_type, size_type)
     */
    ValueType at(size_type batch_id, size_type idx) const noexcept
    {
        return values_.get_const_data()[linearize_index(batch_id, idx)];
    }

    /**
     * Scales the vector with a scalar (aka: BLAS scal).
     *
     * @param alpha  the scalar
     *
     * @note If alpha is 1x1 MultiVector matrix, the entire multi-vector
     *      (all batches) is scaled by alpha.
     *      If it is a MultiVector row vector of values, then i-th column of the
     *      vector is scaled with the i-th element of alpha (the number of
     *      columns of alpha has to match the number of columns of the
     *      multi-vector).
     *      If it is a MultiVector of the same size as this, then an element
     *      wise scaling is performed.
     */
    void scale(ptr_param<const MultiVector<ValueType>> alpha);

    /**
     * Adds `b` scaled by `alpha` to the vector (aka: BLAS axpy).
     *
     * @param alpha  the scalar
     * @param b  a multi-vector of the same dimension as this
     *
     * @note If alpha is 1x1 MultiVector matrix, the entire multi-vector
     *      (all batches) is scaled by alpha. If it is a MultiVector row
     *      vector of values, then i-th column of the vector is scaled with the
     *      i-th element of alpha (the number of columns of alpha has to match
     *      the number of columns of the multi-vector).
     */
    void add_scaled(ptr_param<const MultiVector<ValueType>> alpha,
                    ptr_param<const MultiVector<ValueType>> b);

    /**
     * Computes the column-wise dot product of each multi-vector in this batch
     * and its corresponding entry in `b`.
     *
     * @param b  a MultiVector of same dimension as this
     * @param result  a MultiVector row vector, used to store the dot
     * product
     */
    void compute_dot(ptr_param<const MultiVector<ValueType>> b,
                     ptr_param<MultiVector<ValueType>> result) const;

    /**
     * Computes the column-wise conjugate dot product of each multi-vector in
     * this batch and its corresponding entry in `b`. If the vector has complex
     * value_type, then the conjugate of this is taken.
     *
     * @param b  a MultiVector of same dimension as this
     * @param result  a MultiVector row vector, used to store the dot
     *                product (the number of column in the vector must match the
     *                number of columns of this)
     */
    void compute_conj_dot(ptr_param<const MultiVector<ValueType>> b,
                          ptr_param<MultiVector<ValueType>> result) const;

    /**
     * Computes the Euclidean (L^2) norm of each multi-vector in this batch.
     *
     * @param result  a MultiVector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     */
    void compute_norm2(
        ptr_param<MultiVector<remove_complex<ValueType>>> result) const;

    /**
     * Creates a constant (immutable) batch multi-vector from a constant
     * array.
     *
     * @param exec  the executor to create the vector on
     * @param size  the dimensions of the vector
     * @param values  the value array of the vector
     * @param stride  the row-stride of the vector
     *
     * @return A smart pointer to the constant multi-vector wrapping the input
     * array (if it resides on the same executor as the vector) or a copy of the
     * array on the correct executor.
     */
    static std::unique_ptr<const MultiVector<ValueType>> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        gko::detail::const_array_view<ValueType>&& values);

    /**
     * Fills the input MultiVector with a given value
     *
     * @param value  the value to be filled
     */
    void fill(ValueType value);

private:
    inline size_type compute_num_elems(const batch_dim<2>& size)
    {
        return size.get_num_batch_items() * size.get_common_size()[0] *
               size.get_common_size()[1];
    }

protected:
    /**
     * Sets the size of the MultiVector.
     *
     * @param value  the new size of the operator
     */
    void set_size(const batch_dim<2>& value) noexcept;

    /**
     * Creates an uninitialized multi-vector of the specified
     * size.
     *
     * @param exec  Executor associated to the vector
     * @param size  size of the batch multi vector
     */
    MultiVector(std::shared_ptr<const Executor> exec,
                const batch_dim<2>& size = batch_dim<2>{});

    /**
     * Creates a MultiVector from an already allocated (and
     * initialized) array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the vector
     * @param size  sizes of the batch matrices in a batch_dim object
     * @param values  array of values
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the vector.
     */
    template <typename ValuesArray>
    MultiVector(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
                ValuesArray&& values)
        : EnablePolymorphicObject<MultiVector<ValueType>>(exec),
          batch_size_(size),
          values_{exec, std::forward<ValuesArray>(values)}
    {
        // Ensure that the values array has the correct size
        auto num_elems = compute_num_elems(size);
        GKO_ENSURE_IN_BOUNDS(num_elems, values_.get_size() + 1);
    }

    /**
     * Creates a MultiVector with the same configuration as the
     * callers object.
     *
     * @returns a MultiVector with the same configuration as the
     * caller.
     */
    std::unique_ptr<MultiVector> create_with_same_config() const;

    size_type linearize_index(size_type batch, size_type row,
                              size_type col) const noexcept
    {
        return this->get_cumulative_offset(batch) +
               row * batch_size_.get_common_size()[1] + col;
    }

    size_type linearize_index(size_type batch, size_type idx) const noexcept
    {
        return linearize_index(batch, idx / this->get_common_size()[1],
                               idx % this->get_common_size()[1]);
    }

private:
    batch_dim<2> batch_size_;
    array<value_type> values_;
};


}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_BATCH_MULTI_VECTOR_HPP_
