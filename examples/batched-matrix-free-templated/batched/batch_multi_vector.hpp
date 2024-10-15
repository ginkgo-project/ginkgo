// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_EXAMPLES_BATCHED - MATRIX - FREE - \
    TEMPLATED_BATCHED_BATCH_MULTI_VECTOR_HPP_
#define GKO_EXAMPLES_BATCHED \
    -MATRIX - FREE - TEMPLATED_BATCHED_BATCH_MULTI_VECTOR_HPP_


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
namespace kernels {


template <typename ValueType>
struct multi_vector_view_item {
    ValueType* __restrict__ values;
    int32 stride;
    int32 num_rows;
    int32 num_rhs;

    constexpr operator multi_vector_view_item<std::add_const_t<ValueType>>()
        const
    {
        return {values, stride, num_rows, num_rhs};
    }
};


template <typename ValueType>
struct multi_vector_view {
    ValueType* __restrict__ values;
    int32 num_batches;
    int32 stride;
    int32 num_rows;
    int32 num_rhs;

    constexpr multi_vector_view_item<ValueType> extract_batch_item(
        int32 batch_id) const
    {
        auto batch_offset = batch_id * num_rows * stride;
        return {values + batch_offset, stride, num_rows, num_rhs};
    }
};


}  // namespace kernels
namespace batch_template {


template <typename ValueType = default_precision>
class MultiVector : public EnablePolymorphicObject<MultiVector<ValueType>>,
                    public EnablePolymorphicAssignment<MultiVector<ValueType>> {
    friend class EnablePolymorphicObject<MultiVector>;

public:
    using value_type = ValueType;
    using index_type = int32;
    using unbatch_type = gko::matrix::Dense<ValueType>;

    constexpr kernels::multi_vector_view<value_type> create_view()
    {
        return {get_values(),
                static_cast<int32>(batch_size_.get_num_batch_items()),
                static_cast<int32>(batch_size_.get_common_size()[1]),
                static_cast<int32>(batch_size_.get_common_size()[0]),
                static_cast<int32>(batch_size_.get_common_size()[1])};
    }

    constexpr kernels::multi_vector_view<const value_type> create_view() const
    {
        return {get_const_values(),
                static_cast<int32>(batch_size_.get_num_batch_items()),
                static_cast<int32>(batch_size_.get_common_size()[1]),
                static_cast<int32>(batch_size_.get_common_size()[0]),
                static_cast<int32>(batch_size_.get_common_size()[1])};
    }

    /**
     * Returns the batch size.
     *
     * @return the batch size
     */
    constexpr batch_dim<2> get_size() const { return batch_size_; }

    /**
     * Returns the number of batch items.
     *
     * @return the number of batch items
     */
    constexpr size_type get_num_batch_items() const
    {
        return batch_size_.get_num_batch_items();
    }

    /**
     * Returns the common size of the batch items.
     *
     * @return the common size stored
     */
    constexpr dim<2> get_common_size() const
    {
        return batch_size_.get_common_size();
    }

    /**
     * Returns a pointer to the array of values of the multi-vector
     *
     * @return the pointer to the array of values
     */
    constexpr value_type* get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    constexpr const value_type* get_const_values() const noexcept
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
    constexpr value_type* get_values_for_item(size_type batch_id) noexcept
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
    constexpr const value_type* get_const_values_for_item(
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
    constexpr size_type get_num_stored_elements() const noexcept
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
    constexpr size_type get_cumulative_offset(size_type batch_id) const
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
    constexpr value_type& at(size_type batch_id, size_type row, size_type col)
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_data()[linearize_index(batch_id, row, col)];
    }

    /**
     * @copydoc MultiVector::at(size_type, size_type, size_type)
     */
    constexpr value_type at(size_type batch_id, size_type row,
                            size_type col) const
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
    constexpr ValueType& at(size_type batch_id, size_type idx) noexcept
    {
        return values_.get_data()[linearize_index(batch_id, idx)];
    }

    /**
     * @copydoc MultiVector::at(size_type, size_type, size_type)
     */
    constexpr ValueType at(size_type batch_id, size_type idx) const noexcept
    {
        return values_.get_const_data()[linearize_index(batch_id, idx)];
    }

    /**
     * Creates an uninitialized multi-vector of the specified
     * size.
     *
     * @param exec  Executor associated to the vector
     * @param size  size of the batch multi vector
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<MultiVector> create(
        std::shared_ptr<const Executor> exec,
        const batch_dim<2>& size = batch_dim<2>{})
    {
        return std::unique_ptr<MultiVector>{
            new MultiVector(std::move(exec), size)};
    }

    /**
     * Creates a MultiVector from an already allocated (and
     * initialized) array.
     *
     * @param exec  Executor associated to the vector
     * @param size  sizes of the batch matrices in a batch_dim object
     * @param values  array of values
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the vector.
     */
    static std::unique_ptr<MultiVector> create(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
        array<value_type> values);

    /**
     * @copydoc std::unique_ptr<MultiVector> create(std::shared_ptr<const
     * Executor>, const batch_dim<2>&, array<value_type>)
     */
    template <typename InputValueType>
    GKO_DEPRECATED(
        "explicitly construct the gko::array argument instead of passing an "
        "initializer list")
    static std::unique_ptr<MultiVector> create(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
        std::initializer_list<InputValueType> values)
    {
        return create(exec, size, array<index_type>{exec, std::move(values)});
    }

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
    static std::unique_ptr<const MultiVector> create_const(
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

    MultiVector(std::shared_ptr<const Executor> exec,
                const batch_dim<2>& size = batch_dim<2>{})
        : EnablePolymorphicObject<MultiVector>(exec),
          batch_size_(size),
          values_(exec, compute_num_elems(batch_size_))
    {}

    MultiVector(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
                array<value_type> values);

    /**
     * Creates a MultiVector with the same configuration as the
     * callers object.
     *
     * @returns a MultiVector with the same configuration as the
     * caller.
     */
    std::unique_ptr<MultiVector> create_with_same_config() const;

    constexpr size_type linearize_index(size_type batch, size_type row,
                                        size_type col) const noexcept
    {
        return this->get_cumulative_offset(batch) +
               row * batch_size_.get_common_size()[1] + col;
    }

    constexpr size_type linearize_index(size_type batch,
                                        size_type idx) const noexcept
    {
        return linearize_index(batch, idx / this->get_common_size()[1],
                               idx % this->get_common_size()[1]);
    }

private:
    batch_dim<2> batch_size_;
    array<value_type> values_;
};


}  // namespace batch_template
}  // namespace gko


#endif  // GKO_EXAMPLES_BATCHED-MATRIX-FREE-TEMPLATED_BATCHED_BATCH_MULTI_VECTOR_HPP_
