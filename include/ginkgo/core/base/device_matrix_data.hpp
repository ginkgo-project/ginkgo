// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_DEVICE_MATRIX_DATA_HPP_
#define GKO_PUBLIC_CORE_BASE_DEVICE_MATRIX_DATA_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


namespace gko {


/**
 * This type is a device-side equivalent to matrix_data.
 * It stores the data necessary to initialize any matrix format in Ginkgo in
 * individual value, column and row index arrays together with associated matrix
 * dimensions. matrix_data uses array-of-Structs storage (AoS), while
 * device_matrix_data uses Struct-of-Arrays (SoA).
 *
 * @note To be used with a Ginkgo matrix type, the entry array must be sorted in
 *       row-major order, i.e. by row index, then by column index within rows.
 *       This can be achieved by calling the sort_row_major function.
 * @note The data must not contain any duplicate (row, column) pairs.
 *
 * @tparam ValueType  the type used to store matrix values
 * @tparam IndexType  the type used to store matrix row and column indices
 */
template <typename ValueType, typename IndexType>
class device_matrix_data {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using nonzero_type = matrix_data_entry<value_type, index_type>;
    using host_type = matrix_data<value_type, index_type>;

    /**
     * Initializes a new device_matrix_data object.
     * It uses the given executor to allocate storage for the given number of
     * entries and matrix dimensions.
     *
     * @param exec  the executor to be used to store the matrix entries
     * @param size  the matrix dimensions
     * @param num_entries  the number of entries to be stored
     */
    explicit device_matrix_data(std::shared_ptr<const Executor> exec,
                                dim<2> size = {}, size_type num_entries = 0);

    /**
     * Initializes a device_matrix_data object by copying an existing object on
     * another executor.
     *
     * @param exec  the executor to be used to store the matrix entries
     * @param data  the device_matrix data object to copy, potentially stored on
     * another executor.
     */
    device_matrix_data(std::shared_ptr<const Executor> exec,
                       const device_matrix_data& data);

    /**
     * Initializes a new device_matrix_data object from existing data.
     *
     * @param size  the matrix dimensions
     * @param values  the array containing the matrix values
     * @param col_idxs  the array containing the matrix column indices
     * @param row_idxs  the array containing the matrix row indices
     */
    template <typename ValueArray, typename RowIndexArray,
              typename ColIndexArray>
    device_matrix_data(std::shared_ptr<const Executor> exec, dim<2> size,
                       RowIndexArray&& row_idxs, ColIndexArray&& col_idxs,
                       ValueArray&& values)
        : size_{size},
          row_idxs_{exec, std::forward<RowIndexArray>(row_idxs)},
          col_idxs_{exec, std::forward<ColIndexArray>(col_idxs)},
          values_{exec, std::forward<ValueArray>(values)}
    {
        GKO_ASSERT_EQ(values_.get_size(), row_idxs_.get_size());
        GKO_ASSERT_EQ(values_.get_size(), col_idxs_.get_size());
    }

    /**
     * Copies the device_matrix_data entries to the host to return a regular
     * matrix_data object with the same dimensions and entries.
     *
     * @return a matrix_data object with the same dimensions and entries.
     */
    host_type copy_to_host() const;

    /**
     * Creates a device_matrix_data object from the given host data on the given
     * executor.
     *
     * @param exec  the executor to create the device_matrix_data on.
     * @param data  the data to be wrapped or copied into a device_matrix_data.
     * @return  a device_matrix_data object with the same size and entries as
     *          `data` copied to the device executor.
     */
    static device_matrix_data create_from_host(
        std::shared_ptr<const Executor> exec, const host_type& data);

    /**
     * Sorts the matrix entries in row-major order
     * This means that they will be sorted by row index first, and then by
     * column index inside each row.
     */
    void sort_row_major();

    /**
     * Removes all zero entries from the storage.
     * This does not modify the storage if there are no zero entries, and keeps
     * the relative order of nonzero entries otherwise.
     */
    void remove_zeros();

    /**
     * Sums up all duplicate entries pointing to the same non-zero location.
     * The output will be sorted in row-major order, and it will only reallocate
     * if duplicates exist.
     */
    void sum_duplicates();

    /**
     * Returns the executor used to store the device_matrix_data entries.
     *
     * @return the executor used to store the device_matrix_data entries.
     */
    std::shared_ptr<const Executor> get_executor() const
    {
        return values_.get_executor();
    }

    /**
     * Returns the dimensions of the matrix.
     *
     * @return the dimensions of the matrix.
     */
    dim<2> get_size() const { return size_; }

    /**
     * Returns the number of stored elements of the matrix.
     *
     * @return the number of stored elements of the matrix.
     */
    GKO_DEPRECATED("use get_num_stored_elements()")
    size_type get_num_elems() const { return get_num_stored_elements(); }

    /**
     * Returns the number of stored elements of the matrix.
     *
     * @return the number of stored elements of the matrix.
     */
    size_type get_num_stored_elements() const { return values_.get_size(); }

    /**
     * Returns a pointer to the row index array
     *
     * @return a pointer to the row index array
     */
    index_type* get_row_idxs() { return row_idxs_.get_data(); }

    /**
     * Returns a pointer to the constant row index array
     *
     * @return a pointer to the constant row index array
     */
    const index_type* get_const_row_idxs() const
    {
        return row_idxs_.get_const_data();
    }

    /**
     * Returns a pointer to the column index array
     *
     * @return a pointer to the column index array
     */
    index_type* get_col_idxs() { return col_idxs_.get_data(); }

    /**
     * Returns a pointer to the constant column index array
     *
     * @return a pointer to the constant column index array
     */
    const index_type* get_const_col_idxs() const
    {
        return col_idxs_.get_const_data();
    }

    /**
     * Returns a pointer to the value array
     *
     * @return a pointer to the value array
     */
    value_type* get_values() { return values_.get_data(); }

    /**
     * Returns a pointer to the constant value array
     *
     * @return a pointer to the constant value array
     */
    const value_type* get_const_values() const
    {
        return values_.get_const_data();
    }

    /**
     * Resizes the internal storage to the given number of stored matrix
     * entries. The resulting storage should be assumed uninitialized.
     *
     * @param new_num_entries  the new number of stored matrix entries.
     */
    void resize_and_reset(size_type new_num_entries);

    /**
     * Resizes the matrix and internal storage to the given dimensions.
     * The resulting storage should be assumed uninitialized.
     *
     * @param new_size  the new matrix dimensions.
     * @param new_num_entries  the new number of stored matrix entries.
     */
    void resize_and_reset(dim<2> new_size, size_type new_num_entries);

    /**
     * Stores the internal arrays of a device_matrix_data object.
     */
    struct arrays {
        array<index_type> row_idxs;
        array<index_type> col_idxs;
        array<value_type> values;
    };

    /**
     * Moves out the internal arrays of the device_matrix_data object and resets
     * it to an empty 0x0 matrix.
     *
     * @return a struct containing the internal arrays.
     */
    arrays empty_out();

private:
    dim<2> size_;
    array<index_type> row_idxs_;
    array<index_type> col_idxs_;
    array<value_type> values_;
};


namespace detail {


template <typename ValueType, typename IndexType>
struct temporary_clone_helper<device_matrix_data<ValueType, IndexType>> {
    static std::unique_ptr<device_matrix_data<ValueType, IndexType>> create(
        std::shared_ptr<const Executor> exec,
        device_matrix_data<ValueType, IndexType>* ptr, bool copy_data)
    {
        if (copy_data) {
            return std::make_unique<device_matrix_data<ValueType, IndexType>>(
                std::move(exec), *ptr);
        } else {
            return std::make_unique<device_matrix_data<ValueType, IndexType>>(
                std::move(exec), ptr->get_size(),
                ptr->get_num_stored_elements());
        }
    }
};

template <typename ValueType, typename IndexType>
struct temporary_clone_helper<const device_matrix_data<ValueType, IndexType>> {
    static std::unique_ptr<const device_matrix_data<ValueType, IndexType>>
    create(std::shared_ptr<const Executor> exec,
           const device_matrix_data<ValueType, IndexType>* ptr, bool)
    {
        return std::make_unique<const device_matrix_data<ValueType, IndexType>>(
            std::move(exec), *ptr);
    }
};


// specialization for non-constant device_matrix_data, copying back via
// assignment
template <typename ValueType, typename IndexType>
class copy_back_deleter<device_matrix_data<ValueType, IndexType>> {
public:
    using pointer = device_matrix_data<ValueType, IndexType>*;

    /**
     * Creates a new deleter object.
     *
     * @param original  the origin object where the data will be copied before
     *                  deletion
     */
    copy_back_deleter(pointer original) : original_{original} {}

    /**
     * Copies back the pointed-to object to the original and deletes it.
     *
     * @param ptr  pointer to the object to be copied back and deleted
     */
    void operator()(pointer ptr) const
    {
        *original_ = *ptr;
        delete ptr;
    }

private:
    pointer original_;
};


}  // namespace detail
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_DEVICE_MATRIX_DATA_HPP_
