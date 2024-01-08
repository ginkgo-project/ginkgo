// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_ELL_HPP_
#define GKO_PUBLIC_CORE_MATRIX_ELL_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;

template <typename ValueType, typename IndexType>
class Coo;

template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType, typename IndexType>
class Hybrid;


/**
 * ELL is a matrix format where stride with explicit zeros is used such that
 * all rows have the same number of stored elements. The number of elements
 * stored in each row is the largest number of nonzero elements in any of the
 * rows (obtainable through get_num_stored_elements_per_row() method). This
 * removes the need of a row pointer like in the CSR format, and allows for SIMD
 * processing of the distinct rows. For efficient processing, the nonzero
 * elements and the corresponding column indices are stored in column-major
 * fashion. The columns are padded to the length by user-defined stride
 * parameter whose default value is the number of rows of the matrix.
 *
 * This implementation uses the column index value invalid_index<IndexType>()
 * to mark padding entries that are not part of the sparsity pattern.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup ell
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Ell : public EnableLinOp<Ell<ValueType, IndexType>>,
            public EnableCreateMethod<Ell<ValueType, IndexType>>,
            public ConvertibleTo<Ell<next_precision<ValueType>, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public ConvertibleTo<Csr<ValueType, IndexType>>,
            public DiagonalExtractable<ValueType>,
            public ReadableFromMatrixData<ValueType, IndexType>,
            public WritableToMatrixData<ValueType, IndexType>,
            public EnableAbsoluteComputation<
                remove_complex<Ell<ValueType, IndexType>>> {
    friend class EnableCreateMethod<Ell>;
    friend class EnablePolymorphicObject<Ell, LinOp>;
    friend class Dense<ValueType>;
    friend class Coo<ValueType, IndexType>;
    friend class Csr<ValueType, IndexType>;
    friend class Ell<to_complex<ValueType>, IndexType>;
    friend class Ell<next_precision<ValueType>, IndexType>;
    friend class Hybrid<ValueType, IndexType>;

public:
    using EnableLinOp<Ell>::convert_to;
    using EnableLinOp<Ell>::move_to;
    using ConvertibleTo<Ell<next_precision<ValueType>, IndexType>>::convert_to;
    using ConvertibleTo<Ell<next_precision<ValueType>, IndexType>>::move_to;
    using ConvertibleTo<Dense<ValueType>>::convert_to;
    using ConvertibleTo<Dense<ValueType>>::move_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Ell>;

    void convert_to(
        Ell<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(Ell<next_precision<ValueType>, IndexType>* result) override;

    void convert_to(Dense<ValueType>* other) const override;

    void move_to(Dense<ValueType>* other) override;

    void convert_to(Csr<ValueType, IndexType>* other) const override;

    void move_to(Csr<ValueType, IndexType>* other) override;

    void read(const mat_data& data) override;

    void read(const device_mat_data& data) override;

    void read(device_mat_data&& data) override;

    void write(mat_data& data) const override;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;

    /**
     * Returns the values of the matrix.
     *
     * @return the values of the matrix.
     */
    value_type* get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc Ell::get_values()
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
     * Returns the column indexes of the matrix.
     *
     * @return the column indexes of the matrix.
     */
    index_type* get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc Ell::get_col_idxs()
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
     * Returns the number of stored elements per row.
     *
     * @return the number of stored elements per row.
     */
    size_type get_num_stored_elements_per_row() const noexcept
    {
        return num_stored_elements_per_row_;
    }

    /**
     * Returns the stride of the matrix.
     *
     * @return the stride of the matrix.
     */
    size_type get_stride() const noexcept { return stride_; }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_size();
    }

    /**
     * Returns the `idx`-th non-zero element of the `row`-th row .
     *
     * @param row  the row of the requested element
     * @param idx  the idx-th stored element of the row
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    value_type& val_at(size_type row, size_type idx) noexcept
    {
        return values_.get_data()[this->linearize_index(row, idx)];
    }

    /**
     * @copydoc Ell::val_at(size_type, size_type)
     */
    value_type val_at(size_type row, size_type idx) const noexcept
    {
        return values_.get_const_data()[this->linearize_index(row, idx)];
    }

    /**
     * Returns the `idx`-th column index of the `row`-th row .
     *
     * @param row  the row of the requested element
     * @param idx  the idx-th stored element of the row
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    index_type& col_at(size_type row, size_type idx) noexcept
    {
        return this->get_col_idxs()[this->linearize_index(row, idx)];
    }

    /**
     * @copydoc Ell::col_at(size_type, size_type)
     */
    index_type col_at(size_type row, size_type idx) const noexcept
    {
        return this->get_const_col_idxs()[this->linearize_index(row, idx)];
    }

    /**
     * Creates a constant (immutable) Ell matrix from a set of constant arrays.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param col_idxs  the column index array of the matrix
     * @param num_stored_elements_per_row  the number of stored nonzeros per row
     * @param stride  the column-stride of the value and column index array
     * @returns A smart pointer to the constant matrix wrapping the input arrays
     *          (if they reside on the same executor as the matrix) or a copy of
     *          the arrays on the correct executor.
     */
    static std::unique_ptr<const Ell> create_const(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        gko::detail::const_array_view<ValueType>&& values,
        gko::detail::const_array_view<IndexType>&& col_idxs,
        size_type num_stored_elements_per_row, size_type stride)
    {
        // cast const-ness away, but return a const object afterwards,
        // so we can ensure that no modifications take place.
        return std::unique_ptr<const Ell>(new Ell{
            exec, size, gko::detail::array_const_cast(std::move(values)),
            gko::detail::array_const_cast(std::move(col_idxs)),
            num_stored_elements_per_row, stride});
    }

    /**
     * Copy-assigns an Ell matrix. Preserves the executor, reallocates the
     * matrix with minimal stride if the dimensions don't match, then copies the
     * data over, ignoring padding.
     */
    Ell& operator=(const Ell&);

    /**
     * Move-assigns an Ell matrix. Preserves the executor, moves the data over
     * preserving size and stride. Leaves the moved-from object in an empty
     * state (0x0 with empty Array).
     */
    Ell& operator=(Ell&&);

    /**
     * Copy-constructs an Ell matrix. Inherits executor and dimensions, but
     * copies data without padding.
     */
    Ell(const Ell&);

    /**
     * Move-constructs an Ell matrix. Inherits executor, dimensions and data
     * with padding. The moved-from object is empty (0x0 with empty Array).
     */
    Ell(Ell&&);

protected:
    /**
     * Creates an uninitialized Ell matrix of the specified size.
     *    (The stride is set to the number of rows of the matrix.
     *     The num_stored_elements_per_row is set to the number of cols of the
     * matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Ell(std::shared_ptr<const Executor> exec, const dim<2>& size = dim<2>{})
        : Ell(std::move(exec), size, size[1])
    {}

    /**
     * Creates an uninitialized Ell matrix of the specified size.
     *    (The stride is set to the number of rows of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_stored_elements_per_row   the number of stored elements per
     *                                      row
     */
    Ell(std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type num_stored_elements_per_row)
        : Ell(std::move(exec), size, num_stored_elements_per_row, size[0])
    {}

    /**
     * Creates an uninitialized Ell matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_stored_elements_per_row   the number of stored elements per
     *                                      row
     * @param stride                stride of the rows
     */
    Ell(std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type num_stored_elements_per_row, size_type stride)
        : EnableLinOp<Ell>(exec, size),
          values_(exec, stride * num_stored_elements_per_row),
          col_idxs_(exec, stride * num_stored_elements_per_row),
          num_stored_elements_per_row_(num_stored_elements_per_row),
          stride_(stride)
    {}


    /**
     * Creates an ELL matrix from already allocated (and initialized)
     * column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param col_idxs  array of column indexes
     * @param num_stored_elements_per_row   the number of stored elements per
     *                                      row
     * @param stride  stride of the rows
     *
     * @note If one of `col_idxs` or `values` is not an rvalue, not an array of
     *       IndexType and ValueType, respectively, or is on the wrong executor,
     *       an internal copy of that array will be created, and the original
     *       array data will not be used in the matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray>
    Ell(std::shared_ptr<const Executor> exec, const dim<2>& size,
        ValuesArray&& values, ColIdxsArray&& col_idxs,
        size_type num_stored_elements_per_row, size_type stride)
        : EnableLinOp<Ell>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          num_stored_elements_per_row_{num_stored_elements_per_row},
          stride_{stride}
    {
        GKO_ASSERT_EQ(num_stored_elements_per_row_ * stride_,
                      values_.get_size());
        GKO_ASSERT_EQ(num_stored_elements_per_row_ * stride_,
                      col_idxs_.get_size());
    }

    /**
     * Resizes the matrix to the given dimensions and row nonzero count.
     * If the dimensions or row nonzero count don't match their old values,
     * the column stride will be reset to the number of rows and the internal
     * storage reallocated to match these values.
     *
     * @param new_size  the new matrix dimensions
     * @param max_row_nnz  the new number of nonzeros per row
     */
    void resize(dim<2> new_size, size_type max_row_nnz);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    size_type linearize_index(size_type row, size_type col) const noexcept
    {
        return row + stride_ * col;
    }

private:
    array<value_type> values_;
    array<index_type> col_idxs_;
    size_type num_stored_elements_per_row_;
    size_type stride_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_ELL_HPP_
