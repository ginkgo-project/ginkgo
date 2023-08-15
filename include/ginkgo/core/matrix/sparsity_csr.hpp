// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_SPARSITY_CSR_HPP_
#define GKO_PUBLIC_CORE_MATRIX_SPARSITY_CSR_HPP_


#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
class Csr;


template <typename ValueType>
class Dense;


template <typename ValueType, typename IndexType>
class Fbcsr;


/**
 * SparsityCsr is a matrix format which stores only the sparsity pattern of a
 * sparse matrix by compressing each row of the matrix (compressed sparse row
 * format).
 *
 * The values of the nonzero elements are stored as a value array of length 1.
 * All the values in the matrix are equal to this value. By default, this value
 * is set to 1.0. A row pointer array also stores the linearized starting index
 * of each row. An additional column index array is used to identify the column
 * where a nonzero is present.
 *
 * @tparam ValueType  precision of vectors in apply
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup sparsity
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class SparsityCsr
    : public EnableLinOp<SparsityCsr<ValueType, IndexType>>,
      public EnableCreateMethod<SparsityCsr<ValueType, IndexType>>,
      public ConvertibleTo<Csr<ValueType, IndexType>>,
      public ConvertibleTo<Dense<ValueType>>,
      public ReadableFromMatrixData<ValueType, IndexType>,
      public WritableToMatrixData<ValueType, IndexType>,
      public Transposable {
    friend class EnableCreateMethod<SparsityCsr>;
    friend class EnablePolymorphicObject<SparsityCsr, LinOp>;
    friend class Csr<ValueType, IndexType>;
    friend class Dense<ValueType>;
    friend class Fbcsr<ValueType, IndexType>;

public:
    using EnableLinOp<SparsityCsr>::convert_to;
    using EnableLinOp<SparsityCsr>::move_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::move_to;
    using ConvertibleTo<Dense<ValueType>>::convert_to;
    using ConvertibleTo<Dense<ValueType>>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = SparsityCsr<IndexType, ValueType>;
    using mat_data = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;

    void convert_to(Csr<ValueType, IndexType>* result) const override;

    void move_to(Csr<ValueType, IndexType>* result) override;

    void convert_to(Dense<ValueType>* result) const override;

    void move_to(Dense<ValueType>* result) override;

    void read(const mat_data& data) override;

    void read(const device_mat_data& data) override;

    void read(device_mat_data&& data) override;

    void write(mat_data& data) const override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Transforms the sparsity matrix to an adjacency matrix. As the adjacency
     * matrix has to be square, the input SparsityCsr matrix for this function
     * to work has to be square.
     *
     * @note The adjacency matrix in this case is the sparsity pattern but with
     * the diagonal ones removed. This is mainly used for the
     * reordering/partitioning as taken in by graph libraries such as METIS.
     */
    std::unique_ptr<SparsityCsr> to_adjacency_matrix() const;

    /**
     * Sorts each row by column index
     */
    void sort_by_column_index();

    /*
     * Tests if all col_idxs are sorted by column index
     *
     * @returns True if all col_idxs are sorted.
     */
    bool is_sorted_by_column_index() const;

    /**
     * Returns the column indices of the matrix.
     *
     * @return the column indices of the matrix.
     */
    index_type* get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc SparsityCsr::get_col_idxs()
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
     * Returns the row pointers of the matrix.
     *
     * @return the row pointers of the matrix.
     */
    index_type* get_row_ptrs() noexcept { return row_ptrs_.get_data(); }

    /**
     * @copydoc SparsityCsr::get_row_ptrs()
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
     * Returns the value stored in the matrix.
     *
     * @return the value of the matrix.
     */
    value_type* get_value() noexcept { return value_.get_data(); }

    /**
     * @copydoc SparsityCsr::get_value()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_value() const noexcept
    {
        return value_.get_const_data();
    }


    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_nonzeros() const noexcept
    {
        return col_idxs_.get_num_elems();
    }

    /**
     * Creates a constant (immutable) SparsityCsr matrix from constant arrays.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param col_idxs  the column index array of the matrix
     * @param row_ptrs  the row pointer array of the matrix
     * @param strategy  the strategy the matrix uses for SpMV operations
     * @returns A smart pointer to the constant matrix wrapping the input arrays
     *          (if they reside on the same executor as the matrix) or a copy of
     *          these arrays on the correct executor.
     */
    static std::unique_ptr<const SparsityCsr> create_const(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        gko::detail::const_array_view<IndexType>&& col_idxs,
        gko::detail::const_array_view<IndexType>&& row_ptrs,
        ValueType value = one<ValueType>())
    {
        // cast const-ness away, but return a const object afterwards,
        // so we can ensure that no modifications take place.
        return std::unique_ptr<const SparsityCsr>(new SparsityCsr{
            exec, size, gko::detail::array_const_cast(std::move(col_idxs)),
            gko::detail::array_const_cast(std::move(row_ptrs)), value});
    }

    /**
     * Copy-assigns a SparsityCsr matrix. Preserves executor, copies everything
     * else.
     */
    SparsityCsr& operator=(const SparsityCsr&);

    /**
     * Move-assigns a SparsityCsr matrix. Preserves executor, moves the data and
     * leaves the moved-from object in an empty state (0x0 LinOp with unchanged
     * executor, no nonzeros and valid row pointers).
     */
    SparsityCsr& operator=(SparsityCsr&&);

    /**
     * Copy-constructs a SparsityCsr matrix. Inherits executor, strategy and
     * data.
     */
    SparsityCsr(const SparsityCsr&);

    /**
     * Move-constructs a SparsityCsr matrix. Inherits executor, moves the data
     * and leaves the moved-from object in an empty state (0x0 LinOp with
     * unchanged executor, no nonzeros and valid row pointers).
     */
    SparsityCsr(SparsityCsr&&);

protected:
    /**
     * Creates an uninitialized SparsityCsr matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     */
    SparsityCsr(std::shared_ptr<const Executor> exec,
                const dim<2>& size = dim<2>{}, size_type num_nonzeros = {})
        : EnableLinOp<SparsityCsr>(exec, size),
          col_idxs_(exec, num_nonzeros),
          row_ptrs_(exec, size[0] + 1),
          value_(exec, {one<ValueType>()})
    {
        row_ptrs_.fill(0);
    }

    /**
     * Creates a SparsityCsr matrix from already allocated (and initialized) row
     * pointer and column index arrays.
     *
     * @tparam ColIdxsArray  type of `col_idxs` array
     * @tparam RowPtrsArray  type of `row_ptrs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param col_idxs  array of column indexes
     * @param row_ptrs  array of row pointers
     * @param value  value stored. (same value for all matrix elements)
     *
     * @note If one of `row_ptrs` or `col_idxs` is not an rvalue, not
     *       an array of IndexType and IndexType respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ColIdxsArray, typename RowPtrsArray>
    SparsityCsr(std::shared_ptr<const Executor> exec, const dim<2>& size,
                ColIdxsArray&& col_idxs, RowPtrsArray&& row_ptrs,
                value_type value = one<ValueType>())
        : EnableLinOp<SparsityCsr>(exec, size),
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          row_ptrs_{exec, std::forward<RowPtrsArray>(row_ptrs)},
          value_{exec, {value}}
    {
        GKO_ASSERT_EQ(this->get_size()[0] + 1, row_ptrs_.get_num_elems());
    }

    /**
     * Creates a Sparsity matrix from an existing matrix. Uses the
     * `copy_and_convert_to` functionality.
     *
     * @param exec  Executor associated to the matrix
     * @param matrix The input matrix
     */
    SparsityCsr(std::shared_ptr<const Executor> exec,
                std::shared_ptr<const LinOp> matrix)
        : EnableLinOp<SparsityCsr>(exec, matrix->get_size())
    {
        auto tmp_ = copy_and_convert_to<SparsityCsr>(exec, matrix);
        this->copy_from(tmp_);
    }

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    array<index_type> col_idxs_;
    array<index_type> row_ptrs_;
    array<value_type> value_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_SPARSITY_CSR_HPP_
