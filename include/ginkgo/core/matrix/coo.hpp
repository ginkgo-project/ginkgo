// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_COO_HPP_
#define GKO_PUBLIC_CORE_MATRIX_COO_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
/**
 * @brief The matrix namespace.
 *
 * @ingroup matrix
 */
namespace matrix {


template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType>
class Dense;

template <typename ValueType, typename IndexType>
class CooBuilder;

template <typename ValueType, typename IndexType>
class Hybrid;


/**
 * COO stores a matrix in the coordinate matrix format.
 *
 * The nonzero elements are stored in an array row-wise (but not necessarily
 * sorted by column index within a row). Two extra arrays contain the row and
 * column indexes of each nonzero element of the matrix.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup coo
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Coo : public EnableLinOp<Coo<ValueType, IndexType>>,
            public ConvertibleTo<Coo<next_precision<ValueType>, IndexType>>,
#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
            public ConvertibleTo<Coo<next_precision<ValueType, 2>, IndexType>>,
#endif
#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
            public ConvertibleTo<Coo<next_precision<ValueType, 3>, IndexType>>,
#endif
            public ConvertibleTo<Csr<ValueType, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public DiagonalExtractable<ValueType>,
            public ReadableFromMatrixData<ValueType, IndexType>,
            public WritableToMatrixData<ValueType, IndexType>,
            public Transposable,
            public EnableAbsoluteComputation<
                remove_complex<Coo<ValueType, IndexType>>> {
    friend class EnablePolymorphicObject<Coo, LinOp>;
    friend class Csr<ValueType, IndexType>;
    friend class Dense<ValueType>;
    friend class CooBuilder<ValueType, IndexType>;
    friend class Coo<to_complex<ValueType>, IndexType>;
    friend class Hybrid<ValueType, IndexType>;

public:
    using EnableLinOp<Coo>::convert_to;
    using EnableLinOp<Coo>::move_to;
    using ConvertibleTo<Coo<next_precision<ValueType>, IndexType>>::convert_to;
    using ConvertibleTo<Coo<next_precision<ValueType>, IndexType>>::move_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::move_to;
    using ConvertibleTo<Dense<ValueType>>::convert_to;
    using ConvertibleTo<Dense<ValueType>>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = Coo<ValueType, IndexType>;
    using mat_data = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Coo>;

    friend class Coo<previous_precision<ValueType>, IndexType>;

    void convert_to(
        Coo<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(Coo<next_precision<ValueType>, IndexType>* result) override;

#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
    friend class Coo<previous_precision<ValueType, 2>, IndexType>;
    using ConvertibleTo<
        Coo<next_precision<ValueType, 2>, IndexType>>::convert_to;
    using ConvertibleTo<Coo<next_precision<ValueType, 2>, IndexType>>::move_to;

    void convert_to(
        Coo<next_precision<ValueType, 2>, IndexType>* result) const override;

    void move_to(Coo<next_precision<ValueType, 2>, IndexType>* result) override;
#endif

#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
    friend class Coo<previous_precision<ValueType, 3>, IndexType>;
    using ConvertibleTo<
        Coo<next_precision<ValueType, 3>, IndexType>>::convert_to;
    using ConvertibleTo<Coo<next_precision<ValueType, 3>, IndexType>>::move_to;

    void convert_to(
        Coo<next_precision<ValueType, 3>, IndexType>* result) const override;

    void move_to(Coo<next_precision<ValueType, 3>, IndexType>* result) override;
#endif

    void convert_to(Csr<ValueType, IndexType>* other) const override;

    void move_to(Csr<ValueType, IndexType>* other) override;

    void convert_to(Dense<ValueType>* other) const override;

    void move_to(Dense<ValueType>* other) override;

    void read(const mat_data& data) override;

    void read(const device_mat_data& data) override;

    void read(device_mat_data&& data) override;

    void write(mat_data& data) const override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

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
     * @copydoc Csr::get_values()
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
     * @copydoc Csr::get_col_idxs()
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
     * Returns the row indexes of the matrix.
     *
     * @return the row indexes of the matrix.
     */
    index_type* get_row_idxs() noexcept { return row_idxs_.get_data(); }

    /**
     * @copydoc Csr::get_row_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_row_idxs() const noexcept
    {
        return row_idxs_.get_const_data();
    }

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
     * Applies Coo matrix axpy to a vector (or a sequence of vectors).
     *
     * Performs the operation x = Coo * b + x
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     *
     * @return this
     */
    LinOp* apply2(ptr_param<const LinOp> b, ptr_param<LinOp> x);

    /**
     * @copydoc apply2(cost LinOp *, LinOp *)
     */
    const LinOp* apply2(ptr_param<const LinOp> b, ptr_param<LinOp> x) const;

    /**
     * Performs the operation x = alpha * Coo * b + x.
     *
     * @param alpha  scaling of the result of Coo * b
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     *
     * @return this
     */
    LinOp* apply2(ptr_param<const LinOp> alpha, ptr_param<const LinOp> b,
                  ptr_param<LinOp> x);

    /**
     * @copydoc apply2(const LinOp *, const LinOp *, LinOp *)
     */
    const LinOp* apply2(ptr_param<const LinOp> alpha, ptr_param<const LinOp> b,
                        ptr_param<LinOp> x) const;

    /**
     * Creates an uninitialized COO matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Coo> create(std::shared_ptr<const Executor> exec,
                                       const dim<2>& size = dim<2>{},
                                       size_type num_nonzeros = {});

    /**
     * Creates a COO matrix from already allocated (and initialized) row
     * index, column index and value arrays.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param col_idxs  array of column indexes
     * @param row_idxs  array of row pointers
     *
     * @note If one of `row_idxs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     *
     * @return A smart pointer to the matrix wrapping the input arrays
     *         (if they reside on the same executor as the matrix) or a copy of
     *         these arrays on the correct executor.
     */
    static std::unique_ptr<Coo> create(std::shared_ptr<const Executor> exec,
                                       const dim<2>& size,
                                       array<value_type> values,
                                       array<index_type> col_idxs,
                                       array<index_type> row_idxs);

    /**
     * @copydoc std::unique_ptr<Coo> create(std::shared_ptr<const Executor>,
     * const dim<2>&, array<value_type>, array<index_type>, array<index_type>)
     */
    template <typename InputValueType, typename InputColumnIndexType,
              typename InputRowIndexType>
    GKO_DEPRECATED(
        "explicitly construct the gko::array argument instead of passing "
        "initializer lists")
    static std::unique_ptr<Coo> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        std::initializer_list<InputValueType> values,
        std::initializer_list<InputColumnIndexType> col_idxs,
        std::initializer_list<InputRowIndexType> row_idxs)
    {
        return create(exec, size, array<value_type>{exec, std::move(values)},
                      array<index_type>{exec, std::move(col_idxs)},
                      array<index_type>{exec, std::move(row_idxs)});
    }

    /**
     * Creates a constant (immutable) Coo matrix from a set of constant arrays.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param col_idxs  the column index array of the matrix
     * @param row_ptrs  the row index array of the matrix
     *
     * @return A smart pointer to the constant matrix wrapping the input arrays
     *         (if they reside on the same executor as the matrix) or a copy of
     *         these arrays on the correct executor.
     */
    static std::unique_ptr<const Coo> create_const(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        gko::detail::const_array_view<ValueType>&& values,
        gko::detail::const_array_view<IndexType>&& col_idxs,
        gko::detail::const_array_view<IndexType>&& row_idxs);

protected:
    Coo(std::shared_ptr<const Executor> exec, const dim<2>& size = dim<2>{},
        size_type num_nonzeros = {});

    Coo(std::shared_ptr<const Executor> exec, const dim<2>& size,
        array<value_type> values, array<index_type> col_idxs,
        array<index_type> row_idxs);

    /**
     * Resizes the matrix and associated storage to the given sizes.
     * Internal storage may be reallocated if they don't match the old values.
     *
     * @param new_size  the new matrix dimensions.
     * @param nnz  the new number of nonzeros.
     */
    void resize(dim<2> new_size, size_type nnz);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void apply2_impl(const LinOp* b, LinOp* x) const;

    void apply2_impl(const LinOp* alpha, const LinOp* b, LinOp* x) const;

private:
    array<value_type> values_;
    array<index_type> col_idxs_;
    array<index_type> row_idxs_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_COO_HPP_
