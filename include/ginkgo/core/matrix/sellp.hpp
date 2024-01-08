// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_SELLP_HPP_
#define GKO_PUBLIC_CORE_MATRIX_SELLP_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


constexpr int default_slice_size = 64;
constexpr int default_stride_factor = 1;


template <typename ValueType>
class Dense;

template <typename ValueType, typename IndexType>
class Csr;

/**
 * SELL-P is a matrix format similar to ELL format. The difference is that
 * SELL-P format divides rows into smaller slices and store each slice with ELL
 * format.
 *
 * This implementation uses the column index value invalid_index<IndexType>()
 * to mark padding entries that are not part of the sparsity pattern.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup sellp
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Sellp : public EnableLinOp<Sellp<ValueType, IndexType>>,
              public EnableCreateMethod<Sellp<ValueType, IndexType>>,
              public ConvertibleTo<Sellp<next_precision<ValueType>, IndexType>>,
              public ConvertibleTo<Dense<ValueType>>,
              public ConvertibleTo<Csr<ValueType, IndexType>>,
              public DiagonalExtractable<ValueType>,
              public ReadableFromMatrixData<ValueType, IndexType>,
              public WritableToMatrixData<ValueType, IndexType>,
              public EnableAbsoluteComputation<
                  remove_complex<Sellp<ValueType, IndexType>>> {
    friend class EnableCreateMethod<Sellp>;
    friend class EnablePolymorphicObject<Sellp, LinOp>;
    friend class Dense<ValueType>;
    friend class Csr<ValueType, IndexType>;
    friend class Sellp<to_complex<ValueType>, IndexType>;

public:
    using EnableLinOp<Sellp>::convert_to;
    using EnableLinOp<Sellp>::move_to;
    using ConvertibleTo<
        Sellp<next_precision<ValueType>, IndexType>>::convert_to;
    using ConvertibleTo<Sellp<next_precision<ValueType>, IndexType>>::move_to;
    using ConvertibleTo<Dense<ValueType>>::convert_to;
    using ConvertibleTo<Dense<ValueType>>::move_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Sellp>;

    friend class Sellp<next_precision<ValueType>, IndexType>;

    void convert_to(
        Sellp<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(Sellp<next_precision<ValueType>, IndexType>* result) override;

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
     * @copydoc Sellp::get_values()
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
     * @copydoc Sellp::get_col_idxs()
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
     * Returns the lengths(columns) of slices.
     *
     * @return the lengths(columns) of slices.
     */
    size_type* get_slice_lengths() noexcept
    {
        return slice_lengths_.get_data();
    }

    /**
     * @copydoc Sellp::get_slice_lengths()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const size_type* get_const_slice_lengths() const noexcept
    {
        return slice_lengths_.get_const_data();
    }

    /**
     * Returns the offsets of slices.
     *
     * @return the offsets of slices.
     */
    size_type* get_slice_sets() noexcept { return slice_sets_.get_data(); }

    /**
     * @copydoc Sellp::get_slice_sets()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const size_type* get_const_slice_sets() const noexcept
    {
        return slice_sets_.get_const_data();
    }

    /**
     * Returns the size of a slice.
     *
     * @return the size of a slice.
     */
    size_type get_slice_size() const noexcept { return slice_size_; }

    /**
     * Returns the stride factor(t) of SELL-P.
     *
     * @return the stride factor(t) of SELL-P.
     */
    size_type get_stride_factor() const noexcept { return stride_factor_; }

    /**
     * Returns the total column number.
     *
     * @return the total column number.
     */
    size_type get_total_cols() const noexcept
    {
        return values_.get_size() / slice_size_;
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
     * Returns the `idx`-th non-zero element of the `row`-th row with
     * `slice_set` slice set.
     *
     * @param row  the row of the requested element in the slice
     * @param slice_set  the slice set of the slice
     * @param idx  the idx-th stored element of the row in the slice
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the CPU results in a runtime error)
     */
    value_type& val_at(size_type row, size_type slice_set,
                       size_type idx) noexcept
    {
        return values_.get_data()[this->linearize_index(row, slice_set, idx)];
    }

    /**
     * @copydoc Sellp::val_at(size_type, size_type, size_type)
     */
    value_type val_at(size_type row, size_type slice_set,
                      size_type idx) const noexcept
    {
        return values_
            .get_const_data()[this->linearize_index(row, slice_set, idx)];
    }

    /**
     * Returns the `idx`-th column index of the `row`-th row with `slice_set`
     * slice set.
     *
     * @param row  the row of the requested element in the slice
     * @param slice_set  the slice set of the slice
     * @param idx  the idx-th stored element of the row in the slice
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the CPU results in a runtime error)
     */
    index_type& col_at(size_type row, size_type slice_set,
                       size_type idx) noexcept
    {
        return this->get_col_idxs()[this->linearize_index(row, slice_set, idx)];
    }

    /**
     * @copydoc Sellp::col_at(size_type, size_type, size_type)
     */
    index_type col_at(size_type row, size_type slice_set,
                      size_type idx) const noexcept
    {
        return this
            ->get_const_col_idxs()[this->linearize_index(row, slice_set, idx)];
    }

    /**
     * Copy-assigns a Sellp matrix. Preserves the executor, copies the data and
     * parameters.
     */
    Sellp& operator=(const Sellp&);

    /**
     * Move-assigns a Sellp matrix. Preserves the executor, moves the data and
     * parameters. The moved-from object is empty (0x0 with valid slice_sets and
     * unchanged parameters).
     */
    Sellp& operator=(Sellp&&);

    /**
     * Copy-assigns a Sellp matrix. Inherits the executor, copies the data and
     * parameters.
     */
    Sellp(const Sellp&);

    /**
     * Move-assigns a Sellp matrix. Inherits the executor, moves the data and
     * parameters. The moved-from object is empty (0x0 with valid slice_sets and
     * unchanged parameters).
     */
    Sellp(Sellp&&);

protected:
    /**
     * Creates an uninitialized Sellp matrix of the specified size.
     *    (The total_cols is set to be the number of slice times the number
     *     of cols of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Sellp(std::shared_ptr<const Executor> exec, const dim<2>& size = dim<2>{})
        : Sellp(std::move(exec), size,
                ceildiv(size[0], default_slice_size) * size[1])
    {}

    /**
     * Creates an uninitialized Sellp matrix of the specified size.
     *    (The slice_size and stride_factor are set to the default values.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param total_cols   number of the sum of all cols in every slice.
     */
    Sellp(std::shared_ptr<const Executor> exec, const dim<2>& size,
          size_type total_cols)
        : Sellp(std::move(exec), size, default_slice_size,
                default_stride_factor, total_cols)
    {}

    /**
     * Creates an uninitialized Sellp matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param slice_size  number of rows in each slice
     * @param stride_factor  factor for the stride in each slice (strides
     *                        should be multiples of the stride_factor)
     * @param total_cols   number of the sum of all cols in every slice.
     */
    Sellp(std::shared_ptr<const Executor> exec, const dim<2>& size,
          size_type slice_size, size_type stride_factor, size_type total_cols)
        : EnableLinOp<Sellp>(exec, size),
          values_(exec, slice_size * total_cols),
          col_idxs_(exec, slice_size * total_cols),
          slice_lengths_(exec, ceildiv(size[0], slice_size)),
          slice_sets_(exec, ceildiv(size[0], slice_size) + 1),
          slice_size_(slice_size),
          stride_factor_(stride_factor)
    {
        slice_sets_.fill(0);
        slice_lengths_.fill(0);
    }

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    size_type linearize_index(size_type row, size_type slice_set,
                              size_type col) const noexcept
    {
        return (slice_set + col) * slice_size_ + row;
    }

private:
    array<value_type> values_;
    array<index_type> col_idxs_;
    array<size_type> slice_lengths_;
    array<size_type> slice_sets_;
    size_type slice_size_;
    size_type stride_factor_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_SELLP_HPP_
