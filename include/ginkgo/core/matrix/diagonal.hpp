// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_DIAGONAL_HPP_
#define GKO_PUBLIC_CORE_MATRIX_DIAGONAL_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType>
class Dense;


/**
 * This class is a utility which efficiently implements the diagonal matrix (a
 * linear operator which scales a vector row wise).
 *
 * Objects of the Diagonal class always represent a square matrix, and
 * require one array to store their values.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes of a CSR matrix the diagonal
 *                    is applied or converted to.
 *
 * @ingroup diagonal
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Diagonal
    : public EnableLinOp<Diagonal<ValueType>>,
      public EnableCreateMethod<Diagonal<ValueType>>,
      public ConvertibleTo<Csr<ValueType, int32>>,
      public ConvertibleTo<Csr<ValueType, int64>>,
      public ConvertibleTo<Diagonal<next_precision<ValueType>>>,
      public Transposable,
      public WritableToMatrixData<ValueType, int32>,
      public WritableToMatrixData<ValueType, int64>,
      public ReadableFromMatrixData<ValueType, int32>,
      public ReadableFromMatrixData<ValueType, int64>,
      public EnableAbsoluteComputation<remove_complex<Diagonal<ValueType>>> {
    friend class EnablePolymorphicObject<Diagonal, LinOp>;
    friend class EnableCreateMethod<Diagonal>;
    friend class Csr<ValueType, int32>;
    friend class Csr<ValueType, int64>;
    friend class Diagonal<to_complex<ValueType>>;

public:
    using EnableLinOp<Diagonal>::convert_to;
    using EnableLinOp<Diagonal>::move_to;
    using ConvertibleTo<Csr<ValueType, int32>>::convert_to;
    using ConvertibleTo<Csr<ValueType, int32>>::move_to;
    using ConvertibleTo<Csr<ValueType, int64>>::convert_to;
    using ConvertibleTo<Csr<ValueType, int64>>::move_to;
    using ConvertibleTo<Diagonal<next_precision<ValueType>>>::convert_to;
    using ConvertibleTo<Diagonal<next_precision<ValueType>>>::move_to;

    using value_type = ValueType;
    using index_type = int64;
    using mat_data = matrix_data<ValueType, int64>;
    using mat_data32 = matrix_data<ValueType, int32>;
    using device_mat_data = device_matrix_data<ValueType, int64>;
    using device_mat_data32 = device_matrix_data<ValueType, int32>;
    using absolute_type = remove_complex<Diagonal>;

    friend class Diagonal<next_precision<ValueType>>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    void convert_to(Diagonal<next_precision<ValueType>>* result) const override;

    void move_to(Diagonal<next_precision<ValueType>>* result) override;

    void convert_to(Csr<ValueType, int32>* result) const override;

    void move_to(Csr<ValueType, int32>* result) override;

    void convert_to(Csr<ValueType, int64>* result) const override;

    void move_to(Csr<ValueType, int64>* result) override;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;

    /**
     * Returns a pointer to the array of values of the matrix.
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
     * Applies the diagonal matrix from the right side to a matrix b,
     * which means scales the columns of b with the according diagonal entries.
     *
     * @param b  the input vector(s) on which the diagonal matrix is applied
     * @param x  the output vector(s) where the result is stored
     */
    void rapply(ptr_param<const LinOp> b, ptr_param<LinOp> x) const
    {
        GKO_ASSERT_REVERSE_CONFORMANT(this, b);
        GKO_ASSERT_EQUAL_ROWS(b, x);
        GKO_ASSERT_EQUAL_COLS(this, x);

        this->rapply_impl(b.get(), x.get());
    }

    /**
     * Applies the inverse of the diagonal matrix to a matrix b,
     * which means scales the columns of b with the inverse of the according
     * diagonal entries.
     *
     * @param b  the input vector(s) on which the inverse of the diagonal matrix
     * is applied
     * @param x  the output vector(s) where the result is stored
     */
    void inverse_apply(ptr_param<const LinOp> b, ptr_param<LinOp> x) const
    {
        GKO_ASSERT_CONFORMANT(this, b);
        GKO_ASSERT_EQUAL_ROWS(b, x);
        GKO_ASSERT_EQUAL_ROWS(this, x);

        this->inverse_apply_impl(b.get(), x.get());
    }

    void read(const mat_data& data) override;

    void read(const mat_data32& data) override;

    void read(const device_mat_data& data) override;

    void read(const device_mat_data32& data) override;

    void read(device_mat_data&& data) override;

    void read(device_mat_data32&& data) override;

    void write(mat_data& data) const override;

    void write(mat_data32& data) const override;

    /**
     * Creates a constant (immutable) Diagonal matrix from a constant array.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the size of the square matrix
     * @param values  the value array of the matrix
     * @returns A smart pointer to the constant matrix wrapping the input array
     *          (if it resides on the same executor as the matrix) or a copy of
     *          the array on the correct executor.
     */
    static std::unique_ptr<const Diagonal> create_const(
        std::shared_ptr<const Executor> exec, size_type size,
        gko::detail::const_array_view<ValueType>&& values)
    {
        // cast const-ness away, but return a const object afterwards,
        // so we can ensure that no modifications take place.
        return std::unique_ptr<const Diagonal>(new Diagonal{
            exec, size, gko::detail::array_const_cast(std::move(values))});
    }

protected:
    /**
     * Creates an empty Diagonal matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit Diagonal(std::shared_ptr<const Executor> exec)
        : Diagonal(std::move(exec), size_type{})
    {}

    /**
     * Creates an Diagonal matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Diagonal(std::shared_ptr<const Executor> exec, size_type size)
        : EnableLinOp<Diagonal>(exec, dim<2>{size}), values_(exec, size)
    {}

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
    Diagonal(std::shared_ptr<const Executor> exec, const size_type size,
             ValuesArray&& values)
        : EnableLinOp<Diagonal>(exec, dim<2>(size)),
          values_{exec, std::forward<ValuesArray>(values)}
    {
        GKO_ENSURE_IN_BOUNDS(size - 1, values_.get_size());
    }

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void rapply_impl(const LinOp* b, LinOp* x) const;

    void inverse_apply_impl(const LinOp* b, LinOp* x) const;

private:
    array<value_type> values_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_DIAGONAL_HPP_
