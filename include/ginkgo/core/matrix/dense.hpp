// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/device_views.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class MultiVector;


template <typename ValueType, typename IndexType>
class Coo;

template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType>
class Diagonal;

template <typename ValueType, typename IndexType>
class Ell;

template <typename ValueType, typename IndexType>
class Fbcsr;

template <typename ValueType, typename IndexType>
class Hybrid;

template <typename ValueType, typename IndexType>
class Sellp;

template <typename ValueType, typename IndexType>
class SparsityCsr;


/**
 * Dense is a matrix format which explicitly stores all values of the
 * matrix.
 *
 * The values are stored in row-major format (values belonging to the same row
 * appear consecutive in the memory). Optionally, rows can be padded for better
 * memory access.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup dense
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Dense : public LinOp,
              public EnableCloneable<Dense<ValueType>>,
              public ConvertibleTo<MultiVector<ValueType>>,
              public ConvertibleTo<Dense<next_precision<ValueType>>>,
#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
              public ConvertibleTo<Dense<next_precision<ValueType, 2>>>,
#endif
#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
              public ConvertibleTo<Dense<next_precision<ValueType, 3>>>,
#endif
              public ConvertibleTo<Coo<ValueType, int32>>,
              public ConvertibleTo<Coo<ValueType, int64>>,
              public ConvertibleTo<Csr<ValueType, int32>>,
              public ConvertibleTo<Csr<ValueType, int64>>,
              public ConvertibleTo<Ell<ValueType, int32>>,
              public ConvertibleTo<Ell<ValueType, int64>>,
              public ConvertibleTo<Fbcsr<ValueType, int32>>,
              public ConvertibleTo<Fbcsr<ValueType, int64>>,
              public ConvertibleTo<Hybrid<ValueType, int32>>,
              public ConvertibleTo<Hybrid<ValueType, int64>>,
              public ConvertibleTo<Sellp<ValueType, int32>>,
              public ConvertibleTo<Sellp<ValueType, int64>>,
              public ConvertibleTo<SparsityCsr<ValueType, int32>>,
              public ConvertibleTo<SparsityCsr<ValueType, int64>>,
              public DiagonalExtractable<ValueType>,
              public ReadableFromMatrixData<ValueType, int32>,
              public ReadableFromMatrixData<ValueType, int64>,
              public WritableToMatrixData<ValueType, int32>,
              public WritableToMatrixData<ValueType, int64>,
              public Transposable,
              public ScaledIdentityAddable {
    friend class EnableCloneable<Dense>;
    friend class Dense<to_complex<ValueType>>;
    friend class Dense<previous_precision<ValueType>>;
    friend class MultiVector<ValueType>;
    friend class Coo<ValueType, int32>;
    friend class Coo<ValueType, int64>;
    friend class Csr<ValueType, int32>;
    friend class Csr<ValueType, int64>;
    friend class Ell<ValueType, int32>;
    friend class Ell<ValueType, int64>;
    friend class Fbcsr<ValueType, int32>;
    friend class Fbcsr<ValueType, int64>;
    friend class Hybrid<ValueType, int32>;
    friend class Hybrid<ValueType, int64>;
    friend class Sellp<ValueType, int32>;
    friend class Sellp<ValueType, int64>;
    friend class SparsityCsr<ValueType, int32>;
    friend class SparsityCsr<ValueType, int64>;
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using EnableCloneable<Dense>::convert_to;
    using EnableCloneable<Dense>::move_to;
    using ConvertibleTo<MultiVector<ValueType>>::convert_to;
    using ConvertibleTo<MultiVector<ValueType>>::move_to;
    using ConvertibleTo<Coo<ValueType, int32>>::convert_to;
    using ConvertibleTo<Coo<ValueType, int32>>::move_to;
    using ConvertibleTo<Coo<ValueType, int64>>::convert_to;
    using ConvertibleTo<Coo<ValueType, int64>>::move_to;
    using ConvertibleTo<Csr<ValueType, int32>>::convert_to;
    using ConvertibleTo<Csr<ValueType, int32>>::move_to;
    using ConvertibleTo<Csr<ValueType, int64>>::convert_to;
    using ConvertibleTo<Csr<ValueType, int64>>::move_to;
    using ConvertibleTo<Ell<ValueType, int32>>::convert_to;
    using ConvertibleTo<Ell<ValueType, int32>>::move_to;
    using ConvertibleTo<Ell<ValueType, int64>>::convert_to;
    using ConvertibleTo<Ell<ValueType, int64>>::move_to;
    using ConvertibleTo<Fbcsr<ValueType, int32>>::convert_to;
    using ConvertibleTo<Fbcsr<ValueType, int32>>::move_to;
    using ConvertibleTo<Fbcsr<ValueType, int64>>::convert_to;
    using ConvertibleTo<Fbcsr<ValueType, int64>>::move_to;
    using ConvertibleTo<Hybrid<ValueType, int32>>::convert_to;
    using ConvertibleTo<Hybrid<ValueType, int32>>::move_to;
    using ConvertibleTo<Hybrid<ValueType, int64>>::convert_to;
    using ConvertibleTo<Hybrid<ValueType, int64>>::move_to;
    using ConvertibleTo<Sellp<ValueType, int32>>::convert_to;
    using ConvertibleTo<Sellp<ValueType, int32>>::move_to;
    using ConvertibleTo<Sellp<ValueType, int64>>::convert_to;
    using ConvertibleTo<Sellp<ValueType, int64>>::move_to;
    using ConvertibleTo<SparsityCsr<ValueType, int32>>::convert_to;
    using ConvertibleTo<SparsityCsr<ValueType, int32>>::move_to;
    using ConvertibleTo<SparsityCsr<ValueType, int64>>::convert_to;
    using ConvertibleTo<SparsityCsr<ValueType, int64>>::move_to;

    using value_type = ValueType;
    using index_type = int64;
    using transposed_type = Dense;
    using mat_data64 = matrix_data<value_type, int64>;
    using mat_data32 = matrix_data<value_type, int32>;
    using device_mat_data64 = device_matrix_data<value_type, int64>;
    using device_mat_data32 = device_matrix_data<value_type, int32>;
    using device_view = view::dense<value_type>;
    using const_device_view = view::dense<const value_type>;

    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;

    void convert_to(MultiVector<ValueType>* result) const override;

    void move_to(MultiVector<ValueType>* result) override;

    void convert_to(Dense<next_precision<ValueType>>* result) const override;

    void move_to(Dense<next_precision<ValueType>>* result) override;

#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
    friend class Dense<previous_precision<ValueType, 2>>;
    using ConvertibleTo<Dense<next_precision<ValueType, 2>>>::convert_to;
    using ConvertibleTo<Dense<next_precision<ValueType, 2>>>::move_to;

    void convert_to(Dense<next_precision<ValueType, 2>>* result) const override;

    void move_to(Dense<next_precision<ValueType, 2>>* result) override;
#endif

#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
    friend class Dense<previous_precision<ValueType, 3>>;
    using ConvertibleTo<Dense<next_precision<ValueType, 3>>>::convert_to;
    using ConvertibleTo<Dense<next_precision<ValueType, 3>>>::move_to;

    void convert_to(Dense<next_precision<ValueType, 3>>* result) const override;

    void move_to(Dense<next_precision<ValueType, 3>>* result) override;
#endif

    void convert_to(Coo<ValueType, int32>* result) const override;

    void move_to(Coo<ValueType, int32>* result) override;

    void convert_to(Coo<ValueType, int64>* result) const override;

    void move_to(Coo<ValueType, int64>* result) override;

    void convert_to(Csr<ValueType, int32>* result) const override;

    void move_to(Csr<ValueType, int32>* result) override;

    void convert_to(Csr<ValueType, int64>* result) const override;

    void move_to(Csr<ValueType, int64>* result) override;

    void convert_to(Ell<ValueType, int32>* result) const override;

    void move_to(Ell<ValueType, int32>* result) override;

    void convert_to(Ell<ValueType, int64>* result) const override;

    void move_to(Ell<ValueType, int64>* result) override;

    void convert_to(Fbcsr<ValueType, int32>* result) const override;

    void move_to(Fbcsr<ValueType, int32>* result) override;

    void convert_to(Fbcsr<ValueType, int64>* result) const override;

    void move_to(Fbcsr<ValueType, int64>* result) override;

    void convert_to(Hybrid<ValueType, int32>* result) const override;

    void move_to(Hybrid<ValueType, int32>* result) override;

    void convert_to(Hybrid<ValueType, int64>* result) const override;

    void move_to(Hybrid<ValueType, int64>* result) override;

    void convert_to(Sellp<ValueType, int32>* result) const override;

    void move_to(Sellp<ValueType, int32>* result) override;

    void convert_to(Sellp<ValueType, int64>* result) const override;

    void move_to(Sellp<ValueType, int64>* result) override;

    void convert_to(SparsityCsr<ValueType, int32>* result) const override;

    void move_to(SparsityCsr<ValueType, int32>* result) override;

    void convert_to(SparsityCsr<ValueType, int64>* result) const override;

    void move_to(SparsityCsr<ValueType, int64>* result) override;

    void read(const mat_data32& data) override;

    void read(const mat_data64& data) override;

    void read(const device_mat_data32& data) override;

    void read(const device_mat_data64& data) override;

    void read(device_mat_data32&& data) override;

    void read(device_mat_data64&& data) override;

    void write(mat_data32& data) const override;

    void write(mat_data64& data) const override;

    void validate_data() const override;

    void fill(ValueType value);

    /**
     * Writes the diagonal of this matrix into an existing diagonal matrix.
     *
     * @param output  The output matrix. Its size must match the size of this
     *                matrix's diagonal.
     * @see Dense::extract_diagonal()
     */
    void extract_diagonal(ptr_param<Diagonal<ValueType>> output) const;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    [[nodiscard]] std::unique_ptr<LinOp> transpose() const override;

    [[nodiscard]] std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Writes the transposed matrix into the given output matrix.
     *
     * @param output  The output matrix. It must have the dimensions
     *                `gko::transpose(this->get_size())`
     */
    void transpose(ptr_param<Dense> output) const;

    /**
     * Writes the conjugate-transposed matrix into the given output matrix.
     *
     * @param output  The output matrix. It must have the dimensions
     *                `gko::transpose(this->get_size())`
     */
    void conj_transpose(ptr_param<Dense> output) const;

    void add_scaled(ptr_param<const LinOp> alpha,
                    ptr_param<const Diagonal<value_type>> diag);

    void sub_scaled(ptr_param<const LinOp> alpha,
                    ptr_param<const Diagonal<value_type>> diag);

    [[nodiscard]] static std::unique_ptr<Dense> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size = dim<2>{},
        size_type stride = 0);

    [[nodiscard]] static std::unique_ptr<Dense> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        array<value_type> values, size_type stride);

    [[nodiscard]] static std::unique_ptr<const Dense> create_const(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        ::gko::detail::const_array_view<ValueType>&& values, size_type stride);

    [[nodiscard]] std::unique_ptr<Dense> create_subview(span rows, span cols);

    [[nodiscard]] std::unique_ptr<const Dense> create_subview(span rows,
                                                              span cols) const;

    [[nodiscard]] std::unique_ptr<const Dense> create_const_subview(
        span rows, span cols) const;

    [[nodiscard]] std::unique_ptr<const MultiVector<ValueType>>
    as_const_multivector_view() const;

    [[nodiscard]] std::unique_ptr<MultiVector<ValueType>> as_multivector_view();

    [[nodiscard]] device_view get_device_view();

    [[nodiscard]] const_device_view get_const_device_view() const;

    ValueType* get_values() noexcept { return values_.get_data(); }

    const ValueType* get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    ValueType& at(size_type row, size_type col);

    ValueType at(size_type row, size_type col) const;

    [[nodiscard]] size_type get_stride() const noexcept;

    [[nodiscard]] size_type get_num_stored_elements() const noexcept;

    Dense(const Dense& other);

    Dense(Dense&& other);

    Dense& operator=(const Dense& other);

    Dense& operator=(Dense&& other);

protected:
    Dense(std::shared_ptr<const Executor> exec, const dim<2>& size = dim<2>{},
          size_type stride = 0);

    Dense(std::shared_ptr<const Executor> exec, const dim<2>& size,
          array<value_type> values, size_type stride);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    [[nodiscard]] size_type linearize_index(size_type row,
                                            size_type col) const noexcept;

    void resize(dim<2> new_size);

    template <typename IndexType>
    void convert_impl(Coo<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(Csr<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(Ell<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(Fbcsr<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(Hybrid<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(Sellp<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(SparsityCsr<ValueType, IndexType>* result) const;

private:
    size_type stride_;
    array<value_type> values_;

    void add_scaled_identity_impl(const LinOp* a, const LinOp* b) override;
};


}  // namespace matrix


namespace detail {


template <typename ValueType>
struct temporary_clone_helper<matrix::Dense<ValueType>> {
    static std::unique_ptr<matrix::Dense<ValueType>> create(
        std::shared_ptr<const Executor> exec, matrix::Dense<ValueType>* ptr,
        bool copy_data)
    {
        if (copy_data) {
            return gko::clone(std::move(exec), ptr);
        } else {
            return matrix::Dense<ValueType>::create(exec, ptr->get_size());
        }
    }
};


}  // namespace detail


/**
 * Creates and initializes a column-vector.
 *
 * This function first creates a temporary Dense matrix, fills it with
 * passed in values, and then converts the matrix to the requested type.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix>
 *                 interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param stride  row stride for the temporary Dense matrix
 * @param vals  values used to initialize the vector
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup LinOp
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    size_type stride, std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using dense = matrix::Dense<typename Matrix::value_type>;
    size_type num_rows = vals.size();
    auto tmp = dense::create(exec->get_master(), dim<2>{num_rows, 1}, stride);
    size_type idx = 0;
    for (const auto& elem : vals) {
        tmp->at(idx, 0) = elem;
        ++idx;
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx);
    return mtx;
}

/**
 * Creates and initializes a column-vector.
 *
 * This function first creates a temporary Dense matrix, fills it with
 * passed in values, and then converts the matrix to the requested type. The
 * stride of the intermediate Dense matrix is set to 1.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix>
 *                 interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param vals  values used to initialize the vector
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup LinOp
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    return initialize<Matrix>(1, vals, std::move(exec),
                              std::forward<TArgs>(create_args)...);
}


/**
 * Creates and initializes a matrix.
 *
 * This function first creates a temporary Dense matrix, fills it with
 * passed in values, and then converts the matrix to the requested type.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix>
 *                 interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param stride  row stride for the temporary Dense matrix
 * @param vals  values used to initialize the matrix
 * @param exec  Executor associated to the matrix
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup LinOp
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    size_type stride,
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using dense = matrix::Dense<typename Matrix::value_type>;
    size_type num_rows = vals.size();
    size_type num_cols = num_rows > 0 ? begin(vals)->size() : 1;
    auto tmp =
        dense::create(exec->get_master(), dim<2>{num_rows, num_cols}, stride);
    size_type ridx = 0;
    for (const auto& row : vals) {
        size_type cidx = 0;
        for (const auto& elem : row) {
            tmp->at(ridx, cidx) = elem;
            ++cidx;
        }
        ++ridx;
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx);
    return mtx;
}


/**
 * Creates and initializes a matrix.
 *
 * This function first creates a temporary Dense matrix, fills it with
 * passed in values, and then converts the matrix to the requested type. The
 * stride of the intermediate Dense matrix is set to the number of columns
 * of the initializer list.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix>
 *                 interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param vals  values used to initialize the matrix
 * @param exec  Executor associated to the matrix
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup LinOp
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    return initialize<Matrix>(vals.size() > 0 ? begin(vals)->size() : 0, vals,
                              std::move(exec),
                              std::forward<TArgs>(create_args)...);
}


}  // namespace gko
