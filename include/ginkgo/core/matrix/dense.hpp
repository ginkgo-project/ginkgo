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
              public DiagonalExtractable<ValueType>,
              public ReadableFromMatrixData<ValueType, int32>,
              public ReadableFromMatrixData<ValueType, int64>,
              public WritableToMatrixData<ValueType, int32>,
              public WritableToMatrixData<ValueType, int64>,
              public Transposable {
    friend class EnableCloneable<Dense>;
    friend class Dense<to_complex<ValueType>>;
    friend class Dense<previous_precision<ValueType>>;
    friend class MultiVector<ValueType>;
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using EnableCloneable<Dense>::convert_to;
    using EnableCloneable<Dense>::move_to;
    using ConvertibleTo<MultiVector<ValueType>>::convert_to;
    using ConvertibleTo<MultiVector<ValueType>>::move_to;

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

private:
    size_type stride_;
    array<value_type> values_;
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
}  // namespace gko
