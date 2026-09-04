// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tuple>
#include <variant>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/type_traits.hpp>
#include <ginkgo/core/matrix/device_views.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class MultiVector;


}


/**
 * Helper type for type-erased scalar arguments.
 *
 * An object of this type can hold any a value with any supported value type.
 * It is also possible to construct this object with values that are convertible
 * to double, especially ints.
 */
struct any_scalar {
    using variant_type = syn::variant_from_list<supported_value_types>;

    template <
        typename T,
        std::enable_if_t<std::is_constructible_v<variant_type, T&&>, int> = 0>
    any_scalar(T&& value) : variant(std::forward<T>(value))
    {}

    // Allow constructing from int or similar
    template <typename T,
              std::enable_if_t<!std::is_constructible_v<variant_type, T&&> &&
                                   std::is_convertible_v<T, double>,
                               int> = 1>
    any_scalar(T&& value) : variant(static_cast<double>(value))
    {}

    variant_type variant;
};


/**
 * Abstract interface class for (multi-)vector types.
 *
 * This interface defines the required functions on a vector. LinOps are applied
 * to this type.
 * It has functions in the following categories:
 * - creation
 * - complex handling
 * - pointwise operations (scaling, adding, ...)
 * - reductions (dot product, norm, ...)
 * - (sub-)views
 * - precision conversion
 *
 * The public interface functions are not virtual. Instead they call virtual
 * implementation functions and check for matching dimensions appropriately. For
 * example the function add_sub(scale, other), then an exception is thrown if
 * either:
 * - scale is not 1x1 or 1xc, where c is the number of columns of this,
 * - other has not the same size as this.
 *
 * To simplify the creation of derived classes, the mixin @see EnableMultiVector
 * can be used.
 */
class AbstractMultiVector : public PolymorphicObject, public Cloneable {
public:
    template <typename ValueType>
    using device_view = matrix::view::dense<ValueType>;

    /**
     * Creates an empty multi-vector with the same type, executor, size, stride
     * and precision as other.
     *
     * @param other  The vector whose configuration is used
     *
     * @return An uninitialized vector with the configuration of other
     */
    [[nodiscard]] static std::unique_ptr<AbstractMultiVector>
    create_with_config_of(ptr_param<const AbstractMultiVector> other);

    /**
     * Creates an empty multi-vector with the same type as other on exec.
     *
     * Only the type and precision of other are used, the size and stride of
     * the new vector are both zero.
     *
     * @param other The multi-vector whose type is used.
     * @param exec The executor of the new multi-vector.
     *
     * @return An empty vector with the type of other
     */
    [[nodiscard]] static std::unique_ptr<AbstractMultiVector>
    create_with_type_of(ptr_param<const AbstractMultiVector> other,
                        std::shared_ptr<const Executor> exec);

    /**
     * Creates an empty multi-vector with the same type as other and
     * the given size.
     *
     * The stride of the new vector is set to the number of global columns. For
     * non-distributed types, global_size and local_size have to be equal.
     *
     * @param other The multi-vector whose type is used.
     * @param exec The executor of the new multi-vector.
     * @param global_size The global size of the new multi-vector.
     * @param local_size The local size of the new multi-vector.
     *
     * @throws DimensionMismatch if global_size and local_size don't have
     *                           the same number of columns
     *
     * @return An empty vector with the type of other
     */
    [[nodiscard]] static std::unique_ptr<AbstractMultiVector>
    create_with_type_of(ptr_param<const AbstractMultiVector> other,
                        std::shared_ptr<const Executor> exec,
                        const dim<2>& global_size, const dim<2>& local_size);

    /**
     * Creates an empty multi-vector with the same type as other and
     * the given size and stride.
     *
     * @see create_with_type_of(ptr_param<const AbstractMultiVector>,
     *      std::shared_ptr<const Executor>, const dim<2>&, const dim<2>&)
     *
     * @param other The multi-vector whose type is used.
     * @param exec The executor of the new multi-vector.
     * @param global_size The global size of the new multi-vector.
     * @param local_size The local size of the new multi-vector.
     * @param stride The stride of the new multi-vector.
     *
     * @return An empty vector with the type of other
     */
    [[nodiscard]] static std::unique_ptr<AbstractMultiVector>
    create_with_type_of(ptr_param<const AbstractMultiVector> other,
                        std::shared_ptr<const Executor> exec,
                        const dim<2>& global_size, const dim<2>& local_size,
                        size_type stride);

    /**
     * Creates a copy of this vector on the given executor.
     *
     * @param exec  The executor of the new vector
     *
     * @return A copy of this vector, stored on exec
     */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> clone(
        std::shared_ptr<const Executor> exec) const;

    /**
     * Creates a copy of this vector on this vector's executor.
     *
     * @return A copy of this vector
     */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> clone() const;

    /**
     * Copies the data of other into this vector.
     *
     * The executor and the precision of this vector are preserved, i.e. the
     * data of other is converted if the precisions don't match.
     *
     * @param other  The vector to copy from
     *
     * @return This vector
     */
    AbstractMultiVector* copy_from(ptr_param<const AbstractMultiVector> other);

    /**
     * Moves the data of other into this vector.
     *
     * The executor and the precision of this vector are preserved, i.e. the
     * data of other is copied and converted if the precisions or executors
     * don't match. other is left in an unspecified, but valid, state.
     *
     * @param other  The vector to move from
     *
     * @return This vector
     */
    AbstractMultiVector* move_from(ptr_param<AbstractMultiVector> other);

    /**
     * Creates an empty vector with the same type as this vector
     * on the given executor.
     *
     * @param exec  The executor of the new vector
     *
     * @return An empty vector with the type of this vector
     */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> create_default(
        std::shared_ptr<const Executor> exec) const;

    /**
     * Creates an empty vector with the same type as this vector
     * on this vector's executor.
     *
     * @return An empty vector with the type of this vector
     */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> create_default() const;

    /**
     * Creates a new vector with the element-wise absolute values of this
     * vector.
     *
     * The result has the real precision matching this vector's precision, e.g.
     * a complex_fp32 vector results in a fp32 vector.
     *
     * @return A real vector with the absolute values of this vector
     */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> compute_absolute() const;

    /**
     * Writes the element-wise absolute values of this vector into output.
     *
     * @see compute_absolute()
     *
     * @param output  The vector to write the absolute values into. It must
     *                have the same size as this vector.
     *
     * @throws DimensionMismatch if output doesn't have the same size as this
     *                           vector
     * @throws PrecisionError if the output doesn't have the real precision of
     *                        this
     */
    void compute_absolute(ptr_param<AbstractMultiVector> output) const;

    /**
     * Replaces each element of this vector with its absolute value.
     *
     * The precision of this vector is not changed, i.e. a complex vector stays
     * complex, with zero imaginary parts.
     */
    void compute_absolute_inplace();

    /**
     * Creates a complex copy of this vector.
     *
     * If this vector is real, the imaginary part of the result is zero.
     *
     * @return A complex copy of this vector
     */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> make_complex() const;

    /**
     * Writes a complex copy of this vector into result.
     *
     * @see make_complex()
     *
     * @param result  The complex vector to write into. It must have the same
     *                size as this vector.
     *
     * @throws DimensionMismatch if result doesn't have the same size as this
     *                           vector
     * @throws PrecisionError if the output doesn't have the complex precision
     *                        of this
     */
    void make_complex(ptr_param<AbstractMultiVector> result) const;

    /**
     * Creates a new real vector with the real parts of this vector.
     *
     * @return A real vector with the real parts of this vector
     */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> get_real() const;

    /**
     * Writes the real parts of this vector into result.
     *
     * @param result  The real vector to write into. It must have the same size
     *                as this vector.
     *
     * @throws DimensionMismatch if result doesn't have the same size as this
     *                           vector
     * @throws PrecisionError if the output doesn't have the real precision of
     *                        this
     */
    void get_real(ptr_param<AbstractMultiVector> result) const;

    /**
     * Creates a new real vector with the imaginary parts of this vector.
     *
     * If this vector is real, the result is zero.
     *
     * @return A real vector with the imaginary parts of this vector
     */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> get_imag() const;

    /**
     * Writes the imaginary parts of this vector into result.
     *
     * @see get_imag()
     *
     * @param result  The real vector to write into. It must have the same size
     *                as this vector.
     *
     * @throws DimensionMismatch if result doesn't have the same size as this
     *                           vector
     * @throws PrecisionError if the output doesn't have the real precision of
     *                        this
     */
    void get_imag(ptr_param<AbstractMultiVector> result) const;

    /**
     * Fills this vector with a single value.
     *
     * @param value  The value to fill this vector with.
     */
    void fill(any_scalar value);

    /**
     * Scales this multi-vector element-wise by alpha.
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the entire matrix is scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column of the matrix is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of the matrix).
     *
     * @throws NotSupported  If alpha isn't a matrix::MultiVector
     * @throws DimensionMismatch  If alpha has incompatible dimensions
     */
    void scale(ptr_param<const AbstractMultiVector> alpha);

    /**
     * Divides this multi-vector element-wise by alpha.
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the entire matrix is scaled
     *               by 1 / alpha. If it is a Dense row vector of values,
     *               then i-th column of the matrix is scaled with the inverse
     *               of the i-th element of alpha (the number of columns of
     *               alpha has to match the number of columns of the matrix).
     *
     * @throws NotSupported  If alpha isn't a matrix::MultiVector
     * @throws DimensionMismatch  If alpha has incompatible dimensions
     */
    void inv_scale(ptr_param<const AbstractMultiVector> alpha);

    /**
     * Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the entire matrix is scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column of the matrix is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of the matrix).
     * @param b  a matrix of the same dimension as this
     *
     * @throws NotSupported  If alpha isn't a matrix::MultiVector
     * @throws DimensionMismatch  If alpha or b have incompatible dimensions
     */
    void add_scaled(ptr_param<const AbstractMultiVector> alpha,
                    ptr_param<const AbstractMultiVector> b);

    /**
     * Subtracts `b` scaled by `alpha` from the matrix (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, b is scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column of b is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of the matrix).
     * @param b  a matrix of the same dimension as this
     *
     * @throws NotSupported  If alpha isn't a matrix::MultiVector
     * @throws DimensionMismatch  If alpha or b have incompatible dimensions
     */
    void sub_scaled(ptr_param<const AbstractMultiVector> alpha,
                    ptr_param<const AbstractMultiVector> b);

    /**
     * Computes the column-wise dot product of this matrix and `b`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     *
     * @throws DimensionMismatch  If result or b have incompatible dimensions
     */
    void compute_dot(ptr_param<const AbstractMultiVector> b,
                     ptr_param<AbstractMultiVector> result) const;

    /**
     * Computes the column-wise dot product of this matrix and `b`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     *
     * @throws DimensionMismatch  If result or b have incompatible dimensions
     */
    void compute_dot(ptr_param<const AbstractMultiVector> b,
                     ptr_param<AbstractMultiVector> result,
                     array<char>& tmp) const;

    /**
     * Computes the column-wise dot product of `conj(this matrix)` and `b`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     *
     * @throws DimensionMismatch  If result or b have incompatible dimensions
     */
    void compute_conj_dot(ptr_param<const AbstractMultiVector> b,
                          ptr_param<AbstractMultiVector> result) const;

    /**
     * Computes the column-wise dot product of `conj(this matrix)` and `b`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     *
     * @throws DimensionMismatch  If result or b have incompatible dimensions
     */
    void compute_conj_dot(ptr_param<const AbstractMultiVector> b,
                          ptr_param<AbstractMultiVector> result,
                          array<char>& tmp) const;

    /**
     * Computes the column-wise Euclidean (L^2) norm of this matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     *
     * @throws DimensionMismatch  If result has incompatible dimensions
     */
    void compute_norm2(ptr_param<AbstractMultiVector> result) const;

    /**
     * Computes the column-wise Euclidean (L^2) norm of this matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the
     *                number of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     *
     * @throws DimensionMismatch  If result has incompatible dimensions
     */
    void compute_norm2(ptr_param<AbstractMultiVector> result,
                       array<char>& tmp) const;

    /**
     * Computes the square of the column-wise Euclidean (L^2) norm of this
     * matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     *
     * @throws DimensionMismatch  If result has incompatible dimensions
     */
    void compute_squared_norm2(ptr_param<AbstractMultiVector> result) const;

    /**
     * Computes the square of the column-wise Euclidean (L^2) norm of this
     * matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the
     *                number of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     *
     * @throws DimensionMismatch  If result has incompatible dimensions
     */
    void compute_squared_norm2(ptr_param<AbstractMultiVector> result,
                               array<char>& tmp) const;

    /**
     * Computes the column-wise (L^1) norm of this matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     *
     * @throws DimensionMismatch  If result has incompatible dimensions
     */
    void compute_norm1(ptr_param<AbstractMultiVector> result) const;

    /**
     * Computes the column-wise (L^1) norm of this matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the
     *                number of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     *
     * @throws DimensionMismatch  If result has incompatible dimensions
     */
    void compute_norm1(ptr_param<AbstractMultiVector> result,
                       array<char>& tmp) const;

    /**
     * Create a real view of the (potentially) complex original matrix.
     * If the original matrix is real, nothing changes. If the original matrix
     * is complex, the result is created by viewing the complex matrix with as
     * real with a reinterpret_cast with twice the number of columns and
     * double the stride.
     */
    [[nodiscard]] std::unique_ptr<const AbstractMultiVector> create_real_view()
        const;

    /** @copydoc create_real_view() const */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> create_real_view();

    /** Creates a view containing the selected local rows and columns. */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> create_subview(
        local_span rows, local_span columns);

    /** Creates a const view containing the selected local rows and columns. */
    [[nodiscard]] std::unique_ptr<const AbstractMultiVector> create_subview(
        local_span rows, local_span columns) const;

    /** Creates a view with the selected rows, columns, and global size. */
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> create_subview(
        local_span rows, local_span columns, dim<2> global_size);

    /**
     * Creates a const view with the selected rows, columns, and global size.
     */
    [[nodiscard]] std::unique_ptr<const AbstractMultiVector> create_subview(
        local_span rows, local_span columns, dim<2> global_size) const;

    /**
     * Gets a local device view of this vector for the specified value type.
     *
     * @throws InvalidStateError if the vector isn't stored in the requested
     *                           value type
     *
     * @tparam ValueType The value type of the view
     * @return A device view of the vector
     */
    template <typename ValueType>
    [[nodiscard]] device_view<ValueType> get_local_device_view();

    /** @copydoc get_local_device_view */
    template <typename ValueType>
    [[nodiscard]] device_view<const ValueType> get_const_local_device_view()
        const;

    /**
     * Creates a temporary conversion of this vector with another precision.
     *
     * Allowed conversions:
     * - bf16 <-> fp16 <-> fp32 <-> fp64
     * - complex_bf16 <-> complex_fp16 <-> complex_fp32 <-> complex_fp64
     * No conversions from complex to real or vice versa are supported.
     *
     * @note This non-const overload may cause a copy when the return value
     *       is destroyed.
     *
     * @param p The requested precision
     * @return A vector with the requested precision
     */
    [[nodiscard]] temporary_conversion<AbstractMultiVector> as_precision(
        precision p);

    /**
     * Creates a temporary conversion of this vector with another precision.
     *
     * @see as_precision(precision) for more details
     *
     * @param p Multi vector with the target precision
     * @return A vector with the same precision as p
     */
    [[nodiscard]] temporary_conversion<AbstractMultiVector> as_precision(
        ptr_param<const AbstractMultiVector> p);

    /**
     * @copydoc as_precision(precision)
     *
     * @note The const overload will not cause a copy when the return value
     *       is destroyed.
     */
    [[nodiscard]] temporary_conversion<const AbstractMultiVector> as_precision(
        precision p) const;

    /** @copydoc as_precision(ptr_param<const AbstractMultiVector>) */
    [[nodiscard]] temporary_conversion<const AbstractMultiVector> as_precision(
        ptr_param<const AbstractMultiVector> p) const;

    /**
     * Gets the precision of this vector.
     *
     * @return The precision enum value
     */
    [[nodiscard]] precision get_precision() const noexcept;

    /**
     * Gets the size (number of rows and columns) of this vector.
     *
     * @return The size as a dim<2> object
     */
    [[nodiscard]] dim<2> get_size() const noexcept;

    AbstractMultiVector(const AbstractMultiVector& other);

    AbstractMultiVector(AbstractMultiVector&& other);

    /**
     * Copies the contents of another multi-vector.
     *
     * Preserves the executor and precision on both objects.
     */
    AbstractMultiVector& operator=(const AbstractMultiVector& other);

    /**
     * Moves the contents of another multi-vector.
     *
     * Preserves the executor and precision on both objects.
     */
    AbstractMultiVector& operator=(AbstractMultiVector&& other);

protected:
    explicit AbstractMultiVector(std::shared_ptr<const Executor> exec,
                                 const dim<2>& size = dim<2>{},
                                 precision p = precision::none);

    [[nodiscard]] virtual std::unique_ptr<AbstractMultiVector>
    create_generic_with_same_config_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<AbstractMultiVector>
    create_generic_with_type_of_impl(std::shared_ptr<const Executor> exec,
                                     const dim<2>& global_size,
                                     const dim<2>& local_size,
                                     size_type stride) const = 0;

    [[nodiscard]] virtual std::unique_ptr<AbstractMultiVector>
    compute_absolute_generic_impl() const = 0;

    virtual void compute_absolute_generic_impl(
        AbstractMultiVector* result) const = 0;

    virtual void compute_absolute_inplace_impl() = 0;

    [[nodiscard]] virtual std::unique_ptr<AbstractMultiVector>
    make_complex_generic_impl() const = 0;

    virtual void make_complex_generic_impl(
        AbstractMultiVector* result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<AbstractMultiVector>
    get_real_generic_impl() const = 0;

    virtual void get_real_generic_impl(AbstractMultiVector* result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<AbstractMultiVector>
    get_imag_generic_impl() const = 0;

    virtual void get_imag_generic_impl(AbstractMultiVector* result) const = 0;

    virtual void fill_impl(any_scalar value) = 0;

    virtual void scale_impl(const AbstractMultiVector* alpha) = 0;

    virtual void inv_scale_impl(const AbstractMultiVector* alpha) = 0;

    virtual void add_scaled_impl(const AbstractMultiVector* alpha,
                                 const AbstractMultiVector* b) = 0;

    virtual void sub_scaled_impl(const AbstractMultiVector* alpha,
                                 const AbstractMultiVector* b) = 0;

    virtual void compute_dot_impl(const AbstractMultiVector* b,
                                  AbstractMultiVector* result,
                                  array<char>& tmp) const = 0;

    virtual void compute_conj_dot_impl(const AbstractMultiVector* b,
                                       AbstractMultiVector* result,
                                       array<char>& tmp) const = 0;

    virtual void compute_norm2_impl(AbstractMultiVector* result,
                                    array<char>& tmp) const = 0;

    virtual void compute_squared_norm2_impl(AbstractMultiVector* result,
                                            array<char>& tmp) const = 0;

    virtual void compute_norm1_impl(AbstractMultiVector* result,
                                    array<char>& tmp) const = 0;

    [[nodiscard]] virtual std::unique_ptr<const AbstractMultiVector>
    create_real_view_generic_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<AbstractMultiVector>
    create_real_view_generic_impl() = 0;

    [[nodiscard]] virtual std::unique_ptr<AbstractMultiVector>
    create_subview_generic_impl(local_span rows, local_span columns) = 0;

    [[nodiscard]] virtual std::unique_ptr<const AbstractMultiVector>
    create_subview_generic_impl(local_span rows, local_span columns) const = 0;

    [[nodiscard]] virtual std::unique_ptr<AbstractMultiVector>
    create_subview_generic_impl(local_span rows, local_span columns,
                                dim<2> global_size) = 0;

    [[nodiscard]] virtual std::unique_ptr<const AbstractMultiVector>
    create_subview_generic_impl(local_span rows, local_span columns,
                                dim<2> global_size) const = 0;

    [[nodiscard]] virtual std::variant<
#if GINKGO_ENABLE_HALF
        device_view<half>, device_view<std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        device_view<bfloat16>, device_view<std::complex<bfloat16>>,
#endif
        device_view<float>, device_view<std::complex<float>>,
        device_view<double>, device_view<std::complex<double>>>
    get_local_device_view_generic_impl() = 0;

    [[nodiscard]] virtual std::variant<
#if GINKGO_ENABLE_HALF
        device_view<const half>, device_view<const std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        device_view<const bfloat16>, device_view<const std::complex<bfloat16>>,
#endif
        device_view<const float>, device_view<const std::complex<float>>,
        device_view<const double>, device_view<const std::complex<double>>>
    get_const_local_device_view_generic_impl() const = 0;

    [[nodiscard]] virtual temporary_conversion<AbstractMultiVector>
    as_precision_impl(precision p) = 0;

    [[nodiscard]] virtual temporary_conversion<const AbstractMultiVector>
    as_precision_impl(precision p) const = 0;

    void set_size(const dim<2>& value) noexcept;

private:
    dim<2> size_;
    precision precision_;
};


}  // namespace gko
