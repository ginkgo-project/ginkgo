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


/**
 * Trait class containing type aliases required by EnableMultiVector.
 *
 * @tparam ConcreteType  The vector type derived from EnableMultiVector.
 */
template <typename ConcreteType>
struct vector_traits;

/**
 * Specialization for concrete types with value type template parameter.
 *
 * @tparam ConcreteType The vector type derived from EnableMultiVector. Must be
 *                      a templated type, with the first template argument being
 *                      ValueType
 * @tparam ValueType The value type of the concrete type
 * @tparam Args Other template arguments of the concrete type.
 */
template <template <typename...> typename ConcreteType, typename ValueType,
          typename... Args>
struct vector_traits<ConcreteType<ValueType, Args...>> {
    using value_type = ValueType;
    using absolute_value_type = remove_complex<value_type>;
    using absolute_type = ConcreteType<absolute_value_type, Args...>;
    using real_type = absolute_type;
    using complex_value_type = to_complex<value_type>;
    using complex_type = ConcreteType<complex_value_type, Args...>;
};


/**
 * The allowed matrix::MultiVector<T> type to be used in scaling operations,
 * e.g. in scaled_add.
 *
 * @tparam ValueType The value type of the vector type on which to call the
 *                    scaling operation
 */
template <typename ValueType>
struct scaling_param {
    using variant_type = std::variant<const matrix::MultiVector<ValueType>*>;

    variant_type variant;
};

/**
 * Specialization for complex types, allows both real and complex scaling
 * parameters.
 */
template <typename ValueType>
struct scaling_param<std::complex<ValueType>> {
    using variant_type =
        std::variant<const matrix::MultiVector<ValueType>*,
                     const matrix::MultiVector<std::complex<ValueType>>*>;

    variant_type variant;
};


/**
 * Mixin class to simplify the implementation of the AbstractMultiVector
 * interface.
 *
 * The mixin is used for the following reasons:
 * 1. hide functions that return AbstractMultiVector and instead return
 *    ConcreteType
 * 2. handle the precision dispatch such that implementors are only required to
 *    implement virtual functions with concrete types
 *
 * For example the clone function returns now ConcreteType, instead of an
 * AbstractMultiVector. This means that if the concrete type is available, it is
 * carried through. Here is an example:
 * ```c++
 * auto vec = ConcreteType::create(...);
 * auto clone = vec->clone();  // clone has type ConcreteType
 * ```
 *
 * The mixin handles the dispatch from the abstract interface functions to the
 * concrete overridden functions. Two dispatches happen:
 *
 * 1. To the same precision as this,
 * 2. To the same derived type as this.
 *
 * Classes using this mixin need to only implement functions with concretized
 * arguments. Especially any precision conversion is guaranteed to be
 * unnecessary.
 *
 * For example, the interface defines abstract virtual functions, such
 * as AbstractMultiVector::add_scaled_impl(const AbstractMultiVector* alpha,
 * const AbstractMultiVector* b). This function is overridden (with final) here.
 * Classes using this mixin, instead override functions with concretized
 * arguments, such as add_sca(scaling_param<value_type> alpha,
 * const ConcreteType* b).
 *
 * @tparam ConcreteType Multi vector type to enable the mixin for.
 */
template <typename ConcreteType>
class EnableMultiVector : public AbstractMultiVector,
                          public ConvertibleTo<ConcreteType> {
public:
    using traits = vector_traits<ConcreteType>;
    using value_type = typename traits::value_type;
    using absolute_value_type = typename traits::absolute_value_type;
    using absolute_type = typename traits::absolute_type;
    using real_type = typename traits::real_type;
    using complex_type = typename traits::complex_type;
    using result_type = ConcreteType;
    using norm_type = matrix::MultiVector<absolute_value_type>;
    using device_view = AbstractMultiVector::device_view<value_type>;
    using const_device_view =
        AbstractMultiVector::device_view<const value_type>;

    using ConvertibleTo<result_type>::convert_to;
    using ConvertibleTo<result_type>::move_to;

    [[nodiscard]] static std::unique_ptr<ConcreteType> create_with_config_of(
        ptr_param<const ConcreteType> other);

    [[nodiscard]] static std::unique_ptr<ConcreteType> create_with_type_of(
        ptr_param<const ConcreteType> other,
        std::shared_ptr<const Executor> exec);

    [[nodiscard]] static std::unique_ptr<ConcreteType> create_with_type_of(
        ptr_param<const ConcreteType> other,
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size);

    [[nodiscard]] static std::unique_ptr<ConcreteType> create_with_type_of(
        ptr_param<const ConcreteType> other,
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size, size_type stride);

    [[nodiscard]] std::unique_ptr<ConcreteType> clone(
        std::shared_ptr<const Executor> exec) const;

    [[nodiscard]] std::unique_ptr<ConcreteType> clone() const;

    ConcreteType* copy_from(ptr_param<const ConcreteType> other);

    ConcreteType* move_from(ptr_param<ConcreteType> other);

    [[nodiscard]] std::unique_ptr<ConcreteType> create_default();

    [[nodiscard]] std::unique_ptr<ConcreteType> create_default(
        std::shared_ptr<const Executor> exec);

    [[nodiscard]] std::unique_ptr<ConcreteType> create_subview(
        local_span rows, local_span columns);

    [[nodiscard]] std::unique_ptr<const ConcreteType> create_subview(
        local_span rows, local_span columns) const;

    [[nodiscard]] std::unique_ptr<ConcreteType> create_subview(
        local_span rows, local_span columns, dim<2> global_size);

    [[nodiscard]] std::unique_ptr<const ConcreteType> create_subview(
        local_span rows, local_span columns, dim<2> global_size) const;

    [[nodiscard]] std::unique_ptr<const real_type> create_real_view() const;

    [[nodiscard]] std::unique_ptr<real_type> create_real_view();

    [[nodiscard]] std::unique_ptr<absolute_type> compute_absolute() const;

    void compute_absolute(ptr_param<absolute_type> output) const;

    [[nodiscard]] std::unique_ptr<complex_type> make_complex() const;

    void make_complex(ptr_param<complex_type> output) const;

    [[nodiscard]] std::unique_ptr<real_type> get_real() const;

    void get_real(ptr_param<real_type> output) const;

    [[nodiscard]] std::unique_ptr<real_type> get_imag() const;

    void get_imag(ptr_param<real_type> output) const;

    void convert_to(result_type* result) const override;

    void move_to(result_type* result) override;

    [[nodiscard]] device_view get_local_device_view();

    [[nodiscard]] const_device_view get_const_local_device_view() const;

protected:
    Cloneable* copy_from_impl(const Cloneable* other) override;

    Cloneable* move_from_impl(Cloneable* other) override;

    [[nodiscard]] std::unique_ptr<Cloneable> clone_impl(
        std::shared_ptr<const Executor> exec) const override;

    [[nodiscard]] std::unique_ptr<Cloneable> clone_impl() const override;

    [[nodiscard]] std::unique_ptr<Cloneable> create_default_impl()
        const override;

    [[nodiscard]] std::unique_ptr<Cloneable> create_default_impl(
        std::shared_ptr<const Executor> exec) const override;

    EnableMultiVector(std::shared_ptr<const Executor> exec, dim<2> size = {})
        : AbstractMultiVector(exec, size, precision_v<value_type>)
    {}

    // Concretized virtual function calls

    [[nodiscard]] virtual std::unique_ptr<ConcreteType>
    create_with_same_config_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<ConcreteType>
    create_with_type_of_impl(std::shared_ptr<const Executor> exec,
                             const dim<2>& global_size,
                             const dim<2>& local_size,
                             size_type stride) const = 0;

    [[nodiscard]] virtual std::unique_ptr<ConcreteType> create_subview_impl(
        local_span rows, local_span columns) = 0;

    [[nodiscard]] virtual std::unique_ptr<const ConcreteType>
    create_subview_impl(local_span rows, local_span columns) const = 0;

    [[nodiscard]] virtual std::unique_ptr<ConcreteType> create_subview_impl(
        local_span rows, local_span columns, dim<2> global_size) = 0;

    [[nodiscard]] virtual std::unique_ptr<const ConcreteType>
    create_subview_impl(local_span rows, local_span columns,
                        dim<2> global_size) const = 0;

    [[nodiscard]] virtual std::unique_ptr<const real_type>
    create_real_view_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<real_type>
    create_real_view_impl() = 0;

    [[nodiscard]] virtual std::unique_ptr<absolute_type> compute_absolute_impl()
        const = 0;

    virtual void compute_absolute_impl(absolute_type* result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<complex_type> make_complex_impl()
        const = 0;

    [[nodiscard]] virtual std::unique_ptr<real_type> get_real_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<real_type> get_imag_impl() const = 0;

    virtual void make_complex_impl(complex_type* result) const = 0;

    virtual void get_real_impl(real_type* result) const = 0;

    virtual void get_imag_impl(real_type* result) const = 0;

    virtual void fill_impl(value_type value) = 0;

    virtual void scale_impl(scaling_param<value_type> alpha) = 0;

    virtual void inv_scale_impl(scaling_param<value_type> alpha) = 0;

    virtual void add_scaled_impl(scaling_param<value_type> alpha,
                                 const ConcreteType* b) = 0;

    virtual void sub_scaled_impl(scaling_param<value_type> alpha,
                                 const ConcreteType* b) = 0;

    virtual void compute_dot_impl(const ConcreteType* b,
                                  matrix::MultiVector<value_type>* result,
                                  array<char>& tmp) const = 0;

    virtual void compute_conj_dot_impl(const ConcreteType* b,
                                       matrix::MultiVector<value_type>* result,
                                       array<char>& tmp) const = 0;

    virtual void compute_norm2_impl(norm_type* result,
                                    array<char>& tmp) const = 0;

    virtual void compute_squared_norm2_impl(norm_type* result,
                                            array<char>& tmp) const = 0;

    virtual void compute_norm1_impl(norm_type* result,
                                    array<char>& tmp) const = 0;

    [[nodiscard]] temporary_conversion<AbstractMultiVector> as_precision_impl(
        precision p) override;

    [[nodiscard]] temporary_conversion<const AbstractMultiVector>
    as_precision_impl(precision p) const override;

    virtual AbstractMultiVector::device_view<value_type>
    get_local_device_view_impl() = 0;

    virtual AbstractMultiVector::device_view<const value_type>
    get_const_local_device_view_impl() const = 0;

    [[nodiscard]] std::variant<
#if GINKGO_ENABLE_HALF
        AbstractMultiVector::device_view<half>,
        AbstractMultiVector::device_view<std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        AbstractMultiVector::device_view<bfloat16>,
        AbstractMultiVector::device_view<std::complex<bfloat16>>,
#endif
        AbstractMultiVector::device_view<float>,
        AbstractMultiVector::device_view<std::complex<float>>,
        AbstractMultiVector::device_view<double>,
        AbstractMultiVector::device_view<std::complex<double>>>
    get_local_device_view_generic_impl() override;

    [[nodiscard]] std::variant<
#if GINKGO_ENABLE_HALF
        AbstractMultiVector::device_view<const half>,
        AbstractMultiVector::device_view<const std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        AbstractMultiVector::device_view<const bfloat16>,
        AbstractMultiVector::device_view<const std::complex<bfloat16>>,
#endif
        AbstractMultiVector::device_view<const float>,
        AbstractMultiVector::device_view<const std::complex<float>>,
        AbstractMultiVector::device_view<const double>,
        AbstractMultiVector::device_view<const std::complex<double>>>
    get_const_local_device_view_generic_impl() const override;

    GKO_ENABLE_SELF(ConcreteType);

    // Overridden generic functions

    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    create_generic_with_same_config_impl() const final;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    create_generic_with_type_of_impl(std::shared_ptr<const Executor> exec,
                                     const dim<2>& global_size,
                                     const dim<2>& local_size,
                                     size_type stride) const final;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    create_subview_generic_impl(local_span rows, local_span columns) final;

    [[nodiscard]] std::unique_ptr<const AbstractMultiVector>
    create_subview_generic_impl(local_span rows,
                                local_span columns) const final;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    create_subview_generic_impl(local_span rows, local_span columns,
                                dim<2> global_size) final;

    [[nodiscard]] std::unique_ptr<const AbstractMultiVector>
    create_subview_generic_impl(local_span rows, local_span columns,
                                dim<2> global_size) const final;

    [[nodiscard]] std::unique_ptr<const AbstractMultiVector>
    create_real_view_generic_impl() const final;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    create_real_view_generic_impl() final;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    compute_absolute_generic_impl() const final;

    void compute_absolute_generic_impl(AbstractMultiVector* result) const final;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    make_complex_generic_impl() const final;

    void make_complex_generic_impl(AbstractMultiVector* result) const final;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> get_real_generic_impl()
        const final;

    void get_real_generic_impl(AbstractMultiVector* result) const final;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> get_imag_generic_impl()
        const final;

    void get_imag_generic_impl(AbstractMultiVector* result) const final;

    void fill_impl(any_scalar value) final;

    void scale_impl(const AbstractMultiVector* alpha) final;

    void inv_scale_impl(const AbstractMultiVector* alpha) final;

    void add_scaled_impl(const AbstractMultiVector* alpha,
                         const AbstractMultiVector* b) final;

    void sub_scaled_impl(const AbstractMultiVector* alpha,
                         const AbstractMultiVector* b) final;

    void compute_dot_impl(const AbstractMultiVector* b,
                          AbstractMultiVector* result,
                          array<char>& tmp) const final;

    void compute_conj_dot_impl(const AbstractMultiVector* b,
                               AbstractMultiVector* result,
                               array<char>& tmp) const final;

    void compute_norm2_impl(AbstractMultiVector* result,
                            array<char>& tmp) const final;

    void compute_squared_norm2_impl(AbstractMultiVector* result,
                                    array<char>& tmp) const final;

    void compute_norm1_impl(AbstractMultiVector* result,
                            array<char>& tmp) const final;
};


template <typename ConcreteType>
std::unique_ptr<ConcreteType>
EnableMultiVector<ConcreteType>::create_with_config_of(
    ptr_param<const ConcreteType> other)
{
    return static_cast<const EnableMultiVector*>(other.get())
        ->create_with_same_config_impl();
}


template <typename ConcreteType>
std::unique_ptr<ConcreteType>
EnableMultiVector<ConcreteType>::create_with_type_of(
    ptr_param<const ConcreteType> other, std::shared_ptr<const Executor> exec)
{
    return static_cast<const EnableMultiVector*>(other.get())
        ->create_with_type_of_impl(std::move(exec), {}, {}, 0);
}


template <typename ConcreteType>
std::unique_ptr<ConcreteType>
EnableMultiVector<ConcreteType>::create_with_type_of(
    ptr_param<const ConcreteType> other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size)
{
    GKO_ASSERT_EQUAL_COLS(global_size, local_size);
    return static_cast<const EnableMultiVector*>(other.get())
        ->create_with_type_of_impl(std::move(exec), global_size, local_size,
                                   local_size[1]);
}


template <typename ConcreteType>
std::unique_ptr<ConcreteType>
EnableMultiVector<ConcreteType>::create_with_type_of(
    ptr_param<const ConcreteType> other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size, size_type stride)
{
    return static_cast<const EnableMultiVector*>(other.get())
        ->create_with_type_of_impl(std::move(exec), global_size, local_size,
                                   stride);
}


template <typename ConcreteType>
std::unique_ptr<ConcreteType> EnableMultiVector<ConcreteType>::clone(
    std::shared_ptr<const Executor> exec) const
{
    return as<ConcreteType>(this->clone_impl(std::move(exec)));
}


template <typename ConcreteType>
std::unique_ptr<ConcreteType> EnableMultiVector<ConcreteType>::clone() const
{
    return as<ConcreteType>(this->clone_impl());
}


template <typename ConcreteType>
ConcreteType* EnableMultiVector<ConcreteType>::copy_from(
    ptr_param<const ConcreteType> other)
{
    return as<ConcreteType>(this->copy_from_impl(other.get()));
}


template <typename ConcreteType>
ConcreteType* EnableMultiVector<ConcreteType>::move_from(
    ptr_param<ConcreteType> other)
{
    return as<ConcreteType>(this->move_from_impl(other.get()));
}


template <typename ConcreteType>
std::unique_ptr<ConcreteType> EnableMultiVector<ConcreteType>::create_default()
{
    return as<ConcreteType>(this->create_default_impl());
}

template <typename ConcreteType>
std::unique_ptr<ConcreteType> EnableMultiVector<ConcreteType>::create_default(
    std::shared_ptr<const Executor> exec)
{
    return as<ConcreteType>(this->create_default_impl(std::move(exec)));
}


template <typename ConcreteType>
std::unique_ptr<ConcreteType> EnableMultiVector<ConcreteType>::create_subview(
    local_span rows, local_span columns)
{
    return this->create_subview_impl(rows, columns);
}


template <typename ConcreteType>
std::unique_ptr<const ConcreteType>
EnableMultiVector<ConcreteType>::create_subview(local_span rows,
                                                local_span columns) const
{
    return this->create_subview_impl(rows, columns);
}


template <typename ConcreteType>
std::unique_ptr<ConcreteType> EnableMultiVector<ConcreteType>::create_subview(
    local_span rows, local_span columns, dim<2> global_size)
{
    return this->create_subview_impl(rows, columns, global_size);
}


template <typename ConcreteType>
std::unique_ptr<const ConcreteType>
EnableMultiVector<ConcreteType>::create_subview(local_span rows,
                                                local_span columns,
                                                dim<2> global_size) const
{
    return this->create_subview_impl(rows, columns, global_size);
}


template <typename ConcreteType>
std::unique_ptr<const typename EnableMultiVector<ConcreteType>::real_type>
EnableMultiVector<ConcreteType>::create_real_view() const
{
    return this->create_real_view_impl();
}


template <typename ConcreteType>
std::unique_ptr<typename EnableMultiVector<ConcreteType>::real_type>
EnableMultiVector<ConcreteType>::create_real_view()
{
    return this->create_real_view_impl();
}


template <typename ConcreteType>
std::unique_ptr<typename EnableMultiVector<ConcreteType>::absolute_type>
EnableMultiVector<ConcreteType>::compute_absolute() const
{
    return this->compute_absolute_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_absolute(
    ptr_param<absolute_type> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    auto exec = this->get_executor();
    this->compute_absolute_impl(
        make_temporary_output_clone(exec, output.get()).get());
}


template <typename ConcreteType>
std::unique_ptr<typename EnableMultiVector<ConcreteType>::complex_type>
EnableMultiVector<ConcreteType>::make_complex() const
{
    return this->make_complex_impl();
}

template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::make_complex(
    ptr_param<complex_type> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    auto exec = this->get_executor();
    this->make_complex_impl(
        make_temporary_output_clone(exec, output.get()).get());
}


template <typename ConcreteType>
std::unique_ptr<typename EnableMultiVector<ConcreteType>::real_type>
EnableMultiVector<ConcreteType>::get_real() const
{
    return this->get_real_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::get_real(
    ptr_param<real_type> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    auto exec = this->get_executor();
    this->get_real_impl(make_temporary_output_clone(exec, output.get()).get());
}


template <typename ConcreteType>
std::unique_ptr<typename EnableMultiVector<ConcreteType>::real_type>
EnableMultiVector<ConcreteType>::get_imag() const
{
    return this->get_imag_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::get_imag(
    ptr_param<real_type> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    auto exec = this->get_executor();
    this->get_imag_impl(make_temporary_output_clone(exec, output.get()).get());
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::convert_to(result_type* result) const
{
    *result = *self();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::move_to(result_type* result)
{
    *result = std::move(*self());
}


template <typename ConcreteType>
typename EnableMultiVector<ConcreteType>::device_view
EnableMultiVector<ConcreteType>::get_local_device_view()
{
    return this->get_local_device_view_impl();
}


template <typename ConcreteType>
typename EnableMultiVector<ConcreteType>::const_device_view
EnableMultiVector<ConcreteType>::get_const_local_device_view() const
{
    return this->get_const_local_device_view_impl();
}


template <typename ConcreteType>
Cloneable* EnableMultiVector<ConcreteType>::copy_from_impl(
    const Cloneable* other)
{
    self()->template log<log::Logger::polymorphic_object_copy_started>(
        self()->get_executor().get(),
        dynamic_cast<const PolymorphicObject*>(other),
        dynamic_cast<const PolymorphicObject*>(this));
    as<ConvertibleTo<ConcreteType>>(other)->convert_to(self());
    self()->template log<log::Logger::polymorphic_object_copy_completed>(
        self()->get_executor().get(),
        dynamic_cast<const PolymorphicObject*>(other),
        dynamic_cast<const PolymorphicObject*>(this));
    return this;
}


template <typename ConcreteType>
Cloneable* EnableMultiVector<ConcreteType>::move_from_impl(Cloneable* other)
{
    self()->template log<log::Logger::polymorphic_object_move_started>(
        self()->get_executor().get(),
        dynamic_cast<const PolymorphicObject*>(other),
        dynamic_cast<const PolymorphicObject*>(this));
    as<ConvertibleTo<ConcreteType>>(other)->move_to(self());
    self()->template log<log::Logger::polymorphic_object_move_completed>(
        self()->get_executor().get(),
        dynamic_cast<const PolymorphicObject*>(other),
        dynamic_cast<const PolymorphicObject*>(this));
    return this;
}


template <typename ConcreteType>
std::unique_ptr<Cloneable> EnableMultiVector<ConcreteType>::clone_impl(
    std::shared_ptr<const Executor> exec) const
{
    auto result = this->create_default_impl(exec);
    result->copy_from(this);
    return result;
}


template <typename ConcreteType>
std::unique_ptr<Cloneable> EnableMultiVector<ConcreteType>::clone_impl() const
{
    auto result = this->create_default_impl();
    result->copy_from(this);
    return result;
}


template <typename ConcreteType>
std::unique_ptr<Cloneable>
EnableMultiVector<ConcreteType>::create_default_impl() const
{
    return this->create_with_type_of_impl(this->get_executor(), {}, {}, 0);
}


template <typename ConcreteType>
std::unique_ptr<Cloneable> EnableMultiVector<ConcreteType>::create_default_impl(
    std::shared_ptr<const Executor> exec) const
{
    return this->create_with_type_of_impl(exec, {}, {}, 0);
}


template <typename ConcreteType>
std::unique_ptr<AbstractMultiVector>
EnableMultiVector<ConcreteType>::create_generic_with_same_config_impl() const
{
    return this->create_with_same_config_impl();
}


template <typename ConcreteType>
std::unique_ptr<AbstractMultiVector>
EnableMultiVector<ConcreteType>::create_generic_with_type_of_impl(
    std::shared_ptr<const Executor> exec, const dim<2>& global_size,
    const dim<2>& local_size, size_type stride) const
{
    return this->create_with_type_of_impl(std::move(exec), global_size,
                                          local_size, stride);
}


template <typename ConcreteType>
std::unique_ptr<AbstractMultiVector>
EnableMultiVector<ConcreteType>::create_subview_generic_impl(local_span rows,
                                                             local_span columns)
{
    return this->create_subview_impl(rows, columns);
}


template <typename ConcreteType>
std::unique_ptr<const AbstractMultiVector>
EnableMultiVector<ConcreteType>::create_subview_generic_impl(
    local_span rows, local_span columns) const
{
    return this->create_subview_impl(rows, columns);
}


template <typename ConcreteType>
std::unique_ptr<AbstractMultiVector>
EnableMultiVector<ConcreteType>::create_subview_generic_impl(local_span rows,
                                                             local_span columns,
                                                             dim<2> global_size)
{
    return this->create_subview_impl(rows, columns, global_size);
}


template <typename ConcreteType>
std::unique_ptr<const AbstractMultiVector>
EnableMultiVector<ConcreteType>::create_subview_generic_impl(
    local_span rows, local_span columns, dim<2> global_size) const
{
    return this->create_subview_impl(rows, columns, global_size);
}


template <typename ConcreteType>
std::unique_ptr<const AbstractMultiVector>
EnableMultiVector<ConcreteType>::create_real_view_generic_impl() const
{
    return this->create_real_view_impl();
}


template <typename ConcreteType>
std::unique_ptr<AbstractMultiVector>
EnableMultiVector<ConcreteType>::create_real_view_generic_impl()
{
    return this->create_real_view_impl();
}


template <typename ConcreteType>
std::unique_ptr<AbstractMultiVector>
EnableMultiVector<ConcreteType>::compute_absolute_generic_impl() const
{
    return this->compute_absolute_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_absolute_generic_impl(
    AbstractMultiVector* result) const
{
    this->compute_absolute_impl(as<absolute_type>(result));
}

template <typename ConcreteType>
std::unique_ptr<AbstractMultiVector>
EnableMultiVector<ConcreteType>::make_complex_generic_impl() const
{
    return this->make_complex_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::make_complex_generic_impl(
    AbstractMultiVector* result) const
{
    this->make_complex_impl(as<complex_type>(result));
}


template <typename ConcreteType>
std::unique_ptr<AbstractMultiVector>
EnableMultiVector<ConcreteType>::get_real_generic_impl() const
{
    return this->get_real_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::get_real_generic_impl(
    AbstractMultiVector* result) const
{
    this->get_real_impl(as<absolute_type>(result));
}


template <typename ConcreteType>
std::unique_ptr<AbstractMultiVector>
EnableMultiVector<ConcreteType>::get_imag_generic_impl() const
{
    return this->get_imag_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::get_imag_generic_impl(
    AbstractMultiVector* result) const
{
    this->get_imag_impl(as<absolute_type>(result));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::fill_impl(any_scalar value)
{
    std::visit(
        [this](auto value_v) {
            using snd_value_type = std::decay_t<decltype(value_v)>;
            if constexpr (std::is_convertible_v<snd_value_type, value_type>) {
                this->fill_impl(static_cast<value_type>(value_v));
            } else {
                GKO_NOT_IMPLEMENTED;
            }
        },
        value.variant);
}


namespace detail {


template <typename ValueType, typename AlphaValueType>
struct scaling_factor_target {
    using type = std::conditional_t<is_complex<ValueType>() &&
                                        !is_complex<AlphaValueType>(),
                                    remove_complex<ValueType>, ValueType>;
};

template <typename ValueType, typename AlphaValueType>
using scaling_factor_target_type =
    typename scaling_factor_target<ValueType, AlphaValueType>::type;


}  // namespace detail


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::scale_impl(
    const AbstractMultiVector* alpha)
{
    std::visit(
        [this, alpha](auto p) {
            using alpha_value_type = std::decay_t<decltype(p)>;
            if constexpr (is_complex<alpha_value_type>() &&
                          !is_complex<value_type>()) {
                GKO_NOT_IMPLEMENTED;
            } else {
                auto alpha_v = as<matrix::MultiVector<alpha_value_type>>(alpha);
                this->scale_impl(scaling_param<value_type>{
                    alpha_v
                        ->template as_precision<
                            detail::scaling_factor_target_type<
                                value_type, alpha_value_type>>()
                        .get()});
            }
        },
        precision_to_variant(alpha->get_precision()));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::inv_scale_impl(
    const AbstractMultiVector* alpha)
{
    std::visit(
        [this, alpha](auto p) {
            using alpha_value_type = std::decay_t<decltype(p)>;
            if constexpr (is_complex<alpha_value_type>() &&
                          !is_complex<value_type>()) {
                GKO_NOT_IMPLEMENTED;
            } else {
                auto alpha_v = as<matrix::MultiVector<alpha_value_type>>(alpha);
                this->inv_scale_impl(scaling_param<value_type>{
                    alpha_v
                        ->template as_precision<
                            detail::scaling_factor_target_type<
                                value_type, alpha_value_type>>()
                        .get()});
            }
        },
        precision_to_variant(alpha->get_precision()));
}

template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::add_scaled_impl(
    const AbstractMultiVector* alpha, const AbstractMultiVector* b)
{
    std::visit(
        [this, alpha, b](auto p) {
            using alpha_value_type = std::decay_t<decltype(p)>;
            if constexpr (is_complex<alpha_value_type>() &&
                          !is_complex<value_type>()) {
                GKO_NOT_IMPLEMENTED;
            } else {
                auto alpha_v = as<matrix::MultiVector<alpha_value_type>>(alpha);
                this->add_scaled_impl(
                    scaling_param<value_type>{
                        alpha_v
                            ->template as_precision<
                                detail::scaling_factor_target_type<
                                    value_type, alpha_value_type>>()
                            .get()},
                    as<const ConcreteType>(
                        b->as_precision(this->get_precision()).get()));
            }
        },
        precision_to_variant(alpha->get_precision()));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::sub_scaled_impl(
    const AbstractMultiVector* alpha, const AbstractMultiVector* b)
{
    std::visit(
        [this, alpha, b](auto p) {
            using alpha_value_type = std::decay_t<decltype(p)>;
            if constexpr (is_complex<alpha_value_type>() &&
                          !is_complex<value_type>()) {
                GKO_NOT_IMPLEMENTED;
            } else {
                auto alpha_v = as<matrix::MultiVector<alpha_value_type>>(alpha);
                this->sub_scaled_impl(
                    scaling_param<value_type>{
                        alpha_v
                            ->template as_precision<
                                detail::scaling_factor_target_type<
                                    value_type, alpha_value_type>>()
                            .get()},
                    as<const ConcreteType>(
                        b->as_precision(this->get_precision()).get()));
            }
        },
        precision_to_variant(alpha->get_precision()));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_dot_impl(
    const AbstractMultiVector* b, AbstractMultiVector* result,
    array<char>& tmp) const
{
    this->compute_dot_impl(
        as<const ConcreteType>(b->as_precision(this->get_precision()).get()),
        as<matrix::MultiVector<value_type>>(
            result->as_precision(this->get_precision()).get()),
        tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_conj_dot_impl(
    const AbstractMultiVector* b, AbstractMultiVector* result,
    array<char>& tmp) const
{
    this->compute_conj_dot_impl(
        as<const ConcreteType>(b->as_precision(this->get_precision()).get()),
        as<matrix::MultiVector<value_type>>(
            result->as_precision(this->get_precision()).get()),
        tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_norm2_impl(
    AbstractMultiVector* result, array<char>& tmp) const
{
    this->compute_norm2_impl(
        as<norm_type>(
            result->as_precision(as_real(this->get_precision())).get()),
        tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_squared_norm2_impl(
    AbstractMultiVector* result, array<char>& tmp) const
{
    this->compute_squared_norm2_impl(
        as<norm_type>(
            result->as_precision(as_real(this->get_precision())).get()),
        tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_norm1_impl(
    AbstractMultiVector* result, array<char>& tmp) const
{
    this->compute_norm1_impl(
        as<norm_type>(
            result->as_precision(as_real(this->get_precision())).get()),
        tmp);
}


namespace detail {


template <typename SourceValueType, typename TargetValueType>
struct is_supported_conversion {
    static constexpr bool value =
        is_complex<SourceValueType>() == is_complex<TargetValueType>() ||
        (is_complex<SourceValueType>() &&
         std::is_same_v<SourceValueType, to_complex<TargetValueType>>);
};

template <typename SourceValueType, typename TargetValueType>
constexpr bool is_supported_conversion_v =
    is_supported_conversion<SourceValueType, TargetValueType>::value;


}  // namespace detail


template <typename ConcreteType>
temporary_conversion<AbstractMultiVector>
EnableMultiVector<ConcreteType>::as_precision_impl(precision p)
{
    return std::visit(
        [this](auto v) -> temporary_conversion<AbstractMultiVector> {
            using target_value_type = std::decay_t<decltype(v)>;
            if constexpr (detail::is_supported_conversion_v<
                              value_type, target_value_type>) {
                return temporary_conversion<AbstractMultiVector>::
                    create_from_derived(
                        self()->template as_precision<target_value_type>());
            } else {
                GKO_NOT_IMPLEMENTED;
            }
        },
        precision_to_variant(p));
}


template <typename ConcreteType>
temporary_conversion<const AbstractMultiVector>
EnableMultiVector<ConcreteType>::as_precision_impl(precision p) const
{
    return std::visit(
        [this](auto v) -> temporary_conversion<const AbstractMultiVector> {
            using target_value_type = std::decay_t<decltype(v)>;
            if constexpr (detail::is_supported_conversion_v<
                              value_type, target_value_type>) {
                return temporary_conversion<const AbstractMultiVector>::
                    create_from_derived(
                        self()->template as_precision<target_value_type>());
            } else {
                GKO_NOT_IMPLEMENTED;
            }
        },
        precision_to_variant(p));
}


template <typename ConcreteType>
std::variant<
#if GINKGO_ENABLE_HALF
    AbstractMultiVector::device_view<half>,
    AbstractMultiVector::device_view<std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
    AbstractMultiVector::device_view<bfloat16>,
    AbstractMultiVector::device_view<std::complex<bfloat16>>,
#endif
    AbstractMultiVector::device_view<float>,
    AbstractMultiVector::device_view<std::complex<float>>,
    AbstractMultiVector::device_view<double>,
    AbstractMultiVector::device_view<std::complex<double>>>
EnableMultiVector<ConcreteType>::get_local_device_view_generic_impl()
{
    return this->get_local_device_view_impl();
}


template <typename ConcreteType>
std::variant<
#if GINKGO_ENABLE_HALF
    AbstractMultiVector::device_view<const half>,
    AbstractMultiVector::device_view<const std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
    AbstractMultiVector::device_view<const bfloat16>,
    AbstractMultiVector::device_view<const std::complex<bfloat16>>,
#endif
    AbstractMultiVector::device_view<const float>,
    AbstractMultiVector::device_view<const std::complex<float>>,
    AbstractMultiVector::device_view<const double>,
    AbstractMultiVector::device_view<const std::complex<double>>>
EnableMultiVector<ConcreteType>::get_const_local_device_view_generic_impl()
    const
{
    return this->get_const_local_device_view_impl();
}


}  // namespace gko
