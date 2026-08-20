// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tuple>
#include <variant>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/multivector.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class MultiVector;


}


/**
 * Trait class containing type aliases required by EnableMultiVector.
 *
 * A specialization for a vector type has to define the following aliases:
 * - `value_type`: the value type of the vector,
 * - `absolute_value_type`: the real counterpart of `value_type`, used to store
 *   absolute values and norms,
 * - `absolute_type`: the vector type storing absolute values,
 * - `real_type`: the vector type storing real or imaginary parts,
 * - `complex_value_type`: the complex counterpart of `value_type`,
 * - `complex_type`: the vector type storing complex values.
 *
 * Vector types whose first template parameter is the value type are already
 * covered by the specialization below.
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
 * The stored pointer is never null, and it always points to a vector whose
 * precision is one of the alternatives of variant_type, i.e. no further
 * precision conversion is necessary. Implementations have to handle every
 * alternative, for example by using std::visit.
 *
 * @tparam ValueType The value type of the vector type on which to call the
 *                    scaling operation
 */
template <typename ValueType>
struct scaling_param {
    using variant_type = std::variant<const matrix::MultiVector<ValueType>*>;

    /** The scaling factor with one of the allowed precisions. */
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

    /** The scaling factor with one of the allowed precisions. */
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
 * arguments, such as add_scaled_impl(scaling_param<value_type> alpha,
 * const ConcreteType* b).
 *
 * If a requested combination of precisions is not supported, for example
 * scaling a real vector by a complex factor, then the dispatch throws
 * NotImplemented instead of forwarding to the concretized function.
 *
 * Besides the concretized virtual functions, ConcreteType has to provide:
 * - a specialization of vector_traits,
 * - a member function template `as_precision<ValueType>()` returning a
 *   temporary_conversion of a vector with the requested value type, which is
 *   used to implement AbstractMultiVector::as_precision, and
 * - copy and move assignment, which are used to implement the conversion to
 *   and from ConcreteType.
 *
 * @tparam ConcreteType Multi vector type to enable the mixin for.
 */
template <typename ConcreteType>
class EnableMultiVector : public AbstractMultiVector,
                          public ConvertibleTo<ConcreteType> {
public:
    /** The traits providing the types related to ConcreteType. */
    using traits = vector_traits<ConcreteType>;
    /** The value type of the stored elements. */
    using value_type = typename traits::value_type;
    /** The real counterpart of value_type. */
    using absolute_value_type = typename traits::absolute_value_type;
    /** The vector type holding absolute values of this vector. */
    using absolute_type = typename traits::absolute_type;
    /** The vector type holding real or imaginary parts of this vector. */
    using real_type = typename traits::real_type;
    /** The vector type holding a complex version of this vector. */
    using complex_type = typename traits::complex_type;
    /** The target type of the conversions provided by this mixin. */
    using result_type = ConcreteType;
    /** The vector type used to store the results of reductions to norms. */
    using norm_type = matrix::MultiVector<absolute_value_type>;
    /** The mutable device view type of the local data. */
    using device_view = AbstractMultiVector::device_view<value_type>;
    /** The immutable device view type of the local data. */
    using const_device_view =
        AbstractMultiVector::device_view<const value_type>;

    using ConvertibleTo<result_type>::convert_to;
    using ConvertibleTo<result_type>::move_to;

    /**
     * @copydoc AbstractMultiVector::create_with_config_of(ptr_param<const
     *          ConcreteType>)
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] static std::unique_ptr<ConcreteType> create_with_config_of(
        ptr_param<const ConcreteType> other);

    /**
     * @copydoc AbstractMultiVector::create_with_type_of(ptr_param<const
     *          ConcreteType>, std::shared_ptr<const Executor>)
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] static std::unique_ptr<ConcreteType> create_with_type_of(
        ptr_param<const ConcreteType> other,
        std::shared_ptr<const Executor> exec);

    /**
     * @copydoc AbstractMultiVector::create_with_type_of(ptr_param<const
     *          ConcreteType>, std::shared_ptr<const Executor>, const dim<2>&,
     *          const dim<2>&)
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] static std::unique_ptr<ConcreteType> create_with_type_of(
        ptr_param<const ConcreteType> other,
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size);

    /**
     * @copydoc AbstractMultiVector::create_with_type_of(ptr_param<const
     *          ConcreteType>, std::shared_ptr<const Executor>, const dim<2>&,
     *          const dim<2>&, size_type)
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] static std::unique_ptr<ConcreteType> create_with_type_of(
        ptr_param<const ConcreteType> other,
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size, size_type stride);

    /**
     * @copydoc AbstractMultiVector::clone(std::shared_ptr<const Executor>)
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] std::unique_ptr<ConcreteType> clone(
        std::shared_ptr<const Executor> exec) const;

    /**
     * @copydoc AbstractMultiVector::clone()
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] std::unique_ptr<ConcreteType> clone() const;

    /**
     * @copydoc AbstractMultiVector::copy_from(ptr_param<const ConcreteType>)
     *
     * @note Takes an object with ConcreteType as input
     */
    ConcreteType* copy_from(ptr_param<const ConcreteType> other);


    /**
     * @copydoc AbstractMultiVector::move_from(ptr_param<ConcreteType>)
     *
     * @note Takes an object with ConcreteType as input
     */
    ConcreteType* move_from(ptr_param<ConcreteType> other);

    /**
     * @copydoc AbstractMultiVector::create_default()
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] std::unique_ptr<ConcreteType> create_default();

    /**
     * @copydoc AbstractMultiVector::create_default(ptr_param<const Executor>)
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] std::unique_ptr<ConcreteType> create_default(
        std::shared_ptr<const Executor> exec);

    /**
     * @copydoc AbstractMultiVector::create_subview(local_span, local_span)
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] std::unique_ptr<ConcreteType> create_subview(
        local_span rows, local_span columns);

    /**
     * @copydoc AbstractMultiVector::create_subview(local_span, local_span)
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] std::unique_ptr<const ConcreteType> create_subview(
        local_span rows, local_span columns) const;

    /**
     * @copydoc AbstractMultiVector::create_subview(local_span, local_span,
     *          dim<2>)
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] std::unique_ptr<ConcreteType> create_subview(
        local_span rows, local_span columns, dim<2> global_size);

    /**
     * @copydoc AbstractMultiVector::create_subview(local_span, local_span,
     *          dim<2>)
     *
     * @note Returns an object with ConcreteType
     */
    [[nodiscard]] std::unique_ptr<const ConcreteType> create_subview(
        local_span rows, local_span columns, dim<2> global_size) const;

    /**
     * @copydoc AbstractMultiVector::create_real_view() const
     *
     * @note Returns an object with real_type
     */
    [[nodiscard]] std::unique_ptr<const real_type> create_real_view() const;

    /**
     * @copydoc create_real_view() const
     *
     * @note Returns an object with real_type
     */
    [[nodiscard]] std::unique_ptr<real_type> create_real_view();

    /**
     * @copydoc AbstractMultiVector::compute_absolute()
     *
     * @note Returns an object with absolute_type
     */
    [[nodiscard]] std::unique_ptr<absolute_type> compute_absolute() const;

    /**
     * @copydoc
     * AbstractMultiVector::compute_absolute(ptr_param<AbstractMultiVector>)
     *
     * @note Takes an object with absolute_type as output, so the precision is
     *       checked at compile time
     */
    void compute_absolute(ptr_param<absolute_type> output) const;

    /**
     * @copydoc AbstractMultiVector::make_complex()
     *
     * @note Returns an object with complex_type
     */
    [[nodiscard]] std::unique_ptr<complex_type> make_complex() const;

    /**
     * @copydoc
     * AbstractMultiVector::make_complex(ptr_param<AbstractMultiVector>)
     *
     * @note Takes an object with complex_type as output, so the precision is
     *       checked at compile time
     */
    void make_complex(ptr_param<complex_type> output) const;

    /**
     * @copydoc AbstractMultiVector::get_real()
     *
     * @note Returns an object with real_type
     */
    [[nodiscard]] std::unique_ptr<real_type> get_real() const;

    /**
     * @copydoc AbstractMultiVector::get_real(ptr_param<AbstractMultiVector>)
     *
     * @note Takes an object with real_type as output, so the precision is
     *       checked at compile time
     */
    void get_real(ptr_param<real_type> output) const;

    /**
     * @copydoc AbstractMultiVector::get_imag()
     *
     * @note Returns an object with real_type
     */
    [[nodiscard]] std::unique_ptr<real_type> get_imag() const;

    /**
     * @copydoc AbstractMultiVector::get_imag(ptr_param<AbstractMultiVector>)
     *
     * @note Takes an object with real_type as output, so the precision is
     *       checked at compile time
     */
    void get_imag(ptr_param<real_type> output) const;

    /**
     * @copydoc ConvertibleTo::convert_to(result_type*)
     *
     * @note This uses the copy assignment of ConcreteType.
     */
    void convert_to(result_type* result) const override;

    /**
     * @copydoc ConvertibleTo::move_to(result_type*)
     *
     * @note This uses the move assignment of ConcreteType.
     */
    void move_to(result_type* result) override;

    /**
     * Gets a local device view of this vector.
     *
     * @note In contrast to AbstractMultiVector::get_local_device_view, the
     *       value type is already known, so it does not have to be specified
     *       and no runtime check on the precision is necessary.
     *
     * @return A device view of the local data of this vector
     */
    [[nodiscard]] device_view get_local_device_view();

    /** @copydoc get_local_device_view() */
    [[nodiscard]] const_device_view get_const_local_device_view() const;

protected:
    /**
     * Constructs a vector on the given executor with the given size.
     *
     * The precision is deduced from value_type, so derived classes don't have
     * to pass it on.
     *
     * @param exec  The executor of this vector
     * @param size  The size of this vector
     */
    EnableMultiVector(std::shared_ptr<const Executor> exec, dim<2> size = {})
        : AbstractMultiVector(exec, size, precision_v<value_type>)
    {}

    Cloneable* copy_from_impl(const Cloneable* other) override;

    Cloneable* move_from_impl(Cloneable* other) override;

    [[nodiscard]] std::unique_ptr<Cloneable> clone_impl(
        std::shared_ptr<const Executor> exec) const override;

    [[nodiscard]] std::unique_ptr<Cloneable> clone_impl() const override;

    [[nodiscard]] std::unique_ptr<Cloneable> create_default_impl()
        const override;

    [[nodiscard]] std::unique_ptr<Cloneable> create_default_impl(
        std::shared_ptr<const Executor> exec) const override;

    // Concretized virtual function calls
    //
    // These are the functions that classes using this mixin have to implement.
    // All arguments passed to them are already concretized, i.e. they are
    // stored on the executor of this vector, their dimensions have been
    // checked, and their precision matches the one required by the operation.
    // Thus, implementations don't have to handle any conversion themselves.

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
    //
    // These implement the type-erased interface of AbstractMultiVector by
    // dispatching to the concretized functions above, see the class
    // documentation. They are final, so classes using this mixin can't
    // override them and have to implement the concretized functions instead.
    // Precision combinations that can't be dispatched, e.g. scaling a real
    // vector by a complex factor, throw NotImplemented.

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


/**
 * The value type a scaling factor with AlphaValueType is converted to, before
 * it is passed to a vector with ValueType.
 *
 * A real scaling factor for a complex vector stays real, every other factor is
 * converted to the vector's value type.
 */
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
                // can't scale a real vector with a complex scalar
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
                // can't scale a real vector with a complex scalar
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
                // can't scale a real vector with a complex scalar
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
                // can't scale a real vector with a complex scalar
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


template <typename ConcreteType>
temporary_conversion<AbstractMultiVector>
EnableMultiVector<ConcreteType>::as_precision_impl(precision p)
{
    return std::visit(
        [this](auto v) -> temporary_conversion<AbstractMultiVector> {
            using target_value_type = std::decay_t<decltype(v)>;
            if constexpr (is_complex<value_type>() ==
                          is_complex<target_value_type>()) {
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
            if constexpr (is_complex<value_type>() ==
                          is_complex<target_value_type>()) {
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
