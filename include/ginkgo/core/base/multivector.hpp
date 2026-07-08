// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tuple>
#include <variant>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/matrix/device_views.hpp>

#include "ginkgo/core/base/type_traits.hpp"


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;


}

using supported_value_types =
    std::tuple<double, float, std::complex<double>, std::complex<float>
#if GINKGO_ENABLE_HALF
               ,
               half, std::complex<half>
#endif
#if GINKGO_ENABLE_BFLOAT16
               ,
               bfloat16, std::complex<bfloat16>
#endif
               >;


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


class any_scalar : public syn::variant_from_tuple<supported_value_types> {
public:
    using base_type = syn::variant_from_tuple<supported_value_types>;

    template <typename T, std::enable_if_t<
                              std::is_constructible_v<base_type, T&&>, int> = 0>
    any_scalar(T&& value) : base_type(std::forward<T>(value))
    {}

    template <
        typename T,
        std::enable_if_t<!std::is_constructible_v<base_type, T&&>, int> = 1>
    any_scalar(T&& value) : base_type(static_cast<double>(value))
    {}
};

/**
 * The allowed Dense<T> type to be used in scaling operations, e.g. in
 * scaled_add.
 *
 * @tparam ValueType The value type of the vector type on which to call the
 *                    scaling operation
 */
template <typename ValueType>
struct scaling_param : std::variant<const matrix::Dense<ValueType>*> {};

/**
 * Specialization for complex types, allows both real and complex scaling
 * parameters.
 */
template <typename ValueType>
struct scaling_param<std::complex<ValueType>>
    : std::variant<const matrix::Dense<ValueType>*,
                   const matrix::Dense<std::complex<ValueType>>*> {};


template <typename ConcreteType>
struct vector_traits;

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


class MultiVector : public LinOp, public Cloneable {
public:
    template <typename ValueType>
    using device_view = matrix::view::dense<ValueType>;

    [[nodiscard]] static std::unique_ptr<MultiVector> create_with_config_of(
        ptr_param<const MultiVector> other);

    [[nodiscard]] static std::unique_ptr<MultiVector> create_with_type_of(
        ptr_param<const MultiVector> other,
        std::shared_ptr<const Executor> exec);

    [[nodiscard]] static std::unique_ptr<MultiVector> create_with_type_of(
        ptr_param<const MultiVector> other,
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size);

    [[nodiscard]] static std::unique_ptr<MultiVector> create_with_type_of(
        ptr_param<const MultiVector> other,
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size, size_type stride);

    [[nodiscard]] std::unique_ptr<MultiVector> clone(
        std::shared_ptr<const Executor> exec) const;

    [[nodiscard]] std::unique_ptr<MultiVector> clone() const;

    MultiVector* copy_from(ptr_param<const MultiVector> other);

    MultiVector* move_from(ptr_param<MultiVector> other);

    [[nodiscard]] std::unique_ptr<MultiVector> create_default(
        std::shared_ptr<const Executor> exec) const;

    [[nodiscard]] std::unique_ptr<MultiVector> create_default() const;

    [[nodiscard]] std::unique_ptr<MultiVector> compute_absolute() const;

    void compute_absolute(ptr_param<MultiVector> output) const;

    void compute_absolute_inplace();

    [[nodiscard]] std::unique_ptr<MultiVector> make_complex() const;

    void make_complex(ptr_param<MultiVector> result) const;

    [[nodiscard]] std::unique_ptr<MultiVector> get_real() const;

    void get_real(ptr_param<MultiVector> result) const;

    [[nodiscard]] std::unique_ptr<MultiVector> get_imag() const;

    void get_imag(ptr_param<MultiVector> result) const;

    void fill(any_scalar value);

    // Todo figure out real * complex, since this has a special dense kernel
    void scale(ptr_param<const MultiVector> alpha);

    void inv_scale(ptr_param<const MultiVector> alpha);

    void add_scaled(ptr_param<const MultiVector> alpha,
                    ptr_param<const MultiVector> b);

    void sub_scaled(ptr_param<const MultiVector> alpha,
                    ptr_param<const MultiVector> b);

    // @todo: the result can only be one of Dense<...>
    void compute_dot(ptr_param<const MultiVector> b,
                     ptr_param<MultiVector> result) const;

    void compute_dot(ptr_param<const MultiVector> b,
                     ptr_param<MultiVector> result, array<char>& tmp) const;

    void compute_conj_dot(ptr_param<const MultiVector> b,
                          ptr_param<MultiVector> result) const;

    void compute_conj_dot(ptr_param<const MultiVector> b,
                          ptr_param<MultiVector> result,
                          array<char>& tmp) const;

    void compute_norm2(ptr_param<MultiVector> result) const;

    void compute_norm2(ptr_param<MultiVector> result, array<char>& tmp) const;

    void compute_squared_norm2(ptr_param<MultiVector> result) const;

    void compute_squared_norm2(ptr_param<MultiVector> result,
                               array<char>& tmp) const;

    void compute_norm1(ptr_param<MultiVector> result) const;

    void compute_norm1(ptr_param<MultiVector> result, array<char>& tmp) const;

    [[nodiscard]] std::unique_ptr<const MultiVector> create_real_view() const;

    [[nodiscard]] std::unique_ptr<MultiVector> create_real_view();

    [[nodiscard]] std::unique_ptr<MultiVector> create_subview(
        local_span rows, local_span columns);

    [[nodiscard]] std::unique_ptr<const MultiVector> create_subview(
        local_span rows, local_span columns) const;

    [[nodiscard]] std::unique_ptr<MultiVector> create_subview(
        local_span rows, local_span columns, dim<2> global_size);

    [[nodiscard]] std::unique_ptr<const MultiVector> create_subview(
        local_span rows, local_span columns, dim<2> global_size) const;

    template <typename ValueType>
    [[nodiscard]] device_view<ValueType> get_local_device_view();

    template <typename ValueType>
    [[nodiscard]] device_view<const ValueType> get_const_local_device_view()
        const;

    /**
     * Converts this vector into another precision.
     *
     * Allowed conversions:
     * - bf16 <-> fp16 <-> fp32 <-> fp64
     * - complex_bf16 <-> complex_fp16 <-> complex_fp32 <-> complex_fp64
     * - complex_P -> P, where P = bf16, fp16, fp32, fp64
     *
     * @param p The requested precision
     * @return A vector with the requested precision
     */
    [[nodiscard]] detail::temporary_conversion<MultiVector> as_precision(
        precision p);

    [[nodiscard]] detail::temporary_conversion<MultiVector> as_precision(
        ptr_param<const MultiVector> p);

    [[nodiscard]] detail::temporary_conversion<MultiVector> as_precision(
        ptr_param<const LinOp> p);

    [[nodiscard]] detail::temporary_conversion<const MultiVector> as_precision(
        precision p) const;

    [[nodiscard]] detail::temporary_conversion<const MultiVector> as_precision(
        ptr_param<const MultiVector> p) const;

    [[nodiscard]] detail::temporary_conversion<const MultiVector> as_precision(
        ptr_param<const LinOp> p) const;

protected:
    explicit MultiVector(std::shared_ptr<const Executor> exec,
                         const dim<2>& size = dim<2>{},
                         precision p = precision::none);

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_generic_with_same_config_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_generic_with_type_of_impl(std::shared_ptr<const Executor> exec,
                                     const dim<2>& global_size,
                                     const dim<2>& local_size,
                                     size_type stride) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    compute_absolute_generic_impl() const = 0;

    virtual void compute_absolute_generic_impl(MultiVector* result) const = 0;

    virtual void compute_absolute_inplace_impl() = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    make_complex_generic_impl() const = 0;

    virtual void make_complex_generic_impl(MultiVector* result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> get_real_generic_impl()
        const = 0;

    virtual void get_real_generic_impl(MultiVector* result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> get_imag_generic_impl()
        const = 0;

    virtual void get_imag_generic_impl(MultiVector* result) const = 0;

    virtual void fill_impl(any_scalar value) = 0;

    // @todo: need to fix alpha to a our dense type
    virtual void scale_impl(const MultiVector* alpha) = 0;

    virtual void inv_scale_impl(const MultiVector* alpha) = 0;

    virtual void add_scaled_impl(const MultiVector* alpha,
                                 const MultiVector* b) = 0;

    virtual void sub_scaled_impl(const MultiVector* alpha,
                                 const MultiVector* b) = 0;

    virtual void compute_dot_impl(const MultiVector* b,
                                  MultiVector* result) const = 0;

    virtual void compute_dot_impl(const MultiVector* b, MultiVector* result,
                                  array<char>& tmp) const = 0;

    virtual void compute_conj_dot_impl(const MultiVector* b,
                                       MultiVector* result) const = 0;

    virtual void compute_conj_dot_impl(const MultiVector* b,
                                       MultiVector* result,
                                       array<char>& tmp) const = 0;

    virtual void compute_norm2_impl(MultiVector* result) const = 0;

    virtual void compute_norm2_impl(MultiVector* result,
                                    array<char>& tmp) const = 0;

    virtual void compute_squared_norm2_impl(MultiVector* result) const = 0;

    virtual void compute_squared_norm2_impl(MultiVector* result,
                                            array<char>& tmp) const = 0;

    virtual void compute_norm1_impl(MultiVector* result) const = 0;

    virtual void compute_norm1_impl(MultiVector* result,
                                    array<char>& tmp) const = 0;

    [[nodiscard]] virtual std::unique_ptr<const MultiVector>
    create_real_view_generic_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_real_view_generic_impl() = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_subview_generic_impl(local_span rows, local_span columns) = 0;

    [[nodiscard]] virtual std::unique_ptr<const MultiVector>
    create_subview_generic_impl(local_span rows, local_span columns) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_subview_generic_impl(local_span rows, local_span columns,
                                dim<2> global_size) = 0;

    [[nodiscard]] virtual std::unique_ptr<const MultiVector>
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

    [[nodiscard]] virtual gko::detail::temporary_conversion<MultiVector>
    as_precision_impl(precision p) = 0;

    [[nodiscard]] virtual gko::detail::temporary_conversion<const MultiVector>
    as_precision_impl(precision p) const = 0;
};


template <typename ConcreteType>
class EnableMultiVector : public MultiVector,
                          public ConvertibleTo<ConcreteType> {
public:
    using traits = vector_traits<ConcreteType>;
    using value_type = typename traits::value_type;
    using absolute_value_type = typename traits::absolute_value_type;
    using absolute_type = typename traits::absolute_type;
    using real_type = typename traits::real_type;
    using complex_type = typename traits::complex_type;
    using result_type = ConcreteType;
    using device_view = MultiVector::device_view<value_type>;
    using const_device_view = MultiVector::device_view<const value_type>;

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
        : MultiVector(exec, size, type_to_precision<value_type>)
    {}

    // Concretized function calls
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
                                  matrix::Dense<value_type>* result) const = 0;

    virtual void compute_dot_impl(const ConcreteType* b,
                                  matrix::Dense<value_type>* result,
                                  array<char>& tmp) const = 0;

    virtual void compute_conj_dot_impl(
        const ConcreteType* b, matrix::Dense<value_type>* result) const = 0;

    virtual void compute_conj_dot_impl(const ConcreteType* b,
                                       matrix::Dense<value_type>* result,
                                       array<char>& tmp) const = 0;

    virtual void compute_norm2_impl(
        matrix::Dense<absolute_value_type>* result) const = 0;

    virtual void compute_norm2_impl(matrix::Dense<absolute_value_type>* result,
                                    array<char>& tmp) const = 0;

    virtual void compute_squared_norm2_impl(
        matrix::Dense<absolute_value_type>* result) const = 0;

    virtual void compute_squared_norm2_impl(
        matrix::Dense<absolute_value_type>* result, array<char>& tmp) const = 0;

    virtual void compute_norm1_impl(
        matrix::Dense<absolute_value_type>* result) const = 0;

    virtual void compute_norm1_impl(matrix::Dense<absolute_value_type>* result,
                                    array<char>& tmp) const = 0;

    [[nodiscard]] detail::temporary_conversion<MultiVector> as_precision_impl(
        precision p) override;

    [[nodiscard]] detail::temporary_conversion<const MultiVector>
    as_precision_impl(precision p) const override;

    virtual MultiVector::device_view<value_type>
    get_local_device_view_impl() = 0;

    virtual MultiVector::device_view<const value_type>
    get_const_local_device_view_impl() const = 0;

    [[nodiscard]] std::variant<
#if GINKGO_ENABLE_HALF
        MultiVector::device_view<half>,
        MultiVector::device_view<std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        MultiVector::device_view<bfloat16>,
        MultiVector::device_view<std::complex<bfloat16>>,
#endif
        MultiVector::device_view<float>,
        MultiVector::device_view<std::complex<float>>,
        MultiVector::device_view<double>,
        MultiVector::device_view<std::complex<double>>>
    get_local_device_view_generic_impl() override;

    [[nodiscard]] std::variant<
#if GINKGO_ENABLE_HALF
        MultiVector::device_view<const half>,
        MultiVector::device_view<const std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        MultiVector::device_view<const bfloat16>,
        MultiVector::device_view<const std::complex<bfloat16>>,
#endif
        MultiVector::device_view<const float>,
        MultiVector::device_view<const std::complex<float>>,
        MultiVector::device_view<const double>,
        MultiVector::device_view<const std::complex<double>>>
    get_const_local_device_view_generic_impl() const override;

    GKO_ENABLE_SELF(ConcreteType);

private:
    [[nodiscard]] std::unique_ptr<MultiVector>
    create_generic_with_same_config_impl() const final;

    [[nodiscard]] std::unique_ptr<MultiVector> create_generic_with_type_of_impl(
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size, size_type stride) const final;

    [[nodiscard]] std::unique_ptr<MultiVector> create_subview_generic_impl(
        local_span rows, local_span columns) final;

    [[nodiscard]] std::unique_ptr<const MultiVector>
    create_subview_generic_impl(local_span rows,
                                local_span columns) const final;

    [[nodiscard]] std::unique_ptr<MultiVector> create_subview_generic_impl(
        local_span rows, local_span columns, dim<2> global_size) final;

    [[nodiscard]] std::unique_ptr<const MultiVector>
    create_subview_generic_impl(local_span rows, local_span columns,
                                dim<2> global_size) const final;

    [[nodiscard]] std::unique_ptr<const MultiVector>
    create_real_view_generic_impl() const final;

    [[nodiscard]] std::unique_ptr<MultiVector> create_real_view_generic_impl()
        final;

    [[nodiscard]] std::unique_ptr<MultiVector> compute_absolute_generic_impl()
        const final;

    void compute_absolute_generic_impl(MultiVector* result) const final;

    [[nodiscard]] std::unique_ptr<MultiVector> make_complex_generic_impl()
        const final;

    void make_complex_generic_impl(MultiVector* result) const final;

    [[nodiscard]] std::unique_ptr<MultiVector> get_real_generic_impl()
        const final;

    void get_real_generic_impl(MultiVector* result) const final;

    [[nodiscard]] std::unique_ptr<MultiVector> get_imag_generic_impl()
        const final;

    void get_imag_generic_impl(MultiVector* result) const final;

    void fill_impl(any_scalar value) override final;

    void scale_impl(const MultiVector* alpha) override final;

    void inv_scale_impl(const MultiVector* alpha) override final;

    void add_scaled_impl(const MultiVector* alpha, const MultiVector* b) final;

    void sub_scaled_impl(const MultiVector* alpha, const MultiVector* b) final;

    void compute_dot_impl(const MultiVector* b,
                          MultiVector* result) const final;

    void compute_dot_impl(const MultiVector* b, MultiVector* result,
                          array<char>& tmp) const final;

    void compute_conj_dot_impl(const MultiVector* b,
                               MultiVector* result) const final;

    void compute_conj_dot_impl(const MultiVector* b, MultiVector* result,
                               array<char>& tmp) const final;

    void compute_norm2_impl(MultiVector* result) const final;

    void compute_norm2_impl(MultiVector* result, array<char>& tmp) const final;

    void compute_squared_norm2_impl(MultiVector* result) const final;

    void compute_squared_norm2_impl(MultiVector* result,
                                    array<char>& tmp) const final;

    void compute_norm1_impl(MultiVector* result) const final;

    void compute_norm1_impl(MultiVector* result, array<char>& tmp) const final;
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
    this->compute_absolute_impl(output.get());
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
    this->make_complex_impl(output.get());
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
    this->get_real_impl(output.get());
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
    this->get_imag_impl(output.get());
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
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::create_generic_with_same_config_impl() const
{
    return this->create_with_same_config_impl();
}


template <typename ConcreteType>
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::create_generic_with_type_of_impl(
    std::shared_ptr<const Executor> exec, const dim<2>& global_size,
    const dim<2>& local_size, size_type stride) const
{
    return this->create_with_type_of_impl(std::move(exec), global_size,
                                          local_size, stride);
}


template <typename ConcreteType>
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::create_subview_generic_impl(local_span rows,
                                                             local_span columns)
{
    return this->create_subview_impl(rows, columns);
}


template <typename ConcreteType>
std::unique_ptr<const MultiVector>
EnableMultiVector<ConcreteType>::create_subview_generic_impl(
    local_span rows, local_span columns) const
{
    return this->create_subview_impl(rows, columns);
}


template <typename ConcreteType>
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::create_subview_generic_impl(local_span rows,
                                                             local_span columns,
                                                             dim<2> global_size)
{
    return this->create_subview_impl(rows, columns, global_size);
}


template <typename ConcreteType>
std::unique_ptr<const MultiVector>
EnableMultiVector<ConcreteType>::create_subview_generic_impl(
    local_span rows, local_span columns, dim<2> global_size) const
{
    return this->create_subview_impl(rows, columns, global_size);
}


template <typename ConcreteType>
std::unique_ptr<const MultiVector>
EnableMultiVector<ConcreteType>::create_real_view_generic_impl() const
{
    return this->create_real_view_impl();
}


template <typename ConcreteType>
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::create_real_view_generic_impl()
{
    return this->create_real_view_impl();
}


template <typename ConcreteType>
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::compute_absolute_generic_impl() const
{
    return this->compute_absolute_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_absolute_generic_impl(
    MultiVector* result) const
{
    this->compute_absolute_impl(as<absolute_type>(result));
}

template <typename ConcreteType>
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::make_complex_generic_impl() const
{
    return this->make_complex_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::make_complex_generic_impl(
    MultiVector* result) const
{
    this->make_complex_impl(as<complex_type>(result));
}


template <typename ConcreteType>
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::get_real_generic_impl() const
{
    return this->get_real_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::get_real_generic_impl(
    MultiVector* result) const
{
    this->get_real_impl(as<absolute_type>(result));
}


template <typename ConcreteType>
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::get_imag_generic_impl() const
{
    return this->get_imag_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::get_imag_generic_impl(
    MultiVector* result) const
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
        value);
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
void EnableMultiVector<ConcreteType>::scale_impl(const MultiVector* alpha)
{
    std::visit(
        [this, alpha](auto p) {
            using alpha_value_type = std::decay_t<decltype(p)>;
            if constexpr (is_complex<alpha_value_type>() &&
                          !is_complex<value_type>()) {
                GKO_NOT_IMPLEMENTED;
            } else {
                auto alpha_v = as<matrix::Dense<alpha_value_type>>(alpha);
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
void EnableMultiVector<ConcreteType>::inv_scale_impl(const MultiVector* alpha)
{
    std::visit(
        [this, alpha](auto p) {
            using alpha_value_type = std::decay_t<decltype(p)>;
            if constexpr (is_complex<alpha_value_type>() &&
                          !is_complex<value_type>()) {
                GKO_NOT_IMPLEMENTED;
            } else {
                auto alpha_v = as<matrix::Dense<alpha_value_type>>(alpha);
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
void EnableMultiVector<ConcreteType>::add_scaled_impl(const MultiVector* alpha,
                                                      const MultiVector* b)
{
    std::visit(
        [this, alpha, b](auto p) {
            using alpha_value_type = std::decay_t<decltype(p)>;
            if constexpr (is_complex<alpha_value_type>() &&
                          !is_complex<value_type>()) {
                GKO_NOT_IMPLEMENTED;
            } else {
                auto alpha_v = as<matrix::Dense<alpha_value_type>>(alpha);
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
void EnableMultiVector<ConcreteType>::sub_scaled_impl(const MultiVector* alpha,
                                                      const MultiVector* b)
{
    std::visit(
        [this, alpha, b](auto p) {
            using alpha_value_type = std::decay_t<decltype(p)>;
            if constexpr (is_complex<alpha_value_type>() &&
                          !is_complex<value_type>()) {
                GKO_NOT_IMPLEMENTED;
            } else {
                auto alpha_v = as<matrix::Dense<alpha_value_type>>(alpha);
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
    const MultiVector* b, MultiVector* result) const
{
    this->compute_dot_impl(
        as<const ConcreteType>(b->as_precision(this->get_precision()).get()),
        as<matrix::Dense<value_type>>(
            result->as_precision(this->get_precision()).get()));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_dot_impl(const MultiVector* b,
                                                       MultiVector* result,
                                                       array<char>& tmp) const
{
    this->compute_dot_impl(
        as<const ConcreteType>(b->as_precision(this->get_precision()).get()),
        as<matrix::Dense<value_type>>(
            result->as_precision(this->get_precision()).get()),
        tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_conj_dot_impl(
    const MultiVector* b, MultiVector* result) const
{
    this->compute_conj_dot_impl(
        as<const ConcreteType>(b->as_precision(this->get_precision()).get()),
        as<matrix::Dense<value_type>>(
            result->as_precision(this->get_precision()).get()));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_conj_dot_impl(
    const MultiVector* b, MultiVector* result, array<char>& tmp) const
{
    this->compute_conj_dot_impl(
        as<const ConcreteType>(b->as_precision(this->get_precision()).get()),
        as<matrix::Dense<value_type>>(
            result->as_precision(this->get_precision()).get()),
        tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_norm2_impl(
    MultiVector* result) const
{
    this->compute_norm2_impl(as<matrix::Dense<absolute_value_type>>(
        result->as_precision(as_real(this->get_precision())).get()));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_norm2_impl(MultiVector* result,
                                                         array<char>& tmp) const
{
    this->compute_norm2_impl(
        as<matrix::Dense<absolute_value_type>>(
            result->as_precision(as_real(this->get_precision())).get()),
        tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_squared_norm2_impl(
    MultiVector* result) const
{
    this->compute_squared_norm2_impl(as<matrix::Dense<absolute_value_type>>(
        result->as_precision(as_real(this->get_precision())).get()));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_squared_norm2_impl(
    MultiVector* result, array<char>& tmp) const
{
    this->compute_squared_norm2_impl(
        as<matrix::Dense<absolute_value_type>>(
            result->as_precision(as_real(this->get_precision())).get()),
        tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_norm1_impl(
    MultiVector* result) const
{
    this->compute_norm1_impl(as<matrix::Dense<absolute_value_type>>(
        result->as_precision(as_real(this->get_precision())).get()));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_norm1_impl(MultiVector* result,
                                                         array<char>& tmp) const
{
    this->compute_norm1_impl(
        as<matrix::Dense<absolute_value_type>>(
            result->as_precision(as_real(this->get_precision())).get()),
        tmp);
}


template <typename ConcreteType>
detail::temporary_conversion<MultiVector>
EnableMultiVector<ConcreteType>::as_precision_impl(precision p)
{
    return std::visit(
        [this](auto v) -> detail::temporary_conversion<MultiVector> {
            using target_value_type = std::decay_t<decltype(v)>;
            if constexpr (is_supported_conversion_v<value_type,
                                                    target_value_type>) {
                return detail::temporary_conversion<MultiVector>::
                    create_from_derived(
                        self()->template as_precision<target_value_type>());
            } else {
                GKO_NOT_IMPLEMENTED;
            }
        },
        precision_to_variant(p));
}


template <typename ConcreteType>
detail::temporary_conversion<const MultiVector>
EnableMultiVector<ConcreteType>::as_precision_impl(precision p) const
{
    return std::visit(
        [this](auto v) -> detail::temporary_conversion<const MultiVector> {
            using target_value_type = std::decay_t<decltype(v)>;
            if constexpr (is_supported_conversion_v<value_type,
                                                    target_value_type>) {
                return detail::temporary_conversion<const MultiVector>::
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
    MultiVector::device_view<half>,
    MultiVector::device_view<std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
    MultiVector::device_view<bfloat16>,
    MultiVector::device_view<std::complex<bfloat16>>,
#endif

    MultiVector::device_view<float>,
    MultiVector::device_view<std::complex<float>>,
    MultiVector::device_view<double>,
    MultiVector::device_view<std::complex<double>>>
EnableMultiVector<ConcreteType>::get_local_device_view_generic_impl()
{
    return static_cast<ConcreteType*>(this)->get_local_device_view();
}


template <typename ConcreteType>
std::variant<
#if GINKGO_ENABLE_HALF
    MultiVector::device_view<const half>,
    MultiVector::device_view<const std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
    MultiVector::device_view<const bfloat16>,
    MultiVector::device_view<const std::complex<bfloat16>>,
#endif
    MultiVector::device_view<const float>,
    MultiVector::device_view<const std::complex<float>>,
    MultiVector::device_view<const double>,
    MultiVector::device_view<const std::complex<double>>>
EnableMultiVector<ConcreteType>::get_const_local_device_view_generic_impl()
    const
{
    return static_cast<ConcreteType const*>(this)
        ->get_const_local_device_view();
}


}  // namespace gko
