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


class AbstractMultiVector : public PolymorphicObject, public Cloneable {
public:
    template <typename ValueType>
    using device_view = matrix::view::dense<ValueType>;

    [[nodiscard]] static std::unique_ptr<AbstractMultiVector>
    create_with_config_of(ptr_param<const AbstractMultiVector> other);

    [[nodiscard]] static std::unique_ptr<AbstractMultiVector>
    create_with_type_of(ptr_param<const AbstractMultiVector> other,
                        std::shared_ptr<const Executor> exec);

    [[nodiscard]] static std::unique_ptr<AbstractMultiVector>
    create_with_type_of(ptr_param<const AbstractMultiVector> other,
                        std::shared_ptr<const Executor> exec,
                        const dim<2>& global_size, const dim<2>& local_size);

    [[nodiscard]] static std::unique_ptr<AbstractMultiVector>
    create_with_type_of(ptr_param<const AbstractMultiVector> other,
                        std::shared_ptr<const Executor> exec,
                        const dim<2>& global_size, const dim<2>& local_size,
                        size_type stride);

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> clone(
        std::shared_ptr<const Executor> exec) const;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> clone() const;

    AbstractMultiVector* copy_from(ptr_param<const AbstractMultiVector> other);

    AbstractMultiVector* move_from(ptr_param<AbstractMultiVector> other);

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> create_default(
        std::shared_ptr<const Executor> exec) const;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> create_default() const;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> compute_absolute() const;

    void compute_absolute(ptr_param<AbstractMultiVector> output) const;

    void compute_absolute_inplace();

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> make_complex() const;

    void make_complex(ptr_param<AbstractMultiVector> result) const;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> get_real() const;

    void get_real(ptr_param<AbstractMultiVector> result) const;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> get_imag() const;

    void get_imag(ptr_param<AbstractMultiVector> result) const;

    void fill(any_scalar value);

    void scale(ptr_param<const AbstractMultiVector> alpha);

    void inv_scale(ptr_param<const AbstractMultiVector> alpha);

    void add_scaled(ptr_param<const AbstractMultiVector> alpha,
                    ptr_param<const AbstractMultiVector> b);

    void sub_scaled(ptr_param<const AbstractMultiVector> alpha,
                    ptr_param<const AbstractMultiVector> b);

    void compute_dot(ptr_param<const AbstractMultiVector> b,
                     ptr_param<AbstractMultiVector> result) const;

    void compute_dot(ptr_param<const AbstractMultiVector> b,
                     ptr_param<AbstractMultiVector> result,
                     array<char>& tmp) const;

    void compute_conj_dot(ptr_param<const AbstractMultiVector> b,
                          ptr_param<AbstractMultiVector> result) const;

    void compute_conj_dot(ptr_param<const AbstractMultiVector> b,
                          ptr_param<AbstractMultiVector> result,
                          array<char>& tmp) const;

    void compute_norm2(ptr_param<AbstractMultiVector> result) const;

    void compute_norm2(ptr_param<AbstractMultiVector> result,
                       array<char>& tmp) const;

    void compute_squared_norm2(ptr_param<AbstractMultiVector> result) const;

    void compute_squared_norm2(ptr_param<AbstractMultiVector> result,
                               array<char>& tmp) const;

    void compute_norm1(ptr_param<AbstractMultiVector> result) const;

    void compute_norm1(ptr_param<AbstractMultiVector> result,
                       array<char>& tmp) const;

    [[nodiscard]] std::unique_ptr<const AbstractMultiVector> create_real_view()
        const;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> create_real_view();

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> create_subview(
        local_span rows, local_span columns);

    [[nodiscard]] std::unique_ptr<const AbstractMultiVector> create_subview(
        local_span rows, local_span columns) const;

    [[nodiscard]] std::unique_ptr<AbstractMultiVector> create_subview(
        local_span rows, local_span columns, dim<2> global_size);

    [[nodiscard]] std::unique_ptr<const AbstractMultiVector> create_subview(
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
    [[nodiscard]] temporary_conversion<AbstractMultiVector> as_precision(
        precision p);

    [[nodiscard]] temporary_conversion<AbstractMultiVector> as_precision(
        ptr_param<const AbstractMultiVector> p);

    [[nodiscard]] temporary_conversion<const AbstractMultiVector> as_precision(
        precision p) const;

    [[nodiscard]] temporary_conversion<const AbstractMultiVector> as_precision(
        ptr_param<const AbstractMultiVector> p) const;

    [[nodiscard]] precision get_precision() const noexcept;

    [[nodiscard]] dim<2> get_size() const noexcept;

    AbstractMultiVector(const AbstractMultiVector& other);

    AbstractMultiVector(AbstractMultiVector&& other);

    // Preserves executor and precision on both objects
    AbstractMultiVector& operator=(const AbstractMultiVector& other);

    // Preserves executor and precision on both objects
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
