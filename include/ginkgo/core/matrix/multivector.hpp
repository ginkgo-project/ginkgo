// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/matrix/device_views.hpp>


namespace gko {


// Different type to clarify that only local rows/columns are meant
struct local_span {
    constexpr local_span(size_type point) noexcept
        : local_span{point, point + 1}
    {}

    constexpr local_span(size_type begin, size_type end) noexcept
        : begin{begin}, end{end}
    {}

    constexpr operator span() const { return {begin, end}; }

    constexpr local_span(const span& s) noexcept : local_span(s.begin, s.end) {}

    constexpr bool is_valid() const { return begin <= end; }

    constexpr size_type length() const { return end - begin; }

    size_type begin;
    size_type end;
};


namespace matrix {

template <typename ValueType>
class Dense;


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

using dense_types = syn::apply_to_list<Dense, supported_value_types>;

using any_const_dense_t = syn::variant_from_tuple<syn::apply_to_list<
    ptr_param, syn::apply_to_list<std::add_const_t, dense_types>>>;
using any_dense_type =
    syn::variant_from_tuple<syn::apply_to_list<ptr_param, dense_types>>;

using any_value_t = syn::variant_from_tuple<supported_value_types>;

class MultiVector : public EnableAbstractPolymorphicObject<MultiVector, LinOp> {
public:
    template <typename ValueType>
    using device_view = view::dense<ValueType>;

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

    [[nodiscard]] std::unique_ptr<MultiVector> compute_absolute() const;

    void compute_absolute(ptr_param<MultiVector> output) const;

    void compute_absolute_inplace();

    [[nodiscard]] std::unique_ptr<MultiVector> make_complex() const;

    void make_complex(ptr_param<MultiVector> result) const;

    [[nodiscard]] std::unique_ptr<MultiVector> get_real() const;

    void get_real(ptr_param<MultiVector> result) const;

    [[nodiscard]] std::unique_ptr<MultiVector> get_imag() const;

    void get_imag(ptr_param<MultiVector> result) const;

    void fill(any_value_t value);

    void scale(any_const_dense_t alpha);

    void inv_scale(any_const_dense_t alpha);

    void add_scaled(any_const_dense_t alpha, ptr_param<const MultiVector> b);

    void sub_scaled(any_const_dense_t alpha, ptr_param<const MultiVector> b);

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

    [[nodiscard]] gko::detail::temporary_conversion<MultiVector> as_precision(
        precision p);

    [[nodiscard]] gko::detail::temporary_conversion<const MultiVector>
    as_precision(precision p) const;

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

    virtual void fill_impl(any_value_t value) = 0;

    // @todo: need to fix alpha to a our dense type
    virtual void scale_impl(any_const_dense_t alpha) = 0;

    virtual void inv_scale_impl(any_const_dense_t alpha) = 0;

    virtual void add_scaled_impl(any_const_dense_t alpha,
                                 const MultiVector* b) = 0;

    virtual void sub_scaled_impl(any_const_dense_t alpha,
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
class EnableMultiVector
    : public EnablePolymorphicObject<ConcreteType, MultiVector>,
      public EnablePolymorphicAssignment<ConcreteType> {
public:
    using absolute_type = remove_complex<ConcreteType>;
    using real_type = absolute_type;
    using complex_type = to_complex<ConcreteType>;

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

protected:
    EnableMultiVector(std::shared_ptr<const Executor> exec, dim<2> size = {},
                      precision p = precision::none)
        : EnablePolymorphicObject<ConcreteType, MultiVector>(exec, size, p)
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

    virtual void add_scaled_impl(any_const_dense_t alpha,
                                 const ConcreteType* b) = 0;

    virtual void sub_scaled_impl(any_const_dense_t alpha,
                                 const ConcreteType* b) = 0;

    virtual void compute_dot_impl(const ConcreteType* b,
                                  ConcreteType* result) const = 0;

    virtual void compute_dot_impl(const ConcreteType* b, ConcreteType* result,
                                  array<char>& tmp) const = 0;

    virtual void compute_conj_dot_impl(const ConcreteType* b,
                                       ConcreteType* result) const = 0;

    virtual void compute_conj_dot_impl(const ConcreteType* b,
                                       ConcreteType* result,
                                       array<char>& tmp) const = 0;

    virtual void compute_norm2_impl(absolute_type* result) const = 0;

    virtual void compute_norm2_impl(absolute_type* result,
                                    array<char>& tmp) const = 0;

    virtual void compute_squared_norm2_impl(absolute_type* result) const = 0;

    virtual void compute_squared_norm2_impl(absolute_type* result,
                                            array<char>& tmp) const = 0;

    virtual void compute_norm1_impl(absolute_type* result) const = 0;

    virtual void compute_norm1_impl(absolute_type* result,
                                    array<char>& tmp) const = 0;

    [[nodiscard]] gko::detail::temporary_conversion<MultiVector>
    as_precision_impl(precision p) override;

    [[nodiscard]] gko::detail::temporary_conversion<const MultiVector>
    as_precision_impl(precision p) const override;

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

    void add_scaled_impl(any_const_dense_t alpha, const MultiVector* b) final;

    void sub_scaled_impl(any_const_dense_t alpha, const MultiVector* b) final;

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

    GKO_ENABLE_SELF(ConcreteType);
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
        ->create_with_type_of_impl(std::move(exec), global_size, local_size);
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
void EnableMultiVector<ConcreteType>::add_scaled_impl(any_const_dense_t alpha,
                                                      const MultiVector* b)
{
    this->add_scaled_impl(alpha, as<const ConcreteType>(b));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::sub_scaled_impl(any_const_dense_t alpha,
                                                      const MultiVector* b)
{
    this->sub_scaled_impl(alpha, as<const ConcreteType>(b));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_dot_impl(
    const MultiVector* b, MultiVector* result) const
{
    this->compute_dot_impl(as<const ConcreteType>(b), as<ConcreteType>(result));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_dot_impl(const MultiVector* b,
                                                       MultiVector* result,
                                                       array<char>& tmp) const
{
    this->compute_dot_impl(as<const ConcreteType>(b), as<ConcreteType>(result),
                           tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_conj_dot_impl(
    const MultiVector* b, MultiVector* result) const
{
    this->compute_conj_dot_impl(as<const ConcreteType>(b),
                                as<ConcreteType>(result));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_conj_dot_impl(
    const MultiVector* b, MultiVector* result, array<char>& tmp) const
{
    this->compute_conj_dot_impl(as<const ConcreteType>(b),
                                as<ConcreteType>(result), tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_norm2_impl(
    MultiVector* result) const
{
    this->compute_norm2_impl(as<ConcreteType>(result));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_norm2_impl(MultiVector* result,
                                                         array<char>& tmp) const
{
    this->compute_norm2_impl(as<ConcreteType>(result), tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_squared_norm2_impl(
    MultiVector* result) const
{
    this->compute_squared_norm2_impl(as<ConcreteType>(result));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_squared_norm2_impl(
    MultiVector* result, array<char>& tmp) const
{
    this->compute_squared_norm2_impl(as<ConcreteType>(result), tmp);
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_norm1_impl(
    MultiVector* result) const
{
    this->compute_norm1_impl(as<ConcreteType>(result));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::compute_norm1_impl(MultiVector* result,
                                                         array<char>& tmp) const
{
    this->compute_norm1_impl(as<ConcreteType>(result), tmp);
}


template <typename ConcreteType>
gko::detail::temporary_conversion<MultiVector>
EnableMultiVector<ConcreteType>::as_precision_impl(precision p)
{
    return std::visit(
        [this](auto v) -> gko::detail::temporary_conversion<MultiVector> {
            using fst_value_type = typename ConcreteType::value_type;
            using snd_value_type = std::decay_t<decltype(v)>;
            if constexpr (is_complex_s<fst_value_type>::value ==
                          is_complex_s<snd_value_type>::value) {
                return gko::detail::temporary_conversion<MultiVector>::create(
                    self()->template as_precision<snd_value_type>());
            } else {
                GKO_NOT_IMPLEMENTED;
            }
        },
        precision_to_variant(p));
}


template <typename ConcreteType>
gko::detail::temporary_conversion<const MultiVector>
EnableMultiVector<ConcreteType>::as_precision_impl(precision p) const
{
    return std::visit(
        [this](auto v) -> gko::detail::temporary_conversion<const MultiVector> {
            using fst_value_type = typename ConcreteType::value_type;
            using snd_value_type = std::decay_t<decltype(v)>;
            if constexpr (is_complex_s<fst_value_type>::value ==
                          is_complex_s<snd_value_type>::value) {
                return gko::detail::temporary_conversion<const MultiVector>::
                    create(self()->template as_precision<snd_value_type>());
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


}  // namespace matrix
}  // namespace gko
