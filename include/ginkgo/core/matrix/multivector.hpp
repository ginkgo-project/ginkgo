// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
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


// Different type to clarify that only local rows/columns are meant
struct local_span : span {};

class MultiVector : public EnableAbstractPolymorphicObject<MultiVector, LinOp> {
public:
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
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols);

    [[nodiscard]] std::unique_ptr<const MultiVector> create_subview(
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols) const;

    // @todo: this should return something like a device view
    template <typename ValueType>
    [[nodiscard]] std::unique_ptr<Dense<ValueType>> create_local_view();

    template <typename ValueType>
    [[nodiscard]] std::unique_ptr<const Dense<ValueType>> create_local_view()
        const;

    template <typename ValueType>
    [[nodiscard]] auto temporary_precision() const
        -> std::unique_ptr<const MultiVector>;

    template <typename ValueType>
    [[nodiscard]] auto temporary_precision()
        -> std::unique_ptr<MultiVector, std::function<void(MultiVector*)>>;

    [[nodiscard]] dim<2> get_size() const noexcept;

    [[nodiscard]] size_type get_stride() const noexcept;

protected:
    explicit MultiVector(std::shared_ptr<const Executor> exec,
                         const dim<2>& size = dim<2>{});

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_generic_with_same_config_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_generic_with_type_of_impl(std::shared_ptr<const Executor> exec,
                                     const dim<2>& global_size,
                                     const dim<2>& local_size,
                                     size_type stride) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    compute_absolute_generic_impl() const = 0;

    virtual void compute_absolute_inplace_impl() = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    make_complex_generic_impl() const = 0;

    virtual void make_complex_impl(MultiVector* result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> get_real_generic_impl()
        const = 0;

    virtual void get_real_impl(MultiVector* result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> get_imag_generic_impl()
        const = 0;

    virtual void get_imag_impl(MultiVector* result) const = 0;

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
                                size_type global_rows,
                                size_type globals_cols) = 0;

    [[nodiscard]] virtual std::unique_ptr<const MultiVector>
    create_subview_generic_impl(local_span rows, local_span columns,
                                size_type global_rows,
                                size_type globals_cols) const = 0;

    [[nodiscard]] virtual auto create_local_view_impl(
        syn::variant_from_tuple<supported_value_types> type)
        -> syn::variant_from_tuple<
            syn::apply_to_list<std::unique_ptr, dense_types>> = 0;

    [[nodiscard]] virtual auto create_local_view_impl(
        syn::variant_from_tuple<supported_value_types> type) const
        -> syn::variant_from_tuple<syn::apply_to_list<
            std::unique_ptr,
            syn::apply_to_list<std::add_const_t, dense_types>>> = 0;

    [[nodiscard]] virtual auto get_stride_impl() const -> size_type = 0;

    [[nodiscard]] virtual auto temporary_precision_impl(
        syn::variant_from_tuple<supported_value_types> type)
        -> std::unique_ptr<MultiVector, std::function<void(MultiVector*)>> = 0;

    [[nodiscard]] virtual auto temporary_precision_impl(
        syn::variant_from_tuple<supported_value_types> type) const
        -> std::unique_ptr<const MultiVector> = 0;

    void set_size(const dim<2>& size) noexcept;
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
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols);

    [[nodiscard]] std::unique_ptr<const ConcreteType> create_subview(
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols) const;

    [[nodiscard]] std::unique_ptr<const real_type> create_real_view() const;

    [[nodiscard]] std::unique_ptr<real_type> create_real_view();

    [[nodiscard]] std::unique_ptr<absolute_type> compute_absolute() const;

    [[nodiscard]] std::unique_ptr<complex_type> make_complex() const;

    [[nodiscard]] std::unique_ptr<real_type> get_real() const;

    [[nodiscard]] std::unique_ptr<real_type> get_imag() const;

protected:
    EnableMultiVector(std::shared_ptr<const Executor> exec, dim<2> size = {})
        : EnablePolymorphicObject<ConcreteType, MultiVector>(exec, size)
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
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols) = 0;

    [[nodiscard]] virtual std::unique_ptr<const ConcreteType>
    create_subview_impl(local_span rows, local_span columns,
                        size_type global_rows,
                        size_type globals_cols) const = 0;

    [[nodiscard]] virtual std::unique_ptr<const real_type>
    create_real_view_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<real_type>
    create_real_view_impl() = 0;

    [[nodiscard]] virtual std::unique_ptr<absolute_type> compute_absolute_impl()
        const = 0;

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

    virtual void compute_norm1_impl(absolute_type* result) const = 0;

    virtual void compute_norm1_impl(absolute_type* result,
                                    array<char>& tmp) const = 0;

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
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols) final;

    [[nodiscard]] std::unique_ptr<const MultiVector>
    create_subview_generic_impl(local_span rows, local_span columns,
                                size_type global_rows,
                                size_type globals_cols) const final;

    [[nodiscard]] std::unique_ptr<const MultiVector>
    create_real_view_generic_impl() const final;

    [[nodiscard]] std::unique_ptr<MultiVector> create_real_view_generic_impl()
        final;

    [[nodiscard]] std::unique_ptr<MultiVector> compute_absolute_generic_impl()
        const final;

    [[nodiscard]] std::unique_ptr<MultiVector> make_complex_generic_impl()
        const final;

    [[nodiscard]] std::unique_ptr<MultiVector> get_real_generic_impl()
        const final;

    [[nodiscard]] std::unique_ptr<MultiVector> get_imag_generic_impl()
        const final;

    void make_complex_impl(MultiVector* result) const final;

    void get_real_impl(MultiVector* result) const final;

    void get_imag_impl(MultiVector* result) const final;

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
    local_span rows, local_span columns, size_type global_rows,
    size_type globals_cols)
{
    return this->create_subview_impl(rows, columns, global_rows, globals_cols);
}


template <typename ConcreteType>
std::unique_ptr<const ConcreteType>
EnableMultiVector<ConcreteType>::create_subview(local_span rows,
                                                local_span columns,
                                                size_type global_rows,
                                                size_type globals_cols) const
{
    return this->create_subview_impl(rows, columns, global_rows, globals_cols);
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
std::unique_ptr<typename EnableMultiVector<ConcreteType>::complex_type>
EnableMultiVector<ConcreteType>::make_complex() const
{
    return this->make_complex_impl();
}


template <typename ConcreteType>
std::unique_ptr<typename EnableMultiVector<ConcreteType>::real_type>
EnableMultiVector<ConcreteType>::get_real() const
{
    return this->get_real_impl();
}


template <typename ConcreteType>
std::unique_ptr<typename EnableMultiVector<ConcreteType>::real_type>
EnableMultiVector<ConcreteType>::get_imag() const
{
    return this->get_imag_impl();
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
EnableMultiVector<ConcreteType>::create_subview_generic_impl(
    local_span rows, local_span columns, size_type global_rows,
    size_type globals_cols)
{
    return this->create_subview_impl(rows, columns, global_rows, globals_cols);
}


template <typename ConcreteType>
std::unique_ptr<const MultiVector>
EnableMultiVector<ConcreteType>::create_subview_generic_impl(
    local_span rows, local_span columns, size_type global_rows,
    size_type globals_cols) const
{
    return this->create_subview_impl(rows, columns, global_rows, globals_cols);
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
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::make_complex_generic_impl() const
{
    return this->make_complex_impl();
}


template <typename ConcreteType>
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::get_real_generic_impl() const
{
    return this->get_real_impl();
}


template <typename ConcreteType>
std::unique_ptr<MultiVector>
EnableMultiVector<ConcreteType>::get_imag_generic_impl() const
{
    return this->get_imag_impl();
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::make_complex_impl(
    MultiVector* result) const
{
    this->make_complex_impl(as<ConcreteType>(result));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::get_real_impl(MultiVector* result) const
{
    this->get_real_impl(as<ConcreteType>(result));
}


template <typename ConcreteType>
void EnableMultiVector<ConcreteType>::get_imag_impl(MultiVector* result) const
{
    this->get_imag_impl(as<ConcreteType>(result));
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


}  // namespace matrix
}  // namespace gko
