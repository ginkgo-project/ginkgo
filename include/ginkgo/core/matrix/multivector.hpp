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

    [[nodiscard]] std::unique_ptr<MultiVector> create_subview(
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols);

    [[nodiscard]] dim<2> get_size() const noexcept;

protected:
    explicit MultiVector(std::shared_ptr<const Executor> exec,
                         const dim<2>& size = dim<2>{});

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_with_same_config_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_with_type_of_impl(
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size, size_type stride) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> compute_absolute_impl()
        const = 0;

    virtual void compute_absolute_inplace_impl() = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> make_complex_impl()
        const = 0;

    virtual void make_complex_impl(MultiVector* result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> get_real_impl()
        const = 0;

    virtual void get_real_impl(MultiVector* result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> get_imag_impl()
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
    create_real_view_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_real_view_impl() = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_subview_impl(
        local_span rows, local_span columns) = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_subview_impl(
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols) = 0;

    void set_size(const dim<2>& size) noexcept;
};


}  // namespace matrix
}  // namespace gko
