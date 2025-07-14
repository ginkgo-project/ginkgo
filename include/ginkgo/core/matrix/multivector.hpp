// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


struct local_span : span {};

// @todo: remove ConcreteType
template <typename ValueType, typename ConcreteType>
class MultiVector : public EnableLinOp<ConcreteType> {
public:
    using value_type = ValueType;
    using absolute_type =
        MultiVector<remove_complex<ValueType>, remove_complex<ConcreteType>>;
    using real_type = absolute_type;
    using complex_type =
        MultiVector<to_complex<ValueType>, to_complex<ConcreteType>>;

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

    [[nodiscard]] virtual std::unique_ptr<absolute_type> compute_absolute()
        const = 0;

    virtual void compute_absolute_inplace() = 0;

    [[nodiscard]] virtual std::unique_ptr<complex_type> make_complex()
        const = 0;

    virtual void make_complex(ptr_param<complex_type> result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<real_type> get_real() const = 0;

    virtual void get_real(ptr_param<real_type> result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<real_type> get_imag() const = 0;

    virtual void get_imag(ptr_param<real_type> result) const = 0;

    virtual void scale(ptr_param<const MultiVector> alpha) = 0;

    virtual void inv_scale(ptr_param<const MultiVector> alpha) = 0;

    virtual void add_scaled(ptr_param<const MultiVector> alpha,
                            ptr_param<const MultiVector> b) = 0;

    virtual void sub_scaled(ptr_param<const MultiVector> alpha,
                            ptr_param<const MultiVector> b) = 0;

    virtual void compute_dot(ptr_param<const MultiVector> b,
                             ptr_param<MultiVector> result) const = 0;

    virtual void compute_dot(ptr_param<const MultiVector> b,
                             ptr_param<MultiVector> result,
                             array<char>& tmp) const = 0;

    virtual void compute_conj_dot(ptr_param<const MultiVector> b,
                                  ptr_param<MultiVector> result) const = 0;

    virtual void compute_conj_dot(ptr_param<const MultiVector> b,
                                  ptr_param<MultiVector> result,
                                  array<char>& tmp) const = 0;

    virtual void compute_norm2(ptr_param<absolute_type> result) const = 0;

    virtual void compute_norm2(ptr_param<absolute_type> result,
                               array<char>& tmp) const = 0;

    virtual void compute_norm1(ptr_param<absolute_type> result) const = 0;

    virtual void compute_norm1(ptr_param<absolute_type> result,
                               array<char>& tmp) const = 0;

    [[nodiscard]] virtual std::unique_ptr<const real_type> create_real_view()
        const = 0;

    [[nodiscard]] virtual std::unique_ptr<real_type> create_real_view() = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_subview(
        local_span rows, local_span columns) = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_subview(
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols) = 0;

    [[nodiscard]] dim<2> get_size() const noexcept;

protected:
    explicit MultiVector(std::shared_ptr<const Executor> exec,
                         const dim<2>& size = dim<2>{})
        : EnableLinOp<MultiVector>(std::move(exec), size)
    {}

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_with_same_config_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_with_type_of_impl(
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size, size_type stride) const = 0;

    void apply_impl(const LinOp* b, LinOp* x) const override {}
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {}

    void set_size(const dim<2>& size) noexcept;
};


}  // namespace matrix
}  // namespace gko
