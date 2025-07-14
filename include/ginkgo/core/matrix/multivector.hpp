// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


struct local_span : span {};


template <typename ValueType, typename ConcreteType>
class MultiVector : public EnableLinOp<ConcreteType> {
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

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_absolute_type()
        const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> compute_absolute()
        const = 0;

    virtual void compute_absolute_inplace() = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> make_complex() const = 0;

    virtual void make_complex(ptr_param<MultiVector> result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> get_real() const = 0;

    virtual void get_real(ptr_param<MultiVector> result) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> get_imag() const = 0;

    virtual void get_imag(ptr_param<MultiVector> result) const = 0;

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

    virtual void compute_norm2(ptr_param<MultiVector> result) const = 0;

    virtual void compute_norm2(ptr_param<MultiVector> result,
                               array<char>& tmp) const = 0;

    virtual void compute_norm1(ptr_param<MultiVector> result) const = 0;

    virtual void compute_norm1(ptr_param<MultiVector> result,
                               array<char>& tmp) const = 0;

    [[nodiscard]] virtual std::unique_ptr<const MultiVector> create_real_view()
        const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_real_view() = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_subview(
        local_span rows, local_span columns) = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_subview(
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols) = 0;

    [[nodiscard]] dim<2> get_size() const noexcept;

protected:
    explicit MultiVector(std::shared_ptr<const Executor> exec,
                         const dim<2>& size = dim<2>{})
        : EnableLinOp<MultiVector>(std::move(exec)), size_{size}
    {}

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_with_same_config_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_with_type_of_impl(
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size, size_type stride) const = 0;

    void set_size(const dim<2>& size) noexcept;

private:
    dim<2> size_;
};


}  // namespace matrix
}  // namespace gko
