// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <variant>

#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {

struct local_span : span {};


class MultiVector : public EnableAbstractPolymorphicObject<MultiVector> {
public:
    [[nodiscard]] static std::unique_ptr<MultiVector> create_with_config_of(
        ptr_param<const MultiVector> other)
    {
        return other->create_with_same_config_impl();
    }

    [[nodiscard]] static std::unique_ptr<MultiVector> create_with_type_of(
        ptr_param<const MultiVector> other,
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type stride)
    {
        return other->create_with_type_of_impl(std::move(exec), size, stride);
    }

    [[nodiscard]] static std::unique_ptr<MultiVector> create_view_of(
        ptr_param<MultiVector> other)
    {
        return other->create_view_of_impl();
    }

    [[nodiscard]] static std::unique_ptr<const MultiVector>
    create_const_view_of(ptr_param<const MultiVector> other)
    {
        return other->create_const_view_of_impl();
    }

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

    virtual void scale(ptr_param<const LinOp> alpha) = 0;

    virtual void inv_scale(ptr_param<const LinOp> alpha) = 0;

    virtual void add_scaled(ptr_param<const LinOp> alpha,
                            ptr_param<const LinOp> b) = 0;

    virtual void sub_scaled(ptr_param<const LinOp> alpha,
                            ptr_param<const LinOp> b) = 0;

    virtual void compute_dot(ptr_param<const LinOp> b,
                             ptr_param<LinOp> result) const = 0;

    virtual void compute_dot(ptr_param<const LinOp> b, ptr_param<LinOp> result,
                             array<char>& tmp) const = 0;

    virtual void compute_conj_dot(ptr_param<const LinOp> b,
                                  ptr_param<LinOp> result) const = 0;

    virtual void compute_conj_dot(ptr_param<const LinOp> b,
                                  ptr_param<LinOp> result,
                                  array<char>& tmp) const = 0;

    virtual void compute_norm2(ptr_param<LinOp> result) const = 0;

    virtual void compute_norm2(ptr_param<LinOp> result,
                               array<char>& tmp) const = 0;

    virtual void compute_norm1(ptr_param<LinOp> result) const = 0;

    virtual void compute_norm1(ptr_param<LinOp> result,
                               array<char>& tmp) const = 0;

    [[nodiscard]] virtual std::unique_ptr<const MultiVector> create_real_view()
        const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_real_view() = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_subview(
        local_span rows, local_span columns);

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_subview(
        local_span rows, local_span columns, size_type global_rows,
        size_type globals_cols);

    [[nodiscard]] dim<2> get_size() const noexcept { return size_; }

protected:
    explicit MultiVector(std::shared_ptr<const Executor> exec,
                         const dim<2>& size = dim<2>{})
        : EnableAbstractPolymorphicObject<MultiVector>(std::move(exec)),
          size_{size}
    {}

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_with_same_config_impl() const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector> create_with_type_of_impl(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type stride) const = 0;

    [[nodiscard]] virtual std::unique_ptr<MultiVector>
    create_view_of_impl() = 0;

    [[nodiscard]] virtual std::unique_ptr<const MultiVector>
    create_const_view_of_impl() const = 0;

    void set_size(const dim<2>& size) noexcept { size_ = size; }

private:
    dim<2> size_;
};


}  // namespace matrix
}  // namespace gko
