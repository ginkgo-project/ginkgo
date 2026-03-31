// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/multivector.hpp>

namespace gko {


MultiVector::MultiVector(std::shared_ptr<const Executor> exec,
                         const dim<2>& size)
    : EnableAbstractPolymorphicObject<MultiVector, LinOp>(std::move(exec), size)
{}


std::unique_ptr<MultiVector> MultiVector::create_with_config_of(
    ptr_param<const MultiVector> other)
{
    return other->create_generic_with_same_config_impl();
}


std::unique_ptr<MultiVector> MultiVector::create_with_type_of(
    ptr_param<const MultiVector> other, std::shared_ptr<const Executor> exec)
{
    return other->create_generic_with_type_of_impl(std::move(exec), {}, {}, 0);
}


std::unique_ptr<MultiVector> MultiVector::create_with_type_of(
    ptr_param<const MultiVector> other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size)
{
    GKO_ASSERT_EQUAL_COLS(global_size, local_size);
    return other->create_generic_with_type_of_impl(std::move(exec), global_size,
                                                   local_size, global_size[1]);
}


std::unique_ptr<MultiVector> MultiVector::create_with_type_of(
    ptr_param<const MultiVector> other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size, size_type stride)
{
    return other->create_generic_with_type_of_impl(std::move(exec), global_size,
                                                   local_size, stride);
}


std::unique_ptr<MultiVector> MultiVector::compute_absolute() const
{
    return this->compute_absolute_generic_impl();
}


void MultiVector::compute_absolute(ptr_param<MultiVector> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    auto exec = this->get_executor();
    this->compute_absolute_generic_impl(
        make_temporary_output_clone(exec, output).get());
}


void MultiVector::compute_absolute_inplace()
{
    this->compute_absolute_inplace_impl();
}


std::unique_ptr<MultiVector> MultiVector::make_complex() const
{
    return this->make_complex_generic_impl();
}


void MultiVector::make_complex(ptr_param<MultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
    auto exec = this->get_executor();
    this->make_complex_generic_impl(
        make_temporary_output_clone(exec, result).get());
}


std::unique_ptr<MultiVector> MultiVector::get_real() const
{
    return this->get_real_generic_impl();
}


void MultiVector::get_real(ptr_param<MultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
    auto exec = this->get_executor();
    this->get_real_generic_impl(
        make_temporary_output_clone(exec, result).get());
}


std::unique_ptr<MultiVector> MultiVector::get_imag() const
{
    return this->get_imag_generic_impl();
}


void MultiVector::get_imag(ptr_param<MultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
    auto exec = this->get_executor();
    this->get_imag_generic_impl(
        make_temporary_output_clone(exec, result).get());
}


void MultiVector::fill(any_scalar value) { this->fill_impl(value); }


void MultiVector::scale(any_const_dense_t alpha)
{
    std::visit(
        [this](auto alpha_v) {
            GKO_ASSERT_EQUAL_ROWS(alpha_v, dim<2>(1, 1));
            if (alpha_v->get_size()[1] != 1) {
                // different alpha for each column
                GKO_ASSERT_EQUAL_COLS(this, alpha_v);
            }
        },
        alpha);
    this->scale_impl(alpha);
}


void MultiVector::inv_scale(any_const_dense_t alpha)
{
    std::visit(
        [this](auto alpha_v) {
            GKO_ASSERT_EQUAL_ROWS(alpha_v, dim<2>(1, 1));
            if (alpha_v->get_size()[1] != 1) {
                // different alpha for each column
                GKO_ASSERT_EQUAL_COLS(this, alpha_v);
            }
        },
        alpha);
    this->inv_scale_impl(alpha);
}


void MultiVector::add_scaled(any_const_dense_t alpha,
                             ptr_param<const MultiVector> b)
{
    std::visit(
        [this, b](auto alpha_v) {
            GKO_ASSERT_EQUAL_ROWS(alpha_v, dim<2>(1, 1));
            if (alpha_v->get_size()[1] != 1) {
                // different alpha for each column
                GKO_ASSERT_EQUAL_COLS(this, alpha_v);
            }
            GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
        },
        alpha);
    this->add_scaled_impl(alpha, b.get());
}


void MultiVector::sub_scaled(any_const_dense_t alpha,
                             ptr_param<const MultiVector> b)
{
    std::visit(
        [this, b](auto alpha_v) {
            GKO_ASSERT_EQUAL_ROWS(alpha_v, dim<2>(1, 1));
            if (alpha_v->get_size()[1] != 1) {
                // different alpha for each column
                GKO_ASSERT_EQUAL_COLS(this, alpha_v);
            }
            GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
        },
        alpha);
    this->sub_scaled_impl(alpha, b.get());
}


void MultiVector::compute_dot(ptr_param<const MultiVector> b,
                              ptr_param<MultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    this->compute_dot_impl(make_temporary_clone(exec, b).get(),
                           make_temporary_output_clone(exec, result).get());
}


void MultiVector::compute_dot(ptr_param<const MultiVector> b,
                              ptr_param<MultiVector> result,
                              array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    this->compute_dot_impl(make_temporary_clone(exec, b).get(),
                           make_temporary_output_clone(exec, result).get(),
                           tmp);
}


void MultiVector::compute_conj_dot(ptr_param<const MultiVector> b,
                                   ptr_param<MultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    this->compute_conj_dot_impl(
        make_temporary_clone(exec, b).get(),
        make_temporary_output_clone(exec, result).get());
}


void MultiVector::compute_conj_dot(ptr_param<const MultiVector> b,
                                   ptr_param<MultiVector> result,
                                   array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    this->compute_conj_dot_impl(make_temporary_clone(exec, b).get(),
                                make_temporary_output_clone(exec, result).get(),
                                tmp);
}


void MultiVector::compute_norm2(ptr_param<MultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    this->compute_norm2_impl(make_temporary_output_clone(exec, result).get());
}


void MultiVector::compute_norm2(ptr_param<MultiVector> result,
                                array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    this->compute_norm2_impl(make_temporary_output_clone(exec, result).get(),
                             tmp);
}


void MultiVector::compute_squared_norm2(ptr_param<MultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    this->compute_squared_norm2_impl(
        make_temporary_output_clone(exec, result).get());
}


void MultiVector::compute_squared_norm2(ptr_param<MultiVector> result,
                                        array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    this->compute_squared_norm2_impl(
        make_temporary_output_clone(exec, result).get(), tmp);
}


void MultiVector::compute_norm1(ptr_param<MultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    this->compute_norm1_impl(make_temporary_output_clone(exec, result).get());
}


void MultiVector::compute_norm1(ptr_param<MultiVector> result,
                                array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    this->compute_norm1_impl(make_temporary_output_clone(exec, result).get(),
                             tmp);
}


std::unique_ptr<const MultiVector> MultiVector::create_real_view() const
{
    return this->create_real_view_generic_impl();
}


std::unique_ptr<MultiVector> MultiVector::create_real_view()
{
    return this->create_real_view_generic_impl();
}


std::unique_ptr<MultiVector> MultiVector::create_subview(local_span rows,
                                                         local_span columns)
{
    return this->create_subview_generic_impl(rows, columns);
}


std::unique_ptr<const MultiVector> MultiVector::create_subview(
    local_span rows, local_span columns) const
{
    return this->create_subview_generic_impl(rows, columns);
}


std::unique_ptr<const MultiVector> MultiVector::create_subview(
    local_span rows, local_span columns, dim<2> global_size) const
{
    return this->create_subview_generic_impl(rows, columns, global_size);
}


std::unique_ptr<MultiVector> MultiVector::create_subview(local_span rows,
                                                         local_span columns,
                                                         dim<2> global_size)
{
    return this->create_subview_generic_impl(rows, columns, global_size);
}


}  // namespace gko
