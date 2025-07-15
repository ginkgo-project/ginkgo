// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/multivector.hpp>

namespace gko {
namespace matrix {


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
    this->make_complex_impl(result.get());
}


std::unique_ptr<MultiVector> MultiVector::get_real() const
{
    return this->get_real_generic_impl();
}


void MultiVector::get_real(ptr_param<MultiVector> result) const
{
    this->get_real_impl(result.get());
}


std::unique_ptr<MultiVector> MultiVector::get_imag() const
{
    return this->get_imag_generic_impl();
}


void MultiVector::get_imag(ptr_param<MultiVector> result) const
{
    this->get_imag_impl(result.get());
}


void MultiVector::fill(syn::variant_from_tuple<supported_value_types> value)
{
    this->fill_impl(value);
}


void MultiVector::scale(any_const_dense_t alpha) { this->scale_impl(alpha); }


void MultiVector::inv_scale(any_const_dense_t alpha)
{
    this->inv_scale_impl(alpha);
}


void MultiVector::add_scaled(any_const_dense_t alpha,
                             ptr_param<const MultiVector> b)
{
    this->add_scaled_impl(alpha, b.get());
}


void MultiVector::sub_scaled(any_const_dense_t alpha,
                             ptr_param<const MultiVector> b)
{
    this->sub_scaled_impl(alpha, b.get());
}


void MultiVector::compute_dot(ptr_param<const MultiVector> b,
                              ptr_param<MultiVector> result) const
{
    this->compute_dot_impl(b.get(), result.get());
}


void MultiVector::compute_dot(ptr_param<const MultiVector> b,
                              ptr_param<MultiVector> result,
                              array<char>& tmp) const
{
    this->compute_dot_impl(b.get(), result.get(), tmp);
}


void MultiVector::compute_conj_dot(ptr_param<const MultiVector> b,
                                   ptr_param<MultiVector> result) const
{
    this->compute_conj_dot_impl(b.get(), result.get());
}


void MultiVector::compute_conj_dot(ptr_param<const MultiVector> b,
                                   ptr_param<MultiVector> result,
                                   array<char>& tmp) const
{
    this->compute_conj_dot_impl(b.get(), result.get(), tmp);
}


void MultiVector::compute_norm2(ptr_param<MultiVector> result) const
{
    this->compute_norm2_impl(result.get());
}


void MultiVector::compute_norm2(ptr_param<MultiVector> result,
                                array<char>& tmp) const
{
    this->compute_norm2_impl(result.get(), tmp);
}


void MultiVector::compute_squared_norm2(ptr_param<MultiVector> result) const
{
    this->compute_squared_norm2_impl(result.get());
}


void MultiVector::compute_squared_norm2(ptr_param<MultiVector> result,
                                        array<char>& tmp) const
{
    this->compute_squared_norm2_impl(result.get(), tmp);
}


void MultiVector::compute_norm1(ptr_param<MultiVector> result) const
{
    this->compute_norm1_impl(result.get());
}


void MultiVector::compute_norm1(ptr_param<MultiVector> result,
                                array<char>& tmp) const
{
    this->compute_norm1_impl(result.get(), tmp);
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
    local_span rows, local_span columns, size_type global_rows,
    size_type globals_cols) const
{
    return this->create_subview_generic_impl(rows, columns, global_rows,
                                             globals_cols);
}


std::unique_ptr<MultiVector> MultiVector::create_subview(local_span rows,
                                                         local_span columns,
                                                         size_type global_rows,
                                                         size_type globals_cols)
{
    return this->create_subview_generic_impl(rows, columns, global_rows,
                                             globals_cols);
}


dim<2> MultiVector::get_size() const noexcept { return LinOp::get_size(); }


void MultiVector::set_size(const dim<2>& size) noexcept
{
    LinOp::set_size(size);
}


}  // namespace matrix
}  // namespace gko
