// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/multivector.hpp>

namespace gko {
namespace matrix {


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_config_of(
    ptr_param<const MultiVector> other)
{
    return other->create_with_same_config_impl();
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_type_of(
    ptr_param<const MultiVector> other, std::shared_ptr<const Executor> exec)
{
    return other->create_with_type_of_impl(std::move(exec), {}, {}, 0);
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_type_of(
    ptr_param<const MultiVector> other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size)
{
    GKO_ASSERT_EQUAL_COLS(global_size, local_size);
    return other->create_with_type_of_impl(std::move(exec), global_size,
                                           local_size, global_size[1]);
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_type_of(
    ptr_param<const MultiVector> other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size, size_type stride)
{
    return other->create_with_type_of_impl(std::move(exec), global_size,
                                           local_size, stride);
}


template <typename ValueType>
std::unique_ptr<typename MultiVector<ValueType>::absolute_type>
MultiVector<ValueType>::compute_absolute() const
{
    return this->compute_absolute_impl();
}


template <typename ValueType>
void MultiVector<ValueType>::compute_absolute_inplace()
{
    this->compute_absolute_inplace_impl();
}


template <typename ValueType>
std::unique_ptr<typename MultiVector<ValueType>::complex_type>
MultiVector<ValueType>::make_complex() const
{
    return this->make_complex_impl();
}


template <typename ValueType>
void MultiVector<ValueType>::make_complex(ptr_param<complex_type> result) const
{
    this->make_complex_impl(result.get());
}


template <typename ValueType>
std::unique_ptr<typename MultiVector<ValueType>::real_type>
MultiVector<ValueType>::get_real() const
{
    return this->get_real_impl();
}


template <typename ValueType>
void MultiVector<ValueType>::get_real(ptr_param<real_type> result) const
{
    this->get_real_impl(result.get());
}


template <typename ValueType>
std::unique_ptr<typename MultiVector<ValueType>::real_type>
MultiVector<ValueType>::get_imag() const
{
    return this->get_imag_impl();
}


template <typename ValueType>
void MultiVector<ValueType>::get_imag(ptr_param<real_type> result) const
{
    this->get_imag_impl(result.get());
}


template <typename ValueType>
void MultiVector<ValueType>::fill(ValueType value)
{
    this->fill_impl(value.get());
}


template <typename ValueType>
void MultiVector<ValueType>::scale(any_const_dense_t alpha)
{
    this->scale_impl(alpha);
}


template <typename ValueType>
void MultiVector<ValueType>::inv_scale(any_const_dense_t alpha)
{
    this->inv_scale_impl(alpha);
}


template <typename ValueType>
void MultiVector<ValueType>::add_scaled(any_const_dense_t alpha,
                                        ptr_param<const MultiVector> b)
{
    this->add_scaled_impl(alpha, b.get());
}


template <typename ValueType>
void MultiVector<ValueType>::sub_scaled(any_const_dense_t alpha,
                                        ptr_param<const MultiVector> b)
{
    this->sub_scaled_impl(alpha, b.get());
}


template <typename ValueType>
void MultiVector<ValueType>::compute_dot(ptr_param<const MultiVector> b,
                                         ptr_param<MultiVector> result) const
{
    this->compute_dot_impl(b.get(), result.get());
}


template <typename ValueType>
void MultiVector<ValueType>::compute_dot(ptr_param<const MultiVector> b,
                                         ptr_param<MultiVector> result,
                                         array<char>& tmp) const
{
    this->compute_dot_impl(b.get(), result.get(), tmp);
}


template <typename ValueType>
void MultiVector<ValueType>::compute_conj_dot(
    ptr_param<const MultiVector> b, ptr_param<MultiVector> result) const
{
    this->compute_conj_dot_impl(b.get(), result.get());
}


template <typename ValueType>
void MultiVector<ValueType>::compute_conj_dot(ptr_param<const MultiVector> b,
                                              ptr_param<MultiVector> result,
                                              array<char>& tmp) const
{
    this->compute_conj_dot_impl(b.get(), result.get(), tmp);
}


template <typename ValueType>
void MultiVector<ValueType>::compute_norm2(
    ptr_param<absolute_type> result) const
{
    this->compute_norm2_impl(result.get());
}


template <typename ValueType>
void MultiVector<ValueType>::compute_norm2(ptr_param<absolute_type> result,
                                           array<char>& tmp) const
{
    this->compute_norm2_impl(result.get(), tmp);
}


template <typename ValueType>
void MultiVector<ValueType>::compute_squared_norm2(
    ptr_param<absolute_type> result) const
{
    this->compute_squared_norm2_impl(result.get());
}


template <typename ValueType>
void MultiVector<ValueType>::compute_squared_norm2(
    ptr_param<absolute_type> result, array<char>& tmp) const
{
    this->compute_squared_norm2_impl(result.get(), tmp);
}


template <typename ValueType>
void MultiVector<ValueType>::compute_norm1(
    ptr_param<absolute_type> result) const
{
    this->compute_norm1_impl(result.get());
}


template <typename ValueType>
void MultiVector<ValueType>::compute_norm1(ptr_param<absolute_type> result,
                                           array<char>& tmp) const
{
    this->compute_norm1_impl(result.get(), tmp);
}


template <typename ValueType>
std::unique_ptr<const typename MultiVector<ValueType>::real_type>
MultiVector<ValueType>::create_real_view() const
{
    return this->create_real_view_impl();
}


template <typename ValueType>
std::unique_ptr<typename MultiVector<ValueType>::real_type>
MultiVector<ValueType>::create_real_view()
{
    return this->create_real_view_impl();
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::create_subview(
    local_span rows, local_span columns)
{
    return this->create_subview_impl(rows, columns);
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::create_subview(
    local_span rows, local_span columns, size_type global_rows,
    size_type globals_cols)
{
    return this->create_subview_impl(rows, columns, global_rows, globals_cols);
}

template <typename ValueType>
dim<2> MultiVector<ValueType>::get_size() const noexcept
{
    return LinOp::get_size();
}


template <typename ValueType>
void MultiVector<ValueType>::set_size(const dim<2>& size) noexcept
{
    LinOp::set_size(size);
}


}  // namespace matrix
}  // namespace gko
