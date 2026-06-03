// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/multivector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace gko {


MultiVector::MultiVector(std::shared_ptr<const Executor> exec,
                         const dim<2>& size, precision p)
    : LinOp(std::move(exec), size, p)
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


std::unique_ptr<MultiVector> MultiVector::clone(
    std::shared_ptr<const Executor> exec) const
{
    return std::unique_ptr<MultiVector>(
        as<MultiVector>(this->clone_impl(std::move(exec)).release()));
}


std::unique_ptr<MultiVector> MultiVector::clone() const
{
    return std::unique_ptr<MultiVector>(
        as<MultiVector>(this->clone_impl(this->get_executor()).release()));
}


MultiVector* MultiVector::copy_from(ptr_param<const MultiVector> other)
{
    return as<MultiVector>(this->copy_from_impl(other.get()));
}


MultiVector* MultiVector::move_from(ptr_param<MultiVector> other)
{
    return as<MultiVector>(this->move_from_impl(other.get()));
}


std::unique_ptr<MultiVector> MultiVector::create_default(
    std::shared_ptr<const Executor> exec) const
{
    return std::unique_ptr<MultiVector>(
        as<MultiVector>(this->create_default_impl(std::move(exec)).release()));
}


std::unique_ptr<MultiVector> MultiVector::create_default() const
{
    return std::unique_ptr<MultiVector>(as<MultiVector>(
        this->create_default_impl(this->get_executor()).release()));
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


#define GKO_ASSERT_IS_DENSE(alpha)                                           \
    {                                                                        \
        bool is_dense = std::visit(                                          \
            [alpha](auto p) {                                                \
                using value_type = std::decay_t<decltype(p)>;                \
                return dynamic_cast<const matrix::Dense<value_type>*>(       \
                           alpha.get()) != nullptr;                          \
            },                                                               \
            precision_to_variant(alpha->get_precision()));                   \
        if (!is_dense) {                                                     \
            GKO_NOT_SUPPORTED(alpha);                                        \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


void MultiVector::scale(ptr_param<const MultiVector> alpha)
{
    GKO_ASSERT_IS_DENSE(alpha);
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    auto exec = this->get_executor();
    this->scale_impl(make_temporary_clone(exec, alpha).get());
}


void MultiVector::inv_scale(ptr_param<const MultiVector> alpha)
{
    GKO_ASSERT_IS_DENSE(alpha);
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    auto exec = this->get_executor();
    this->inv_scale_impl(make_temporary_clone(exec, alpha).get());
}


void MultiVector::add_scaled(ptr_param<const MultiVector> alpha,
                             ptr_param<const MultiVector> b)
{
    GKO_ASSERT_IS_DENSE(alpha);
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    auto exec = this->get_executor();
    this->add_scaled_impl(make_temporary_clone(exec, alpha).get(),
                          make_temporary_clone(exec, b).get());
}


void MultiVector::sub_scaled(ptr_param<const MultiVector> alpha,
                             ptr_param<const MultiVector> b)
{
    GKO_ASSERT_IS_DENSE(alpha);
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    auto exec = this->get_executor();
    this->sub_scaled_impl(make_temporary_clone(exec, alpha).get(),
                          make_temporary_clone(exec, b).get());
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


gko::detail::temporary_conversion<MultiVector> MultiVector::as_precision(
    precision p)
{
    return this->as_precision_impl(p);
}


detail::temporary_conversion<MultiVector> MultiVector::as_precision(
    ptr_param<const MultiVector> p)
{
    return this->as_precision_impl(p->get_precision());
}


detail::temporary_conversion<MultiVector> MultiVector::as_precision(
    ptr_param<const LinOp> p)
{
    return this->as_precision_impl(p->get_precision());
}


gko::detail::temporary_conversion<const MultiVector> MultiVector::as_precision(
    precision p) const
{
    return this->as_precision_impl(p);
}


detail::temporary_conversion<const MultiVector> MultiVector::as_precision(
    ptr_param<const MultiVector> p) const
{
    return this->as_precision_impl(p->get_precision());
}


detail::temporary_conversion<const MultiVector> MultiVector::as_precision(
    ptr_param<const LinOp> p) const
{
    return this->as_precision_impl(p->get_precision());
}


template <typename ValueType>
MultiVector::device_view<ValueType> MultiVector::get_local_device_view()
{
    if (this->get_precision() != type_to_precision<ValueType>) {
        GKO_INVALID_STATE("Multivector doesn't have the requested precision");
    }
    using return_type = device_view<ValueType>;
    auto variant = this->get_local_device_view_generic_impl();
    return std::move(std::get<return_type>(variant));
}

#define GKO_DECLARE_MULTIVECTOR_CREATE_LOCAL_VIEW(ValueType) \
    MultiVector::device_view<ValueType> MultiVector::get_local_device_view()
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIVECTOR_CREATE_LOCAL_VIEW);


template <typename ValueType>
MultiVector::device_view<const ValueType>
MultiVector::get_const_local_device_view() const
{
    if (this->get_precision() != type_to_precision<ValueType>) {
        GKO_INVALID_STATE("Multivector doesn't have the requested precision");
    }
    using return_type = device_view<const ValueType>;
    auto variant = this->get_const_local_device_view_generic_impl();
    return std::move(std::get<return_type>(variant));
}

#define GKO_DECLARE_MULTIVECTOR_CREATE_LOCAL_VIEW_CONST(ValueType) \
    MultiVector::device_view<const ValueType>                      \
    MultiVector::get_const_local_device_view() const
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_CREATE_LOCAL_VIEW_CONST);


}  // namespace gko
