// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/multivector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace gko {


void LinOp::apply(ptr_param<const AbstractMultiVector> b,
                  ptr_param<AbstractMultiVector> x) const
{
    this->template log<log::Logger::linop_apply_started>(this, b.get(),
                                                         x.get());
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    this->template log<log::Logger::linop_apply_completed>(this, b.get(),
                                                           x.get());
}


void LinOp::apply(ptr_param<const AbstractMultiVector> alpha,
                  ptr_param<const AbstractMultiVector> b,
                  ptr_param<const AbstractMultiVector> beta,
                  ptr_param<AbstractMultiVector> x) const

{
    this->template log<log::Logger::linop_advanced_apply_started>(
        this, alpha.get(), b.get(), beta.get(), x.get());
    this->validate_application_parameters(alpha.get(), b.get(), beta.get(),
                                          x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, alpha).get(),
                     make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, beta).get(),
                     make_temporary_clone(exec, x).get());
    this->template log<log::Logger::linop_advanced_apply_completed>(
        this, alpha.get(), b.get(), beta.get(), x.get());
}


LinOp& LinOp::operator=(LinOp&& other)
{
    if (this != &other) {
        PolymorphicObject::operator=(std::move(other));
        this->set_size(other.get_size());
        other.set_size({});
    }
    return *this;
}


LinOp::LinOp(LinOp&& other)
    : PolymorphicObject(std::move(other)),
      size_{std::exchange(other.size_, dim<2>{})}
{}


LinOp::LinOp(std::shared_ptr<const Executor> exec, const dim<2>& size)
    : PolymorphicObject(exec), size_{size}
{}


void LinOp::set_size(const dim<2>& value) noexcept { size_ = value; }


void LinOp::apply_impl(const AbstractMultiVector* alpha,
                       const AbstractMultiVector* b,
                       const AbstractMultiVector* beta,
                       AbstractMultiVector* x) const
{
    auto x_clone = x->clone();
    this->apply_impl(b, x_clone.get());
    x->scale(beta);
    x->add_scaled(alpha, x_clone);
}


void LinOp::validate_application_parameters(const AbstractMultiVector* b,
                                            const AbstractMultiVector* x) const

{
    GKO_ASSERT_CONFORMANT(this, b);
    GKO_ASSERT_EQUAL_ROWS(this, x);
    GKO_ASSERT_EQUAL_COLS(b, x);
}


void LinOp::validate_application_parameters(const LinOp* b,
                                            const LinOp* x) const
{
    GKO_ASSERT_CONFORMANT(this, b);
    GKO_ASSERT_EQUAL_ROWS(this, x);
    GKO_ASSERT_EQUAL_COLS(b, x);
}


void LinOp::validate_application_parameters(const AbstractMultiVector* alpha,
                                            const AbstractMultiVector* b,
                                            const AbstractMultiVector* beta,
                                            const AbstractMultiVector* x) const

{
    this->validate_application_parameters(b, x);
    GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(beta, dim<2>(1, 1));
}


void ScaledIdentityAddable::add_scaled_identity(
    ptr_param<const AbstractMultiVector> a,
    ptr_param<const AbstractMultiVector> b)
{
    GKO_ASSERT_IS_SCALAR(a);
    GKO_ASSERT_IS_SCALAR(b);
    auto ae =
        make_temporary_clone(as<PolymorphicObject>(this)->get_executor(), a);
    auto be =
        make_temporary_clone(as<PolymorphicObject>(this)->get_executor(), b);
    add_scaled_identity_impl(ae.get(), be.get());
}


}  // namespace gko
