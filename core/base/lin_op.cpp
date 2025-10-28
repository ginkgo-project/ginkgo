// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>

namespace gko {


LinOp* LinOp::apply(ptr_param<const LinOp> b, ptr_param<LinOp> x)
{
    this->template log<log::Logger::linop_apply_started>(this, b.get(),
                                                         x.get());
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    this->template log<log::Logger::linop_apply_completed>(this, b.get(),
                                                           x.get());
    return this;
}


const LinOp* LinOp::apply(ptr_param<const LinOp> b, ptr_param<LinOp> x) const
{
    this->template log<log::Logger::linop_apply_started>(this, b.get(),
                                                         x.get());
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    this->template log<log::Logger::linop_apply_completed>(this, b.get(),
                                                           x.get());
    return this;
}


LinOp* LinOp::apply(ptr_param<const LinOp> alpha, ptr_param<const LinOp> b,
                    ptr_param<const LinOp> beta, ptr_param<LinOp> x)
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
    return this;
}


const LinOp* LinOp::apply(ptr_param<const LinOp> alpha,
                          ptr_param<const LinOp> b, ptr_param<const LinOp> beta,
                          ptr_param<LinOp> x) const
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
    return this;
}


LinOp& LinOp::operator=(const LinOp&) = default;


LinOp& LinOp::operator=(LinOp&& other)
{
    if (this != &other) {
        EnableAbstractPolymorphicObject<LinOp>::operator=(std::move(other));
        this->set_size(other.get_size());
        other.set_size({});
    }
    return *this;
}


LinOp::LinOp(const LinOp&) = default;


LinOp::LinOp(LinOp&& other)
    : EnableAbstractPolymorphicObject<LinOp>(std::move(other)),
      size_{std::exchange(other.size_, dim<2>{})}
{}


LinOp::LinOp(std::shared_ptr<const Executor> exec, const dim<2>& size)
    : EnableAbstractPolymorphicObject<LinOp>(exec), size_{size}
{}


void LinOp::set_size(const dim<2>& value) noexcept { size_ = value; }


void LinOp::validate_application_parameters(const LinOp* b,
                                            const LinOp* x) const
{
    GKO_ASSERT_CONFORMANT(this, b);
    GKO_ASSERT_EQUAL_ROWS(this, x);
    GKO_ASSERT_EQUAL_COLS(b, x);
}


void LinOp::validate_application_parameters(const LinOp* alpha, const LinOp* b,
                                            const LinOp* beta,
                                            const LinOp* x) const
{
    this->validate_application_parameters(b, x);
    GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(beta, dim<2>(1, 1));
}


LinOpFactory::ReuseData::ReuseData() = default;


LinOpFactory::ReuseData::~ReuseData() = default;


std::unique_ptr<LinOp> LinOpFactory::generate(
    std::shared_ptr<const LinOp> input) const
{
    this->template log<log::Logger::linop_factory_generate_started>(
        this, input.get());
    const auto exec = this->get_executor();
    std::unique_ptr<LinOp> generated;
    if (input->get_executor() == exec) {
        generated = this->AbstractFactory::generate(input);
    } else {
        generated = this->AbstractFactory::generate(gko::clone(exec, input));
    }
    this->template log<log::Logger::linop_factory_generate_completed>(
        this, input.get(), generated.get());
    return generated;
}


std::unique_ptr<LinOpFactory::ReuseData> LinOpFactory::create_empty_reuse_data()
    const
{
    return std::make_unique<LinOpFactory::ReuseData>();
}


void LinOpFactory::check_reuse_consistent(const LinOp* /*input*/,
                                          ReuseData& /*data*/) const
{}


std::unique_ptr<LinOp> LinOpFactory::generate_reuse(
    std::shared_ptr<const LinOp> input, ReuseData& reuse_data) const
{
    this->check_reuse_consistent(input.get(), reuse_data);
    this->template log<log::Logger::linop_factory_generate_started>(
        this, input.get());
    const auto exec = this->get_executor();
    std::unique_ptr<LinOp> generated;
    if (input->get_executor() == exec) {
        generated = this->generate_reuse_impl(input, reuse_data);
    } else {
        generated =
            this->generate_reuse_impl(gko::clone(exec, input), reuse_data);
    }
    this->template log<log::Logger::linop_factory_generate_completed>(
        this, input.get(), generated.get());
    return generated;
}


std::unique_ptr<LinOp> LinOpFactory::generate_reuse_impl(
    std::shared_ptr<const LinOp> input, ReuseData& /*reuse_data*/) const
{
    return this->generate_impl(input);
}


}  // namespace gko
