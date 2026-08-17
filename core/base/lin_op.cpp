// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/lin_op.hpp"
#include <ginkgo/core/base/multivector.hpp>

#include <memory>
#include <utility>


namespace gko {


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


LinOpFactory::ReuseData::ReuseData() = default;


LinOpFactory::ReuseData::~ReuseData() = default;


std::unique_ptr<LinOpFactory::ReuseData> LinOpFactory::create_empty_reuse_data()
    const
{
    return std::make_unique<ReuseData>();
}


std::unique_ptr<LinOp> LinOpFactory::generate_reuse(
    std::shared_ptr<const LinOp> input, ReuseData& reuse_data) const
{
    this->template log<log::Logger::linop_factory_generate_started>(
        this, input.get());
    const auto exec = this->get_executor();
    std::shared_ptr<const LinOp> local_input = input;
    if (input->get_executor() != exec) {
        local_input = gko::clone(exec, input);
    }
    this->check_reuse_consistent(local_input.get(), reuse_data);
    auto generated = this->generate_reuse_impl(local_input, reuse_data);
    // same as AbstractFactory::generate, which generate() goes through
    for (auto logger : this->loggers_) {
        generated->add_logger(logger);
    }
    this->template log<log::Logger::linop_factory_generate_completed>(
        this, input.get(), generated.get());
    return generated;
}


void LinOpFactory::check_reuse_consistent(const LinOp*, const ReuseData&) const
{}


std::unique_ptr<LinOp> LinOpFactory::generate_reuse_impl(
    std::shared_ptr<const LinOp> input, ReuseData&) const
{
    // generate_impl, not AbstractFactory::generate: generate_reuse adds the
    // loggers itself
    return this->generate_impl(std::move(input));
}


}  // namespace gko
