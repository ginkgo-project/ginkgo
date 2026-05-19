// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/lin_op.hpp"

#include <memory>
#include <utility>

#include <ginkgo/core/solver/workspace_tree.hpp>


namespace gko {


LinOpGenerateComponents::LinOpGenerateComponents(
    std::shared_ptr<const LinOp> matrix)
    : system_matrix{std::move(matrix)}
{}

LinOpGenerateComponents::LinOpGenerateComponents(
    std::shared_ptr<const LinOp> matrix, std::unique_ptr<solver::Workspace> ws)
    : system_matrix{std::move(matrix)}, owned_workspace_{std::move(ws)}
{}

LinOpGenerateComponents::LinOpGenerateComponents(
    std::shared_ptr<const LinOp> matrix, solver::Workspace* view)
    : system_matrix{std::move(matrix)}, view_workspace_{view}
{}

LinOpGenerateComponents::~LinOpGenerateComponents() = default;

LinOpGenerateComponents::LinOpGenerateComponents(
    LinOpGenerateComponents&&) noexcept = default;

LinOpGenerateComponents& LinOpGenerateComponents::operator=(
    LinOpGenerateComponents&&) noexcept = default;


std::unique_ptr<LinOp> LinOpFactory::generate(
    std::shared_ptr<const LinOp> input) const
{
    this->log<log::Logger::linop_factory_generate_started>(this, input.get());
    const auto exec = this->get_executor();
    std::unique_ptr<LinOp> generated;
    if (input->get_executor() == exec) {
        generated =
            this->AbstractFactory::generate(LinOpGenerateComponents{input});
    } else {
        generated = this->AbstractFactory::generate(
            LinOpGenerateComponents{gko::clone(exec, input)});
    }
    this->log<log::Logger::linop_factory_generate_completed>(this, input.get(),
                                                             generated.get());
    return generated;
}

std::unique_ptr<LinOp> LinOpFactory::generate(
    std::shared_ptr<const LinOp> input,
    std::unique_ptr<solver::Workspace> ws) const
{
    this->log<log::Logger::linop_factory_generate_started>(this, input.get());
    const auto exec = this->get_executor();
    std::unique_ptr<LinOp> generated;
    if (input->get_executor() == exec) {
        generated = this->AbstractFactory::generate(
            LinOpGenerateComponents{input, std::move(ws)});
    } else {
        generated = this->AbstractFactory::generate(
            LinOpGenerateComponents{gko::clone(exec, input), std::move(ws)});
    }
    this->log<log::Logger::linop_factory_generate_completed>(this, input.get(),
                                                             generated.get());
    return generated;
}

std::unique_ptr<LinOp> LinOpFactory::generate(
    std::shared_ptr<const LinOp> input, solver::Workspace* workspace_view) const
{
    this->log<log::Logger::linop_factory_generate_started>(this, input.get());
    const auto exec = this->get_executor();
    std::unique_ptr<LinOp> generated;
    if (input->get_executor() == exec) {
        generated = this->AbstractFactory::generate(
            LinOpGenerateComponents{input, workspace_view});
    } else {
        generated = this->AbstractFactory::generate(
            LinOpGenerateComponents{gko::clone(exec, input), workspace_view});
    }
    this->log<log::Logger::linop_factory_generate_completed>(this, input.get(),
                                                             generated.get());
    return generated;
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
