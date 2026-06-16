// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/solver/workspace.hpp>


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

std::unique_ptr<solver::Workspace>
LinOpGenerateComponents::take_owned_workspace()
{
    return std::move(owned_workspace_);
}

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

namespace solver {
namespace detail {


std::unique_ptr<LinOp> generate_with_view(const LinOpFactory* factory,
                                          std::shared_ptr<const LinOp> input,
                                          Workspace* view)
{
    factory->log<log::Logger::linop_factory_generate_started>(factory,
                                                              input.get());
    const auto exec = factory->get_executor();
    auto on_exec =
        (input->get_executor() == exec) ? input : gko::clone(exec, input);
    auto generated = factory->AbstractFactory::generate(
        LinOpGenerateComponents{std::move(on_exec), view});
    factory->log<log::Logger::linop_factory_generate_completed>(
        factory, input.get(), generated.get());
    return generated;
}


}  // namespace detail
}  // namespace solver
}  // namespace gko
