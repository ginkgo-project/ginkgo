// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/lin_op.hpp"

#include <memory>
#include <utility>

#include <ginkgo/core/base/workspace.hpp>


namespace gko {


LinOpGenerateComponents::LinOpGenerateComponents(
    std::shared_ptr<const LinOp> matrix)
    : system_matrix{std::move(matrix)}
{}

LinOpGenerateComponents::LinOpGenerateComponents(
    std::shared_ptr<const LinOp> matrix, std::unique_ptr<gko::Workspace> ws)
    : system_matrix{std::move(matrix)}, owned_workspace_{std::move(ws)}
{}

LinOpGenerateComponents::LinOpGenerateComponents(
    std::shared_ptr<const LinOp> matrix, gko::Workspace* view)
    : system_matrix{std::move(matrix)}
{
    GKO_THROW_IF_INVALID(!view->weak_from_this().expired(),
                         "only child nodes of a workspace can be borrowed");
    view_workspace_ = view->shared_from_this();
}

LinOpGenerateComponents::~LinOpGenerateComponents() = default;

std::unique_ptr<gko::Workspace> LinOpGenerateComponents::take_owned_workspace()
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
    std::unique_ptr<gko::Workspace> ws) const
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
    return this->generate_reuse_with_components(
        LinOpGenerateComponents{std::move(input)}, reuse_data);
}


std::unique_ptr<LinOp> LinOpFactory::generate_reuse(
    std::shared_ptr<const LinOp> input, ReuseData& reuse_data,
    std::unique_ptr<gko::Workspace> ws) const
{
    return this->generate_reuse_with_components(
        LinOpGenerateComponents{std::move(input), std::move(ws)}, reuse_data);
}


std::unique_ptr<LinOp> LinOpFactory::generate_reuse_with_components(
    LinOpGenerateComponents components, ReuseData& reuse_data) const
{
    const auto input = components.system_matrix;
    this->template log<log::Logger::linop_factory_generate_started>(
        this, input.get());
    const auto exec = this->get_executor();
    if (input->get_executor() != exec) {
        components.system_matrix = gko::clone(exec, input);
    }
    this->check_reuse_consistent(components.system_matrix.get(), reuse_data);
    auto generated =
        this->generate_reuse_impl(std::move(components), reuse_data);
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
    LinOpGenerateComponents components, ReuseData&) const
{
    // generate_impl, not AbstractFactory::generate: generate_reuse adds the
    // loggers itself
    return this->generate_impl(std::move(components));
}


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


class LinOpFactoryAccessor {
public:
    static std::unique_ptr<LinOp> generate_reuse_with_view(
        const LinOpFactory* factory, std::shared_ptr<const LinOp> input,
        LinOpFactory::ReuseData& reuse_data, Workspace* view)
    {
        return factory->generate_reuse_with_components(
            LinOpGenerateComponents{std::move(input), view}, reuse_data);
    }
};


std::unique_ptr<LinOp> generate_reuse_with_view(
    const LinOpFactory* factory, std::shared_ptr<const LinOp> input,
    LinOpFactory::ReuseData& reuse_data, Workspace* view)
{
    return LinOpFactoryAccessor::generate_reuse_with_view(
        factory, std::move(input), reuse_data, view);
}


}  // namespace detail
}  // namespace gko
