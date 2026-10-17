// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/workspace.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace solver {
namespace detail {


SolverBaseLinOp::SolverBaseLinOp(std::shared_ptr<const Executor> exec)
{
    owned_workspace_ = Workspace::create(std::move(exec));
    node_ = owned_workspace_.get();
}

SolverBaseLinOp::SolverBaseLinOp(const SolverBaseLinOp& other)
{
    if (other.node_) {
        owned_workspace_ = Workspace::create(other.node_->get_executor());
        node_ = owned_workspace_.get();
    }
}

SolverBaseLinOp::SolverBaseLinOp(SolverBaseLinOp&& other) noexcept
{
    owned_workspace_ = std::move(other.owned_workspace_);
    node_ = owned_workspace_ ? owned_workspace_.get() : nullptr;
    other.node_ = nullptr;
}

SolverBaseLinOp& SolverBaseLinOp::operator=(const SolverBaseLinOp& /*other*/)
{
    return *this;
}

SolverBaseLinOp& SolverBaseLinOp::operator=(SolverBaseLinOp&& other) noexcept
{
    if (this != &other) {
        owned_workspace_ = std::move(other.owned_workspace_);
        node_ = owned_workspace_ ? owned_workspace_.get() : nullptr;
        other.node_ = nullptr;
    }
    return *this;
}

std::unique_ptr<Workspace> SolverBaseLinOp::extract_workspace()
{
    node_ = nullptr;
    return std::move(owned_workspace_);
}

void SolverBaseLinOp::adopt_workspace(LinOpGenerateComponents& components,
                                      std::shared_ptr<const Executor> exec)
{
    if (components.has_owned_workspace()) {
        owned_workspace_ = components.take_owned_workspace();
        node_ = owned_workspace_.get();
        owned_workspace_->reset(std::move(exec));
    } else if (components.has_view_workspace()) {
        owned_workspace_ = nullptr;
        node_ = components.get_view_workspace();
    }
}


}  // namespace detail
}  // namespace solver
}  // namespace gko
