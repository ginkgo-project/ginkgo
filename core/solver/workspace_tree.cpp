// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/solver/workspace_tree.hpp>


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
        owned_workspace_ =
            Workspace::create(other.node_->get_local_storage().get_executor());
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
        owned_workspace_->get_local_storage().set_executor(std::move(exec));
    } else if (components.has_view_workspace()) {
        owned_workspace_ = nullptr;
        node_ = components.get_view_workspace();
    }
}


}  // namespace detail


void Workspace::describe(std::ostream& os, int indent) const
{
    std::string prefix(indent, ' ');
    os << prefix << "Workspace";
    if (!tag_.empty()) {
        os << " [" << tag_ << "]";
    }
    os << " (children=" << children_.size() << ")\n";
    for (const auto& pair : children_) {
        pair.second->describe(os, indent + 2);
    }
}


std::unique_ptr<Workspace> invalidate_and_extract_workspace(
    std::unique_ptr<LinOp>& solver)
{
    auto* solver_base = dynamic_cast<detail::SolverBaseLinOp*>(solver.get());
    GKO_THROW_IF_INVALID(solver_base != nullptr,
                         "solver does not support workspace extraction");
    auto ws = solver_base->extract_workspace();
    GKO_THROW_IF_INVALID(ws != nullptr,
                         "solver has no workspace to extract (inner solver?)");
    solver.reset();
    return ws;
}


std::unique_ptr<Workspace> Workspace::create(
    std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<Workspace>(new Workspace(std::move(exec)));
}


}  // namespace solver
}  // namespace gko
