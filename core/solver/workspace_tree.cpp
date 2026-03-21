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
    owned_workspace_ = Workspace::create();
    node_ = owned_workspace_->root();
    node_->bind_executor(std::move(exec));
}

SolverBaseLinOp::SolverBaseLinOp(const SolverBaseLinOp& /*other*/)
{
    owned_workspace_ = Workspace::create();
    node_ = owned_workspace_->root();
}

SolverBaseLinOp::SolverBaseLinOp(SolverBaseLinOp&& other) noexcept
{
    owned_workspace_ = std::move(other.owned_workspace_);
    node_ = owned_workspace_ ? owned_workspace_->root() : nullptr;
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
        node_ = owned_workspace_ ? owned_workspace_->root() : nullptr;
        other.node_ = nullptr;
    }
    return *this;
}

void SolverBaseLinOp::set_workspace(std::unique_ptr<Workspace> ws)
{
    owned_workspace_ = std::move(ws);
    node_ = owned_workspace_ ? owned_workspace_->root() : nullptr;
}

void SolverBaseLinOp::set_workspace_node(WorkspaceNode* node)
{
    owned_workspace_ = nullptr;
    node_ = node;
}

std::unique_ptr<Workspace> SolverBaseLinOp::extract_workspace()
{
    node_ = nullptr;
    return std::move(owned_workspace_);
}


void WorkspaceNode::describe(std::ostream& os, int indent) const
{
    std::string prefix(indent, ' ');
    os << prefix << "WorkspaceNode";
    if (!tag_.empty()) {
        os << " [" << tag_ << "]";
    }
    os << " (num_rhs=" << num_rhs_ << ", children=" << children_.size()
       << ")\n";
    for (const auto& pair : children_) {
        pair.second->describe(os, indent + 2);
    }
}


}  // namespace detail


std::unique_ptr<Workspace> Workspace::create(size_type num_rhs)
{
    std::unique_ptr<Workspace> ws(new Workspace());
    ws->owned_root_ = std::make_unique<detail::WorkspaceNode>();
    ws->owned_root_->set_num_rhs(num_rhs);
    ws->node_ = ws->owned_root_.get();
    return ws;
}


std::unique_ptr<Workspace> Workspace::create_non_owning(
    detail::WorkspaceNode* node)
{
    std::unique_ptr<Workspace> ws(new Workspace());
    ws->node_ = node;
    return ws;
}


void Workspace::describe(std::ostream& os) const { node_->describe(os, 0); }


}  // namespace solver
}  // namespace gko
