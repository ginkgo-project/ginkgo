// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_WORKSPACE_TREE_HPP_
#define GKO_PUBLIC_CORE_SOLVER_WORKSPACE_TREE_HPP_


#include <iostream>
#include <map>
#include <memory>
#include <string>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/workspace.hpp>


namespace gko {


// Forward declarations for generate_with_node
class LinOp;
class LinOpFactory;


namespace solver {
namespace detail {


class WorkspaceNode {
public:
    explicit WorkspaceNode(std::shared_ptr<const Executor> exec)
        : local_storage_{std::move(exec)}
    {
        GKO_ASSERT(local_storage_.get_executor() != nullptr);
    }

    WorkspaceNode* get_or_create_child(const std::string& tag)
    {
        auto it = children_.find(tag);
        if (it != children_.end()) {
            return it->second.get();
        }
        auto child =
            std::make_unique<WorkspaceNode>(local_storage_.get_executor());
        child->tag_ = tag;
        child->num_rhs_ = num_rhs_;
        auto* ptr = child.get();
        children_.emplace(tag, std::move(child));
        return ptr;
    }

    WorkspaceNode* get_child(const std::string& tag) const
    {
        auto it = children_.find(tag);
        if (it != children_.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    bool has_child(const std::string& tag) const
    {
        return children_.find(tag) != children_.end();
    }

    size_type get_num_rhs() const { return num_rhs_; }

    void set_num_rhs(size_type num_rhs) { num_rhs_ = num_rhs; }

    void bind_executor(std::shared_ptr<const Executor> exec)
    {
        local_storage_.set_executor(std::move(exec));
    }

    void describe(std::ostream& os, int indent = 0) const;

    workspace& get_local_storage() { return local_storage_; }

    const workspace& get_local_storage() const { return local_storage_; }

private:
    friend class SolverBaseLinOp;

    workspace local_storage_;
    std::map<std::string, std::unique_ptr<WorkspaceNode>> children_;
    std::string tag_;
    size_type num_rhs_ = 0;
};


/**
 * Generates a LinOp from a factory, passing a workspace node to the
 * generated object for workspace tree propagation.
 */
std::unique_ptr<LinOp> generate_with_node(const LinOpFactory* factory,
                                          std::shared_ptr<const LinOp> matrix,
                                          WorkspaceNode* node);


}  // namespace detail


class Workspace {
public:
    static std::unique_ptr<Workspace> create(
        std::shared_ptr<const Executor> exec, size_type num_rhs = 1);
    static std::unique_ptr<Workspace> create_non_owning(
        detail::WorkspaceNode* node);

    size_type get_num_rhs() const { return node_->get_num_rhs(); }

    detail::WorkspaceNode* root() const { return node_; }

    void describe(std::ostream& os) const;

private:
    Workspace() = default;
    std::unique_ptr<detail::WorkspaceNode> owned_root_;
    detail::WorkspaceNode* node_ = nullptr;
};


/**
 * Extracts the workspace from a solver, invalidating the solver.
 * The solver unique_ptr is reset to nullptr after extraction.
 * Only works on top-level solvers that own their workspace.
 *
 * @param solver  the solver to extract from (will be set to nullptr)
 * @return the extracted workspace
 * @throws InvalidStateError if the solver is not workspace-aware or has no
 *         owned workspace (e.g., it is an inner solver)
 */
std::unique_ptr<Workspace> invalidate_and_extract_workspace(
    std::unique_ptr<LinOp>& solver);


}  // namespace solver
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_SOLVER_WORKSPACE_TREE_HPP_
