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


class LinOp;
class LinOpFactory;


namespace solver {


namespace detail {


class SolverBaseLinOp;


}  // namespace detail


/**
 * The Workspace is a node in a solver's temporary-storage tree. It owns a
 * flat slot container for operators and arrays (detail::workspace) plus a
 * map of named child Workspaces for sub-solvers. Every node is bound to an
 * executor at construction; children inherit their parent's executor.
 *
 * Top-level workspaces are constructed via Workspace::create and passed into
 * LinOpFactory::generate(matrix, unique_ptr<Workspace>). Inner solvers
 * receive a non-owning Workspace* (a child created via get_or_create_child on
 * the parent) and are wired in via LinOpFactory::generate(matrix, Workspace*).
 */
class Workspace {
public:
    explicit Workspace(std::shared_ptr<const Executor> exec)
        : local_storage_{std::move(exec)}
    {
        GKO_ASSERT(local_storage_.get_executor() != nullptr);
    }

    static std::unique_ptr<Workspace> create(
        std::shared_ptr<const Executor> exec, size_type num_rhs = 1);

    Workspace* get_or_create_child(const std::string& tag)
    {
        auto it = children_.find(tag);
        if (it != children_.end()) {
            return it->second.get();
        }
        auto child = std::unique_ptr<Workspace>(
            new Workspace(local_storage_.get_executor()));
        child->tag_ = tag;
        child->num_rhs_ = num_rhs_;
        auto* ptr = child.get();
        children_.emplace(tag, std::move(child));
        return ptr;
    }

    Workspace* get_child(const std::string& tag) const
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

    detail::workspace& get_local_storage() { return local_storage_; }

    const detail::workspace& get_local_storage() const
    {
        return local_storage_;
    }

private:
    detail::workspace local_storage_;
    std::map<std::string, std::unique_ptr<Workspace>> children_;
    std::string tag_;
    size_type num_rhs_ = 0;
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
