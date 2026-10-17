// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/workspace.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace detail {


void claim_workspace(Workspace& node, const std::type_info& user)
{
    const std::type_index user_type{user};
    if (node.user_type_ != user_type) {
        node.clear();
        node.user_type_ = user_type;
    }
}


}  // namespace detail


Workspace::Workspace(std::shared_ptr<const Executor> exec)
    : exec_{std::move(exec)}
{
    GKO_ASSERT(exec_ != nullptr);
}


Workspace* Workspace::get_or_create_child(const std::string& tag)
{
    auto it = children_.find(tag);
    if (it != children_.end()) {
        return it->second.get();
    }
    auto child = std::unique_ptr<Workspace>(new Workspace(exec_));
    child->tag_ = tag;
    auto* ptr = child.get();
    children_.emplace(tag, std::move(child));
    return ptr;
}


Workspace* Workspace::get_child(const std::string& tag) const
{
    auto it = children_.find(tag);
    if (it != children_.end()) {
        return it->second.get();
    }
    return nullptr;
}


bool Workspace::has_child(const std::string& tag) const
{
    return children_.find(tag) != children_.end();
}


void Workspace::reset(std::shared_ptr<const Executor> exec)
{
    if (exec_ == exec) {
        return;
    }
    this->clear();
    exec_ = std::move(exec);
}


void Workspace::set_size(int num_operators, int num_arrays)
{
    operators_.resize(num_operators);
    arrays_.resize(num_arrays);
}


const LinOp* Workspace::get_const_op(int op_id) const
{
    GKO_ASSERT(op_id >= 0 && op_id < operators_.size());
    return operators_[op_id].get();
}


LinOp* Workspace::get_mutable_op(int op_id)
{
    GKO_ASSERT(op_id >= 0 && op_id < operators_.size());
    return operators_[op_id].get();
}


bool Workspace::empty() const
{
    for (const auto& op : operators_) {
        if (op) {
            return false;
        }
    }
    for (const auto& arr : arrays_) {
        if (!arr.empty()) {
            return false;
        }
    }
    return true;
}


void Workspace::clear()
{
    for (auto& op : operators_) {
        op.reset();
    }
    for (auto& array : arrays_) {
        array.clear();
    }
}


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
    std::unique_ptr<LinOp>&& solver)
{
    auto* solver_base =
        dynamic_cast<solver::detail::SolverBaseLinOp*>(solver.get());
    GKO_THROW_IF_INVALID(solver_base != nullptr,
                         "solver does not support workspace extraction");
    auto ws = solver_base->extract_workspace();
    GKO_THROW_IF_INVALID(
        ws != nullptr,
        "solver has no owned workspace to extract: inner solvers hold a "
        "non-owning view of their parent's workspace and cannot be extracted "
        "from directly. Extract from the outermost solver instead.");
    solver.reset();
    return ws;
}


std::unique_ptr<Workspace> Workspace::create(
    std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<Workspace>(new Workspace(std::move(exec)));
}


}  // namespace gko
