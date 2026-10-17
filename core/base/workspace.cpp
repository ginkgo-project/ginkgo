// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <utility>

#include <ginkgo/core/base/workspace.hpp>


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
    auto child = std::shared_ptr<Workspace>(new Workspace(exec_));
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


void Workspace::replace_op(int op_id, std::unique_ptr<LinOp> op)
{
    GKO_ASSERT(op_id >= 0 && op_id < operators_.size());
    operators_[op_id] = std::move(op);
}


namespace detail {


WorkspaceHolder::WorkspaceHolder(std::shared_ptr<const Executor> exec)
    : exec_{std::move(exec)}
{}


WorkspaceHolder::WorkspaceHolder(std::shared_ptr<const Executor> exec,
                                 LinOpGenerateComponents& components)
    : exec_{std::move(exec)}
{
    if (components.has_owned_workspace()) {
        this->own(components.take_owned_workspace());
    } else if (components.has_view_workspace()) {
        this->borrow(components.get_view_workspace());
    }
}


WorkspaceHolder::WorkspaceHolder(const WorkspaceHolder& other)
    : exec_{other.exec_}
{}


WorkspaceHolder::WorkspaceHolder(WorkspaceHolder&& other) noexcept
    : exec_{other.exec_},
      owned_{std::move(other.owned_)},
      borrowed_{std::move(other.borrowed_)},
      node_{std::exchange(other.node_, nullptr)},
      extracted_{std::exchange(other.extracted_, false)}
{}


WorkspaceHolder& WorkspaceHolder::operator=(const WorkspaceHolder&)
{
    return *this;
}


WorkspaceHolder& WorkspaceHolder::operator=(WorkspaceHolder&& other) noexcept
{
    if (this != &other) {
        owned_ = std::move(other.owned_);
        borrowed_ = std::move(other.borrowed_);
        node_ = std::exchange(other.node_, nullptr);
        extracted_ = std::exchange(other.extracted_, false);
    }
    return *this;
}


Workspace* WorkspaceHolder::get_workspace_node() const
{
    if (!node_) {
        GKO_THROW_IF_INVALID(
            !extracted_,
            "the workspace has been extracted; this LinOp is invalidated");
        owned_ = Workspace::create(exec_);
        node_ = owned_.get();
    }
    return node_;
}


void WorkspaceHolder::set_workspace_size(int num_operators,
                                         int num_arrays) const
{
    auto node = this->get_workspace_node();
    // another user of a shared node may have bound it to its own executor
    node->reset(exec_);
    claim_workspace(*node, typeid(*this));
    node->set_size(num_operators, num_arrays);
}


void WorkspaceHolder::own(std::unique_ptr<Workspace> root)
{
    root->reset(exec_);
    borrowed_.reset();
    owned_ = std::move(root);
    node_ = owned_.get();
    extracted_ = false;
}


void WorkspaceHolder::borrow(Workspace* node)
{
    GKO_THROW_IF_INVALID(!node->weak_from_this().expired(),
                         "only child nodes of a workspace can be borrowed");
    node->reset(exec_);
    owned_.reset();
    borrowed_ = node->shared_from_this();
    node_ = node;
    extracted_ = false;
}


std::unique_ptr<Workspace> WorkspaceHolder::extract_workspace()
{
    if (borrowed_) {
        return nullptr;
    }
    // a holder that never used its workspace hands out an empty root
    this->get_workspace_node();
    node_ = nullptr;
    extracted_ = true;
    return std::move(owned_);
}


void attach_workspace(LinOp* op, Workspace* node)
{
    if (auto holder = dynamic_cast<WorkspaceHolder*>(op)) {
        holder->borrow(node);
    }
}


void give_workspace(LinOp* op, std::unique_ptr<Workspace> root)
{
    if (auto holder = dynamic_cast<WorkspaceHolder*>(op)) {
        holder->own(std::move(root));
    }
}


}  // namespace detail


std::unique_ptr<Workspace> invalidate_and_extract_workspace(
    std::unique_ptr<LinOp>&& op)
{
    auto holder = dynamic_cast<detail::WorkspaceHolder*>(op.get());
    GKO_THROW_IF_INVALID(holder != nullptr,
                         "solver does not support workspace extraction");
    auto ws = holder->extract_workspace();
    GKO_THROW_IF_INVALID(
        ws != nullptr,
        "solver has no owned workspace to extract: inner solvers hold a "
        "non-owning view of their parent's workspace and cannot be extracted "
        "from directly. Extract from the outermost solver instead.");
    op.reset();
    return ws;
}


std::unique_ptr<Workspace> Workspace::create(
    std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<Workspace>(new Workspace(std::move(exec)));
}


}  // namespace gko
