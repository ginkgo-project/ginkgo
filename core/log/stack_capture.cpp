// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/stack_capture.hpp>


#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace log {


void StackCapture::on_allocation_started(const Executor* exec,
                                         const size_type& size) const
{
    push(stringify_object(exec).append("::allocate"));
}


void StackCapture::on_allocation_completed(const Executor* exec,
                                           const size_type& size,
                                           const uintptr& uintptr) const
{
    pop(stringify_object(exec).append("::allocate"));
}


void StackCapture::on_free_started(const Executor* exec,
                                   const uintptr& uintptr) const
{
    push(stringify_object(exec).append("::free"));
}


void StackCapture::on_free_completed(const Executor* exec,
                                     const uintptr& uintptr) const
{
    pop(stringify_object(exec).append("::free"));
}


void StackCapture::on_copy_started(const Executor* from, const Executor* to,
                                   const uintptr& ptr1, const uintptr& ptr2,
                                   const size_type& size) const
{
    push(stringify_object(from).append("::copy(").append(
        stringify_object(to).append(")")));
}


void StackCapture::on_copy_completed(const Executor* from, const Executor* to,
                                     const uintptr& ptr1, const uintptr& ptr2,
                                     const size_type& size) const
{
    pop(stringify_object(from).append("::copy(").append(
        stringify_object(to).append(")")));
}


void StackCapture::on_operation_launched(const Executor* exec,
                                         const Operation* operation) const
{
    push(stringify_object(exec).append("::").append(operation->get_name()));
}


void StackCapture::on_operation_completed(const Executor* exec,
                                          const Operation* operation) const
{
    pop(stringify_object(exec).append("::").append(operation->get_name()));
}


void StackCapture::on_polymorphic_object_copy_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << stringify_object(exec) << "::copy(" << stringify_object(from) << ", "
       << stringify_object(to) << ")";
    push(ss.str());
}


void StackCapture::on_polymorphic_object_copy_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << stringify_object(exec) << "::copy(" << stringify_object(from) << ", "
       << stringify_object(to) << ")";
    pop(ss.str());
}


void StackCapture::on_polymorphic_object_move_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << stringify_object(exec) << "::move(" << stringify_object(from) << ", "
       << stringify_object(to) << ")";
    push(ss.str());
}


void StackCapture::on_polymorphic_object_move_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << stringify_object(exec) << "::move(" << stringify_object(from) << ", "
       << stringify_object(to) << ")";
    pop(ss.str());
}


void StackCapture::on_linop_apply_started(const LinOp* A, const LinOp* b,
                                          const LinOp* x) const
{
    push(stringify_object(A).append("::apply"));
    if (dynamic_cast<const solver::IterativeBase*>(A)) {
        push(stringify_object(A).append("::iteration"));
    }
}


void StackCapture::on_linop_apply_completed(const LinOp* A, const LinOp* b,
                                            const LinOp* x) const
{
    pop(stringify_object(A).append("::apply"));
    if (dynamic_cast<const solver::IterativeBase*>(A)) {
        pop(stringify_object(A).append("::iteration"));
    }
}


void StackCapture::on_linop_advanced_apply_started(const LinOp* A,
                                                   const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   const LinOp* x) const
{
    push(stringify_object(A).append("::advanced_apply"));
    if (dynamic_cast<const solver::IterativeBase*>(A)) {
        push(stringify_object(A).append("::iteration"));
    }
}


void StackCapture::on_linop_advanced_apply_completed(const LinOp* A,
                                                     const LinOp* alpha,
                                                     const LinOp* b,
                                                     const LinOp* beta,
                                                     const LinOp* x) const
{
    pop(stringify_object(A).append("::advanced_apply"));
    if (dynamic_cast<const solver::IterativeBase*>(A)) {
        pop(stringify_object(A).append("::iteration"));
    }
}


void StackCapture::on_linop_factory_generate_started(
    const LinOpFactory* factory, const LinOp* input) const
{
    push(stringify_object(factory).append("::generate"));
}


void StackCapture::on_linop_factory_generate_completed(
    const LinOpFactory* factory, const LinOp* input, const LinOp* output) const
{
    pop(stringify_object(factory).append("::generate"));
}


void StackCapture::on_criterion_check_started(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stopping_id, const bool& set_finalized) const
{
    push(stringify_object(criterion).append("::check"));
}


void StackCapture::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stopping_id, const bool& set_finalized,
    const array<stopping_status>* status, const bool& one_changed,
    const bool& all_stopped) const
{
    pop(stringify_object(criterion).append("::check"));
}


void StackCapture::on_iteration_complete(
    const LinOp* solver, const LinOp* right_hand_side, const LinOp* solution,
    const size_type& num_iterations, const LinOp* residual,
    const LinOp* residual_norm, const LinOp* implicit_sq_residual_norm,
    const array<stopping_status>* status, bool stopped) const
{
    if (num_iterations > 0 &&
        dynamic_cast<const solver::IterativeBase*>(solver) && !stopped) {
        pop(stringify_object(solver).append("::iteration"));
        push(stringify_object(solver).append("::iteration"));
    }
}


void StackCapture::on_iteration_complete(const LinOp* solver,
                                         const size_type& num_iterations,
                                         const LinOp* residual,
                                         const LinOp* solution,
                                         const LinOp* residual_norm) const
{
    this->on_iteration_complete(solver, nullptr, solution, num_iterations,
                                residual, residual_norm, nullptr, nullptr,
                                false);
}


void StackCapture::on_iteration_complete(
    const LinOp* solver, const size_type& num_iterations, const LinOp* residual,
    const LinOp* solution, const LinOp* residual_norm,
    const LinOp* implicit_sq_residual_norm) const
{
    this->on_iteration_complete(solver, nullptr, solution, num_iterations,
                                residual, residual_norm,
                                implicit_sq_residual_norm, nullptr, false);
}


bool StackCapture::needs_propagation() const { return true; }


void StackCapture::push(const std::string& name) const
{
    auto it = name_map_.find(name);
    if (it == name_map_.end()) {
        const auto new_id = static_cast<int64>(name_map_.size());
        it = name_map_.emplace_hint(it, name, new_id);
        inverse_name_map_.emplace(new_id, name);
    }
    stack_.push_back(it->second);
}


void StackCapture::pop(const std::string& name) const
{
    if (name_map_.at(name) == stack_.back()) {
        stack_.pop_back();
    } else {
        // @TODO: catch mismatching push-pop
        auto id = name_map_.at(name);
        stack_.erase(std::find(stack_.begin(), stack_.end(), id));
    }
}


template <typename T>
std::string StackCapture::stringify_object(const T* obj) const
{
    if (!obj) {
        return "nullptr";
    }
    auto ptr = static_cast<std::uintptr_t>(obj);
    auto it = obj_names_.find(ptr);
    if (it == obj_names_.end()) {
        it = obj_names_.emplace_hint(it, ptr,
                                     name_demangling::get_dynamic_type(*obj));
    }
    return it->second;
}


}  // namespace log
}  // namespace gko
