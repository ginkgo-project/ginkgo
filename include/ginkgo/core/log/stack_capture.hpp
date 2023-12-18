// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_LOG_STACK_CAPTURE_HPP_
#define GKO_PUBLIC_CORE_LOG_STACK_CAPTURE_HPP_


#include <unordered_map>


#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace log {


class StackCapture : Logger {
public:
    void on_allocation_started(const Executor* exec,
                               const size_type&) const override;

    void on_allocation_completed(const Executor* exec, const size_type&,
                                 const uintptr&) const override;

    void on_free_started(const Executor* exec, const uintptr&) const override;

    void on_free_completed(const Executor* exec, const uintptr&) const override;

    void on_copy_started(const Executor* from, const Executor* to,
                         const uintptr&, const uintptr&,
                         const size_type&) const override;

    void on_copy_completed(const Executor* from, const Executor* to,
                           const uintptr&, const uintptr&,
                           const size_type&) const override;

    /* Operation events */
    void on_operation_launched(const Executor* exec,
                               const Operation* operation) const override;

    void on_operation_completed(const Executor* exec,
                                const Operation* operation) const override;

    /* PolymorphicObject events */
    void on_polymorphic_object_copy_started(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override;

    void on_polymorphic_object_copy_completed(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override;

    void on_polymorphic_object_move_started(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override;

    void on_polymorphic_object_move_completed(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override;

    /* LinOp events */
    void on_linop_apply_started(const LinOp* A, const LinOp* b,
                                const LinOp* x) const override;

    void on_linop_apply_completed(const LinOp* A, const LinOp* b,
                                  const LinOp* x) const override;

    void on_linop_advanced_apply_started(const LinOp* A, const LinOp* alpha,
                                         const LinOp* b, const LinOp* beta,
                                         const LinOp* x) const override;

    void on_linop_advanced_apply_completed(const LinOp* A, const LinOp* alpha,
                                           const LinOp* b, const LinOp* beta,
                                           const LinOp* x) const override;

    /* LinOpFactory events */
    void on_linop_factory_generate_started(const LinOpFactory* factory,
                                           const LinOp* input) const override;

    void on_linop_factory_generate_completed(
        const LinOpFactory* factory, const LinOp* input,
        const LinOp* output) const override;

    /* Criterion events */
    void on_criterion_check_started(const stop::Criterion* criterion,
                                    const size_type& num_iterations,
                                    const LinOp* residual,
                                    const LinOp* residual_norm,
                                    const LinOp* solution,
                                    const uint8& stopping_id,
                                    const bool& set_finalized) const override;

    void on_criterion_check_completed(
        const stop::Criterion* criterion, const size_type& num_iterations,
        const LinOp* residual, const LinOp* residual_norm,
        const LinOp* solution, const uint8& stopping_id,
        const bool& set_finalized, const array<stopping_status>* status,
        const bool& one_changed, const bool& all_stopped) const override;

    void on_criterion_check_completed(
        const stop::Criterion* criterion, const size_type& num_iterations,
        const LinOp* residual, const LinOp* residual_norm,
        const LinOp* implicit_sq_resnorm, const LinOp* solution,
        const uint8& stopping_id, const bool& set_finalized,
        const array<stopping_status>* status, const bool& one_changed,
        const bool& all_stopped) const override;

    /* Internal solver events */
    void on_iteration_complete(
        const LinOp* solver, const LinOp* right_hand_side,
        const LinOp* solution, const size_type& num_iterations,
        const LinOp* residual, const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm,
        const array<stopping_status>* status, bool stopped) const override;

    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    void on_iteration_complete(const LinOp* solver,
                               const size_type& num_iterations,
                               const LinOp* residual, const LinOp* solution,
                               const LinOp* residual_norm) const override;

    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    void on_iteration_complete(
        const LinOp* solver, const size_type& num_iterations,
        const LinOp* residual, const LinOp* solution,
        const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm) const override;

    bool needs_propagation() const override;

    const std::vector<int64>& get_stack() const { return stack_; }

    const std::unordered_map<int64, std::string>& get_inverse_name_map() const
    {
        return inverse_name_map_;
    }

private:
    void push(const std::string& name) const;

    void pop(const std::string& name) const;

    template <typename T>
    std::string stringify_object(const T* obj) const;

    mutable std::vector<int64> stack_;
    mutable std::unordered_map<std::string, int64> name_map_;
    mutable std::unordered_map<int64, std::string> inverse_name_map_;
    mutable std::unordered_map<std::uintptr_t, std::string> obj_names_;
};

}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_STACK_CAPTURE_HPP_
