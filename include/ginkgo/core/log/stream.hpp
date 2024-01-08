// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_LOG_STREAM_HPP_
#define GKO_PUBLIC_CORE_LOG_STREAM_HPP_


#include <fstream>
#include <iostream>


#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace log {


/**
 * Stream is a Logger which logs every event to a stream. This can typically be
 * used to log to a file or to the console.
 *
 * @tparam ValueType  the type of values stored in the class (i.e. ValueType
 *                    template parameter of the concrete Loggable this class
 *                    will log)
 *
 * @ingroup log
 */
template <typename ValueType = default_precision>
class Stream : public Logger {
public:
    /* Executor events */
    void on_allocation_started(const Executor* exec,
                               const size_type& num_bytes) const override;

    void on_allocation_completed(const Executor* exec,
                                 const size_type& num_bytes,
                                 const uintptr& location) const override;

    void on_free_started(const Executor* exec,
                         const uintptr& location) const override;

    void on_free_completed(const Executor* exec,
                           const uintptr& location) const override;

    void on_copy_started(const Executor* from, const Executor* to,
                         const uintptr& location_from,
                         const uintptr& location_to,
                         const size_type& num_bytes) const override;

    void on_copy_completed(const Executor* from, const Executor* to,
                           const uintptr& location_from,
                           const uintptr& location_to,
                           const size_type& num_bytes) const override;

    /* Operation events */
    void on_operation_launched(const Executor* exec,
                               const Operation* operation) const override;

    void on_operation_completed(const Executor* exec,
                                const Operation* operation) const override;

    /* PolymorphicObject events */
    void on_polymorphic_object_create_started(
        const Executor*, const PolymorphicObject* po) const override;

    void on_polymorphic_object_create_completed(
        const Executor* exec, const PolymorphicObject* input,
        const PolymorphicObject* output) const override;

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

    void on_polymorphic_object_deleted(
        const Executor* exec, const PolymorphicObject* po) const override;

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
        const bool& one_changed, const bool& all_converged) const override;

    /* Internal solver events */
    void on_iteration_complete(const LinOp* solver, const LinOp* b,
                               const LinOp* x, const size_type& num_iterations,
                               const LinOp* residual,
                               const LinOp* residual_norm,
                               const LinOp* implicit_resnorm_sq,
                               const array<stopping_status>* status,
                               bool stopped) const override;

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

    /**
     * Creates a Stream logger. This dynamically allocates the memory,
     * constructs the object and returns an std::unique_ptr to this object.
     *
     * @param exec  the executor
     * @param enabled_events  the events enabled for this logger. By default all
     *                        events.
     * @param os  the stream used for this logger
     * @param verbose  whether we want detailed information or not. This
     *                 includes always printing residuals and other information
     *                 which can give a large output.
     *
     * @return an std::unique_ptr to the the constructed object
     *
     * @internal here I cannot use EnableCreateMethod due to complex circular
     * dependencies. At the same time, this method is short enough that it
     * shouldn't be a problem.
     */
    GKO_DEPRECATED("use three-parameter create")
    static std::unique_ptr<Stream> create(
        std::shared_ptr<const Executor> exec,
        const Logger::mask_type& enabled_events = Logger::all_events_mask,
        std::ostream& os = std::cout, bool verbose = false)
    {
        return std::unique_ptr<Stream>(new Stream(enabled_events, os, verbose));
    }

    /**
     * Creates a Stream logger. This dynamically allocates the memory,
     * constructs the object and returns an std::unique_ptr to this object.
     *
     * @param exec  the executor
     * @param enabled_events  the events enabled for this logger. By default all
     *                        events.
     * @param os  the stream used for this logger
     * @param verbose  whether we want detailed information or not. This
     *                 includes always printing residuals and other information
     *                 which can give a large output.
     *
     * @return an std::unique_ptr to the the constructed object
     *
     * @internal here I cannot use EnableCreateMethod due to complex circular
     * dependencies. At the same time, this method is short enough that it
     * shouldn't be a problem.
     */
    static std::unique_ptr<Stream> create(
        const Logger::mask_type& enabled_events = Logger::all_events_mask,
        std::ostream& os = std::cerr, bool verbose = false)
    {
        return std::unique_ptr<Stream>(new Stream(enabled_events, os, verbose));
    }

protected:
    /**
     * Creates a Stream logger.
     *
     * @param exec  the executor
     * @param enabled_events  the events enabled for this logger. By default all
     *                        events.
     * @param os  the stream used for this logger
     * @param verbose  whether we want detailed information or not. This
     *                 includes always printing residuals and other information
     *                 which can give a large output.
     */
    GKO_DEPRECATED("use three-parameter constructor")
    explicit Stream(
        std::shared_ptr<const gko::Executor> exec,
        const Logger::mask_type& enabled_events = Logger::all_events_mask,
        std::ostream& os = std::cerr, bool verbose = false)
        : Stream(enabled_events, os, verbose)
    {}

    /**
     * Creates a Stream logger.
     *
     * @param exec  the executor
     * @param enabled_events  the events enabled for this logger. By default all
     *                        events.
     * @param os  the stream used for this logger
     * @param verbose  whether we want detailed information or not. This
     *                 includes always printing residuals and other information
     *                 which can give a large output.
     */
    explicit Stream(
        const Logger::mask_type& enabled_events = Logger::all_events_mask,
        std::ostream& os = std::cerr, bool verbose = false)
        : Logger(enabled_events), os_(&os), verbose_(verbose)
    {}


private:
    std::ostream* os_;
    static constexpr const char* prefix_ = "[LOG] >>> ";
    bool verbose_;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_STREAM_HPP_
