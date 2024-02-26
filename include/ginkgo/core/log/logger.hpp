// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_LOG_LOGGER_HPP_
#define GKO_PUBLIC_CORE_LOG_LOGGER_HPP_


#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils_helper.hpp>


namespace gko {


/* Eliminate circular dependencies the hard way */
template <typename ValueType>
class array;
class Executor;
class LinOp;
class LinOpFactory;
class PolymorphicObject;
class Operation;
class stopping_status;


namespace batch {


class BatchLinOp;
class BatchLinOpFactory;

template <typename ValueType>
class MultiVector;


}  // namespace batch


/**
 * @brief The Stopping criterion namespace.
 * @ref stop
 * @ingroup stop
 */
namespace stop {
class Criterion;
}  // namespace stop


namespace log {


/**
 * @addtogroup log
 *
 * The Logger class represents a simple Logger object. It comprises all masks
 * and events internally. Every new logging event addition should be done here.
 * The Logger class also provides a default implementation for most events which
 * do nothing, therefore it is not an obligation to change all classes which
 * derive from Logger, although it is good practice.
 * The logger class is built using event masks to control which events should be
 * logged, and which should not.
 *
 * @internal The class uses bitset to facilitate picking a combination of events
 * to log. In addition, the class design allows to not propagate empty messages
 * for events which are not tracked.
 * See #GKO_LOGGER_REGISTER_EVENT(_id, _event_name, ...).
 */
class Logger {
public:
    /** @internal std::bitset allows to store any number of bits */
    using mask_type = gko::uint64;

    /**
     * Maximum amount of events (bits) with the current implementation
     */
    static constexpr size_type event_count_max = sizeof(mask_type) * byte_size;

    /**
     * Bitset Mask which activates all events
     */
    static constexpr mask_type all_events_mask = ~mask_type{0};

    /**
     * Helper macro to define functions and masks for each event.
     * A mask named _event_name##_mask is created for each event. `_id` is
     * the number assigned to this event and should be unique.
     *
     * @internal the templated function `on(Params)` will trigger the event
     * call only if the user activates this event through the mask. If the
     * event is activated, we rely on polymorphism and the virtual method
     * `on_##_event_name()` to either call the Logger class's function,
     * which does nothing, or the overridden version in the derived class if
     * any. Therefore, to support a new event in any Logger (i.e. class
     * which derive from this class), the function `on_##_event_name()`
     * should be overridden and implemented.
     *
     * @param _id  the unique id of the event
     *
     * @param _event_name  the name of the event
     *
     * @param ...  a variable list of arguments representing the event's
     *             arguments
     */
#define GKO_LOGGER_REGISTER_EVENT(_id, _event_name, ...)             \
protected:                                                           \
    virtual void on_##_event_name(__VA_ARGS__) const {}              \
                                                                     \
public:                                                              \
    template <size_type Event, typename... Params>                   \
    std::enable_if_t<Event == _id && (_id < event_count_max)> on(    \
        Params&&... params) const                                    \
    {                                                                \
        if (enabled_events_ & (mask_type{1} << _id)) {               \
            this->on_##_event_name(std::forward<Params>(params)...); \
        }                                                            \
    }                                                                \
    static constexpr size_type _event_name{_id};                     \
    static constexpr mask_type _event_name##_mask{mask_type{1} << _id};

    /**
     * Executor's allocation started event.
     *
     * @param exec  the executor used
     * @param num_bytes  the number of bytes to allocate
     */
    GKO_LOGGER_REGISTER_EVENT(0, allocation_started, const Executor* exec,
                              const size_type& num_bytes)

    /**
     * Executor's allocation completed event.
     *
     * @param exec  the executor used
     * @param num_bytes  the number of bytes allocated
     * @param location  the address at which the data was allocated
     */
    GKO_LOGGER_REGISTER_EVENT(1, allocation_completed, const Executor* exec,
                              const size_type& num_bytes,
                              const uintptr& location)

    /**
     * Executor's free started event.
     *
     * @param exec  the executor used
     * @param location  the address at which the data will be freed
     */
    GKO_LOGGER_REGISTER_EVENT(2, free_started, const Executor* exec,
                              const uintptr& location)

    /**
     * Executor's free completed event.
     *
     * @param exec  the executor used
     * @param location  the address at which the data was freed
     */
    GKO_LOGGER_REGISTER_EVENT(3, free_completed, const Executor* exec,
                              const uintptr& location)

    /**
     * Executor's copy started event.

     * @param exec_from  the executor to be copied from
     * @param exec_to  the executor to be copied to
     * @param loc_from  the address at which the data will be copied from
     * @param loc_to  the address at which the data will be copied to
     * @param num_bytes  the number of bytes to be copied
     */
    GKO_LOGGER_REGISTER_EVENT(4, copy_started, const Executor* exec_from,
                              const Executor* exec_to, const uintptr& loc_from,
                              const uintptr& loc_to, const size_type& num_bytes)

    /**
     * Executor's copy completed event.
     *
     * @param exec_from  the executor copied from
     * @param exec_to  the executor copied to
     * @param loc_from  the address at which the data was copied from
     * @param loc_to  the address at which the data was copied to
     * @param num_bytes  the number of bytes copied
     */
    GKO_LOGGER_REGISTER_EVENT(5, copy_completed, const Executor* exec_from,
                              const Executor* exec_to, const uintptr& loc_from,
                              const uintptr& loc_to, const size_type& num_bytes)

    /**
     * Executor's operation launched event (method run).
     *
     * @param exec  the executor used
     * @param op  the operation launched
     */
    GKO_LOGGER_REGISTER_EVENT(6, operation_launched, const Executor* exec,
                              const Operation* op)

    /**
     * Executor's operation completed event (method run).
     *
     * @param exec  the executor used
     * @param op  the completed operation
     *
     * @note For the GPU, to be certain that the operation completed it is
     * required to call synchronize. This burden falls on the logger. Most of
     * the loggers will do lightweight logging, and therefore this operation for
     * the GPU just notes that the Operation has been sent to the GPU.
     */
    GKO_LOGGER_REGISTER_EVENT(7, operation_completed, const Executor* exec,
                              const Operation* op)

    /**
     * PolymorphicObject's create started event.
     *
     * @param exec  the executor used
     * @param po  the PolymorphicObject to be created
     */
    GKO_LOGGER_REGISTER_EVENT(8, polymorphic_object_create_started,
                              const Executor* exec, const PolymorphicObject* po)

    /**
     * PolymorphicObject's create completed event.
     *
     * @param exec  the executor used
     * @param input  the PolymorphicObject used as model for the creation
     * @param output  the PolymorphicObject which was created
     */
    GKO_LOGGER_REGISTER_EVENT(9, polymorphic_object_create_completed,
                              const Executor* exec,
                              const PolymorphicObject* input,
                              const PolymorphicObject* output)

    /**
     * PolymorphicObject's copy started event.
     *
     * @param exec  the executor used
     * @param input  the PolymorphicObject to be copied from
     * @param output  the PolymorphicObject to be copied to
     */
    GKO_LOGGER_REGISTER_EVENT(10, polymorphic_object_copy_started,
                              const Executor* exec,
                              const PolymorphicObject* input,
                              const PolymorphicObject* output)

    /**
     * PolymorphicObject's copy completed event.
     *
     * @param exec  the executor used
     * @param input  the PolymorphicObject to be copied from
     * @param output  the PolymorphicObject to be copied to
     */
    GKO_LOGGER_REGISTER_EVENT(11, polymorphic_object_copy_completed,
                              const Executor* exec,
                              const PolymorphicObject* input,
                              const PolymorphicObject* output)

    /**
     * PolymorphicObject's deleted event.

     * @param exec  the executor used
     * @param po  the PolymorphicObject to be deleted
     */
    GKO_LOGGER_REGISTER_EVENT(12, polymorphic_object_deleted,
                              const Executor* exec, const PolymorphicObject* po)

    /**
     * LinOp's apply started event.
     *
     * @param A  the system matrix
     * @param b  the input vector(s)
     * @param x  the output vector(s)
     */
    GKO_LOGGER_REGISTER_EVENT(13, linop_apply_started, const LinOp* A,
                              const LinOp* b, const LinOp* x)

    /**
     * LinOp's apply completed event.
     *
     * @param A  the system matrix
     * @param b  the input vector(s)
     * @param x  the output vector(s)
     */
    GKO_LOGGER_REGISTER_EVENT(14, linop_apply_completed, const LinOp* A,
                              const LinOp* b, const LinOp* x)

    /**
     * LinOp's advanced apply started event.
     *
     * @param A  the system matrix
     * @param alpha  scaling of the result of op(b)
     * @param b  the input vector(s)
     * @param beta  scaling of the input x
     * @param x  the output vector(s)
     */
    GKO_LOGGER_REGISTER_EVENT(15, linop_advanced_apply_started, const LinOp* A,
                              const LinOp* alpha, const LinOp* b,
                              const LinOp* beta, const LinOp* x)

    /**
     * LinOp's advanced apply completed event.
     *
     * @param A  the system matrix
     * @param alpha  scaling of the result of op(b)
     * @param b  the input vector(s)
     * @param beta  scaling of the input x
     * @param x  the output vector(s)
     */
    GKO_LOGGER_REGISTER_EVENT(16, linop_advanced_apply_completed,
                              const LinOp* A, const LinOp* alpha,
                              const LinOp* b, const LinOp* beta, const LinOp* x)

    /**
     * LinOp Factory's generate started event.
     *
     * @param factory  the factory used
     * @param input  the LinOp object used as input for the generation (usually
     *               a system matrix)
     */
    GKO_LOGGER_REGISTER_EVENT(17, linop_factory_generate_started,
                              const LinOpFactory* factory, const LinOp* input)

    /**
     * LinOp Factory's generate completed event.
     *
     * @param factory  the factory used
     * @param input  the LinOp object used as input for the generation (usually
     *               a system matrix)
     * @param output  the generated LinOp object
     */
    GKO_LOGGER_REGISTER_EVENT(18, linop_factory_generate_completed,
                              const LinOpFactory* factory, const LinOp* input,
                              const LinOp* output)

    /**
     * stop::Criterion's check started event.
     *
     * @param criterion  the criterion used
     * @param it  the current iteration count
     * @param r  the residual
     * @param tau  the residual norm
     * @param x  the solution
     * @param stopping_id  the id of the stopping criterion
     * @param set_finalized  whether this finalizes the iteration
     */
    GKO_LOGGER_REGISTER_EVENT(19, criterion_check_started,
                              const stop::Criterion* criterion,
                              const size_type& it, const LinOp* r,
                              const LinOp* tau, const LinOp* x,
                              const uint8& stopping_id,
                              const bool& set_finalized)

    /**
     * stop::Criterion's check completed event. Parameters are the Criterion,
     * the stoppingId, the finalized boolean, the stopping status, plus the
     * output one_changed boolean and output all_converged boolean.
     *
     * @param criterion  the criterion used
     * @param it  the current iteration count
     * @param r  the residual
     * @param tau  the residual norm
     * @param x  the solution
     * @param stopping_id  the id of the stopping criterion
     * @param set_finalized  whether this finalizes the iteration
     * @param status  the stopping status of the right hand sides
     * @param one_changed  whether at least one right hand side converged or not
     * @param all_converged  whether all right hand sides
     *
     * @note The on_criterion_check_completed function that this macro declares
     * is deprecated. Please use the one with the additional implicit_tau_sq
     * parameter as below.
     */
    GKO_LOGGER_REGISTER_EVENT(
        20, criterion_check_completed, const stop::Criterion* criterion,
        const size_type& it, const LinOp* r, const LinOp* tau, const LinOp* x,
        const uint8& stopping_id, const bool& set_finalized,
        const array<stopping_status>* status, const bool& one_changed,
        const bool& all_converged)
protected:
    /**
     * stop::Criterion's check completed event. Parameters are the Criterion,
     * the stoppingId, the finalized boolean, the stopping status, plus the
     * output one_changed boolean and output all_converged boolean.
     *
     * @param criterion  the criterion used
     * @param it  the current iteration count
     * @param r  the residual
     * @param tau  the residual norm
     * @param implicit_tau_sq  the implicit residual norm squared
     * @param x  the solution
     * @param stopping_id  the id of the stopping criterion
     * @param set_finalized  whether this finalizes the iteration
     * @param status  the stopping status of the right hand sides
     * @param one_changed  whether at least one right hand side converged or not
     * @param all_converged  whether all right hand sides are converged
     */
    virtual void on_criterion_check_completed(
        const stop::Criterion* criterion, const size_type& it, const LinOp* r,
        const LinOp* tau, const LinOp* implicit_tau_sq, const LinOp* x,
        const uint8& stopping_id, const bool& set_finalized,
        const array<stopping_status>* status, const bool& one_changed,
        const bool& all_converged) const
    {
        this->on_criterion_check_completed(criterion, it, r, tau, x,
                                           stopping_id, set_finalized, status,
                                           one_changed, all_converged);
    }

public:
    static constexpr size_type iteration_complete{21};
    static constexpr mask_type iteration_complete_mask{mask_type{1} << 21};

    template <size_type Event, typename... Params>
    std::enable_if_t<Event == 21 && (21 < event_count_max)> on(
        Params&&... params) const
    {
        if (enabled_events_ & (mask_type{1} << 21)) {
            this->on_iteration_complete(std::forward<Params>(params)...);
        }
    }

protected:
    /**
     * Register the `iteration_complete` event which logs every completed
     * iterations.
     *
     * @param it  the current iteration count
     * @param r  the residual
     * @param x  the solution vector (optional)
     * @param tau  the residual norm (optional)
     *
     * @warning This on_iteration_complete function that this macro declares is
     * deprecated. Please use the version with the stopping information.
     */
    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    virtual void on_iteration_complete(const LinOp* solver, const size_type& it,
                                       const LinOp* r, const LinOp* x = nullptr,
                                       const LinOp* tau = nullptr) const
    {}

    /**
     * Register the `iteration_complete` event which logs every completed
     * iterations.
     *
     * @param it  the current iteration count
     * @param r  the residual
     * @param x  the solution vector (optional)
     * @param tau  the residual norm (optional)
     * @param implicit_tau_sq  the implicit residual norm squared (optional)
     *
     * @warning This on_iteration_complete function that this macro declares is
     * deprecated. Please use the version with the stopping information.
     */
    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    virtual void on_iteration_complete(const LinOp* solver, const size_type& it,
                                       const LinOp* r, const LinOp* x,
                                       const LinOp* tau,
                                       const LinOp* implicit_tau_sq) const
    {
        GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS
        this->on_iteration_complete(solver, it, r, x, tau);
        GKO_END_DISABLE_DEPRECATION_WARNINGS
    }

    /**
     * Register the `iteration_complete` event which logs every completed
     * iterations.
     *
     * @param solver  the solver executing the iteration
     * @param b  the right-hand-side vector
     * @param x  the solution vector
     * @param it  the current iteration count
     * @param r  the residual (optional)
     * @param tau  the implicit residual norm squared (optional)
     * @param implicit_tau_sq  the residual norm (optional)
     * @param status  the stopping status of the right hand sides (optional)
     * @param stopped  whether all right hand sides have stopped (invalid if
     *                 status is not provided)
     */
    virtual void on_iteration_complete(const LinOp* solver, const LinOp* b,
                                       const LinOp* x, const size_type& it,
                                       const LinOp* r, const LinOp* tau,
                                       const LinOp* implicit_tau_sq,
                                       const array<stopping_status>* status,
                                       bool stopped) const
    {
        GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS
        this->on_iteration_complete(solver, it, r, x, tau, implicit_tau_sq);
        GKO_END_DISABLE_DEPRECATION_WARNINGS
    }

public:
    /**
     * PolymorphicObject's move started event.
     *
     * @param exec  the executor used
     * @param input  the PolymorphicObject to be move from
     * @param output  the PolymorphicObject to be move into
     */
    GKO_LOGGER_REGISTER_EVENT(22, polymorphic_object_move_started,
                              const Executor* exec,
                              const PolymorphicObject* input,
                              const PolymorphicObject* output)

    /**
     * PolymorphicObject's move completed event.
     *
     * @param exec  the executor used
     * @param input  the PolymorphicObject to be move from
     * @param output  the PolymorphicObject to be move into
     */
    GKO_LOGGER_REGISTER_EVENT(23, polymorphic_object_move_completed,
                              const Executor* exec,
                              const PolymorphicObject* input,
                              const PolymorphicObject* output)

    /**
     * BatchLinOp Factory's generate started event.
     *
     * @param factory  the factory used
     * @param input  the BatchLinOp object used as input for the generation
     * (usually a system matrix)
     */
    GKO_LOGGER_REGISTER_EVENT(24, batch_linop_factory_generate_started,
                              const batch::BatchLinOpFactory* factory,
                              const batch::BatchLinOp* input)

    /**
     * BatchLinOp Factory's generate completed event.
     *
     * @param factory  the factory used
     * @param input  the BatchLinOp object used as input for the generation
     * (usually a system matrix)
     * @param output  the generated BatchLinOp object
     */
    GKO_LOGGER_REGISTER_EVENT(25, batch_linop_factory_generate_completed,
                              const batch::BatchLinOpFactory* factory,
                              const batch::BatchLinOp* input,
                              const batch::BatchLinOp* output)

public:
    static constexpr size_type batch_solver_completed{26};
    static constexpr mask_type batch_solver_completed_mask{mask_type{1} << 26};

    template <size_type Event, typename... Params>
    std::enable_if_t<Event == 26 && (26 < event_count_max)> on(
        Params&&... params) const
    {
        if (enabled_events_ & batch_solver_completed_mask) {
            this->on_batch_solver_completed(std::forward<Params>(params)...);
        }
    }

protected:
    /**
     * Batch solver's event that records the iteration count and the residual
     * norm.
     *
     * @param iters  the array of iteration counts.
     * @param residual_norms  the array storing the residual norms.
     */
    virtual void on_batch_solver_completed(
        const array<int>& iters, const array<double>& residual_norms) const
    {}

    /**
     * Batch solver's event that records the iteration count and the residual
     * norm.
     *
     * @param iters  the array of iteration counts.
     * @param residual_norms  the array storing the residual norms.
     */
    virtual void on_batch_solver_completed(
        const array<int>& iters, const array<float>& residual_norms) const
    {}

public:
#undef GKO_LOGGER_REGISTER_EVENT

    /**
     * Bitset Mask which activates all executor events
     */
    static constexpr mask_type executor_events_mask =
        allocation_started_mask | allocation_completed_mask |
        free_started_mask | free_completed_mask | copy_started_mask |
        copy_completed_mask;

    /**
     * Bitset Mask which activates all operation events
     */
    static constexpr mask_type operation_events_mask =
        operation_launched_mask | operation_completed_mask;

    /**
     * Bitset Mask which activates all polymorphic object events
     */
    static constexpr mask_type polymorphic_object_events_mask =
        polymorphic_object_create_started_mask |
        polymorphic_object_create_completed_mask |
        polymorphic_object_copy_started_mask |
        polymorphic_object_copy_completed_mask |
        polymorphic_object_move_started_mask |
        polymorphic_object_move_completed_mask |
        polymorphic_object_deleted_mask;

    /**
     * Bitset Mask which activates all linop events
     */
    static constexpr mask_type linop_events_mask =
        linop_apply_started_mask | linop_apply_completed_mask |
        linop_advanced_apply_started_mask | linop_advanced_apply_completed_mask;

    /**
     * Bitset Mask which activates all linop factory events
     */
    static constexpr mask_type linop_factory_events_mask =
        linop_factory_generate_started_mask |
        linop_factory_generate_completed_mask;

    /**
     * Bitset Mask which activates all batch linop factory events
     */
    static constexpr mask_type batch_linop_factory_events_mask =
        batch_linop_factory_generate_started_mask |
        batch_linop_factory_generate_completed_mask;

    /**
     * Bitset Mask which activates all criterion events
     */
    static constexpr mask_type criterion_events_mask =
        criterion_check_started_mask | criterion_check_completed_mask;

    /**
     * Returns true if this logger, when attached to an Executor, needs to be
     * forwarded all events from objects on this executor.
     */
    virtual bool needs_propagation() const { return false; }

    virtual ~Logger() = default;

protected:
    /**
     * Constructor for a Logger object.
     *
     * @param enabled_events  the events enabled for this Logger. These can be
     *                        of the following form:
     *                        1. `all_event_mask` which logs every event;
     *                        2. an OR combination of masks, e.g.
     *                           `iteration_complete_mask|linop_apply_started_mask`
     *                           which activates both of these events;
     *                        3. all events with exclusion through XOR, e.g.
     *                           `all_event_mask^linop_apply_started_mask` which
     *                           logs every event except linop's apply started
     *                           event.
     */
    GKO_DEPRECATED("use single-parameter constructor")
    explicit Logger(std::shared_ptr<const gko::Executor> exec,
                    const mask_type& enabled_events = all_events_mask)
        : Logger{enabled_events}
    {}

    /**
     * Constructor for a Logger object.
     *
     * @param enabled_events  the events enabled for this Logger. These can be
     *                        of the following form:
     *                        1. `all_event_mask` which logs every event;
     *                        2. an OR combination of masks, e.g.
     *                           `iteration_complete_mask|linop_apply_started_mask`
     *                           which activates both of these events;
     *                        3. all events with exclusion through XOR, e.g.
     *                           `all_event_mask^linop_apply_started_mask` which
     *                           logs every event except linop's apply started
     *                           event.
     */
    explicit Logger(const mask_type& enabled_events = all_events_mask)
        : enabled_events_{enabled_events}
    {}

private:
    mask_type enabled_events_;
};


/**
 * Loggable class is an interface which should be implemented by classes wanting
 * to support logging. For most cases, one can rely on the EnableLogging mixin
 * which provides a default implementation of this interface.
 */
class Loggable {
public:
    virtual ~Loggable() = default;

    /**
     * Adds a new logger to the list of subscribed loggers.
     *
     * @param logger  the logger to add
     */
    virtual void add_logger(std::shared_ptr<const Logger> logger) = 0;

    /**
     * Removes a logger from the list of subscribed loggers.
     *
     * @param logger the logger to remove
     *
     * @note The comparison is done using the logger's object unique identity.
     *       Thus, two loggers constructed in the same way are not considered
     *       equal.
     */
    virtual void remove_logger(const Logger* logger) = 0;

    void remove_logger(ptr_param<const Logger> logger)
    {
        remove_logger(logger.get());
    }

    /**
     * Returns the vector containing all loggers registered at this object.
     *
     * @return the vector containing all registered loggers.
     */
    virtual const std::vector<std::shared_ptr<const Logger>>& get_loggers()
        const = 0;

    /** Remove all loggers registered at this object. */
    virtual void clear_loggers() = 0;
};


/**
 * EnableLogging is a mixin which should be inherited by any class which wants
 * to enable logging. All the received events are passed to the loggers this
 * class contains.
 *
 * @tparam ConcreteLoggable  the object being logged [CRTP parameter]
 *
 * @tparam PolymorphicBase  the polymorphic base of this class. By default
 *                          it is Loggable. Change it if you want to use a new
 *                          superclass of `Loggable` as polymorphic base of this
 *                          class.
 */
template <typename ConcreteLoggable, typename PolymorphicBase = Loggable>
class EnableLogging : public PolymorphicBase {
public:
    void add_logger(std::shared_ptr<const Logger> logger) override
    {
        loggers_.push_back(logger);
    }

    void remove_logger(const Logger* logger) override
    {
        auto idx =
            find_if(begin(loggers_), end(loggers_),
                    [&logger](const auto& l) { return l.get() == logger; });
        if (idx != end(loggers_)) {
            loggers_.erase(idx);
        } else {
            throw OutOfBoundsError(__FILE__, __LINE__, loggers_.size(),
                                   loggers_.size());
        }
    }

    void remove_logger(ptr_param<const Logger> logger)
    {
        remove_logger(logger.get());
    }

    const std::vector<std::shared_ptr<const Logger>>& get_loggers()
        const override
    {
        return loggers_;
    }

    void clear_loggers() override { loggers_.clear(); }

private:
    /**
     * @internal
     * This struct is used to differentiate between objects that have an
     * associated executor (PolymorphicObject) and ones that don't (Executor).
     * For the ones with executor, it handles the event propagation via template
     * specialization/SFINAE.
     */
    template <size_type Event, typename ConcreteLoggableT, typename = void>
    struct propagate_log_helper {
        template <typename... Args>
        static void propagate_log(const ConcreteLoggableT*, Args&&...)
        {}
    };

    template <size_type Event, typename ConcreteLoggableT>
    struct propagate_log_helper<
        Event, ConcreteLoggableT,
        xstd::void_t<
            decltype(std::declval<ConcreteLoggableT>().get_executor())>> {
        template <typename... Args>
        static void propagate_log(const ConcreteLoggableT* loggable,
                                  Args&&... args)
        {
            const auto exec = loggable->get_executor();
            if (exec->should_propagate_log()) {
                for (auto& logger : exec->get_loggers()) {
                    if (logger->needs_propagation()) {
                        logger->template on<Event>(std::forward<Args>(args)...);
                    }
                }
            }
        }
    };

protected:
    template <size_type Event, typename... Params>
    void log(Params&&... params) const
    {
        propagate_log_helper<Event, ConcreteLoggable>::propagate_log(
            static_cast<const ConcreteLoggable*>(this),
            std::forward<Params>(params)...);
        for (auto& logger : loggers_) {
            logger->template on<Event>(std::forward<Params>(params)...);
        }
    }

    std::vector<std::shared_ptr<const Logger>> loggers_;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_LOGGER_HPP_
