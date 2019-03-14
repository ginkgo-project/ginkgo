/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_LOGGER_HPP_
#define GKO_CORE_LOGGER_HPP_


#include <algorithm>
#include <memory>
#include <string>
#include <vector>


#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


/* Eliminate circular dependencies the hard way */
template <typename ValueType>
class Array;
class Executor;
class LinOp;
class LinOpFactory;
class PolymorphicObject;
class Operation;
class stopping_status;

/**
 * @brief The Stopping criterion namespace .
 * @ref stop
 * @ingroup stop
 */
namespace stop {
class Criterion;
}  // namespace stop


namespace log {


/**
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
 *
 * @ingroup log
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
     * which does nothing, or the overriden version in the derived class if
     * any. Therefore, to support a new event in any Logger (i.e. class
     * which derive from this class), the function `on_##_event_name()`
     * should be overriden and implemented.
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
    xstd::enable_if_t<Event == _id && (_id < event_count_max)> on(   \
        Params &&... params) const                                   \
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
    GKO_LOGGER_REGISTER_EVENT(0, allocation_started, const Executor *exec,
                              const size_type &num_bytes)

    /**
     * Executor's allocation completed event.
     *
     * @param exec  the executor used
     * @param num_bytes  the number of bytes allocated
     * @param location  the address at which the data was allocated
     */
    GKO_LOGGER_REGISTER_EVENT(1, allocation_completed, const Executor *exec,
                              const size_type &num_bytes,
                              const uintptr &location)

    /**
     * Executor's free started event.
     *
     * @param exec  the executor used
     * @param location  the address at which the data will be freed
     */
    GKO_LOGGER_REGISTER_EVENT(2, free_started, const Executor *exec,
                              const uintptr &location)

    /**
     * Executor's free completed event.
     *
     * @param exec  the executor used
     * @param location  the address at which the data was freed
     */
    GKO_LOGGER_REGISTER_EVENT(3, free_completed, const Executor *exec,
                              const uintptr &location)

    /**
     * Executor's copy started event.

     * @param exec_from  the executor to be copied from
     * @param exec_to  the executor to be copied to
     * @param loc_from  the address at which the data will be copied from
     * @param loc_to  the address at which the data will be copied to
     * @param num_bytes  the number of bytes to be copied
     */
    GKO_LOGGER_REGISTER_EVENT(4, copy_started, const Executor *exec_from,
                              const Executor *exec_to, const uintptr &loc_from,
                              const uintptr &loc_to, const size_type &num_bytes)

    /**
     * Executor's copy completed event.
     *
     * @param exec_from  the executor copied from
     * @param exec_to  the executor copied to
     * @param loc_from  the address at which the data was copied from
     * @param loc_to  the address at which the data was copied to
     * @param num_bytes  the number of bytes copied
     */
    GKO_LOGGER_REGISTER_EVENT(5, copy_completed, const Executor *exec_from,
                              const Executor *exec_to, const uintptr &loc_from,
                              const uintptr &loc_to, const size_type &num_bytes)

    /**
     * Executor's operation launched event (method run).
     *
     * @param exec  the executor used
     * @param op  the operation launched
     */
    GKO_LOGGER_REGISTER_EVENT(6, operation_launched, const Executor *exec,
                              const Operation *op)

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
    GKO_LOGGER_REGISTER_EVENT(7, operation_completed, const Executor *exec,
                              const Operation *op)

    /**
     * PolymorphicObject's create started event.
     *
     * @param exec  the executor used
     * @param po  the PolymorphicObject to be created
     */
    GKO_LOGGER_REGISTER_EVENT(8, polymorphic_object_create_started,
                              const Executor *exec, const PolymorphicObject *po)

    /**
     * PolymorphicObject's create completed event.
     *
     * @param exec  the executor used
     * @param input  the PolymorphicObject used as model for the creation
     * @param output  the PolymorphicObject which was created
     */
    GKO_LOGGER_REGISTER_EVENT(9, polymorphic_object_create_completed,
                              const Executor *exec,
                              const PolymorphicObject *input,
                              const PolymorphicObject *output)

    /**
     * PolymorphicObject's copy started event.
     *
     * @param exec  the executor used
     * @param input  the PolymorphicObject to be copied from
     * @param output  the PolymorphicObject to be copied to
     */
    GKO_LOGGER_REGISTER_EVENT(10, polymorphic_object_copy_started,
                              const Executor *exec,
                              const PolymorphicObject *input,
                              const PolymorphicObject *output)

    /**
     * PolymorphicObject's copy completed event.
     *
     * @param exec  the executor used
     * @param input  the PolymorphicObject to be copied from
     * @param output  the PolymorphicObject to be copied to
     */
    GKO_LOGGER_REGISTER_EVENT(11, polymorphic_object_copy_completed,
                              const Executor *exec,
                              const PolymorphicObject *input,
                              const PolymorphicObject *output)

    /**
     * PolymorphicObject's deleted event.

     * @param exec  the executor used
     * @param po  the PolymorphicObject to be deleted
     */
    GKO_LOGGER_REGISTER_EVENT(12, polymorphic_object_deleted,
                              const Executor *exec, const PolymorphicObject *po)

    /**
     * LinOp's apply started event.
     *
     * @param A  the system matrix
     * @param b  the input vector(s)
     * @param x  the output vector(s)
     */
    GKO_LOGGER_REGISTER_EVENT(13, linop_apply_started, const LinOp *A,
                              const LinOp *b, const LinOp *x)

    /**
     * LinOp's apply completed event.
     *
     * @param A  the system matrix
     * @param b  the input vector(s)
     * @param x  the output vector(s)
     */
    GKO_LOGGER_REGISTER_EVENT(14, linop_apply_completed, const LinOp *A,
                              const LinOp *b, const LinOp *x)

    /**
     * LinOp's advanced apply started event.
     *
     * @param A  the system matrix
     * @param alpha  scaling of the result of op(b)
     * @param b  the input vector(s)
     * @param beta  scaling of the input x
     * @param x  the output vector(s)
     */
    GKO_LOGGER_REGISTER_EVENT(15, linop_advanced_apply_started, const LinOp *A,
                              const LinOp *alpha, const LinOp *b,
                              const LinOp *beta, const LinOp *x)

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
                              const LinOp *A, const LinOp *alpha,
                              const LinOp *b, const LinOp *beta, const LinOp *x)

    /**
     * LinOp Factory's generate started event.
     *
     * @param factory  the factory used
     * @param input  the LinOp object used as input for the generation (usually
     *               a system matrix)
     */
    GKO_LOGGER_REGISTER_EVENT(17, linop_factory_generate_started,
                              const LinOpFactory *factory, const LinOp *input)

    /**
     * LinOp Factory's generate completed event.
     *
     * @param factory  the factory used
     * @param input  the LinOp object used as input for the generation (usually
     *               a system matrix)
     * @param output  the generated LinOp object
     */
    GKO_LOGGER_REGISTER_EVENT(18, linop_factory_generate_completed,
                              const LinOpFactory *factory, const LinOp *input,
                              const LinOp *output)

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
                              const stop::Criterion *criterion,
                              const size_type &it, const LinOp *r,
                              const LinOp *tau, const LinOp *x,
                              const uint8 &stopping_id,
                              const bool &set_finalized)

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
     */
    GKO_LOGGER_REGISTER_EVENT(
        20, criterion_check_completed, const stop::Criterion *criterion,
        const size_type &it, const LinOp *r, const LinOp *tau, const LinOp *x,
        const uint8 &stopping_id, const bool &set_finalized,
        const Array<stopping_status> *status, const bool &one_changed,
        const bool &all_converged)

    /**
     * Register the `iteration_complete` event which logs every completed
     * iterations.
     *
     * @param it  the current iteration count
     * @param r  the residual
     * @param x  the solution vector (optional)
     * @param tau  the residual norm (optional)
     */
    GKO_LOGGER_REGISTER_EVENT(21, iteration_complete, const LinOp *solver,
                              const size_type &it, const LinOp *r,
                              const LinOp *x = nullptr,
                              const LinOp *tau = nullptr)


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
     * Bitset Mask which activates all criterion events
     */
    static constexpr mask_type criterion_events_mask =
        criterion_check_started_mask | criterion_check_completed_mask;

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
    explicit Logger(std::shared_ptr<const gko::Executor> exec,
                    const mask_type &enabled_events = all_events_mask)
        : exec_{exec}, enabled_events_{enabled_events}
    {}

private:
    std::shared_ptr<const Executor> exec_;
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
    virtual void remove_logger(const Logger *logger) = 0;
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
class EnableLogging : public Loggable {
public:
    void add_logger(std::shared_ptr<const Logger> logger) override
    {
        loggers_.push_back(logger);
    }

    void remove_logger(const Logger *logger) override
    {
        auto idx = find_if(begin(loggers_), end(loggers_),
                           [&logger](std::shared_ptr<const Logger> l) {
                               return lend(l) == logger;
                           });
        if (idx != end(loggers_)) {
            loggers_.erase(idx);
        } else {
            throw OutOfBoundsError(__FILE__, __LINE__, loggers_.size(),
                                   loggers_.size());
        }
    }

protected:
    template <size_type Event, typename... Params>
    void log(Params &&... params) const
    {
        for (auto &logger : loggers_) {
            logger->template on<Event>(std::forward<Params>(params)...);
        }
    }

    std::vector<std::shared_ptr<const Logger>> loggers_;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOGGER_HPP_
