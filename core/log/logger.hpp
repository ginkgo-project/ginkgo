/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_LOGGER_HPP_
#define GKO_CORE_LOGGER_HPP_


#include <memory>
#include <string>
#include <vector>


#include "core/base/std_extensions.hpp"
#include "core/base/types.hpp"


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
     * Executor's allocation started event. Parameters are the executor and
     * number of bytes to allocate.
     */
    GKO_LOGGER_REGISTER_EVENT(0, allocation_started, const Executor *,
                              const size_type &)

    /**
     * Executor's allocation completed event. Parameters are the executor,
     * number of bytes allocated, memory location.
     */
    GKO_LOGGER_REGISTER_EVENT(1, allocation_completed, const Executor *,
                              const size_type &, const uintptr &)

    /**
     * Executor's free started event. Parameters are the executor and memory
     * location.
     */
    GKO_LOGGER_REGISTER_EVENT(2, free_started, const Executor *,
                              const uintptr &)

    /**
     * Executor's free completed event. Parameters are the executor and memory
     * location.
     */
    GKO_LOGGER_REGISTER_EVENT(3, free_completed, const Executor *,
                              const uintptr &)

    /**
     * Executor's copy started event. Parameters are the executor from, the
     * executor to, the location from, the, location to, the number of bytes.
     */
    GKO_LOGGER_REGISTER_EVENT(4, copy_started, const Executor *,
                              const Executor *, const uintptr &,
                              const uintptr &, const size_type &)

    /**
     * Executor's copy completed event. Parameters are the executor from, the
     * executor to, the location from, the, location to, the number of bytes.
     */
    GKO_LOGGER_REGISTER_EVENT(5, copy_completed, const Executor *,
                              const Executor *, const uintptr &,
                              const uintptr &, const size_type &)

    /**
     * Executor's operation launched event (method run). Parameters are the
     * executor and the Operation.
     */
    GKO_LOGGER_REGISTER_EVENT(6, operation_launched, const Executor *,
                              const Operation *)

    /**
     * Executor's operation completed event (method run). Parameters are the
     * executor and the Operation.
     *
     * @note For the GPU, to be certain that the operation completed it is
     * required to call synchronize. This burden falls on the logger. Most of
     * the loggers will do lightweight logging, and therefore this operation for
     * the GPU just notes that the Operation has been sent to the GPU.
     */
    GKO_LOGGER_REGISTER_EVENT(7, operation_completed, const Executor *,
                              const Operation *)

    /**
     * PolymorphicObject's create started event. Parameters are the executor and
     * the PolymorphicObject.
     */
    GKO_LOGGER_REGISTER_EVENT(8, polymorphic_object_create_started,
                              const Executor *, const PolymorphicObject *)

    /**
     * PolymorphicObject's create completed event. Parameters are the executor,
     * the model PolymorphicObject and the output.
     */
    GKO_LOGGER_REGISTER_EVENT(9, polymorphic_object_create_completed,
                              const Executor *, const PolymorphicObject *,
                              const PolymorphicObject *)

    /**
     * PolymorphicObject's copy started event. Parameters are the Executor, the
     * input PolymorphicObject and the output PolymorphicObject.
     */
    GKO_LOGGER_REGISTER_EVENT(10, polymorphic_object_copy_started,
                              const Executor *, const PolymorphicObject *,
                              const PolymorphicObject *)

    /**
     * PolymorphicObject's copy completed event. Parameters are the Executor,
     * the input PolymorphicObject and the output PolymorphicObject.
     */
    GKO_LOGGER_REGISTER_EVENT(11, polymorphic_object_copy_completed,
                              const Executor *, const PolymorphicObject *,
                              const PolymorphicObject *)

    /**
     * PolymorphicObject's deleted event. Parameters are the Executor and the
     * deleted PolymorphicObject.
     */
    GKO_LOGGER_REGISTER_EVENT(12, polymorphic_object_deleted, const Executor *,
                              const PolymorphicObject *)

    /**
     * LinOp's apply started event. Parameters are the LinOp, b and X.
     */
    GKO_LOGGER_REGISTER_EVENT(13, linop_apply_started, const LinOp *,
                              const LinOp *, const LinOp *)

    /**
     * LinOp's apply completed event. Parameters are the LinOp, b and x.
     */
    GKO_LOGGER_REGISTER_EVENT(14, linop_apply_completed, const LinOp *,
                              const LinOp *, const LinOp *)

    /**
     * LinOp's advanced apply started event. Parameters are the LinOp, alpha,
     * b, beta and x.
     */
    GKO_LOGGER_REGISTER_EVENT(15, linop_advanced_apply_started, const LinOp *,
                              const LinOp *, const LinOp *, const LinOp *,
                              const LinOp *)

    /**
     * LinOp's advanced apply completed event. Parameters are the LinOp, alpha,
     * b, beta and x.
     */
    GKO_LOGGER_REGISTER_EVENT(16, linop_advanced_apply_completed, const LinOp *,
                              const LinOp *, const LinOp *, const LinOp *,
                              const LinOp *)

    /**
     * LinOp Factory's generate started event. Parameters are the LinOpFactory
     * and the input operator.
     */
    GKO_LOGGER_REGISTER_EVENT(17, linop_factory_generate_started,
                              const LinOpFactory *, const LinOp *)

    /**
     * LinOp Factory's generate completed event. Parameters are the
     * LinOpFactory, the input operator and the output operator.
     */
    GKO_LOGGER_REGISTER_EVENT(18, linop_factory_generate_completed,
                              const LinOpFactory *, const LinOp *,
                              const LinOp *)

    /**
     * stop::Criterion's check started event. Parameters are the Criterion,
     * the stoppingId, the finalized boolean.
     */
    GKO_LOGGER_REGISTER_EVENT(19, criterion_check_started,
                              const stop::Criterion *, const uint8 &,
                              const bool &)

    /**
     * stop::Criterion's check completed event. Parameters are the Criterion,
     * the stoppingId, the finalized boolean, the stopping status, plus the
     * output one_changed boolean and output all_converged boolean.
     */
    GKO_LOGGER_REGISTER_EVENT(20, criterion_check_completed,
                              const stop::Criterion *, const uint8 &,
                              const bool &, const Array<stopping_status> *,
                              const bool &, const bool &)

    /**
     * Register the `iteration_complete` event which logs every completed
     * iterations. The parameters are the solver, the iteration count, the
     * residual, the solution vector (optional), the residual norm (optional)
     */
    GKO_LOGGER_REGISTER_EVENT(21, iteration_complete, const LinOp *,
                              const size_type &, const LinOp *,
                              const LinOp * = nullptr, const LinOp * = nullptr)


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
        criterion_check_started_mask ^ criterion_check_completed_mask;

protected:
    /**
     * Constructor for a Logger object.
     *
     * @param enabled_events  the events enabled for this Logger. These can be
     *                        of the following form:
     *                        1. `all_event_mask` which logs every event
     *                        2. an OR combination of masks, e.g.
     *                           `iteration_complete_mask|apply_mask` which
     *                            activates both of these events.
     *                        3. all events with exclusion through XOR, e.g.
     *                           `all_event_mask^apply_mask` which logs every
     *                           event except the apply event
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
     * Adds a Logger object to log events to.
     * @param logger  a shared_ptr to the logger object
     */
    virtual void add_logger(std::shared_ptr<const Logger> logger) = 0;
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
