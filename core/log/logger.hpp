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


#include "core/base/lin_op.hpp"
#include "core/base/std_extensions.hpp"
#include "core/base/types.hpp"


#include <bitset>
#include <memory>
#include <vector>


namespace gko {
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
 * to logs. In addition, the class design allows to not propagate empty messages
 * for events which are not tracked.
 * See #GKO_LOGGER_REGISTER_EVENT(_id, _event_name, ...).
 */
class Logger {
public:
    /**
     * The amount of events (bits) contained in the bitset. Increase for every
     * new event
     */
    static constexpr size_type event_count = 3;

    /* @internal std::bitset allows to store any number of bits */
    using mask_type = std::bitset<event_count>;

    /* Bitset Mask which activates all events */
    static const mask_type all_events_mask;

    /**
     * Constructor for a Logger object. Parameters are event mask which can be
     * of the form:
     * 1. `all_event_mask` which logs every event
     * 2. an OR combination of masks, e.g. `iteration_complete_mask|apply_mask`
     * which activates both of these events.
     * 3. all events with exclusion through XOR, e.g.
     * `all_event_mask^apply_mask` which logs every event except the apply event
     *
     * @param enabled_events the events enabled for this Logger
     */
    Logger(const mask_type &enabled_events = all_events_mask)
        : enabled_events_{enabled_events}
    {}

    virtual ~Logger() = default;

    /**
     * Helper macro to define functions and masks for each event.
     * A mask named _event_name##_mask is created for each event. `_id` is the
     * number assigned to this event and should be unique.
     *
     * @internal the templated function `on(Params)` will trigged the event call
     * only if the user activates this event through the mask. If the event is
     * activated, we rely on polymorphism and the virtual method
     * `on_##_event_name` to either call the Logger class's function, which does
     * nothing, or the overriden version in the derived class if any.
     */
#define GKO_LOGGER_REGISTER_EVENT(_id, _event_name, ...)             \
protected:                                                           \
    virtual void on_##_event_name(__VA_ARGS__) const {}              \
                                                                     \
public:                                                              \
    template <size_type Event, typename... Params>                   \
    xstd::enable_if_t<Event == _id> on(Params &&... params) const    \
    {                                                                \
        if (enabled_events_[_id]) {                                  \
            this->on_##_event_name(std::forward<Params>(params)...); \
        }                                                            \
    }                                                                \
    static constexpr size_type _event_name{_id};                     \
    static constexpr mask_type _event_name##_mask                    \
    {                                                                \
        mask_type { 0b1 << _id }                                     \
    }

    GKO_LOGGER_REGISTER_EVENT(0, iteration_complete,
                              const size_type num_iterations);
    GKO_LOGGER_REGISTER_EVENT(1, apply);
    GKO_LOGGER_REGISTER_EVENT(2, converged, const size_type at_iteration,
                              const LinOp *residual);
    // register other events

#undef GKO_LOGGER_REGISTER_EVENT

private:
    mask_type enabled_events_;
};


/**
 * Loggable class is an interface used as base class to EnableLogging.
 */
class Loggable {
public:
    virtual ~Loggable() = default;

    /**
     * Adds a Logger object to log events to.
     * @param logger a shared_ptr to the logger object
     */
    virtual void add_logger(std::shared_ptr<const Logger> logger) = 0;
};


/**
 * EnableLogging should be inherited by any class which wants to enable logging.
 * All the received events are passed to the loggers this class contains.
 */
class EnableLogging : public Loggable {
public:
    void add_logger(std::shared_ptr<const Logger> logger) override
    {
        loggers_.push_back(std::move(logger));
    }

protected:
    template <size_type Event, typename... Params>
    void log(Params &&... params) const
    {
        for (auto &logger : loggers_) {
            logger->on<Event>(std::forward<Params>(params)...);
        }
    }

    std::vector<std::shared_ptr<const Logger>> loggers_;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOGGER_HPP_
