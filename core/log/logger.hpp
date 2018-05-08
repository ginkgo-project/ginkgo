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


class Logger {
public:
    static constexpr size_type event_count = 3;

    using mask_type = std::bitset<event_count>;

    static const mask_type all_events_mask;

    Logger(const mask_type &enabled_events = all_events_mask)
        : enabled_events_{enabled_events}
    {}

    virtual ~Logger() = default;

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

    GKO_LOGGER_REGISTER_EVENT(0, iteration_complete, size_type num_iterations);
    GKO_LOGGER_REGISTER_EVENT(1, apply);
    GKO_LOGGER_REGISTER_EVENT(2, converged, size_type at_iteration,
                              LinOp *residual);
    // register other events

#undef GKO_LOGGER_REGISTER_EVENT

private:
    mask_type enabled_events_;
};


class Loggable {
public:
    virtual ~Loggable() = default;

    virtual void add_logger(std::shared_ptr<const Logger> logger) = 0;
};


class EnableLogging : public Loggable {
public:
    void add_logger(std::shared_ptr<const Logger> logger) override
    {
        loggers.push_back(std::move(logger));
    }

protected:
    template <size_type Event, typename... Params>
    void log(Params &&... params) const
    {
        for (auto &logger : loggers) {
            logger->on<Event>(std::forward<Params>(params)...);
        }
    }

    std::vector<std::shared_ptr<const Logger>> loggers;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOGGER_HPP_
