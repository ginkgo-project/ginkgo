// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_EVENT_HPP_
#define GKO_CORE_BASE_EVENT_HPP_

#include <memory>

#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace detail {

/**
 * NotAsyncEvent is to provide an Event implementation on unsupported executor
 * like reference. It will ensure the kernels are finished when recording this
 * event.
 */
class NotAsyncEvent : public Event {
public:
    NotAsyncEvent(std::shared_ptr<const Executor> exec) { exec->synchronize(); }

    void synchronize() const override
    {
        // we have sync in the recording phase
    }
};


}  // namespace detail
}  // namespace gko


#endif  // #ifndef GKO_CORE_BASE_EVENT_HPP_
