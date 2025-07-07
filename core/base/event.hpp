// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_EVENT_HPP_
#define GKO_CORE_BASE_EVENT_HPP_

#include <memory>

#include <ginkgo/core/base/event.hpp>


namespace gko {


class Executor;


/**
 * NotAsyncEvent is to provide an Event implementation on unsupported executor
 * like reference. It will ensure the kernels are finished when recording this
 * event.
 */
class NotAsyncEvent : public Event {
public:
    NotAsyncEvent(std::shared_ptr<const Executor> exec);

    void synchronize() const override;
};


}  // namespace gko


#endif  // #ifndef GKO_CORE_BASE_EVENT_HPP_
