// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_EVENT_HPP_
#define GKO_PUBLIC_CORE_BASE_EVENT_HPP_


#include <memory>


namespace gko {


class Executor;


namespace detail {


/**
 * Event is to create a object to record between kernels. It provides
 * synchronize functions such that we can ensure the kernels before the event in
 * the same pipeline are finished.
 */
class Event {
public:
    /**
     * synchronize on this event, all function before recording must be finished
     * before return from this function.
     */
    virtual void synchronize() const = 0;

    /**
     * Makes `exec`'s queue wait for this event without blocking the host.
     * Defaults to synchronize(), which is correct but blocking.
     *
     * @param exec  the executor whose queue should wait.
     */
    virtual void stream_wait(std::shared_ptr<const Executor> exec) const
    {
        this->synchronize();
    }

    virtual ~Event() = default;
};


}  // namespace detail
}  // namespace gko


#endif  // #ifndef GKO_PUBLIC_CORE_BASE_EVENT_HPP_
