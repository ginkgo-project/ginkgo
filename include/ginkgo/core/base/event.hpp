// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_EVENT_HPP_
#define GKO_PUBLIC_CORE_BASE_EVENT_HPP_


#include <memory>

namespace gko {


class Executor;

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
};


}  // namespace gko


#endif  // #ifndef GKO_PUBLIC_CORE_BASE_EVENT_HPP_
