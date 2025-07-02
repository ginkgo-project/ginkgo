// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_EVENT_HPP_
#define GKO_PUBLIC_CORE_BASE_EVENT_HPP_


#include <memory>

namespace gko {


class Executor;


class Event {
public:
    virtual void synchronize() const = 0;
};


class NotAsyncEvent : public Event {
public:
    NotAsyncEvent(std::shared_ptr<const Executor> exec);

    void synchronize() const {

    };
};


}  // namespace gko


#endif  // #ifndef GKO_PUBLIC_CORE_BASE_EVENT_HPP_
