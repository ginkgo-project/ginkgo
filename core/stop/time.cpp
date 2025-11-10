// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/stop/time.hpp"

#include <ginkgo/core/base/abstract_factory.hpp>


namespace gko {
namespace stop {


bool Time::check_impl(uint8 stoppingId, bool setFinalized,
                      array<stopping_status>* stop_status, bool* one_changed,
                      const Updater& updater)
{
    bool result = clock::now() - start_ >= time_limit_;
    if (result) {
        this->set_all_statuses(stoppingId, setFinalized, stop_status);
        *one_changed = true;
    }
    return result;
}


deferred_factory_parameter<Time::Factory> time_sec(double time)
{
    return Time::build().with_time_limit(
        std::chrono::nanoseconds{static_cast<long>(time * 1e9)});
}


deferred_factory_parameter<Time::Factory> time_ms(double time)
{
    return Time::build().with_time_limit(
        std::chrono::nanoseconds{static_cast<long>(time * 1e6)});
}


}  // namespace stop
}  // namespace gko
