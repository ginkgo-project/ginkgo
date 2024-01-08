// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/time.hpp>


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


}  // namespace stop
}  // namespace gko
