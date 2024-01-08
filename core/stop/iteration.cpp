// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/iteration.hpp>


namespace gko {
namespace stop {


bool Iteration::check_impl(uint8 stoppingId, bool setFinalized,
                           array<stopping_status>* stop_status,
                           bool* one_changed, const Updater& updater)
{
    bool result = updater.num_iterations_ >= parameters_.max_iters;
    if (result) {
        this->set_all_statuses(stoppingId, setFinalized, stop_status);
        *one_changed = true;
    }
    return result;
}


}  // namespace stop
}  // namespace gko
