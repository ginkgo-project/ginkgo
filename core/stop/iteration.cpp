// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/stop/iteration.hpp"

#include "ginkgo/core/base/abstract_factory.hpp"


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


deferred_factory_parameter<Iteration::Factory> iteration(size_type count)
{
    return Iteration::build().with_max_iters(count);
}


}  // namespace stop
}  // namespace gko
