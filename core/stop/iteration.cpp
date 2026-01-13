// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/stop/iteration.hpp"

#include "core/stop/iteration.hpp"


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


deferred_factory_parameter<const Iteration::Factory> max_iters(size_type count)
{
    return Iteration::build().with_max_iters(count);
}


deferred_factory_parameter<const CriterionFactory> min_iters(
    size_type count,
    deferred_factory_parameter<const CriterionFactory> criterion)
{
    return MinIterationWrapper::build()
        .with_min_iters(count)
        .with_inner_criterion(criterion);
}


}  // namespace stop
}  // namespace gko
