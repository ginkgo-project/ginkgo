// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/criterion.hpp>


#include "core/stop/criterion_kernels.hpp"


namespace gko {
namespace stop {
namespace criterion {
namespace {


GKO_REGISTER_OPERATION(set_all_statuses, set_all_statuses::set_all_statuses);


}  // anonymous namespace
}  // namespace criterion


void Criterion::set_all_statuses(uint8 stoppingId, bool setFinalized,
                                 array<stopping_status>* stop_status)
{
    this->get_executor()->run(criterion::make_set_all_statuses(
        stoppingId, setFinalized, stop_status));
}


}  // namespace stop
}  // namespace gko
