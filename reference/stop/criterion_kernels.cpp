// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/stop/criterion_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief Setting of all statuses.
 * @ref set_status
 * @ingroup set_all_statuses
 */
namespace set_all_statuses {


void set_all_statuses(std::shared_ptr<const ReferenceExecutor> exec,
                      uint8 stoppingId, bool setFinalized,
                      array<stopping_status>* stop_status)
{
    for (int i = 0; i < stop_status->get_size(); i++) {
        stop_status->get_data()[i].stop(stoppingId, setFinalized);
    }
}


}  // namespace set_all_statuses
}  // namespace reference
}  // namespace kernels
}  // namespace gko
