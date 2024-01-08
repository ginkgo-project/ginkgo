// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/stop/criterion_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Setting of all statuses namespace.
 * @ref set_status
 * @ingroup set_all_statuses
 */
namespace set_all_statuses {


void set_all_statuses(std::shared_ptr<const OmpExecutor> exec, uint8 stoppingId,
                      bool setFinalized, array<stopping_status>* stop_status)
{
#pragma omp parallel for
    for (int i = 0; i < stop_status->get_size(); i++) {
        stop_status->get_data()[i].stop(stoppingId, setFinalized);
    }
}


}  // namespace set_all_statuses
}  // namespace omp
}  // namespace kernels
}  // namespace gko
