// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/stop/criterion_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Setting of all statuses namespace.
 * @ref set_status
 * @ingroup set_all_statuses
 */
namespace set_all_statuses {


void set_all_statuses(std::shared_ptr<const DpcppExecutor> exec,
                      uint8 stoppingId, bool setFinalized,
                      array<stopping_status>* stop_status)
{
    auto size = stop_status->get_size();
    stopping_status* __restrict__ stop_status_ptr = stop_status->get_data();
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{size}, [=](sycl::id<1> idx_id) {
            const auto idx = idx_id[0];
            stop_status_ptr[idx].stop(stoppingId, setFinalized);
        });
    });
}


}  // namespace set_all_statuses
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
