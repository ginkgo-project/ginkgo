// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/stop/criterion_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Set all statuses namespace.
 * @ref set_status
 * @ingroup set_all_statuses
 */
namespace set_all_statuses {


constexpr int default_block_size = 512;


__global__ __launch_bounds__(default_block_size) void set_all_statuses(
    size_type num_elems, uint8 stoppingId, bool setFinalized,
    stopping_status* stop_status)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_elems) {
        stop_status[tidx].stop(stoppingId, setFinalized);
    }
}


void set_all_statuses(std::shared_ptr<const HipExecutor> exec, uint8 stoppingId,
                      bool setFinalized, array<stopping_status>* stop_status)
{
    const auto block_size = default_block_size;
    const auto grid_size = ceildiv(stop_status->get_size(), block_size);

    if (grid_size > 0) {
        set_all_statuses<<<grid_size, block_size, 0, exec->get_stream()>>>(
            stop_status->get_size(), stoppingId, setFinalized,
            as_device_type(stop_status->get_data()));
    }
}


}  // namespace set_all_statuses
}  // namespace hip
}  // namespace kernels
}  // namespace gko
