// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/device.hpp>


#include <hip/hip_runtime.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/stream.hpp>


#include "hip/base/scoped_device_id.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


void reset_device(int device_id)
{
    gko::detail::hip_scoped_device_id_guard guard{device_id};
    hipDeviceReset();
}


void destroy_event(GKO_HIP_EVENT_STRUCT* event)
{
    GKO_ASSERT_NO_HIP_ERRORS(hipEventDestroy(event));
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko
