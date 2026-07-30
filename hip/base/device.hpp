// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_DEVICE_HPP_
#define GKO_HIP_BASE_DEVICE_HPP_


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/export_hip.hpp>

namespace gko {
namespace kernels {
namespace hip {


/** calls hipDeviceReset on the given device. */
GKO_HIP_EXPORT void reset_device(int device_id);


/** calls hipEventDestroy on the given event. */
GKO_HIP_EXPORT void destroy_event(GKO_HIP_EVENT_STRUCT* event);


/** returns hipDeviceProp.name for the given device */
GKO_HIP_EXPORT std::string get_device_name(int device_id);


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_DEVICE_HPP_
