// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_DEVICE_HPP_
#define GKO_CUDA_BASE_DEVICE_HPP_


#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace kernels {
namespace cuda {


/** calls cudaDeviceReset on the given device. */
GKO_CUDA_EXPORT void reset_device(int device_id);


/** calls cudaEventDestroy on the given event. */
GKO_CUDA_EXPORT void destroy_event(CUevent_st* event);


/** returns cudaDeviceProp.name for the given device */
GKO_CUDA_EXPORT std::string get_device_name(int device_id);


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_DEVICE_HPP_
