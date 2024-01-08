// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_DEVICE_HPP_
#define GKO_CUDA_BASE_DEVICE_HPP_


#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace kernels {
namespace cuda {


/** calls cudaDeviceReset on the given device. */
void reset_device(int device_id);


/** calls cudaEventDestroy on the given event. */
void destroy_event(CUevent_st* event);


/** returns cudaDeviceProp.name for the given device */
std::string get_device_name(int device_id);


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_DEVICE_HPP_
