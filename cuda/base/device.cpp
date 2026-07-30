// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cuda/base/device.hpp"

#include <cuda_runtime.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/export_cuda.hpp>

#include "cuda/base/scoped_device_id.hpp"


namespace gko {
namespace kernels {
namespace cuda {


GKO_CUDA_EXPORT void reset_device(int device_id)
{
    gko::detail::cuda_scoped_device_id_guard guard{device_id};
    cudaDeviceReset();
}


GKO_CUDA_EXPORT void destroy_event(CUevent_st* event)
{
    GKO_ASSERT_NO_CUDA_ERRORS(cudaEventDestroy(event));
}


GKO_CUDA_EXPORT std::string get_device_name(int device_id)
{
    cudaDeviceProp prop;
    GKO_ASSERT_NO_CUDA_ERRORS(cudaGetDeviceProperties(&prop, device_id));
    return {prop.name};
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko
