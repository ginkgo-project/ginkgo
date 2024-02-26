// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <exception>
#include <utility>


#include <cuda_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/scoped_device_id.hpp"


namespace gko {
namespace detail {


cuda_scoped_device_id_guard::cuda_scoped_device_id_guard(int device_id)
    : original_device_id_{}, need_reset_{}
{
    GKO_ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&original_device_id_));
    if (original_device_id_ != device_id) {
        GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(device_id));
        need_reset_ = true;
    }
}


cuda_scoped_device_id_guard::~cuda_scoped_device_id_guard()
{
    if (need_reset_) {
        auto error_code = cudaSetDevice(original_device_id_);
        if (error_code != cudaSuccess) {
#if GKO_VERBOSE_LEVEL >= 1
            std::cerr
                << "Unrecoverable CUDA error while resetting the device id to "
                << original_device_id_ << " in " << __func__ << ": "
                << cudaGetErrorName(error_code) << ": "
                << cudaGetErrorString(error_code) << std::endl
                << "Exiting program" << std::endl;
#endif  // GKO_VERBOSE_LEVEL >= 1
            std::exit(error_code);
        }
    }
}


cuda_scoped_device_id_guard::cuda_scoped_device_id_guard(
    gko::detail::cuda_scoped_device_id_guard&& other) noexcept
{
    *this = std::move(other);
}


cuda_scoped_device_id_guard& cuda_scoped_device_id_guard::operator=(
    gko::detail::cuda_scoped_device_id_guard&& other) noexcept
{
    if (this != &other) {
        original_device_id_ = std::exchange(other.original_device_id_, 0);
        need_reset_ = std::exchange(other.need_reset_, false);
    }
    return *this;
}


}  // namespace detail


scoped_device_id_guard::scoped_device_id_guard(const CudaExecutor* exec,
                                               int device_id)
    : scope_(std::make_unique<detail::cuda_scoped_device_id_guard>(device_id))
{}


}  // namespace gko
