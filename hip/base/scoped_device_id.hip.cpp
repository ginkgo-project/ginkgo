// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <exception>
#include <utility>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "hip/base/scoped_device_id.hip.hpp"


namespace gko {
namespace detail {


hip_scoped_device_id_guard::hip_scoped_device_id_guard(int device_id)
    : original_device_id_{}, need_reset_{}
{
    GKO_ASSERT_NO_HIP_ERRORS(hipGetDevice(&original_device_id_));
    if (original_device_id_ != device_id) {
        GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(device_id));
        need_reset_ = true;
    }
}


hip_scoped_device_id_guard::~hip_scoped_device_id_guard()
{
    if (need_reset_) {
        auto error_code = hipSetDevice(original_device_id_);
        if (error_code != hipSuccess) {
#if GKO_VERBOSE_LEVEL >= 1
            std::cerr
                << "Unrecoverable CUDA error while resetting the device id to "
                << original_device_id_ << " in " << __func__ << ": "
                << hipGetErrorName(error_code) << ": "
                << hipGetErrorString(error_code) << std::endl
                << "Exiting program" << std::endl;
#endif  // GKO_VERBOSE_LEVEL >= 1
            std::exit(error_code);
        }
    }
}


hip_scoped_device_id_guard::hip_scoped_device_id_guard(
    hip_scoped_device_id_guard&& other) noexcept
{
    *this = std::move(other);
}


hip_scoped_device_id_guard& hip_scoped_device_id_guard::operator=(
    gko::detail::hip_scoped_device_id_guard&& other) noexcept
{
    if (this != &other) {
        original_device_id_ = std::exchange(other.original_device_id_, 0);
        need_reset_ = std::exchange(other.need_reset_, false);
    }
    return *this;
}


}  // namespace detail


scoped_device_id_guard::scoped_device_id_guard(const HipExecutor* exec,
                                               int device_id)
    : scope_(std::make_unique<detail::hip_scoped_device_id_guard>(device_id))
{}


}  // namespace gko
