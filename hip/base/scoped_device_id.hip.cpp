/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
