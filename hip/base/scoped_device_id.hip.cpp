/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_HIP_BASE_SCOPED_DEVICE_ID_HIP_HPP_
#define GKO_HIP_BASE_SCOPED_DEVICE_ID_HIP_HPP_


#include <exception>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/scoped_device_id.hpp>


namespace gko {
namespace detail {


hip_scoped_device_id::hip_scoped_device_id(int device_id)
    : original_device_id_{}, need_reset_{}
{
    GKO_ASSERT_NO_HIP_ERRORS(hipGetDevice(&original_device_id_));
    if (original_device_id_ != device_id) {
        GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(device_id));
        need_reset_ = true;
    }
}


hip_scoped_device_id::~hip_scoped_device_id() noexcept(false)
{
    if (need_reset_) {
        /* Ignore the error during stack unwinding for this call */
        if (std::uncaught_exception()) {
            hipSetDevice(original_device_id_);
        } else {
            GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(original_device_id_));
        }
    }
}


hip_scoped_device_id::hip_scoped_device_id(
    hip_scoped_device_id&& other) noexcept
{
    *this = std::move(other);
}


hip_scoped_device_id& hip_scoped_device_id::operator=(
    hip_scoped_device_id&& other) noexcept
{
    if (this != &other) {
        original_device_id_ = std::exchange(other.original_device_id_, 0);
        need_reset_ = std::exchange(other.need_reset_, false);
    }
    return *this;
}


}  // namespace detail
}  // namespace gko


#endif  // GKO_HIP_BASE_SCOPED_DEVICE_ID_HIP_HPP_
