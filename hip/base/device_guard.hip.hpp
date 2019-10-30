/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_HIP_BASE_DEVICE_GUARD_HIP_HPP_
#define GKO_HIP_BASE_DEVICE_GUARD_HIP_HPP_


#include <exception>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace hip {


/**
 * This class defines a device guard for the hip functions and the hip module.
 * The guard is used to make sure that the device code is run on the correct
 * hip device, when run with multiple devices. The class records the current
 * device id and uses `hipSetDevice` to set the device id to the one being
 * passed in. After the scope has been exited, the destructor sets the device_id
 * back to the one before entering the scope.
 */
class device_guard {
public:
    device_guard(int device_id)
    {
        GKO_ASSERT_NO_HIP_ERRORS(hipGetDevice(&original_device_id));
        GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(device_id));
    }

    device_guard(device_guard &other) = delete;

    device_guard &operator=(const device_guard &other) = delete;

    device_guard(device_guard &&other) = delete;

    device_guard const &operator=(device_guard &&other) = delete;

    ~device_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (std::uncaught_exception()) {
            hipSetDevice(original_device_id);
        } else {
            GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(original_device_id));
        }
    }

private:
    int original_device_id{};
};


}  // namespace hip
}  // namespace gko


#endif  // GKO_HIP_BASE_DEVICE_GUARD_HIP_HPP_
