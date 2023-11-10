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


std::string get_device_name(int device_id)
{
    hipDeviceProp_t prop;
    GKO_ASSERT_NO_HIP_ERRORS(hipGetDeviceProperties(&prop, device_id));
    return {prop.name};
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko
