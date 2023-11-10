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

#include <ginkgo/core/base/stream.hpp>


#include <hip/hip_runtime.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/device.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "hip/base/scoped_device_id.hip.hpp"


namespace gko {


hip_stream::hip_stream() : stream_{}, device_id_{} {}


hip_stream::hip_stream(int device_id) : stream_{}, device_id_(device_id)
{
    detail::hip_scoped_device_id_guard g(device_id_);
    GKO_ASSERT_NO_HIP_ERRORS(hipStreamCreate(&stream_));
}


hip_stream::~hip_stream()
{
    if (stream_) {
        detail::hip_scoped_device_id_guard g(device_id_);
        hipStreamDestroy(stream_);
    }
}


hip_stream::hip_stream(hip_stream&& other)
    : stream_{std::exchange(other.stream_, nullptr)},
      device_id_{std::exchange(other.device_id_, 0)}
{}


GKO_HIP_STREAM_STRUCT* hip_stream::get() const { return stream_; }


}  // namespace gko
