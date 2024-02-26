// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
