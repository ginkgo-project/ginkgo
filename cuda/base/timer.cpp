// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/timer.hpp>


#include <cuda.h>
#include <cuda_runtime_api.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/scoped_device_id.hpp"


namespace gko {


CudaTimer::CudaTimer(std::shared_ptr<const CudaExecutor> exec)
    : device_id_{exec->get_device_id()}, stream_{exec->get_stream()}
{}


void CudaTimer::init_time_point(time_point& time)
{
    detail::cuda_scoped_device_id_guard guard{device_id_};
    time.type_ = time_point::type::cuda;
    GKO_ASSERT_NO_CUDA_ERRORS(cudaEventCreate(&time.data_.cuda_event));
}


void CudaTimer::record(time_point& time)
{
    detail::cuda_scoped_device_id_guard guard{device_id_};
    GKO_ASSERT(time.type_ == time_point::type::cuda);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaEventRecord(time.data_.cuda_event, stream_));
}


void CudaTimer::wait(time_point& time)
{
    detail::cuda_scoped_device_id_guard guard{device_id_};
    GKO_ASSERT(time.type_ == time_point::type::cuda);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaEventSynchronize(time.data_.cuda_event));
}


std::chrono::nanoseconds CudaTimer::difference_async(const time_point& start,
                                                     const time_point& stop)
{
    detail::cuda_scoped_device_id_guard guard{device_id_};
    GKO_ASSERT(start.type_ == time_point::type::cuda);
    GKO_ASSERT(stop.type_ == time_point::type::cuda);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaEventSynchronize(stop.data_.cuda_event));
    float ms{};
    GKO_ASSERT_NO_CUDA_ERRORS(cudaEventElapsedTime(&ms, start.data_.cuda_event,
                                                   stop.data_.cuda_event));
    return std::chrono::nanoseconds{static_cast<int64>(ms * double{1e6})};
}


}  // namespace gko
