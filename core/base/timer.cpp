// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/timer.hpp>


#include <chrono>
#include <memory>
#include <utility>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "cuda/base/device.hpp"
#include "dpcpp/base/device.hpp"
#include "hip/base/device.hpp"


namespace gko {


time_point::data_union::data_union() : chrono{} {}


time_point::time_point() : type_{type::cpu}, data_{} {}


time_point::~time_point()
{
    switch (type_) {
    case type::cuda:
        kernels::cuda::destroy_event(data_.cuda_event);
        break;
    case type::hip:
        kernels::hip::destroy_event(data_.hip_event);
        break;
    case type::dpcpp:
        kernels::dpcpp::destroy_event(data_.dpcpp_event);
        break;
    case type::cpu:
    default:
        break;
    }
}


time_point::time_point(time_point&& other)
    : type_{std::exchange(other.type_, type::cpu)},
      data_{std::exchange(other.data_, decltype(data_){})}
{}


time_point& time_point::operator=(time_point&& other)
{
    // make sure we release the data via other
    std::swap(type_, other.type_);
    std::swap(data_, other.data_);
    return *this;
}


time_point Timer::create_time_point()
{
    time_point time;
    this->init_time_point(time);
    return time;
}


std::chrono::nanoseconds Timer::difference(time_point& start, time_point& stop)
{
    this->wait(stop);
    return this->difference_async(start, stop);
}


void CpuTimer::init_time_point(time_point& time)
{
    time.type_ = time_point::type::cpu;
}


void CpuTimer::record(time_point& time)
{
    GKO_ASSERT(time.type_ == time_point::type::cpu);
    time.data_.chrono = std::chrono::steady_clock::now();
}


void CpuTimer::wait(time_point& time) {}


std::chrono::nanoseconds CpuTimer::difference_async(const time_point& start,
                                                    const time_point& stop)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        stop.data_.chrono - start.data_.chrono);
}


std::unique_ptr<Timer> Timer::create_for_executor(
    std::shared_ptr<const Executor> exec)
{
    if (auto cuda_exec = std::dynamic_pointer_cast<const CudaExecutor>(exec)) {
        return std::make_unique<CudaTimer>(cuda_exec);
    } else if (auto hip_exec =
                   std::dynamic_pointer_cast<const HipExecutor>(exec)) {
        return std::make_unique<HipTimer>(hip_exec);
    } else if (auto dpcpp_exec =
                   std::dynamic_pointer_cast<const DpcppExecutor>(exec)) {
        return std::make_unique<DpcppTimer>(dpcpp_exec);
    } else {
        return std::make_unique<CpuTimer>();
    }
}


}  // namespace gko
