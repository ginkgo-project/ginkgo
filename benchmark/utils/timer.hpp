/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/ginkgo.hpp>


#include <chrono>
#include <memory>


#include <gflags/gflags.h>


#ifdef HAS_CUDA


#include <cuda.h>
#include <cuda_runtime.h>


#include "cuda/base/device_guard.hpp"


#endif  // HAS_CUDA


#ifdef HAS_HIP


#include <hip/hip_runtime.h>


#include "hip/base/device_guard.hip.hpp"


#endif  // HAS_CUDA


DEFINE_bool(gpu_timer, false,
            "use gpu timer based on event. It is valid only when "
            "executor is cuda or hip");


class Timer {
public:
    void tic()
    {
        assert(tic_called_ == false);
        this->tic_impl();
        tic_called_ = true;
    }

    std::size_t toc()
    {
        assert(tic_called_ == true);
        auto ns = this->toc_impl();
        tic_called_ = false;
        this->add_record(ns);
        return ns;
    }

    std::size_t get_total_time() { return total_duration_ns_; }

    std::size_t get_tictoc_num() { return duration_ns_.size(); }

    double get_average_time()
    {
        return static_cast<double>(this->get_total_time()) /
               this->get_tictoc_num();
    }

    void clear()
    {
        duration_ns_.clear();
        tic_called_ = false;
        total_duration_ns_ = 0;
    }

    Timer() : tic_called_(false), total_duration_ns_(0) {}

protected:
    void add_record(std::size_t ns)
    {
        // add the result;
        duration_ns_.emplace_back(ns);
        total_duration_ns_ += ns;
    }

    virtual void tic_impl() = 0;

    virtual std::size_t toc_impl() = 0;

private:
    std::vector<std::size_t> duration_ns_;
    bool tic_called_;
    std::size_t total_duration_ns_;
};


class CpuTimer : public Timer {
public:
    CpuTimer(std::shared_ptr<const gko::Executor> exec) : Timer(), exec_(exec)
    {}

protected:
    void tic_impl() override
    {
        exec_->synchronize();
        start_ = std::chrono::steady_clock::now();
    }

    std::size_t toc_impl() override
    {
        exec_->synchronize();
        auto stop = std::chrono::steady_clock::now();
        auto duration_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start_)
                .count();
        return static_cast<std::size_t>(duration_time);
    }

private:
    std::shared_ptr<const gko::Executor> exec_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};


#ifdef HAS_CUDA


class CudaTimer : public Timer {
public:
    CudaTimer(std::shared_ptr<const gko::Executor> exec)
        : CudaTimer(std::dynamic_pointer_cast<const gko::CudaExecutor>(exec))
    {}

    CudaTimer(std::shared_ptr<const gko::CudaExecutor> exec) : Timer()
    {
        assert(exec != nullptr);
        exec_ = exec;
        id_ = exec_->get_device_id();
        gko::cuda::device_guard g{id_};
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventCreate(&start_));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventCreate(&stop_));
    }

protected:
    void tic_impl() override
    {
        gko::cuda::device_guard g{id_};
        // Currently, gko::CudaExecutor always use default stream.
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventRecord(start_));
    }

    std::size_t toc_impl() override
    {
        gko::cuda::device_guard g{id_};
        // Currently, gko::CudaExecutor always use default stream.
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventRecord(stop_));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventSynchronize(stop_));
        float duration_time = 0;
        // cudaEventElapsedTime gives the duration_time in milliseconds with a
        // resolution of around 0.5 microseconds
        GKO_ASSERT_NO_CUDA_ERRORS(
            cudaEventElapsedTime(&duration_time, start_, stop_));
        return static_cast<std::size_t>(duration_time * 1e6);
    }

private:
    std::shared_ptr<const gko::CudaExecutor> exec_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
    int id_;
};


#endif  // HAS_CUDA


#ifdef HAS_HIP


class HipTimer : public Timer {
public:
    HipTimer(std::shared_ptr<const gko::Executor> exec)
        : HipTimer(std::dynamic_pointer_cast<const gko::HipExecutor>(exec))
    {}

    HipTimer(std::shared_ptr<const gko::HipExecutor> exec) : Timer()
    {
        assert(exec != nullptr);
        exec_ = exec;
        id_ = exec_->get_device_id();
        gko::hip::device_guard g{id_};
        GKO_ASSERT_NO_HIP_ERRORS(hipEventCreate(&start_));
        GKO_ASSERT_NO_HIP_ERRORS(hipEventCreate(&stop_));
    }

protected:
    void tic_impl() override
    {
        gko::hip::device_guard g{id_};
        // Currently, gko::HipExecutor always use default stream.
        GKO_ASSERT_NO_HIP_ERRORS(hipEventRecord(start_));
    }

    std::size_t toc_impl() override
    {
        gko::hip::device_guard g{id_};
        // Currently, gko::HipExecutor always use default stream.
        GKO_ASSERT_NO_HIP_ERRORS(hipEventRecord(stop_));
        GKO_ASSERT_NO_HIP_ERRORS(hipEventSynchronize(stop_));
        float duration_time = 0;
        // hipEventElapsedTime gives the duration_time in milliseconds with a
        // resolution of around 0.5 microseconds
        GKO_ASSERT_NO_HIP_ERRORS(
            hipEventElapsedTime(&duration_time, start_, stop_));
        return static_cast<std::size_t>(duration_time * 1e6);
    }

private:
    std::shared_ptr<const gko::HipExecutor> exec_;
    hipEvent_t start_;
    hipEvent_t stop_;
    int id_;
};


#endif  // HAS_HIP


std::shared_ptr<Timer> get_timer(std::shared_ptr<const gko::Executor> exec,
                                 bool use_gpu_timer)
{
    if (use_gpu_timer) {
#ifdef HAS_CUDA
        if (auto cuda =
                std::dynamic_pointer_cast<const gko::CudaExecutor>(exec)) {
            return std::make_shared<CudaTimer>(cuda);
        }
#endif  // HAS_CUDA

#ifdef HAS_HIP
        if (auto hip =
                std::dynamic_pointer_cast<const gko::HipExecutor>(exec)) {
            return std::make_shared<HipTimer>(hip);
        }
#endif  // HAS_HIP
    }
    // Not use gpu_timer or not cuda/hip executor
    return std::make_shared<CpuTimer>(exec);
}
