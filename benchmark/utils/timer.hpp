/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#endif  // HAS_HIP


// Command-line arguments
DEFINE_bool(gpu_timer, false,
            "use gpu timer based on event. It is valid only when "
            "executor is cuda or hip");


/**
 * Timer stores the timing information
 */
class Timer {
public:
    /**
     * Start the timer
     */
    void tic()
    {
        assert(tic_called_ == false);
        this->tic_impl();
        tic_called_ = true;
    }

    /**
     * Finish the timer
     */
    void toc()
    {
        assert(tic_called_ == true);
        auto sec = this->toc_impl();
        tic_called_ = false;
        this->add_record(sec);
    }

    /**
     * Get the summation of each time in seconds.
     *
     * @return the seconds of total time
     */
    double get_total_time() const { return total_duration_sec_; }

    /**
     * Get the number of repetitions.
     *
     * @return the number of repetitions
     */
    std::int64_t get_num_repetitions() const { return duration_sec_.size(); }

    /**
     * Compute the average time of repetitions in seconds
     *
     * @return the average time in seconds
     */
    double compute_average_time() const
    {
        return this->get_total_time() / this->get_num_repetitions();
    }

    /**
     * Get the vector containing the time of each repetition in seconds.
     *
     * @return the vector of time for each repetition in seconds
     */
    std::vector<double> get_time_detail() const { return duration_sec_; }

    /**
     * Get the latest result in seconds. If there is no result yet, return
     * 0.
     *
     * @return the latest result in seconds
     */
    double get_latest_time() const
    {
        if (duration_sec_.size() >= 1) {
            return duration_sec_.back();
        } else {
            return 0;
        }
    }

    /**
     * Clear the results of timer
     */
    void clear()
    {
        duration_sec_.clear();
        tic_called_ = false;
        total_duration_sec_ = 0;
    }

    /**
     * Create a timer
     */
    Timer() : tic_called_(false), total_duration_sec_(0) {}

protected:
    /**
     * Put the second result into vector
     *
     * @param sec  the second result to insert
     */
    void add_record(double sec)
    {
        // add the result;
        duration_sec_.emplace_back(sec);
        total_duration_sec_ += sec;
    }

    /**
     * The implementation of tic.
     */
    virtual void tic_impl() = 0;

    /**
     * The implementation of toc. Return the seconds result.
     *
     * @return the seconds result
     */
    virtual double toc_impl() = 0;

private:
    std::vector<double> duration_sec_;
    bool tic_called_;
    double total_duration_sec_;
};


/**
 * CpuTimer uses the synchronize of the executor and std::chrono to measure the
 * timing.
 */
class CpuTimer : public Timer {
public:
    /**
     * Create a CpuTimer
     *
     * @param exec  Executor associated to the timer
     */
    CpuTimer(std::shared_ptr<const gko::Executor> exec) : Timer(), exec_(exec)
    {}

protected:
    void tic_impl() override
    {
        exec_->synchronize();
        start_ = std::chrono::steady_clock::now();
    }

    double toc_impl() override
    {
        exec_->synchronize();
        auto stop = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration_time = stop - start_;
        return duration_time.count();
    }

private:
    std::shared_ptr<const gko::Executor> exec_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};


#ifdef HAS_CUDA


/**
 * CudaTimer uses cuda executor and cudaEvent to measure the timing.
 */
class CudaTimer : public Timer {
public:
    /**
     * Create a CudaTimer.
     *
     * @param exec  Executor which should be a CudaExecutor
     */
    CudaTimer(std::shared_ptr<const gko::Executor> exec)
        : CudaTimer(std::dynamic_pointer_cast<const gko::CudaExecutor>(exec))
    {}

    /**
     * Create a CudaTimer.
     *
     * @param exec  CudaExecutor associated to the timer
     */
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

    double toc_impl() override
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
        constexpr int sec_in_ms = 1e3;
        return static_cast<double>(duration_time) / sec_in_ms;
    }

private:
    std::shared_ptr<const gko::CudaExecutor> exec_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
    int id_;
};


#endif  // HAS_CUDA


#ifdef HAS_HIP


/**
 * HipTimer uses hip executor and hipEvent to measure the timing.
 */
class HipTimer : public Timer {
public:
    /**
     * Create a HipTimer.
     *
     * @param exec  Executor which should be a HipExecutor
     */
    HipTimer(std::shared_ptr<const gko::Executor> exec)
        : HipTimer(std::dynamic_pointer_cast<const gko::HipExecutor>(exec))
    {}

    /**
     * Create a HipTimer.
     *
     * @param exec  HipExecutor associated to the timer
     */
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

    double toc_impl() override
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
        constexpr int sec_in_ms = 1e3;
        return static_cast<double>(duration_time) / sec_in_ms;
    }

private:
    std::shared_ptr<const gko::HipExecutor> exec_;
    hipEvent_t start_;
    hipEvent_t stop_;
    int id_;
};


#endif  // HAS_HIP


/**
 * Get the timer. If the executor does not support gpu timer, still return the
 * cpu timer.
 *
 * @param exec  Executor associated to the timer
 * @param use_gpu_timer  whether to use the gpu timer
 */
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
    // No cuda/hip executor available or no gpu_timer used
    return std::make_shared<CpuTimer>(exec);
}
