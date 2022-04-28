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

#include <cuda.h>
#include <cuda_runtime.h>


#include "benchmark/utils/timer_impl.hpp"


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
        auto guard = exec_->get_scoped_device_id_guard();
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventCreate(&start_));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventCreate(&stop_));
    }

protected:
    void tic_impl() override
    {
        exec_->synchronize();
        auto guard = exec_->get_scoped_device_id_guard();
        // Currently, gko::CudaExecutor always use default stream.
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventRecord(start_));
    }

    double toc_impl() override
    {
        auto guard = exec_->get_scoped_device_id_guard();
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
};


std::shared_ptr<Timer> get_cuda_timer(
    std::shared_ptr<const gko::CudaExecutor> exec)
{
    return std::make_shared<CudaTimer>(exec);
}
