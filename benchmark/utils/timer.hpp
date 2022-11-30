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

#ifndef GKO_BENCHMARK_UTILS_TIMER_HPP_
#define GKO_BENCHMARK_UTILS_TIMER_HPP_


#include <ginkgo/ginkgo.hpp>


#include <memory>


#include <gflags/gflags.h>


#include "benchmark/utils/timer_impl.hpp"


// Command-line arguments
DEFINE_bool(gpu_timer, false,
            "use gpu timer based on event. It is valid only when "
            "executor is cuda or hip");


#ifdef HAS_CUDA_TIMER
std::shared_ptr<Timer> get_cuda_timer(
    std::shared_ptr<const gko::CudaExecutor> exec);
#endif  // HAS_CUDA_TIMER


#ifdef HAS_HIP_TIMER
std::shared_ptr<Timer> get_hip_timer(
    std::shared_ptr<const gko::HipExecutor> exec);
#endif  // HAS_HIP_TIMER


#ifdef HAS_DPCPP_TIMER
std::shared_ptr<Timer> get_dpcpp_timer(
    std::shared_ptr<const gko::DpcppExecutor> exec);
#endif  // HAS_DPCPP_TIMER


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
#ifdef HAS_CUDA_TIMER
        if (auto cuda =
                std::dynamic_pointer_cast<const gko::CudaExecutor>(exec)) {
            return get_cuda_timer(cuda);
        }
#endif  // HAS_CUDA_TIMER

#ifdef HAS_HIP_TIMER
        if (auto hip =
                std::dynamic_pointer_cast<const gko::HipExecutor>(exec)) {
            return get_hip_timer(hip);
        }
#endif  // HAS_HIP_TIMER

#ifdef HAS_DPCPP_TIMER
        if (auto dpcpp =
                std::dynamic_pointer_cast<const gko::DpcppExecutor>(exec)) {
            return get_dpcpp_timer(dpcpp);
        }
#endif  // HAS_DPCPP_TIMER
    }
    // No cuda/hip/dpcpp executor available or no gpu_timer used
    return std::make_shared<CpuTimer>(exec);
}


#if GINKGO_BUILD_MPI


/**
 * Get the timer. If the executor does not support gpu timer, still return the
 * cpu timer.
 *
 * @param exec  Executor associated to the timer
 * @param use_gpu_timer  whether to use the gpu timer
 */
std::shared_ptr<Timer> get_mpi_timer(std::shared_ptr<const gko::Executor> exec,
                                     gko::experimental::mpi::communicator comm,
                                     bool use_gpu_timer)
{
    return std::make_shared<MpiWrappedTimer>(exec, std::move(comm),
                                             get_timer(exec, use_gpu_timer));
}


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_BENCHMARK_UTILS_TIMER_HPP_
