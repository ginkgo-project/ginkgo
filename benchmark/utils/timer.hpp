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

DEFINE_string(
    timer_method, "average",
    "The statistical method for output of timer. Available options: "
    "average, median, min, max. Note. If repetition_growth_factor > 1, the "
    "overhead operations may be different among repetitions");


/**
 * Get the timer. If the executor does not support gpu timer, still return the
 * cpu timer.
 *
 * @param exec  Executor associated to the timer
 * @param use_gpu_timer  whether to use the gpu timer
 */
std::shared_ptr<Timer> get_timer(std::shared_ptr<const gko::Executor> exec,
                                 bool use_gpu_timer);


#endif  // GKO_BENCHMARK_UTILS_TIMER_HPP_
