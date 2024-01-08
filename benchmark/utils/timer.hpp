// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
