// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_ITERATION_CONTROL_HPP_
#define GKO_BENCHMARK_UTILS_ITERATION_CONTROL_HPP_


#include <ginkgo/ginkgo.hpp>


#include <memory>
#include <string>
#include <utility>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"
#include "core/distributed/helpers.hpp"


/**
 * A class for controlling the number warmup and timed iterations.
 *
 * The behavior is determined by the following flags
 * - 'repetitions' switch between fixed and adaptive number of iterations
 * - 'warmup' warmup iterations, applies in fixed and adaptive case
 * - 'min_repetitions' minimal number of repetitions (adaptive case)
 * - 'max_repetitions' maximal number of repetitions (adaptive case)
 * - 'min_runtime' minimal total runtime (adaptive case)
 * - 'repetition_growth_factor' controls the increase between two successive
 *   timings
 *
 * Usage:
 * `IterationControl` exposes the member functions:
 * - `warmup_run()`: controls run defined by `warmup` flag
 * - `run(bool)`: controls run defined by all other flags
 * - `get_timer()`: access to underlying timer
 * The first two methods return an object that is to be used in a range-based
 * for loop:
 * ```
 * IterationControl ic(get_timer(...));
 *
 * // warmup run always uses fixed number of iteration and does not issue
 * // timings
 * for(auto status: ic.warmup_run()){
 *   // execute benchmark
 * }
 * // run may use adaptive number of iterations (depending on cmd line flag)
 * // and issues timing (unless manage_timings is false)
 * for(auto status: ic.run(manage_timings [default is true])){
 *   if(! manage_timings) ic.get_timer->tic();
 *   // execute benchmark
 *   if(! manage_timings) ic.get_timer->toc();
 * }
 *
 * ```
 * At the beginning of both methods, the timer is reset.
 * The `status` object exposes the member
 * - `cur_it`, containing the current iteration number,
 * and the methods
 * - `is_finished`, checks if the benchmark is finished,
 */
class IterationControl {
    using IndexType = unsigned int;  //!< to be compatible with GFLAGS type

    class run_control;

public:
    /**
     * Creates an `IterationControl` object.
     *
     * Uses the commandline flags to setup the stopping criteria for the
     * warmup and timed run.
     *
     * @param timer  the timer that is to be used for the timings
     */
    explicit IterationControl(const std::shared_ptr<Timer>& timer)
    {
        status_warmup_ = {TimerManager{timer, false}, FLAGS_warmup,
                          FLAGS_warmup, 0., 0};
        if (FLAGS_repetitions == "auto") {
            status_run_ = {TimerManager{timer, true}, FLAGS_min_repetitions,
                           FLAGS_max_repetitions, FLAGS_min_runtime};
        } else {
            const auto reps =
                static_cast<unsigned int>(std::stoi(FLAGS_repetitions));
            status_run_ = {TimerManager{timer, true}, reps, reps, 0., 0};
        }
    }

    IterationControl() = default;
    IterationControl(const IterationControl&) = default;
    IterationControl(IterationControl&&) = default;

    /**
     * Creates iterable `run_control` object for the warmup run.
     *
     * This run uses always a fixed number of iterations.
     */
    run_control warmup_run()
    {
        status_warmup_.cur_it = 0;
        status_warmup_.managed_timer.clear();
        return run_control{&status_warmup_};
    }

    /**
     * Creates iterable `run_control` object for the timed run.
     *
     * This run may be adaptive, depending on the commandline flags.
     *
     * @param manage_timings If true, the timer calls (`tic/toc`) are handled
     * by the `run_control` object, otherwise they need to be executed outside
     */
    run_control run(bool manage_timings = true)
    {
        status_run_.cur_it = 0;
        status_run_.managed_timer.clear();
        status_run_.managed_timer.manage_timings = manage_timings;
        return run_control{&status_run_};
    }

    std::shared_ptr<Timer> get_timer() const
    {
        return status_run_.managed_timer.timer;
    }

    /**
     * Compute the time from the given statistical method
     *
     * @param method  the statistical method. If the timer does not have the
     *                same iteration as the IterationControl, it can only use
     *                average from the IterationControl.
     *
     * @return the statistical time
     */
    double compute_time(const std::string& method = "average") const
    {
        if (status_run_.managed_timer.timer->get_num_repetitions() ==
            this->get_num_repetitions()) {
            return status_run_.managed_timer.compute_time(method);
        } else {
            assert(method == "average");
            return status_run_.managed_timer.get_total_time() /
                   this->get_num_repetitions();
        }
    }

    IndexType get_num_repetitions() const { return status_run_.cur_it; }

private:
    struct TimerManager {
        std::shared_ptr<Timer> timer;
        bool manage_timings = false;

        void tic()
        {
            if (manage_timings) {
                timer->tic();
            }
        }
        void toc(unsigned int num = 1)
        {
            if (manage_timings) {
                timer->toc(num);
            }
        }

        void clear() { timer->clear(); }

        double get_total_time() const { return timer->get_total_time(); }

        double compute_time(const std::string& method = "average") const
        {
            return timer->compute_time(method);
        }
    };

    /**
     * Stores stopping criteria of the adaptive benchmark run as well as the
     * current iteration number.
     */
    struct status {
        TimerManager managed_timer{};

        IndexType min_it = 0;
        IndexType max_it = 0;
        double max_runtime = 0.;

        IndexType cur_it = 0;

        /**
         * checks if the adaptive run is complete
         *
         * the adaptive run is complete if:
         * - the minimum number of iteration is reached
         * - and either:
         *   - the maximum number of repetitions is reached
         *   - the total runtime is above the threshold
         *
         * @return completeness state of the adaptive run
         */
        bool is_finished() const
        {
            return cur_it >= min_it &&
                   (cur_it >= max_it ||
                    managed_timer.get_total_time() >= max_runtime);
        }
    };

    /**
     * Iterable class managing the benchmark iteration.
     *
     * Has to be used in a range-based for loop.
     */
    struct run_control {
        struct iterator {
            /**
             * Increases the current iteration count and finishes timing if
             * necessary.
             *
             * As `++it` is the last step of a for-loop, the managed_timer is
             * stopped, if enough iterations have passed since the last timing.
             * The interval between two timings is steadily increased to
             * reduce the timing overhead.
             */
            iterator operator++()
            {
                cur_info->cur_it++;
                if (cur_info->cur_it >= next_timing && !stopped) {
                    cur_info->managed_timer.toc(
                        static_cast<unsigned>(cur_info->cur_it - start_timing));
                    stopped = true;
                    next_timing = static_cast<IndexType>(std::ceil(
                        next_timing * FLAGS_repetition_growth_factor));
                    // If repetition_growth_factor <= 1, next_timing will be
                    // next iteration.
                    if (next_timing <= cur_info->cur_it) {
                        next_timing = cur_info->cur_it + 1;
                    }
                }
                return *this;
            }

            status operator*() const { return *cur_info; }

            /**
             * Checks if the benchmark is finished and handles timing, if
             * necessary.
             *
             * As `begin != end` is the first step in a for-loop, the
             * managed_timer is started, if it was previously stopped.
             * Additionally, if the benchmark is complete and the managed_timer
             * is still running it is stopped. (This may occur if the maximal
             * number of repetitions is surpassed)
             *
             * Uses only the information from the `status` object, i.e.
             * the right hand side is ignored.
             *
             * @return true if benchmark is not finished, else false
             */
            bool operator!=(const iterator&)
            {
                const bool is_finished = cur_info->is_finished();
                if (!is_finished && stopped) {
                    stopped = false;
                    cur_info->managed_timer.tic();
                    start_timing = cur_info->cur_it;
                } else if (is_finished && !stopped) {
                    cur_info->managed_timer.toc(
                        static_cast<unsigned>(cur_info->cur_it - start_timing));
                    stopped = true;
                }
                return !is_finished;
            }

            status* cur_info;
            IndexType next_timing = 1;   //!< next iteration to stop timing
            IndexType start_timing = 0;  //!< iteration for starting timing
            bool stopped = true;
        };

        iterator begin() const { return iterator{info}; }

        // not used, could potentially be used in c++17 as a sentinel
        iterator end() const { return iterator{}; }

        status* info;
    };

    status status_warmup_;
    status status_run_;
};


#endif  // GKO_BENCHMARK_UTILS_ITERATION_CONTROL_HPP_
