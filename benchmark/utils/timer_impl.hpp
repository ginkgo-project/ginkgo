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

#ifndef GKO_BENCHMARK_UTILS_TIMER_IMPL_HPP_
#define GKO_BENCHMARK_UTILS_TIMER_IMPL_HPP_


#include <ginkgo/ginkgo.hpp>


#include <chrono>
#include <memory>


class MpiWrappedTimer;


/**
 * Timer stores the timing information
 */
class Timer {
    friend MpiWrappedTimer;

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


class MpiWrappedTimer : public Timer {
public:
    MpiWrappedTimer(std::shared_ptr<const gko::Executor> exec,
                    gko::experimental::mpi::communicator comm,
                    std::shared_ptr<Timer> concrete_timer)
        : exec_(std::move(exec)),
          comm_(std::move(comm)),
          concrete_timer_(std::move(concrete_timer))
    {}

protected:
    void tic_impl() override { concrete_timer_->tic_impl(); }

    double toc_impl() override
    {
        auto local_duration = concrete_timer_->toc_impl();
        double duration = local_duration;
        comm_.template all_reduce(exec_, &local_duration, &duration, 1,
                                  MPI_MAX);
        return duration;
    }

private:
    std::shared_ptr<const gko::Executor> exec_;
    gko::experimental::mpi::communicator comm_;

    std::shared_ptr<Timer> concrete_timer_;
};


#endif  // GKO_BENCHMARK_UTILS_TIMER_IMPL_HPP_
