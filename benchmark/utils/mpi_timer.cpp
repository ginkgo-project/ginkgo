// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/mpi.hpp>


#include "benchmark/utils/timer_impl.hpp"


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
        comm_.all_reduce(exec_, &local_duration, &duration, 1, MPI_MAX);
        return duration;
    }

private:
    std::shared_ptr<const gko::Executor> exec_;
    gko::experimental::mpi::communicator comm_;

    std::shared_ptr<Timer> concrete_timer_;
};


std::shared_ptr<Timer> get_mpi_timer(std::shared_ptr<const gko::Executor> exec,
                                     gko::experimental::mpi::communicator comm,
                                     bool use_gpu_timer)
{
    return std::make_shared<MpiWrappedTimer>(exec, std::move(comm),
                                             get_timer(exec, use_gpu_timer));
}
