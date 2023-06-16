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
