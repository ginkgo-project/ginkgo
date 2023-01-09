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

#include <ginkgo/core/log/batch_convergence.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/test/utils.hpp"


namespace {


/*
 * In actual code, this would be a solver class such as BatchRichardson.
 */
struct DummyLogged : gko::log::EnableLogging<DummyLogged> {
    using value_type = float;
    const gko::size_type nbatch = 2;
    const int nrhs = 2;
    std::shared_ptr<const gko::Executor> exec;

    DummyLogged() : exec(gko::ReferenceExecutor::create()) {}

    int get_num_loggers() { return loggers_.size(); }

    void apply()
    {
        gko::array<int> iter_counts(exec, nbatch * nrhs);
        int* const itervals = iter_counts.get_data();
        const gko::batch_dim<> sizes(nbatch, gko::dim<2>{1, nrhs});
        auto res_norms =
            gko::matrix::BatchDense<value_type>::create(exec, sizes);
        // value_type *const resnormvals = res_norms->get_values();
        itervals[0] = 10;
        itervals[1] = 9;
        itervals[2] = 5;
        itervals[3] = 6;
        res_norms->at(0, 0, 0) = 2.0;
        res_norms->at(0, 0, 1) = 0.25;
        res_norms->at(1, 0, 0) = 0.125;
        res_norms->at(1, 0, 1) = 4.0;
        this->log<gko::log::Logger::batch_solver_completed>(iter_counts,
                                                            res_norms.get());
    }
};


TEST(BatchConvergence, CanGetEmptyData)
{
    using value_type = DummyLogged::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::BatchConvergence<value_type>::create(exec);

    ASSERT_EQ(logger->get_num_iterations().get_num_elems(), 0);
    ASSERT_EQ(logger->get_residual_norm()->get_size().at(0),
              (gko::dim<2>{0, 0}));
}

TEST(BatchConvergence, CanGetLogData)
{
    using value_type = DummyLogged::value_type;
    auto exec = gko::ReferenceExecutor::create();
    std::shared_ptr<const gko::log::BatchConvergence<value_type>> logger =
        gko::log::BatchConvergence<value_type>::create(exec);
    DummyLogged dum;
    dum.add_logger(logger);

    dum.apply();
    const int* const iters = logger->get_num_iterations().get_const_data();
    const auto resnorms = logger->get_residual_norm();

    ASSERT_EQ(logger->get_num_iterations().get_num_elems(),
              dum.nbatch * dum.nrhs);
    ASSERT_EQ(logger->get_residual_norm()->get_num_batch_entries(), dum.nbatch);
    ASSERT_EQ(logger->get_residual_norm()->get_size().at(0),
              (gko::dim<2>{1, dum.nrhs}));
    ASSERT_EQ(logger->get_residual_norm()->get_size().at(1),
              (gko::dim<2>{1, dum.nrhs}));
    ASSERT_EQ(iters[0], 10);
    ASSERT_EQ(iters[1], 9);
    ASSERT_EQ(iters[2], 5);
    ASSERT_EQ(iters[3], 6);
    ASSERT_EQ(resnorms->at(0, 0, 0), 2.0);
    ASSERT_EQ(resnorms->at(0, 0, 1), 0.25);
    ASSERT_EQ(resnorms->at(1, 0, 0), 0.125);
    ASSERT_EQ(resnorms->at(1, 0, 1), 4.0);
}


}  // namespace
