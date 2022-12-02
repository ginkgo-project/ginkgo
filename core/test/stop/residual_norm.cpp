/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/stop/residual_norm.hpp>


#include <gtest/gtest.h>


namespace {


class AutomaticResidualNorm : public ::testing::Test {
protected:
    AutomaticResidualNorm()
        : exec{gko::ReferenceExecutor::create()}, stopping_status{exec, 1}
    {
        stopping_status.get_data()[0].reset();
    }

    std::shared_ptr<const gko::Executor> exec;
    gko::array<gko::stopping_status> stopping_status;
    std::shared_ptr<gko::LinOp> system_matrix;
    std::shared_ptr<gko::LinOp> b =
        gko::initialize<gko::matrix::Dense<>>({0}, exec);
    std::unique_ptr<gko::LinOp> x;
    std::unique_ptr<gko::LinOp> r;
    std::unique_ptr<gko::LinOp> res_norm =
        gko::initialize<gko::matrix::Dense<>>({0}, exec);
};


TEST_F(AutomaticResidualNorm, CanSelectDirectResidualNorm)
{
    auto logger = gko::stop::AutomaticResidualNorm<>::build()
                      .with_baseline(gko::stop::mode::absolute)
                      .on(exec)
                      ->generate(system_matrix, b, x.get(), r.get(),
                                 gko::stop::residual_norm_criteria::direct);
    bool one_changed = false;


    ASSERT_NO_THROW(
        logger->check(gko::uint8(0), false, &stopping_status, &one_changed,
                      logger->update().residual_norm(res_norm.get())));
    ASSERT_THROW(
        logger->check(
            gko::uint8(0), false, &stopping_status, &one_changed,
            logger->update().implicit_sq_residual_norm(res_norm.get())),
        gko::NotSupported);
}


TEST_F(AutomaticResidualNorm, CanSelectImplicitResidualNorm)
{
    auto logger = gko::stop::AutomaticResidualNorm<>::build()
                      .with_baseline(gko::stop::mode::absolute)
                      .on(exec)
                      ->generate(system_matrix, b, x.get(), r.get(),
                                 gko::stop::residual_norm_criteria::implicit);
    bool one_changed = false;

    ASSERT_NO_THROW(logger->check(
        gko::uint8(0), false, &stopping_status, &one_changed,
        logger->update().implicit_sq_residual_norm(res_norm.get())));
    ASSERT_THROW(
        logger->check(gko::uint8(0), false, &stopping_status, &one_changed,
                      logger->update().residual_norm(res_norm.get())),
        gko::NotSupported);
}


}  // namespace
