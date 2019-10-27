/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/log/convergence.hpp>


#include <gtest/gtest.h>


#include <core/test/utils/assertions.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/iteration.hpp>


namespace {


TEST(Record, CatchesCriterionCheckCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Convergence<>::create(
        exec, exec->get_mem_space(),
        gko::log::Logger::criterion_check_completed_mask);
    auto criterion =
        gko::stop::Iteration::build().with_max_iters(3u).on(exec)->generate(
            nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::Array<gko::stopping_status> stop_status(exec, 1);
    using Mtx = gko::matrix::Dense<>;
    auto residual = gko::initialize<Mtx>({1.0, 2.0, 2.0}, exec);

    logger->on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), 1, residual.get(), nullptr, nullptr,
        RelativeStoppingId, true, &stop_status, true, true);

    ASSERT_EQ(logger->get_num_iterations(), 1);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(logger->get_residual()),
                        l({1.0, 2.0, 2.0}), 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(logger->get_residual_norm()), l({3.0}),
                        0.0);
}


}  // namespace
