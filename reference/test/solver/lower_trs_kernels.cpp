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

#include <ginkgo/core/solver/lower_trs.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class LowerTrs : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    LowerTrs()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1, 0.0, 0.0}, {3.0, 1, 0.0}, {1.0, 2.0, 1}}, exec)),
          mtx2(gko::initialize<Mtx>(
              {{2, 0.0, 0.0}, {3.0, 3, 0.0}, {1.0, 2.0, 4}}, exec)),
          lower_trs_factory(gko::solver::LowerTrs<>::build().on(exec)),
          mtx_big(gko::initialize<Mtx>({{124.0, 0.0, 0.0, 0.0, 0.0},
                                        {43.0, -789.0, 0.0, 0.0, 0.0},
                                        {134.5, -651.0, 654.0, 0.0, 0.0},
                                        {-642.0, 684.0, 68.0, 387.0, 0.0},
                                        {365.0, 97.0, -654.0, 8.0, 91.0}},
                                       exec)),
          lower_trs_factory_big(gko::solver::LowerTrs<>::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx2;
    std::shared_ptr<Mtx> mtx_big;
    std::unique_ptr<gko::solver::LowerTrs<>::Factory> lower_trs_factory;
    std::unique_ptr<gko::solver::LowerTrs<>::Factory> lower_trs_factory_big;
};


TEST_F(LowerTrs, SolvesTriangularSystem)
{
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, exec);
    auto solver = lower_trs_factory->generate(mtx, b);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, -1.0, 2.0}), 1e-14);
}


TEST_F(LowerTrs, SolvesMultipleTriangularSystems)
{
    std::shared_ptr<Mtx> b =
        gko::initialize<Mtx>({{3.0, 4.0}, {1.0, 0.0}, {1.0, -1.0}}, exec);
    auto x = gko::initialize<Mtx>({{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, exec);
    auto solver = lower_trs_factory->generate(mtx, b);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{3.0, 4.0}, {-8.0, -12.0}, {14.0, 19.0}}), 1e-14);
}


TEST_F(LowerTrs, SolvesNonUnitTriangularSystem)
{
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({2.0, 12.0, 3.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, exec);
    auto solver = lower_trs_factory->generate(mtx2, b);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, -1.0}), 1e-14);
}

TEST_F(LowerTrs, SolvesTriangularSystemUsingAdvancedApply)
{
    auto alpha = gko::initialize<Mtx>({2.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0}, exec);
    auto x = gko::initialize<Mtx>({1.0, -1.0, 1.0}, exec);
    auto solver = lower_trs_factory->generate(mtx, b);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, -1.0, 3.0}), 1e-14);
}


TEST_F(LowerTrs, SolvesMultipleTriangularSystemsUsingAdvancedApply)
{
    auto alpha = gko::initialize<Mtx>({-1.0}, exec);
    auto beta = gko::initialize<Mtx>({2.0}, exec);
    std::shared_ptr<Mtx> b =
        gko::initialize<Mtx>({{3.0, 4.0}, {1.0, 0.0}, {1.0, -1.0}}, exec);
    auto x =
        gko::initialize<Mtx>({{1.0, 2.0}, {-1.0, -1.0}, {0.0, -2.0}}, exec);
    auto solver = lower_trs_factory->generate(mtx, b);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{-1.0, 0.0}, {6.0, 10.0}, {-14.0, -23.0}}),
                        1e-14);
}


TEST_F(LowerTrs, SolvesBigDenseSystem)
{
    std::shared_ptr<Mtx> b =
        gko::initialize<Mtx>({-124.0, -3199.0, 3147.5, 5151.0, -6021.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0}, exec);
    auto solver = lower_trs_factory_big->generate(mtx_big, b);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({-1.0, 4.0, 9.0, 3.0, -2.0}), 1e-10);
}


}  // namespace
