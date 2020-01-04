/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/solver/upper_trs.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/solver/upper_trs_kernels.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


class UpperTrs : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    UpperTrs()
        : exec(gko::ReferenceExecutor::create()),
          ref(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1, 3.0, 1.0}, {0.0, 1, 2.0}, {0.0, 0.0, 1}}, exec)),
          mtx2(gko::initialize<Mtx>(
              {{2, 3.0, 1.0}, {0.0, 3, 2.0}, {0.0, 0.0, 4}}, exec)),
          upper_trs_factory(gko::solver::UpperTrs<>::build().on(exec)),
          upper_trs_factory_mrhs(
              gko::solver::UpperTrs<>::build().with_num_rhs(2u).on(exec)),
          mtx_big(gko::initialize<Mtx>({{365.0, 97.0, -654.0, 8.0, 91.0},
                                        {0.0, -642.0, 684.0, 68.0, 387.0},
                                        {0.0, 0.0, 134, -651.0, 654.0},
                                        {0.0, 0.0, 0.0, 43.0, -789.0},
                                        {0.0, 0.0, 0.0, 0.0, 124.0}},
                                       exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx2;
    std::shared_ptr<Mtx> mtx_big;
    std::unique_ptr<gko::solver::UpperTrs<>::Factory> upper_trs_factory;
    std::unique_ptr<gko::solver::UpperTrs<>::Factory> upper_trs_factory_mrhs;
};


TEST_F(UpperTrs, RefUpperTrsFlagCheckIsCorrect)
{
    bool trans_flag = true;
    bool expected_flag = false;

    gko::kernels::reference::upper_trs::should_perform_transpose(ref,
                                                                 trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TEST_F(UpperTrs, SolvesTriangularSystem)
{
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({4.0, 2.0, 3.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, exec);
    auto solver = upper_trs_factory->generate(mtx);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({13.0, -4.0, 3.0}), 1e-14);
}


TEST_F(UpperTrs, SolvesMultipleTriangularSystems)
{
    std::shared_ptr<Mtx> b =
        gko::initialize<Mtx>({{4.0, 2.0}, {2.0, 1.0}, {3.0, -1.0}}, exec);
    auto x = gko::initialize<Mtx>({{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, exec);
    auto solver = upper_trs_factory_mrhs->generate(mtx);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{13.0, -6.0}, {-4.0, 3.0}, {3.0, -1.0}}), 1e-14);
}


TEST_F(UpperTrs, SolvesNonUnitTriangularSystem)
{
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({10.0, 7.0, -4.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, exec);
    auto solver = upper_trs_factory->generate(mtx2);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, -1.0}), 1e-14);
}


TEST_F(UpperTrs, SolvesTriangularSystemUsingAdvancedApply)
{
    auto alpha = gko::initialize<Mtx>({2.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({4.0, 2.0, 3.0}, exec);
    auto x = gko::initialize<Mtx>({1.0, -1.0, 1.0}, exec);
    auto solver = upper_trs_factory->generate(mtx);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({25.0, -7.0, 5.0}), 1e-14);
}


TEST_F(UpperTrs, SolvesMultipleTriangularSystemsUsingAdvancedApply)
{
    auto alpha = gko::initialize<Mtx>({-1.0}, exec);
    auto beta = gko::initialize<Mtx>({2.0}, exec);
    std::shared_ptr<Mtx> b =
        gko::initialize<Mtx>({{4.0, 1.0}, {1.0, 2.0}, {2.0, 3.0}}, exec);
    auto x =
        gko::initialize<Mtx>({{1.0, 2.0}, {-1.0, -1.0}, {1.0, -2.0}}, exec);
    auto solver = upper_trs_factory_mrhs->generate(mtx);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{-9.0, -6.0}, {1.0, 2.0}, {0.0, -7.0}}), 1e-14);
}


TEST_F(UpperTrs, SolvesBigDenseSystem)
{
    std::shared_ptr<Mtx> b =
        gko::initialize<Mtx>({-6021.0, 3018.0, -2055.0, 1707.0, -248.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0}, exec);
    auto solver = upper_trs_factory->generate(mtx_big);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({-1.0, 4.0, 9.0, 3.0, -2.0}), 1e-10);
}


}  // namespace
