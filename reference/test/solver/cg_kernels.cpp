/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/solver/cg.hpp>

#include <core/test/utils/assertions.hpp>

#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>


namespace {


class Cg : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Cg()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec,
                          {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}})),
          cg_factory(gko::solver::CgFactory<>::create(exec, 4, 1e-15))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<gko::solver::CgFactory<>> cg_factory;
};


TEST_F(Cg, SolvesStencilSystem)
{
    auto solver = cg_factory->generate(mtx);
    auto b = Mtx::create(exec, {-1.0, 3.0, 1.0});
    auto x = Mtx::create(exec, {0.0, 0.0, 0.0});

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), 1e-14);
    // EXPECT_NEAR(x->at(0), 1.0, 1e-14);
    // EXPECT_NEAR(x->at(1), 3.0, 1e-14);
    // EXPECT_NEAR(x->at(2), 2.0, 1e-14);
}


TEST_F(Cg, SolvesMultipleStencilSystems)
{
    auto solver = cg_factory->generate(mtx);
    auto b = Mtx::create(exec, {{-1.0, 1.0}, {3.0, 0.0}, {1.0, 1.0}});
    auto x = Mtx::create(exec, {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}});

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({{1.0, 1.0}, {3.0, 1.0}, {2.0, 1.0}}), 1e-14);
    /*EXPECT_NEAR(x->at(0, 0), 1.0, 1e-14);
    EXPECT_NEAR(x->at(1, 0), 3.0, 1e-14);
    EXPECT_NEAR(x->at(2, 0), 2.0, 1e-14);
    EXPECT_NEAR(x->at(0, 1), 1.0, 1e-14);
    EXPECT_NEAR(x->at(1, 1), 1.0, 1e-14);
    EXPECT_NEAR(x->at(2, 1), 1.0, 1e-14);*/
}


TEST_F(Cg, SolvesStencilSystemUsingAdvancedApply)
{
    auto solver = cg_factory->generate(mtx);
    auto alpha = Mtx::create(exec, {2.0});
    auto beta = Mtx::create(exec, {-1.0});
    auto b = Mtx::create(exec, {-1.0, 3.0, 1.0});
    auto x = Mtx::create(exec, {0.5, 1.0, 2.0});

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), 1e-14);
    /*EXPECT_NEAR(x->at(0), 1.5, 1e-14);
    EXPECT_NEAR(x->at(1), 5.0, 1e-14);
    EXPECT_NEAR(x->at(2), 2.0, 1e-14);*/
}


TEST_F(Cg, SolvesMultipleStencilSystemsUsingAdvancedApply)
{
    auto solver = cg_factory->generate(mtx);
    auto alpha = Mtx::create(exec, {2.0});
    auto beta = Mtx::create(exec, {-1.0});
    auto b = Mtx::create(exec, {{-1.0, 1.0}, {3.0, 0.0}, {1.0, 1.0}});
    auto x = Mtx::create(exec, {{0.5, 1.0}, {1.0, 2.0}, {2.0, 3.0}});

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(x, l({{1.5, 1.0}, {5.0, 0.0}, {2.0, -1.0}}), 1e-14);
    /*EXPECT_NEAR(x->at(0, 0), 1.5, 1e-14);
    EXPECT_NEAR(x->at(1, 0), 5.0, 1e-14);
    EXPECT_NEAR(x->at(2, 0), 2.0, 1e-14);
    EXPECT_NEAR(x->at(0, 1), 1.0, 1e-14);
    EXPECT_NEAR(x->at(1, 1), 0.0, 1e-14);
    EXPECT_NEAR(x->at(2, 1), -1.0, 1e-14);*/
}


}  // namespace
