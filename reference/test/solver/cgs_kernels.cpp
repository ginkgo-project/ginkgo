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

#include <core/solver/cgs.hpp>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


class Cgs : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Cgs()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, -3.0, 0.0}, {-4.0, 1.0, -3.0}, {2.0, -1.0, 2.0}}, exec)),
          cgs_factory(gko::solver::CgsFactory<>::create(exec, 40, 1e-15))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<gko::solver::CgsFactory<>> cgs_factory;
};


TEST_F(Cgs, SolvesDenseSystem)
{
    auto solver = cgs_factory->generate(mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), 1e-8);
}


TEST_F(Cgs, SolvesMultipleDenseSystem)
{
    auto solver = cgs_factory->generate(mtx);
    auto b =
        gko::initialize<Mtx>({{-1.0, -5.0}, {3.0, 1.0}, {1.0, -2.0}}, exec);
    auto x = gko::initialize<Mtx>({{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({{-4.0, 1.0}, {-1.0, 2.0}, {4.0, -1.0}}), 1e-8);
}


TEST_F(Cgs, SolvesDenseSystemUsingAdvancedApply)
{
    auto solver = cgs_factory->generate(mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(x, l({-8.5, -3.0, 6.0}), 1e-8);
}


TEST_F(Cgs, SolvesMultipleDenseSystemsUsingAdvancedApply)
{
    auto solver = cgs_factory->generate(mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    auto b =
        gko::initialize<Mtx>({{-1.0, -5.0}, {3.0, 1.0}, {1.0, -2.0}}, exec);
    auto x = gko::initialize<Mtx>({{0.5, 1.0}, {1.0, 2.0}, {2.0, 3.0}}, exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(x, l({{-8.5, 1.0}, {-3.0, 2.0}, {6.0, -5.0}}), 1e-8);
}


}  // namespace
