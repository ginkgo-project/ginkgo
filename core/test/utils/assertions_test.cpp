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

#include <core/test/utils/assertions.hpp>


#include <gtest/gtest.h>


#include <core/matrix/dense.hpp>


namespace {


class MatricesNear : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    MatricesNear()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(gko::initialize<gko::matrix::Dense<>>(
              {{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}, exec)),
          mtx2(gko::initialize<gko::matrix::Dense<>>(
              {{1.0, 2.0, 3.0}, {4.0, 0.0, 4.0}}, exec)),
          mtx3(gko::initialize<gko::matrix::Dense<>>(
              {{1.0, 2.0, 3.0}, {0.0, 4.1, 0.0}}, exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
};


TEST_F(MatricesNear, SuceedsIfSame)
{
    ASSERT_PRED_FORMAT3(gko::test::assertions::matrices_near, mtx1.get(),
                        mtx1.get(), 0.0);
}


TEST_F(MatricesNear, FailsIfDifferent)
{
    ASSERT_PRED_FORMAT3(!gko::test::assertions::matrices_near, mtx1.get(),
                        mtx2.get(), 0.0);
}


TEST_F(MatricesNear, SucceedsIfClose)
{
    ASSERT_PRED_FORMAT3(!gko::test::assertions::matrices_near, mtx1.get(),
                        mtx3.get(), 0.0);
    ASSERT_PRED_FORMAT3(gko::test::assertions::matrices_near, mtx1.get(),
                        mtx3.get(), 0.1);
}


TEST_F(MatricesNear, CanUseShortNotation)
{
    EXPECT_MTX_NEAR(mtx1, mtx1, 0.0);
    ASSERT_MTX_NEAR(mtx1, mtx3, 0.1);
}


TEST_F(MatricesNear, CanPassInitializerList)
{
    EXPECT_MTX_NEAR(mtx1, l({{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}), 0.0);
    ASSERT_MTX_NEAR(mtx1, l({{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}), 0.0);
}


}  // namespace
