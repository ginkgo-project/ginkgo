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

#include <core/base/combination.hpp>


#include <gtest/gtest.h>


#include <vector>


#include <core/matrix/dense.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


class Combination : public ::testing::Test {
protected:
    using mtx = gko::matrix::Dense<>;

    Combination()
        : exec{gko::ReferenceExecutor::create()},
          coefficients{gko::initialize<mtx>({1}, exec),
                       gko::initialize<mtx>({2}, exec)},
          operators{gko::initialize<mtx>({{2.0, 3.0}, {1.0, 4.0}}, exec),
                    gko::initialize<mtx>({{3.0, 2.0}, {2.0, 0.0}}, exec)}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::vector<std::shared_ptr<gko::LinOp>> coefficients;
    std::vector<std::shared_ptr<gko::LinOp>> operators;
};


TEST_F(Combination, AppliesToVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    auto cmb = gko::Combination<>::create(coefficients[0], operators[0],
                                          coefficients[1], operators[1]);
    auto x = gko::initialize<mtx>({1.0, 2.0}, exec);
    auto res = clone(x);

    cmb->apply(lend(x), lend(res));

    ASSERT_MTX_NEAR(res, l({22.0, 13.0}), 1e-15);
}


TEST_F(Combination, AppliesLinearCombinationToVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    auto cmb = gko::Combination<>::create(coefficients[0], operators[0],
                                          coefficients[1], operators[1]);
    auto alpha = gko::initialize<mtx>({3.0}, exec);
    auto beta = gko::initialize<mtx>({-1.0}, exec);
    auto x = gko::initialize<mtx>({1.0, 2.0}, exec);
    auto res = clone(x);

    cmb->apply(lend(alpha), lend(x), lend(beta), lend(res));

    ASSERT_MTX_NEAR(res, l({65.0, 37.0}), 1e-15);
}


}  // namespace
