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

#include <ginkgo/core/base/composition.hpp>


#include <vector>


#include <gtest/gtest.h>


#include <core/test/utils/assertions.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class Composition : public ::testing::Test {
protected:
    using mtx = gko::matrix::Dense<>;

    Composition()
        : exec{gko::ReferenceExecutor::create()},
          operators{gko::initialize<mtx>({2.0, 1.0}, exec),
                    gko::initialize<mtx>({{3.0, 2.0}}, exec)}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::vector<std::shared_ptr<gko::LinOp>> coefficients;
    std::vector<std::shared_ptr<gko::LinOp>> operators;
};


TEST_F(Composition, AppliesToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ]
              [ 1 ]
    */
    auto cmp = gko::Composition<>::create(operators[0], operators[1]);
    auto x = gko::initialize<mtx>({1.0, 2.0}, exec);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({14.0, 7.0}), 1e-15);
}


TEST_F(Composition, AppliesLinearCombinationToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ]
              [ 1 ]
    */
    auto cmp = gko::Composition<>::create(operators[0], operators[1]);
    auto alpha = gko::initialize<mtx>({3.0}, exec);
    auto beta = gko::initialize<mtx>({-1.0}, exec);
    auto x = gko::initialize<mtx>({1.0, 2.0}, exec);
    auto res = clone(x);

    cmp->apply(lend(alpha), lend(x), lend(beta), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({41.0, 19.0}), 1e-15);
}


}  // namespace
