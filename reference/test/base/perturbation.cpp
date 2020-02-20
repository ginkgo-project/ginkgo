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

#include <ginkgo/core/base/perturbation.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include <core/test/utils.hpp>


namespace {


template <typename T>
class Perturbation : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;

    Perturbation()
        : exec{gko::ReferenceExecutor::create()},
          basis{gko::initialize<Mtx>({2.0, 1.0}, exec)},
          projector{gko::initialize<Mtx>({I<T>({3.0, 2.0})}, exec)},
          scalar{gko::initialize<Mtx>({2.0}, exec)}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<gko::LinOp> basis;
    std::shared_ptr<gko::LinOp> projector;
    std::shared_ptr<gko::LinOp> scalar;
};


TYPED_TEST_CASE(Perturbation, gko::test::ValueTypes);


TYPED_TEST(Perturbation, AppliesToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = Mtx::create_with_config_of(gko::lend(x));

    cmp->apply(gko::lend(x), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({29.0, 16.0}), r<TypeParam>::value);
}


TYPED_TEST(Perturbation, AppliesLinearCombinationToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto alpha = gko::initialize<Mtx>({3.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = gko::clone(x);

    cmp->apply(gko::lend(alpha), gko::lend(x), gko::lend(beta), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({86.0, 46.0}), r<TypeParam>::value);
}


TYPED_TEST(Perturbation, ConstructionByBasisAppliesToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 2 1 ]
                      [ 1 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = Mtx::create_with_config_of(gko::lend(x));

    cmp->apply(gko::lend(x), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({17.0, 10.0}), r<TypeParam>::value);
}


TYPED_TEST(Perturbation, ConstructionByBasisAppliesLinearCombinationToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 2 1 ]
                      [ 1 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis);
    auto alpha = gko::initialize<Mtx>({3.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = gko::clone(x);

    cmp->apply(gko::lend(alpha), gko::lend(x), gko::lend(beta), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({50.0, 28.0}), r<TypeParam>::value);
}


}  // namespace
