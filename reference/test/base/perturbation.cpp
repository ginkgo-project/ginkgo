/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#include "core/test/utils.hpp"


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

TYPED_TEST_SUITE(Perturbation, gko::test::ValueTypes);


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


TYPED_TEST(Perturbation, AppliesToMixedVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using Mtx = gko::matrix::Dense<gko::next_precision<TypeParam>>;
    using value_type = typename Mtx::value_type;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = Mtx::create_with_config_of(gko::lend(x));

    cmp->apply(gko::lend(x), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({29.0, 16.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Perturbation, AppliesToComplexVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using value_type = gko::to_complex<TypeParam>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto x = gko::initialize<Mtx>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}}, this->exec);
    auto res = Mtx::create_with_config_of(gko::lend(x));

    cmp->apply(gko::lend(x), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res,
                        l({value_type{29.0, -58.0}, value_type{16.0, -32.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Perturbation, AppliesToMixedComplexVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using value_type = gko::to_complex<gko::next_precision<TypeParam>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto x = gko::initialize<Mtx>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}}, this->exec);
    auto res = Mtx::create_with_config_of(gko::lend(x));

    cmp->apply(gko::lend(x), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res,
                        l({value_type{29.0, -58.0}, value_type{16.0, -32.0}}),
                        (r_mixed<value_type, TypeParam>()));
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


TYPED_TEST(Perturbation, AppliesLinearCombinationToMixedVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using value_type = gko::next_precision<TypeParam>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto alpha = gko::initialize<Mtx>({3.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = gko::clone(x);

    cmp->apply(gko::lend(alpha), gko::lend(x), gko::lend(beta), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({86.0, 46.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Perturbation, AppliesLinearCombinationToComplexVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using Dense = typename TestFixture::Mtx;
    using DenseComplex = gko::to_complex<Dense>;
    using value_type = typename DenseComplex::value_type;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto alpha = gko::initialize<Dense>({3.0}, this->exec);
    auto beta = gko::initialize<Dense>({-1.0}, this->exec);
    auto x = gko::initialize<DenseComplex>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}}, this->exec);
    auto res = gko::clone(x);

    cmp->apply(gko::lend(alpha), gko::lend(x), gko::lend(beta), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res,
                        l({value_type{86.0, -172.0}, value_type{46.0, -92.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Perturbation, AppliesLinearCombinationToMixedComplexVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using MixedDense = gko::matrix::Dense<gko::next_precision<TypeParam>>;
    using MixedDenseComplex = gko::to_complex<MixedDense>;
    using value_type = typename MixedDenseComplex::value_type;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto alpha = gko::initialize<MixedDense>({3.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto x = gko::initialize<MixedDenseComplex>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}}, this->exec);
    auto res = gko::clone(x);

    cmp->apply(gko::lend(alpha), gko::lend(x), gko::lend(beta), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res,
                        l({value_type{86.0, -172.0}, value_type{46.0, -92.0}}),
                        (r_mixed<value_type, TypeParam>()));
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
