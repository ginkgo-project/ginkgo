// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/perturbation.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class Perturbation : public ::testing::Test {
protected:
    using Vec = gko::matrix::MultiVector<T>;
    using Dense = gko::matrix::Dense<T>;

    Perturbation()
        : exec{gko::ReferenceExecutor::create()},
          basis{gko::initialize<Dense>({2.0, 1.0}, exec)},
          scalar{gko::initialize<Vec>({2.0}, exec)}
    {
        projector = gko::initialize<Dense>({I<T>({3.0, 2.0})}, exec);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Dense> basis;
    std::shared_ptr<Dense> projector;
    std::shared_ptr<Vec> scalar;
};

TYPED_TEST_SUITE(Perturbation, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Perturbation, CopiesOnSameExecutor)
{
    using Mtx = typename TestFixture::Vec;
    auto per = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto out = per->create_default();

    per->convert_to(out);

    ASSERT_EQ(out->get_size(), per->get_size());
    ASSERT_EQ(out->get_executor(), per->get_executor());
    ASSERT_EQ(out->get_scalar(), per->get_scalar());
    ASSERT_EQ(out->get_basis(), per->get_basis());
    ASSERT_EQ(out->get_projector(), per->get_projector());
}


TYPED_TEST(Perturbation, MovesOnSameExecutor)
{
    using Vec = typename TestFixture::Vec;
    auto per = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto per2 = per->clone();
    auto out = per->create_default();

    per->move_to(out);

    ASSERT_EQ(out->get_size(), per2->get_size());
    ASSERT_EQ(out->get_executor(), per2->get_executor());
    ASSERT_EQ(out->get_scalar(), per2->get_scalar());
    ASSERT_EQ(out->get_basis(), per2->get_basis());
    ASSERT_EQ(out->get_projector(), per2->get_projector());
    // same executor, empty object
    ASSERT_EQ(per->get_size(), gko::dim<2>{});
    ASSERT_EQ(per->get_executor(), per2->get_executor());
    ASSERT_EQ(per->get_scalar(), nullptr);
    ASSERT_EQ(per->get_basis(), nullptr);
    ASSERT_EQ(per->get_projector(), nullptr);
}


TYPED_TEST(Perturbation, AppliesToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = Vec::create_with_config_of(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({29.0, 16.0}), r<TypeParam>::value);
}


TYPED_TEST(Perturbation, AppliesToMixedVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using Vec = gko::matrix::MultiVector<gko::next_precision<TypeParam>>;
    using value_type = typename Vec::value_type;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = Vec::create_with_config_of(x);

    cmp->apply(x, res);

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
    using Vec = gko::matrix::MultiVector<value_type>;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto x = gko::initialize<Vec>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}}, this->exec);
    auto res = Vec::create_with_config_of(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res,
                        l({value_type{29.0, -58.0}, value_type{16.0, -32.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Perturbation, AppliesLinearCombinationToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = gko::clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({86.0, 46.0}), r<TypeParam>::value);
}


TYPED_TEST(Perturbation, AppliesLinearCombinationToMixedVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using value_type = gko::next_precision<TypeParam>;
    using Vec = gko::matrix::MultiVector<value_type>;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = gko::clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({86.0, 46.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Perturbation, AppliesLinearCombinationToComplexVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    using MultiVector = typename TestFixture::Vec;
    using MultiVectorComplex = gko::to_complex<MultiVector>;
    using value_type = typename MultiVectorComplex::value_type;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis,
                                                    this->projector);
    auto alpha = gko::initialize<MultiVector>({3.0}, this->exec);
    auto beta = gko::initialize<MultiVector>({-1.0}, this->exec);
    auto x = gko::initialize<MultiVectorComplex>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}}, this->exec);
    auto res = gko::clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res,
                        l({value_type{86.0, -172.0}, value_type{46.0, -92.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Perturbation, ConstructionByBasisAppliesToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 2 1 ]
                      [ 1 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = Vec::create_with_config_of(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({17.0, 10.0}), r<TypeParam>::value);
}


TYPED_TEST(Perturbation, ConstructionByBasisAppliesLinearCombinationToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 2 1 ]
                      [ 1 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Perturbation<TypeParam>::create(this->scalar, this->basis);
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = gko::clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({50.0, 28.0}), r<TypeParam>::value);
}


}  // namespace
