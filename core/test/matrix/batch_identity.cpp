// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/batch_identity.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"


template <typename T>
class Identity : public ::testing::Test {
protected:
    using value_type = T;
    using MVec = gko::batch::MultiVector<value_type>;
    using size_type = gko::size_type;
    Identity()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::batch::matrix::Identity<value_type>::create(
              exec, gko::batch_dim<2>(2, gko::dim<2>(3, 3)))),
          mvec(gko::batch::initialize<gko::batch::MultiVector<value_type>>(
              {{{-1.0, 2.0, 3.0}, {-1.0, 8.0, 3.0}, {-1.5, 2.5, 3.5}},
               {{-1.0, 3.0, 2.0}, {8.0, 5.5, 7.0}, {1.0, 2.0, 5.0}}},
              exec))
    {}


    static void assert_equal_to_original_mtx(
        gko::batch::matrix::Identity<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 2);
        ASSERT_EQ(m->get_common_size(), gko::dim<2>(3, 3));
    }

    static void assert_empty(gko::batch::matrix::Identity<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::batch::matrix::Identity<value_type>> mtx;
    std::unique_ptr<gko::batch::MultiVector<value_type>> mvec;
};

TYPED_TEST_SUITE(Identity, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Identity, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Identity, CanBeEmpty)
{
    auto empty = gko::batch::matrix::Identity<TypeParam>::create(this->exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(Identity, CanBeCopied)
{
    auto mtx_copy = gko::batch::matrix::Identity<TypeParam>::create(this->exec);

    mtx_copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Identity, CanBeMoved)
{
    auto mtx_copy = gko::batch::matrix::Identity<TypeParam>::create(this->exec);

    this->mtx->move_to(mtx_copy);

    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Identity, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(Identity, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Identity, CanBeConstructedWithSize)
{
    auto m = gko::batch::matrix::Identity<TypeParam>::create(
        this->exec, gko::batch_dim<2>(4, gko::dim<2>{4, 4}));

    ASSERT_EQ(m->get_num_batch_items(), 4);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(4, 4));
}


TYPED_TEST(Identity, FailsToConstructForRectangularSizes)
{
    ASSERT_THROW(gko::batch::matrix::Identity<TypeParam>::create(
                     this->exec, gko::batch_dim<2>(4, gko::dim<2>{3, 4})),
                 gko::BadDimension);
}


TYPED_TEST(Identity, CanApplytoMultiVector)
{
    using MVec = typename TestFixture::MVec;
    using value_type = typename TestFixture::value_type;
    auto x = this->mvec->clone();
    x->fill(gko::zero<value_type>());
    ASSERT_EQ(x->at(0, 0, 0), value_type{0.0});

    this->mtx->apply(this->mvec, x);

    GKO_ASSERT_BATCH_MTX_NEAR(this->mvec, x, 0.0);
}


TYPED_TEST(Identity, CanAdvancedApplytoMultiVector)
{
    using MVec = typename TestFixture::MVec;
    using value_type = typename TestFixture::value_type;
    auto x = this->mvec->clone();
    x->fill(gko::one<value_type>());
    ASSERT_EQ(x->at(0, 0, 0), value_type{1.0});
    auto alpha = gko::batch::initialize<MVec>({{1.0}, {-1.0}}, this->exec);
    auto beta = gko::batch::initialize<MVec>({{2.0}, {-4.0}}, this->exec);
    auto axpby = x->clone();
    axpby->scale(beta);
    axpby->add_scaled(alpha, this->mvec);

    this->mtx->apply(alpha, this->mvec, beta, x);

    GKO_ASSERT_BATCH_MTX_NEAR(axpby, x, 0.0);
}
