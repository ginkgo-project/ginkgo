// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/batch_diagonal.hpp>


// Copyright (c) 2017-2023, the Ginkgo authors
#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"


template <typename T>
class Diagonal : public ::testing::Test {
protected:
    using value_type = T;
    using MVec = gko::batch::MultiVector<value_type>;
    using Mtx = gko::batch::matrix::Diagonal<value_type>;
    using size_type = gko::size_type;
    Diagonal()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::batch::initialize<Mtx>(
              // clang-format off
              {{{-1.0, 0.0, 0.0},
                {0.0, 2.5, 0.0},
                {0.0, 0.0, 3.5}},
               {{1.0, 0.0, 0.0},
                {0.0, 2.5, 0.0},
                {0.0, 0.0, 4.0}}},
              // clang-format on
              exec))
    {}


    static void assert_equal_to_original_mtx(
        gko::batch::matrix::Diagonal<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 2);
        ASSERT_EQ(m->get_common_size(), gko::dim<2>(3, 3));
        ASSERT_EQ(m->get_values()[0], value_type{-1.0});
        ASSERT_EQ(m->get_values()[1], value_type{2.5});
        ASSERT_EQ(m->get_values()[2], value_type{3.5});
        ASSERT_EQ(m->get_values()[3], value_type{1.0});
        ASSERT_EQ(m->get_values()[4], value_type{2.5});
        ASSERT_EQ(m->get_values()[5], value_type{4.0});
    }

    static void assert_empty(gko::batch::matrix::Diagonal<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(Diagonal, gko::test::ValueTypes);


TYPED_TEST(Diagonal, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Diagonal, CanBeEmpty)
{
    auto empty = gko::batch::matrix::Diagonal<TypeParam>::create(this->exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(Diagonal, CanBeCopied)
{
    auto mtx_copy = gko::batch::matrix::Diagonal<TypeParam>::create(this->exec);

    mtx_copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Diagonal, CanBeMoved)
{
    auto mtx_copy = gko::batch::matrix::Diagonal<TypeParam>::create(this->exec);

    this->mtx->move_to(mtx_copy);

    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Diagonal, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(Diagonal, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Diagonal, CanBeConstructedWithSize)
{
    auto m = gko::batch::matrix::Diagonal<TypeParam>::create(
        this->exec, gko::batch_dim<2>(4, gko::dim<2>{4, 4}));

    ASSERT_EQ(m->get_num_batch_items(), 4);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(4, 4));
}


TYPED_TEST(Diagonal, FailsToConstructForRectangularSizes)
{
    ASSERT_THROW(gko::batch::matrix::Diagonal<TypeParam>::create(
                     this->exec, gko::batch_dim<2>(4, gko::dim<2>{3, 4})),
                 gko::BadDimension);
}
