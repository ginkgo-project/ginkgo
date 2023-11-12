/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/matrix/batch_diagonal.hpp>


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
