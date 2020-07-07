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

#include <ginkgo/core/matrix/dense.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Dense : public ::testing::Test {
protected:
    using value_type = T;
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<gko::matrix::Dense<value_type>>(
              4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}}, exec))
    {}


    static void assert_equal_to_original_mtx(gko::matrix::Dense<value_type> *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_stride(), 4);
        ASSERT_EQ(m->get_num_stored_elements(), 2 * 4);
        EXPECT_EQ(m->at(0, 0), value_type{1.0});
        EXPECT_EQ(m->at(0, 1), value_type{2.0});
        EXPECT_EQ(m->at(0, 2), value_type{3.0});
        EXPECT_EQ(m->at(1, 0), value_type{1.5});
        EXPECT_EQ(m->at(1, 1), value_type{2.5});
        ASSERT_EQ(m->at(1, 2), value_type{3.5});
    }

    static void assert_empty(gko::matrix::Dense<value_type> *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Dense<value_type>> mtx;
};

TYPED_TEST_CASE(Dense, gko::test::ValueTypes);


TYPED_TEST(Dense, CanBeEmpty)
{
    auto empty = gko::matrix::Dense<TypeParam>::create(this->exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(Dense, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::matrix::Dense<TypeParam>::create(this->exec);
    ASSERT_EQ(empty->get_const_values(), nullptr);
}


TYPED_TEST(Dense, CanBeConstructedWithSize)
{
    auto m =
        gko::matrix::Dense<TypeParam>::create(this->exec, gko::dim<2>{2, 3});

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    EXPECT_EQ(m->get_stride(), 3);
    ASSERT_EQ(m->get_num_stored_elements(), 6);
}


TYPED_TEST(Dense, CanBeConstructedWithSizeAndStride)
{
    auto m =
        gko::matrix::Dense<TypeParam>::create(this->exec, gko::dim<2>{2, 3}, 4);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    EXPECT_EQ(m->get_stride(), 4);
    ASSERT_EQ(m->get_num_stored_elements(), 8);
}


TYPED_TEST(Dense, CanBeConstructedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    // clang-format off
    value_type data[] = {
        1.0, 2.0, -1.0,
        3.0, 4.0, -1.0,
        5.0, 6.0, -1.0};
    // clang-format on

    auto m = gko::matrix::Dense<TypeParam>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::Array<value_type>::view(this->exec, 9, data), 3);

    ASSERT_EQ(m->get_const_values(), data);
    ASSERT_EQ(m->at(2, 1), value_type{6.0});
}


TYPED_TEST(Dense, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Dense, CanBeListConstructed)
{
    using value_type = typename TestFixture::value_type;
    auto m =
        gko::initialize<gko::matrix::Dense<TypeParam>>({1.0, 2.0}, this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 2);
    EXPECT_EQ(m->at(0), value_type{1});
    EXPECT_EQ(m->at(1), value_type{2});
}


TYPED_TEST(Dense, CanBeListConstructedWithstride)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(2, {1.0, 2.0},
                                                            this->exec);
    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
}


TYPED_TEST(Dense, CanBeDoubleListConstructed)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(
        {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 6);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
    EXPECT_EQ(m->at(2), value_type{3.0});
    ASSERT_EQ(m->at(3), value_type{4.0});
    EXPECT_EQ(m->at(4), value_type{5.0});
}


TYPED_TEST(Dense, CanBeDoubleListConstructedWithstride)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
    EXPECT_EQ(m->at(2), value_type{3.0});
    ASSERT_EQ(m->at(3), value_type{4.0});
    EXPECT_EQ(m->at(4), value_type{5.0});
}


TYPED_TEST(Dense, CanBeCopied)
{
    auto mtx_copy = gko::matrix::Dense<TypeParam>::create(this->exec);
    mtx_copy->copy_from(this->mtx.get());
    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->at(0) = 7;
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Dense, CanBeMoved)
{
    auto mtx_copy = gko::matrix::Dense<TypeParam>::create(this->exec);
    mtx_copy->copy_from(std::move(this->mtx));
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Dense, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();
    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(Dense, CanBeCleared)
{
    this->mtx->clear();
    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Dense, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::Dense<TypeParam>::create(this->exec);
    m->read(gko::matrix_data<TypeParam>{{2, 3},
                                        {{0, 0, 1.0},
                                         {0, 1, 3.0},
                                         {0, 2, 2.0},
                                         {1, 0, 0.0},
                                         {1, 1, 5.0},
                                         {1, 2, 0.0}}});

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_num_stored_elements(), 6);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(1, 0), value_type{0.0});
    EXPECT_EQ(m->at(0, 1), value_type{3.0});
    EXPECT_EQ(m->at(1, 1), value_type{5.0});
    EXPECT_EQ(m->at(0, 2), value_type{2.0});
    ASSERT_EQ(m->at(1, 2), value_type{0.0});
}


TYPED_TEST(Dense, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using tpl = typename gko::matrix_data<TypeParam>::nonzero_type;
    gko::matrix_data<TypeParam> data;

    this->mtx->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 6);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 0, value_type{1.5}));
    EXPECT_EQ(data.nonzeros[4], tpl(1, 1, value_type{2.5}));
    EXPECT_EQ(data.nonzeros[5], tpl(1, 2, value_type{3.5}));
}


TYPED_TEST(Dense, CanCreateSubmatrix)
{
    using value_type = typename TestFixture::value_type;
    auto submtx = this->mtx->create_submatrix(gko::span{0, 1}, gko::span{1, 2});

    EXPECT_EQ(submtx->at(0, 0), value_type{2.0});
    EXPECT_EQ(submtx->at(0, 1), value_type{3.0});
    EXPECT_EQ(submtx->at(1, 0), value_type{2.5});
    EXPECT_EQ(submtx->at(1, 1), value_type{3.5});
}


TYPED_TEST(Dense, CanCreateSubmatrixWithStride)
{
    using value_type = typename TestFixture::value_type;
    auto submtx =
        this->mtx->create_submatrix(gko::span{0, 1}, gko::span{1, 2}, 3);

    EXPECT_EQ(submtx->at(0, 0), value_type{2.0});
    EXPECT_EQ(submtx->at(0, 1), value_type{3.0});
    EXPECT_EQ(submtx->at(1, 0), value_type{1.5});
    EXPECT_EQ(submtx->at(1, 1), value_type{2.5});
}


}  // namespace
