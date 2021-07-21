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

#include <ginkgo/core/base/overlap.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Overlap : public ::testing::Test {
protected:
    Overlap()
        : exec(gko::ReferenceExecutor::create()),
          x(exec, gko::Array<T>{exec, I<T>{3, 2}},
            gko::Array<bool>{exec, I<bool>{true, false}},
            gko::Array<bool>{exec, I<bool>{false, true}})
    {}

    static void assert_equal_to_original(gko::Overlap<T> &a)
    {
        ASSERT_EQ(a.get_num_elems(), 2);
        EXPECT_EQ(a.get_overlaps()[0], T{3});
        EXPECT_EQ(a.get_overlaps()[1], T{2});
        EXPECT_EQ(a.get_unidirectional_array()[0], true);
        EXPECT_EQ(a.get_unidirectional_array()[1], false);
        EXPECT_EQ(a.get_overlap_at_start_array()[0], false);
        EXPECT_EQ(a.get_overlap_at_start_array()[1], true);
    }

    std::shared_ptr<const gko::Executor> exec;
    gko::Overlap<T> x;
};

TYPED_TEST_SUITE(Overlap, gko::test::IndexAndSizeTypes);


TYPED_TEST(Overlap, CanBeCreatedWithoutAnExecutor)
{
    gko::Overlap<TypeParam> a;

    ASSERT_EQ(a.get_executor(), nullptr);
    ASSERT_EQ(a.get_num_elems(), 0);
}


TYPED_TEST(Overlap, CanBeEmpty)
{
    gko::Overlap<TypeParam> a(this->exec);

    ASSERT_EQ(a.get_num_elems(), 0);
}


TYPED_TEST(Overlap, ReturnsNullWhenEmpty)
{
    gko::Overlap<TypeParam> a(this->exec);

    ASSERT_EQ(a.get_overlaps(), nullptr);
    ASSERT_EQ(a.get_unidirectional_array(), nullptr);
    ASSERT_EQ(a.get_overlap_at_start_array(), nullptr);
}


TYPED_TEST(Overlap, CanBeCopyConstructed)
{
    gko::Overlap<TypeParam> a(this->x);

    this->assert_equal_to_original(a);
}


TYPED_TEST(Overlap, CanBeMoveConstructed)
{
    gko::Overlap<TypeParam> a(std::move(this->x));

    this->assert_equal_to_original(a);
}


TYPED_TEST(Overlap, CanBeCopyConstructedToADifferentExecutor)
{
    gko::Overlap<TypeParam> a{this->exec, this->x};

    this->assert_equal_to_original(a);
}


TYPED_TEST(Overlap, CanBeMoveConstructedToADifferentExecutor)
{
    gko::Overlap<TypeParam> a{this->exec, std::move(this->x)};

    this->assert_equal_to_original(a);
}


TYPED_TEST(Overlap, CanBeCopied)
{
    auto omp = gko::OmpExecutor::create();
    gko::Overlap<TypeParam> a(omp, gko::size_type(3), TypeParam{2});

    a = this->x;

    this->assert_equal_to_original(a);
}


TYPED_TEST(Overlap, CanBeCopiedToExecutorlessOverlap)
{
    gko::Overlap<TypeParam> a;

    a = this->x;

    ASSERT_EQ(a.get_executor(), this->x.get_executor());
    this->assert_equal_to_original(a);
}


TYPED_TEST(Overlap, CanBeCopiedFromExecutorlessOverlap)
{
    gko::Overlap<TypeParam> a;

    this->x = a;

    ASSERT_NE(this->x.get_executor(), nullptr);
    ASSERT_EQ(this->x.get_num_elems(), 0);
}


TYPED_TEST(Overlap, CanBeMoved)
{
    auto omp = gko::OmpExecutor::create();
    gko::Overlap<TypeParam> a(omp, gko::size_type(3), TypeParam{2});

    a = std::move(this->x);

    this->assert_equal_to_original(a);
}


TYPED_TEST(Overlap, CanBeMovedToExecutorlessOverlap)
{
    gko::Overlap<TypeParam> a;

    a = std::move(this->x);

    ASSERT_NE(a.get_executor(), nullptr);
    this->assert_equal_to_original(a);
}


TYPED_TEST(Overlap, CanBeMovedFromExecutorlessOverlap)
{
    gko::Overlap<TypeParam> a;

    this->x = std::move(a);

    ASSERT_NE(this->x.get_executor(), nullptr);
    ASSERT_EQ(this->x.get_num_elems(), 0);
}


TYPED_TEST(Overlap, KnowsItsSize) { ASSERT_EQ(this->x.get_num_elems(), 2); }


TYPED_TEST(Overlap, CanGetSingleOverlap)
{
    ASSERT_EQ(this->x.get_overlap(), 3);
}


TYPED_TEST(Overlap, CanCheckUnidir)
{
    ASSERT_EQ(this->x.is_unidirectional(), true);
}


TYPED_TEST(Overlap, CanCheckStartEnd)
{
    ASSERT_EQ(this->x.is_overlap_at_start(), false);
}


TYPED_TEST(Overlap, CanBeCreatedFromInitializerList)
{
    this->assert_equal_to_original(this->x);
}


TYPED_TEST(Overlap, CanBeCreatedWithDuplicatedElements)
{
    auto a = gko::Overlap<TypeParam>(this->exec, 3, 2, true, false);
    ASSERT_EQ(a.get_num_elems(), 3);
    EXPECT_EQ(a.get_overlaps()[0], TypeParam{2});
    EXPECT_EQ(a.get_overlaps()[1], TypeParam{2});
    EXPECT_EQ(a.get_overlaps()[2], TypeParam{2});
    EXPECT_EQ(a.get_unidirectional_array()[0], true);
    EXPECT_EQ(a.get_unidirectional_array()[1], true);
    EXPECT_EQ(a.get_unidirectional_array()[2], true);
    EXPECT_EQ(a.get_overlap_at_start_array()[0], false);
    EXPECT_EQ(a.get_overlap_at_start_array()[1], false);
    EXPECT_EQ(a.get_overlap_at_start_array()[2], true);
}


}  // namespace
