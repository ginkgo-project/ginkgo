/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, Karlsruhe Institute of Technology
Copyright (c) 2017-2019, Universitat Jaume I
Copyright (c) 2017-2019, University of Tennessee
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

#include <ginkgo/core/base/array.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


namespace {


class Array : public ::testing::Test {
protected:
    Array() : exec(gko::ReferenceExecutor::create()), x(exec, 2)
    {
        x.get_data()[0] = 5;
        x.get_data()[1] = 2;
    }

    static void assert_equal_to_original_x(gko::Array<int> &a)
    {
        ASSERT_EQ(a.get_num_elems(), 2);
        EXPECT_EQ(a.get_data()[0], 5);
        EXPECT_EQ(a.get_data()[1], 2);
        EXPECT_EQ(a.get_const_data()[0], 5);
        EXPECT_EQ(a.get_const_data()[1], 2);
    }

    std::shared_ptr<const gko::Executor> exec;
    gko::Array<int> x;
};


TEST_F(Array, CanBeCreatedWithoutAnExecutor)
{
    gko::Array<int> a;

    ASSERT_EQ(a.get_executor(), nullptr);
    ASSERT_EQ(a.get_num_elems(), 0);
}


TEST_F(Array, CanBeEmpty)
{
    gko::Array<int> a(exec);

    ASSERT_EQ(a.get_num_elems(), 0);
}


TEST_F(Array, ReturnsNullWhenEmpty)
{
    gko::Array<int> a(exec);

    EXPECT_EQ(a.get_const_data(), nullptr);
    ASSERT_EQ(a.get_data(), nullptr);
}


TEST_F(Array, CanBeCreatedFromExistingData)
{
    gko::Array<int> a{exec, 3, new int[3], std::default_delete<int[]>{}};

    EXPECT_EQ(a.get_num_elems(), 3);
}


TEST_F(Array, CanBeCreatedFromDataOnExecutor)
{
    gko::Array<int> a{exec, 3, exec->alloc<int>(3)};

    EXPECT_EQ(a.get_num_elems(), 3);
}


TEST_F(Array, CanBeCreatedFromRange)
{
    using std::begin;
    auto data = {1, 2, 3};

    gko::Array<int> a{exec, begin(data), end(data)};

    EXPECT_EQ(a.get_const_data()[0], 1);
    EXPECT_EQ(a.get_const_data()[1], 2);
    ASSERT_EQ(a.get_const_data()[2], 3);
}


TEST_F(Array, CanBeCreatedFromInitializerList)
{
    gko::Array<int> a{exec, {1, 2, 3}};

    EXPECT_EQ(a.get_const_data()[0], 1);
    EXPECT_EQ(a.get_const_data()[1], 2);
    ASSERT_EQ(a.get_const_data()[2], 3);
}


TEST_F(Array, KnowsItsSize) { ASSERT_EQ(x.get_num_elems(), 2); }


TEST_F(Array, ReturnsValidDataPtr)
{
    EXPECT_EQ(x.get_data()[0], 5);
    EXPECT_EQ(x.get_data()[1], 2);
}


TEST_F(Array, ReturnsValidConstDataPtr)
{
    EXPECT_EQ(x.get_const_data()[0], 5);
    EXPECT_EQ(x.get_const_data()[1], 2);
}


TEST_F(Array, KnowsItsExecutor) { ASSERT_EQ(x.get_executor(), exec); }


TEST_F(Array, CanBeCopyConstructed)
{
    gko::Array<int> a(x);
    x.get_data()[0] = 7;

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeMoveConstructed)
{
    gko::Array<int> a(std::move(x));

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeCopyConstructedToADifferentExecutor)
{
    gko::Array<int> a{exec, x};

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeMoveConstructedToADifferentExecutor)
{
    gko::Array<int> a{exec, std::move(x)};

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeCopied)
{
    auto omp = gko::OmpExecutor::create();
    gko::Array<int> a(omp, 3);

    a = x;
    x.get_data()[0] = 7;

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeCopiedToExecutorlessArray)
{
    gko::Array<int> a;

    a = x;

    ASSERT_EQ(a.get_executor(), x.get_executor());
    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeCopiedFromExecutorlessArray)
{
    gko::Array<int> a;

    x = a;

    ASSERT_NE(x.get_executor(), nullptr);
    ASSERT_EQ(x.get_num_elems(), 0);
}


TEST_F(Array, CanBeMoved)
{
    auto omp = gko::OmpExecutor::create();
    gko::Array<int> a(omp, 3);

    a = std::move(x);

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeMovedToExecutorlessArray)
{
    gko::Array<int> a;

    a = std::move(x);

    ASSERT_NE(a.get_executor(), nullptr);
    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeMovedFromExecutorlessArray)
{
    gko::Array<int> a;

    x = std::move(a);

    ASSERT_NE(x.get_executor(), nullptr);
    ASSERT_EQ(x.get_num_elems(), 0);
}


TEST_F(Array, CanBeCleared)
{
    x.clear();

    ASSERT_EQ(x.get_num_elems(), 0);
    ASSERT_EQ(x.get_data(), nullptr);
    ASSERT_EQ(x.get_const_data(), nullptr);
}


TEST_F(Array, CanBeResized)
{
    x.resize_and_reset(3);

    x.get_data()[0] = 1;
    x.get_data()[1] = 8;
    x.get_data()[2] = 7;

    EXPECT_EQ(x.get_const_data()[0], 1);
    EXPECT_EQ(x.get_const_data()[1], 8);
    EXPECT_EQ(x.get_const_data()[2], 7);
}


TEST_F(Array, CanBeAssignedAnExecutor)
{
    gko::Array<int> a;

    a.set_executor(exec);

    ASSERT_EQ(a.get_executor(), exec);
}


TEST_F(Array, ChangesExecutors)
{
    auto omp = gko::OmpExecutor::create();
    x.set_executor(omp);

    ASSERT_EQ(x.get_executor(), omp);
    assert_equal_to_original_x(x);
}


TEST_F(Array, CanCreateView)
{
    int data[] = {1, 2, 3};

    auto view = gko::Array<int>::view(exec, 3, data);
    view = gko::Array<int>{exec, {5, 4, 2}};

    EXPECT_EQ(data[0], 5);
    EXPECT_EQ(data[1], 4);
    EXPECT_EQ(data[2], 2);
}


}  // namespace
