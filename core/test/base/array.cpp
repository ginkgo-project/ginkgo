/*
 * Copyright 2017-2018
 *
 * Karlsruhe Institute of Technology
 *
 * Universitat Jaume I
 *
 * University of Tennessee
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <core/base/array.hpp>


#include <gtest/gtest.h>


#include <core/base/executor.hpp>


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


TEST_F(Array, CanBeCopied)
{
    auto cpu = gko::CpuExecutor::create();
    gko::Array<int> a(cpu, 3);
    a = x;
    x.get_data()[0] = 7;

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeMoved)
{
    auto cpu = gko::CpuExecutor::create();
    gko::Array<int> a(cpu, 3);
    a = std::move(x);

    assert_equal_to_original_x(a);
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


TEST_F(Array, ManagesExternalData)
{
    int *data = nullptr;
    ASSERT_NE(data = reinterpret_cast<int *>(std::malloc(3 * sizeof(int))),
              nullptr);
    data[0] = 1;
    data[1] = 8;
    data[2] = 7;
    x.manage(3, data);

    ASSERT_EQ(x.get_num_elems(), 3);
    EXPECT_EQ(x.get_const_data()[0], 1);
    EXPECT_EQ(x.get_const_data()[1], 8);
    EXPECT_EQ(x.get_const_data()[2], 7);
}


TEST_F(Array, ReleasesData)
{
    int *data = x.get_data();

    ASSERT_NO_THROW(x.release());
    ASSERT_NO_THROW(std::free(data));
    ASSERT_EQ(x.get_data(), nullptr);
    ASSERT_EQ(x.get_const_data(), nullptr);
    ASSERT_EQ(x.get_num_elems(), 0);
}


TEST_F(Array, ChangesExecutors)
{
    auto cpu = gko::CpuExecutor::create();
    x.set_executor(cpu);

    ASSERT_EQ(x.get_executor(), cpu);
    assert_equal_to_original_x(x);
}


}  // namespace
