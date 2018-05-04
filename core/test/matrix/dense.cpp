/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/matrix/dense.hpp>


#include <gtest/gtest.h>


#include <core/base/executor.hpp>
#include <core/base/mtx_reader.hpp>


namespace {


class Dense : public ::testing::Test {
protected:
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<gko::matrix::Dense<>>(
              4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}}, exec))
    {}


    static void assert_equal_to_original_mtx(gko::matrix::Dense<> *m)
    {
        ASSERT_EQ(m->get_size().num_rows, 2);
        ASSERT_EQ(m->get_size().num_cols, 3);
        ASSERT_EQ(m->get_stride(), 4);
        ASSERT_EQ(m->get_num_stored_elements(), 2 * 4);
        EXPECT_EQ(m->at(0, 0), 1.0);
        EXPECT_EQ(m->at(0, 1), 2.0);
        EXPECT_EQ(m->at(0, 2), 3.0);
        EXPECT_EQ(m->at(1, 0), 1.5);
        EXPECT_EQ(m->at(1, 1), 2.5);
        ASSERT_EQ(m->at(1, 2), 3.5);
    }

    static void assert_empty(gko::matrix::Dense<> *m)
    {
        EXPECT_EQ(m->get_size().num_rows, 0);
        EXPECT_EQ(m->get_size().num_cols, 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Dense<>> mtx;
};


TEST_F(Dense, CanBeEmpty)
{
    auto empty = gko::matrix::Dense<>::create(exec);
    assert_empty(empty.get());
}


TEST_F(Dense, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::matrix::Dense<>::create(exec);
    ASSERT_EQ(empty->get_const_values(), nullptr);
}


TEST_F(Dense, CanBeConstructedWithSize)
{
    auto m = gko::matrix::Dense<>::create(exec, gko::dim{2, 3});

    EXPECT_EQ(m->get_size().num_rows, 2);
    EXPECT_EQ(m->get_size().num_cols, 3);
    EXPECT_EQ(m->get_stride(), 3);
    ASSERT_EQ(m->get_num_stored_elements(), 6);
}


TEST_F(Dense, CanBeConstructedWithSizeAndstride)
{
    auto m = gko::matrix::Dense<>::create(exec, gko::dim{2, 3}, 4);

    EXPECT_EQ(m->get_size().num_rows, 2);
    EXPECT_EQ(m->get_size().num_cols, 3);
    EXPECT_EQ(m->get_stride(), 4);
    ASSERT_EQ(m->get_num_stored_elements(), 8);
}


TEST_F(Dense, KnowsItsSizeAndValues)
{
    assert_equal_to_original_mtx(mtx.get());
}


TEST_F(Dense, CanBeListConstructed)
{
    auto m = gko::initialize<gko::matrix::Dense<>>({1.0, 2.0}, exec);

    EXPECT_EQ(m->get_size().num_rows, 2);
    EXPECT_EQ(m->get_size().num_cols, 1);
    ASSERT_EQ(m->get_num_stored_elements(), 2);
    EXPECT_EQ(m->at(0), 1);
    EXPECT_EQ(m->at(1), 2);
}


TEST_F(Dense, CanBeListConstructedWithstride)
{
    auto m = gko::initialize<gko::matrix::Dense<>>(2, {1.0, 2.0}, exec);
    EXPECT_EQ(m->get_size().num_rows, 2);
    EXPECT_EQ(m->get_size().num_cols, 1);
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0), 1.0);
    EXPECT_EQ(m->at(1), 2.0);
}


TEST_F(Dense, CanBeDoubleListConstructed)
{
    auto m = gko::initialize<gko::matrix::Dense<>>(
        {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, exec);

    EXPECT_EQ(m->get_size().num_rows, 3);
    EXPECT_EQ(m->get_size().num_cols, 2);
    ASSERT_EQ(m->get_num_stored_elements(), 6);
    EXPECT_EQ(m->at(0), 1.0);
    EXPECT_EQ(m->at(1), 2.0);
    EXPECT_EQ(m->at(2), 3.0);
    ASSERT_EQ(m->at(3), 4.0);
    EXPECT_EQ(m->at(4), 5.0);
}


TEST_F(Dense, CanBeDoubleListConstructedWithstride)
{
    auto m = gko::initialize<gko::matrix::Dense<>>(
        4, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, exec);

    EXPECT_EQ(m->get_size().num_rows, 3);
    EXPECT_EQ(m->get_size().num_cols, 2);
    ASSERT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0), 1.0);
    EXPECT_EQ(m->at(1), 2.0);
    EXPECT_EQ(m->at(2), 3.0);
    ASSERT_EQ(m->at(3), 4.0);
    EXPECT_EQ(m->at(4), 5.0);
}


TEST_F(Dense, CanBeCopied)
{
    auto mtx_copy = gko::matrix::Dense<>::create(exec);
    mtx_copy->copy_from(mtx.get());
    assert_equal_to_original_mtx(mtx.get());
    mtx->at(0) = 7;
    assert_equal_to_original_mtx(mtx_copy.get());
}


TEST_F(Dense, CanBeMoved)
{
    auto mtx_copy = gko::matrix::Dense<>::create(exec);
    mtx_copy->copy_from(std::move(mtx));
    assert_equal_to_original_mtx(mtx_copy.get());
}


TEST_F(Dense, CanBeCloned)
{
    auto mtx_clone = mtx->clone();
    assert_equal_to_original_mtx(
        dynamic_cast<decltype(mtx.get())>(mtx_clone.get()));
}


TEST_F(Dense, CanBeCleared)
{
    mtx->clear();
    assert_empty(mtx.get());
}


TEST_F(Dense, CanBeReadFromMatrixData)
{
    auto m = gko::matrix::Dense<>::create(exec);
    m->read(gko::matrix_data<>{{2, 3},
                               {{0, 0, 1.0},
                                {0, 1, 3.0},
                                {0, 2, 2.0},
                                {1, 0, 0.0},
                                {1, 1, 5.0},
                                {1, 2, 0.0}}});

    ASSERT_EQ(m->get_size().num_rows, 2);
    ASSERT_EQ(m->get_size().num_cols, 3);
    ASSERT_EQ(m->get_num_stored_elements(), 6);
    EXPECT_EQ(m->at(0, 0), 1.0);
    EXPECT_EQ(m->at(1, 0), 0.0);
    EXPECT_EQ(m->at(0, 1), 3.0);
    EXPECT_EQ(m->at(1, 1), 5.0);
    EXPECT_EQ(m->at(0, 2), 2.0);
    ASSERT_EQ(m->at(1, 2), 0.0);
}


TEST_F(Dense, GeneratesCorrectMatrixData)
{
    using tpl = gko::matrix_data<>::nonzero_type;
    gko::matrix_data<> data;

    mtx->write(data);

    ASSERT_EQ(data.size, gko::dim(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 6);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, 1.0));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, 2.0));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, 3.0));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 0, 1.5));
    EXPECT_EQ(data.nonzeros[4], tpl(1, 1, 2.5));
    EXPECT_EQ(data.nonzeros[5], tpl(1, 2, 3.5));
}


}  // namespace
