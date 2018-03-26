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

#include <core/matrix/coo.hpp>


#include <gtest/gtest.h>


#include <core/base/mtx_reader.hpp>


namespace {


class Coo : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Coo<>;

    Coo()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Coo<>::create(exec, 2, 3, 4))
    {
        Mtx::value_type *v = mtx->get_values();
        Mtx::index_type *c = mtx->get_col_idxs();
        Mtx::index_type *r = mtx->get_row_idxs();
        r[0] = 0;
        r[1] = 0;
        r[2] = 0;
        r[3] = 1;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 5.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_idxs();
        ASSERT_EQ(m->get_num_rows(), 2);
        ASSERT_EQ(m->get_num_cols(), 3);
        ASSERT_EQ(m->get_num_stored_elements(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 0);
        EXPECT_EQ(r[2], 0);
        EXPECT_EQ(r[3], 1);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 3.0);
        EXPECT_EQ(v[2], 2.0);
        EXPECT_EQ(v[3], 5.0);
    }

    void assert_empty(const Mtx *m)
    {
        ASSERT_EQ(m->get_num_rows(), 0);
        ASSERT_EQ(m->get_num_cols(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_EQ(m->get_const_row_idxs(), nullptr);
    }
};


TEST_F(Coo, KnowsItsSize)
{
    ASSERT_EQ(mtx->get_num_rows(), 2);
    ASSERT_EQ(mtx->get_num_cols(), 3);
    ASSERT_EQ(mtx->get_num_stored_elements(), 4);
}


TEST_F(Coo, ContainsCorrectData) { assert_equal_to_original_mtx(mtx.get()); }


TEST_F(Coo, CanBeEmpty)
{
    auto mtx = Mtx::create(exec);

    assert_empty(mtx.get());
}


TEST_F(Coo, CanBeCopied)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(mtx.get());

    assert_equal_to_original_mtx(mtx.get());
    mtx->get_values()[1] = 5.0;
    assert_equal_to_original_mtx(copy.get());
}


TEST_F(Coo, CanBeMoved)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(std::move(mtx));

    assert_equal_to_original_mtx(copy.get());
}


TEST_F(Coo, CanBeCloned)
{
    auto clone = mtx->clone();

    assert_equal_to_original_mtx(mtx.get());
    mtx->get_values()[1] = 5.0;
    assert_equal_to_original_mtx(dynamic_cast<Mtx *>(clone.get()));
}

TEST_F(Coo, CanBeCleared)
{
    mtx->clear();

    assert_empty(mtx.get());
}


TEST_F(Coo, CanBeReadFromMatrixData)
{
    auto m = Mtx::create(exec);
    m->read({2,
             3,
             {{0, 0, 1.0},
              {0, 1, 3.0},
              {0, 2, 2.0},
              {1, 0, 0.0},
              {1, 1, 5.0},
              {1, 2, 0.0}}});

    assert_equal_to_original_mtx(m.get());
}


TEST_F(Coo, GeneratesCorrectMatrixData)
{
    using tpl = gko::matrix_data<>::nonzero_type;
    gko::matrix_data<> data;

    mtx->write(data);

    ASSERT_EQ(data.num_rows, 2);
    ASSERT_EQ(data.num_cols, 3);
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, 1.0));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, 3.0));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, 2.0));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, 5.0));
}


}  // namespace
