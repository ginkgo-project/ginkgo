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

#include <core/matrix/ell.hpp>


#include <gtest/gtest.h>


namespace {


class Ell : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Ell<>;

    Ell()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Ell<>::create(exec, 2, 3, 3))
    {
        Mtx::value_type *v = mtx->get_values();
        Mtx::index_type *c = mtx->get_col_idxs();
        c[0] = 0;
        c[1] = 1;
        c[2] = 1;
        c[3] = 0;
        c[4] = 2;
        c[5] = 0;
        v[0] = 1.0;
        v[1] = 5.0;
        v[2] = 3.0;
        v[3] = 0.0;
        v[4] = 2.0;
        v[5] = 0.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto n = m->get_max_nonzeros_per_row();
        auto p = m->get_stride();
        ASSERT_EQ(m->get_num_rows(), 2);
        ASSERT_EQ(m->get_num_cols(), 3);
        ASSERT_EQ(m->get_num_stored_elements(), 6);
        EXPECT_EQ(n, 3);
        EXPECT_EQ(p, 2);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 1);
        EXPECT_EQ(c[3], 0);
        EXPECT_EQ(c[4], 2);
        EXPECT_EQ(c[5], 0);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 5.0);
        EXPECT_EQ(v[2], 3.0);
        EXPECT_EQ(v[3], 0.0);
        EXPECT_EQ(v[4], 2.0);
        EXPECT_EQ(v[5], 0.0);
    }

    void assert_empty(const Mtx *m)
    {
        ASSERT_EQ(m->get_num_rows(), 0);
        ASSERT_EQ(m->get_num_cols(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_EQ(m->get_max_nonzeros_per_row(), 0);
        ASSERT_EQ(m->get_stride(), 0);
    }
};


TEST_F(Ell, KnowsItsSize)
{
    ASSERT_EQ(mtx->get_num_rows(), 2);
    ASSERT_EQ(mtx->get_num_cols(), 3);
    ASSERT_EQ(mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(mtx->get_max_nonzeros_per_row(), 3);
    ASSERT_EQ(mtx->get_stride(), 2);
}


TEST_F(Ell, ContainsCorrectData) { assert_equal_to_original_mtx(mtx.get()); }


TEST_F(Ell, CanBeEmpty)
{
    auto mtx = Mtx::create(exec);

    assert_empty(mtx.get());
}


TEST_F(Ell, CanBeCopied)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(mtx.get());

    assert_equal_to_original_mtx(mtx.get());
    mtx->get_values()[1] = 5.0;
    assert_equal_to_original_mtx(copy.get());
}


TEST_F(Ell, CanBeMoved)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(std::move(mtx));

    assert_equal_to_original_mtx(copy.get());
}


TEST_F(Ell, CanBeCloned)
{
    auto clone = mtx->clone();

    assert_equal_to_original_mtx(mtx.get());
    mtx->get_values()[1] = 5.0;
    assert_equal_to_original_mtx(static_cast<Mtx *>(clone.get()));
}


TEST_F(Ell, CanBeCleared)
{
    mtx->clear();

    assert_empty(mtx.get());
}


TEST_F(Ell, CanBeReadFromMtx)
{
    auto m = Mtx::create(exec);

    m->read_from_mtx("../base/data/dense_real.mtx");

    assert_equal_to_original_mtx(m.get());
}


}  // namespace
