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

#include <core/matrix/sliced_ell.hpp>


#include <gtest/gtest.h>


namespace {


class Sliced_ell : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Sliced_ell<>;

    Sliced_ell()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Sliced_ell<>::create(exec, 2, 3, 4))
    {
        Mtx::value_type *v = mtx->get_values();
        Mtx::index_type *c = mtx->get_col_idxs();
        Mtx::index_type *l = mtx->get_slice_lens();
        Mtx::index_type *s = mtx->get_slice_sets();
        l[0] = 3;
        s[0] = 0;
        c[0] = 0;
        c[1] = 1;
        c[default_slice_size] = 1;
        c[default_slice_size+1] = 1;
        c[2*default_slice_size] = 2;
        c[2*default_slice_size+1] = 1;
        v[0] = 1.0;
        v[1] = 5.0;
        v[default_slice_size] = 3.0;
        v[default_slice_size+1] = 0.0;
        v[2*default_slice_size] = 2.0;
        v[2*default_slice_size+1] = 0.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto l = m->get_const_slice_lens();
        auto s = m->get_const_slice_sets();
        ASSERT_EQ(m->get_num_rows(), 2);
        ASSERT_EQ(m->get_num_cols(), 3);
        ASSERT_EQ(m->get_num_stored_elements(), 4);
        EXPECT_EQ(l[0], 3);
        EXPECT_EQ(s[0], 0);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[default_slice_size], 1);
        EXPECT_EQ(c[default_slice_size+1], 1);
        EXPECT_EQ(c[2*default_slice_size], 2);
        EXPECT_EQ(c[2*default_slice_size+1], 1);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 5.0);
        EXPECT_EQ(v[default_slice_size], 3.0);
        EXPECT_EQ(v[default_slice_size+1], 0.0);
        EXPECT_EQ(v[2*default_slice_size], 2.0);
        EXPECT_EQ(v[2*default_slice_size+1], 0.0);
    }

    void assert_empty(const Mtx *m)
    {
        ASSERT_EQ(m->get_num_rows(), 0);
        ASSERT_EQ(m->get_num_cols(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_EQ(m->get_const_slice_lens(), nullptr);
        ASSERT_EQ(m->get_const_slice_sets(), nullptr);
    }
};


TEST_F(Sliced_ell, KnowsItsSize)
{
    ASSERT_EQ(mtx->get_num_rows(), 2);
    ASSERT_EQ(mtx->get_num_cols(), 3);
    ASSERT_EQ(mtx->get_num_stored_elements(), 4);
}


TEST_F(Sliced_ell, ContainsCorrectData) { assert_equal_to_original_mtx(mtx.get()); }


TEST_F(Sliced_ell, CanBeEmpty)
{
    auto mtx = Mtx::create(exec);

    assert_empty(mtx.get());
}


TEST_F(Sliced_ell, CanBeCopied)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(mtx.get());

    assert_equal_to_original_mtx(mtx.get());
    mtx->get_values()[1] = 5.0;
    assert_equal_to_original_mtx(copy.get());
}


TEST_F(Sliced_ell, CanBeMoved)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(std::move(mtx));

    assert_equal_to_original_mtx(copy.get());
}


TEST_F(Sliced_ell, CanBeCloned)
{
    auto clone = mtx->clone();

    assert_equal_to_original_mtx(mtx.get());
    mtx->get_values()[1] = 5.0;
    assert_equal_to_original_mtx(dynamic_cast<Mtx *>(clone.get()));
}


TEST_F(Sliced_ell, CanBeCleared)
{
    mtx->clear();

    assert_empty(mtx.get());
}


TEST_F(Sliced_ell, CanBeReadFromMtx)
{
    auto m = Mtx::create(exec);

    m->read_from_mtx("../base/data/dense_real.mtx");

    assert_equal_to_original_mtx(m.get());
}


}  // namespace
