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

#include <core/matrix/sellp.hpp>


#include <gtest/gtest.h>


namespace {


class Sellp : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Sellp<>;

    Sellp()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Sellp<>::create(exec, gko::dim<2>{2, 3}, 3))
    {
        Mtx::value_type *v = mtx->get_values();
        Mtx::index_type *c = mtx->get_col_idxs();
        gko::size_type *l = mtx->get_slice_lengths();
        gko::size_type *s = mtx->get_slice_sets();
        l[0] = gko::matrix::default_stride_factor *
               gko::ceildiv(3, gko::matrix::default_stride_factor);
        s[0] = 0;
        s[1] = l[0];
        c[0] = 0;
        c[1] = 1;
        c[gko::matrix::default_slice_size] = 1;
        c[gko::matrix::default_slice_size + 1] = 0;
        c[2 * gko::matrix::default_slice_size] = 2;
        c[2 * gko::matrix::default_slice_size + 1] = 0;
        v[0] = 1.0;
        v[1] = 5.0;
        v[gko::matrix::default_slice_size] = 3.0;
        v[gko::matrix::default_slice_size + 1] = 0.0;
        v[2 * gko::matrix::default_slice_size] = 2.0;
        v[2 * gko::matrix::default_slice_size + 1] = 0.0;
        for (int i = 3; i < l[0]; i++) {
            for (int j = 0; j < 2; j++) {
                c[i * gko::matrix::default_slice_size + j] =
                    c[2 * gko::matrix::default_slice_size + j];
                v[i * gko::matrix::default_slice_size + j] = 0;
            }
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto l = m->get_const_slice_lengths();
        auto s = m->get_const_slice_sets();
        auto slice_size = m->get_slice_size();
        auto stride_factor = m->get_stride_factor();
        auto total_cols = m->get_total_cols();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 192);
        ASSERT_EQ(m->get_slice_size(), gko::matrix::default_slice_size);
        ASSERT_EQ(m->get_stride_factor(), gko::matrix::default_stride_factor);
        ASSERT_EQ(m->get_total_cols(), 3);
        EXPECT_EQ(l[0], 3);
        EXPECT_EQ(s[0], 0);
        EXPECT_EQ(s[1], 3);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
        EXPECT_EQ(c[gko::matrix::default_slice_size + 1], 0);
        EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
        EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], 0);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 5.0);
        EXPECT_EQ(v[gko::matrix::default_slice_size], 3.0);
        EXPECT_EQ(v[gko::matrix::default_slice_size + 1], 0.0);
        EXPECT_EQ(v[2 * gko::matrix::default_slice_size], 2.0);
        EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], 0.0);
    }

    void assert_equal_to_original_mtx_with_slice_size_and_stride_factor(
        const Mtx *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto l = m->get_const_slice_lengths();
        auto s = m->get_const_slice_sets();
        auto slice_size = m->get_slice_size();
        auto stride_factor = m->get_stride_factor();
        auto total_cols = m->get_total_cols();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 8);
        ASSERT_EQ(m->get_slice_size(), 2);
        ASSERT_EQ(m->get_stride_factor(), 2);
        ASSERT_EQ(m->get_total_cols(), 4);
        EXPECT_EQ(l[0], 4);
        EXPECT_EQ(s[0], 0);
        EXPECT_EQ(s[1], 4);
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
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_total_cols(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_EQ(m->get_const_slice_lengths(), nullptr);
        ASSERT_EQ(m->get_const_slice_sets(), nullptr);
    }
};


TEST_F(Sellp, KnowsItsSize)
{
    ASSERT_EQ(mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(mtx->get_num_stored_elements(), 192);
    ASSERT_EQ(mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(mtx->get_stride_factor(), gko::matrix::default_stride_factor);
    ASSERT_EQ(mtx->get_total_cols(), 3);
}


TEST_F(Sellp, ContainsCorrectData) { assert_equal_to_original_mtx(mtx.get()); }


TEST_F(Sellp, CanBeEmpty)
{
    auto mtx = Mtx::create(exec);

    assert_empty(mtx.get());
}

TEST_F(Sellp, CanBeConstructedWithSliceSizeAndStrideFactor)
{
    auto mtx = Mtx::create(exec, gko::dim<2>{2, 3}, 2, 2, 3);

    ASSERT_EQ(mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(mtx->get_slice_size(), 2);
    ASSERT_EQ(mtx->get_stride_factor(), 2);
    ASSERT_EQ(mtx->get_total_cols(), 3);
}


TEST_F(Sellp, CanBeCopied)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(mtx.get());

    assert_equal_to_original_mtx(mtx.get());
    mtx->get_values()[1] = 5.0;
    assert_equal_to_original_mtx(copy.get());
}


TEST_F(Sellp, CanBeMoved)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(std::move(mtx));

    assert_equal_to_original_mtx(copy.get());
}


TEST_F(Sellp, CanBeCloned)
{
    auto clone = mtx->clone();

    assert_equal_to_original_mtx(mtx.get());
    mtx->get_values()[1] = 5.0;
    assert_equal_to_original_mtx(dynamic_cast<Mtx *>(clone.get()));
}


TEST_F(Sellp, CanBeCleared)
{
    mtx->clear();

    assert_empty(mtx.get());
}


TEST_F(Sellp, CanBeReadFromMatrixData)
{
    auto m = Mtx::create(exec);
    m->read({{2, 3},
             {{0, 0, 1.0},
              {0, 1, 3.0},
              {0, 2, 2.0},
              {1, 0, 0.0},
              {1, 1, 5.0},
              {1, 2, 0.0}}});

    assert_equal_to_original_mtx(m.get());
}

TEST_F(Sellp, CanBeReadFromMatrixDataWithSliceSizeAndStrideFactor)
{
    auto m = Mtx::create(exec, gko::dim<2>{2, 3}, 2, 2, 3);
    m->read({{2, 3},
             {{0, 0, 1.0},
              {0, 1, 3.0},
              {0, 2, 2.0},
              {1, 0, 0.0},
              {1, 1, 5.0},
              {1, 2, 0.0}}});

    assert_equal_to_original_mtx_with_slice_size_and_stride_factor(m.get());
}

TEST_F(Sellp, GeneratesCorrectMatrixData)
{
    using tpl = gko::matrix_data<>::nonzero_type;
    gko::matrix_data<> data;

    mtx->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, 1.0));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, 3.0));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, 2.0));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, 5.0));
}


}  // namespace
