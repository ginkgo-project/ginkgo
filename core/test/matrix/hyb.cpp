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

#include <core/matrix/hyb.hpp>


#include <gtest/gtest.h>


#include <core/base/mtx_reader.hpp>


namespace {


class Hyb : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Hyb<>;

    Hyb()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Hyb<>::create(exec, gko::dim{2, 3}, 2, 2, 1))
    {
        Mtx::value_type *v = mtx->get_ell_values();
        Mtx::index_type *c = mtx->get_ell_col_idxs();
        c[0] = 0;
        c[1] = 1;
        c[2] = 1;
        c[3] = 0;
        v[0] = 1.0;
        v[1] = 5.0;
        v[2] = 3.0;
        v[3] = 0.0;
        mtx->get_coo_values()[0] = 2.0;
        mtx->get_coo_col_idxs()[0] = 2;
        mtx->get_coo_row_idxs()[0] = 0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx *m)
    {
        auto v = m->get_const_ell_values();
        auto c = m->get_const_ell_col_idxs();
        auto n = m->get_ell_max_nonzeros_per_row();
        auto p = m->get_ell_stride();
        ASSERT_EQ(m->get_size(), gko::dim(2, 3));
        ASSERT_EQ(m->get_ell_num_stored_elements(), 4);
        ASSERT_EQ(m->get_coo_num_stored_elements(), 1);
        EXPECT_EQ(n, 2);
        EXPECT_EQ(p, 2);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 1);
        EXPECT_EQ(c[3], 0);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 5.0);
        EXPECT_EQ(v[2], 3.0);
        EXPECT_EQ(v[3], 0.0);
        EXPECT_EQ(m->get_const_coo_values()[0], 2.0);
        EXPECT_EQ(m->get_const_coo_col_idxs()[0], 2);
        EXPECT_EQ(m->get_const_coo_row_idxs()[0], 0);
    }

    void assert_empty(const Mtx *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim(0, 0));
        ASSERT_EQ(m->get_ell_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_ell_values(), nullptr);
        ASSERT_EQ(m->get_const_ell_col_idxs(), nullptr);
        ASSERT_EQ(m->get_ell_max_nonzeros_per_row(), 0);
        ASSERT_EQ(m->get_ell_stride(), 0);
        ASSERT_EQ(m->get_coo_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_coo_values(), nullptr);
        ASSERT_EQ(m->get_const_coo_col_idxs(), nullptr);
    }
};


TEST_F(Hyb, KnowsItsSize)
{
    ASSERT_EQ(mtx->get_size(), gko::dim(2, 3));
    ASSERT_EQ(mtx->get_ell_num_stored_elements(), 4);
    ASSERT_EQ(mtx->get_ell_max_nonzeros_per_row(), 2);
    ASSERT_EQ(mtx->get_ell_stride(), 2);
    ASSERT_EQ(mtx->get_coo_num_stored_elements(), 1);
}


TEST_F(Hyb, ContainsCorrectData) { assert_equal_to_original_mtx(mtx.get()); }


TEST_F(Hyb, CanBeEmpty)
{
    auto mtx = Mtx::create(exec);

    assert_empty(mtx.get());
}


TEST_F(Hyb, CanBeCopied)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(mtx.get());

    assert_equal_to_original_mtx(mtx.get());
    mtx->get_ell_values()[1] = 5.0;
    assert_equal_to_original_mtx(copy.get());
}


TEST_F(Hyb, CanBeMoved)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(std::move(mtx));

    assert_equal_to_original_mtx(copy.get());
}


TEST_F(Hyb, CanBeCloned)
{
    auto clone = mtx->clone();

    assert_equal_to_original_mtx(mtx.get());
    mtx->get_ell_values()[1] = 5.0;
    assert_equal_to_original_mtx(static_cast<Mtx *>(clone.get()));
}


TEST_F(Hyb, CanBeCleared)
{
    mtx->clear();

    assert_empty(mtx.get());
}


// TEST_F(Hyb, CanBeReadFromMatrixData)
// {
//     auto m = Mtx::create(exec);
//     m->read({{2, 3},
//              {{0, 0, 1.0},
//               {0, 1, 3.0},
//               {0, 2, 2.0},
//               {1, 0, 0.0},
//               {1, 1, 5.0},
//               {1, 2, 0.0}}});

//     assert_equal_to_original_mtx(m.get());
// }


// TEST_F(Hyb, GeneratesCorrectMatrixData)
// {
//     using tpl = gko::matrix_data<>::nonzero_type;
//     gko::matrix_data<> data;

//     mtx->write(data);

//     ASSERT_EQ(data.size, gko::dim(2, 3));
//     ASSERT_EQ(data.nonzeros.size(), 4);
//     EXPECT_EQ(data.nonzeros[0], tpl(0, 0, 1.0));
//     EXPECT_EQ(data.nonzeros[1], tpl(0, 1, 3.0));
//     EXPECT_EQ(data.nonzeros[2], tpl(0, 2, 2.0));
//     EXPECT_EQ(data.nonzeros[3], tpl(1, 1, 5.0));
// }


}  // namespace
