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

#include <ginkgo/core/matrix/hybrid.hpp>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Hybrid : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Hybrid<value_type, index_type>;

    Hybrid()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Hybrid<value_type, index_type>::create(
              exec, gko::dim<2>{2, 3}, 2, 2, 1))
    {
        value_type *v = mtx->get_ell_values();
        index_type *c = mtx->get_ell_col_idxs();
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
        auto n = m->get_ell_num_stored_elements_per_row();
        auto p = m->get_ell_stride();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_ell_num_stored_elements(), 4);
        ASSERT_EQ(m->get_coo_num_stored_elements(), 1);
        EXPECT_EQ(n, 2);
        EXPECT_EQ(p, 2);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 1);
        EXPECT_EQ(c[3], 0);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{5.0});
        EXPECT_EQ(v[2], value_type{3.0});
        EXPECT_EQ(v[3], value_type{0.0});
        EXPECT_EQ(m->get_const_coo_values()[0], value_type{2.0});
        EXPECT_EQ(m->get_const_coo_col_idxs()[0], 2);
        EXPECT_EQ(m->get_const_coo_row_idxs()[0], 0);
    }

    void assert_empty(const Mtx *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_ell_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_ell_values(), nullptr);
        ASSERT_EQ(m->get_const_ell_col_idxs(), nullptr);
        ASSERT_EQ(m->get_ell_num_stored_elements_per_row(), 0);
        ASSERT_EQ(m->get_ell_stride(), 0);
        ASSERT_EQ(m->get_coo_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_coo_values(), nullptr);
        ASSERT_EQ(m->get_const_coo_col_idxs(), nullptr);
    }
};

TYPED_TEST_CASE(Hybrid, gko::test::ValueIndexTypes);


TYPED_TEST(Hybrid, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_ell_num_stored_elements(), 4);
    ASSERT_EQ(this->mtx->get_ell_num_stored_elements_per_row(), 2);
    ASSERT_EQ(this->mtx->get_ell_stride(), 2);
    ASSERT_EQ(this->mtx->get_coo_num_stored_elements(), 1);
}


TYPED_TEST(Hybrid, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Hybrid, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(Hybrid, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_ell_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(Hybrid, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(this->mtx));

    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(Hybrid, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_ell_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(static_cast<Mtx *>(clone.get()));
}


TYPED_TEST(Hybrid, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Hybrid, CanBeReadFromMatrixDataAutomatically)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto m =
        Mtx::create(this->exec, std::make_shared<typename Mtx::automatic>());
    m->read({{2, 3},
             {{0, 0, 1.0},
              {0, 1, 3.0},
              {0, 2, 2.0},
              {1, 0, 0.0},
              {1, 1, 5.0},
              {1, 2, 0.0}}});

    auto v = m->get_const_coo_values();
    auto c = m->get_const_coo_col_idxs();
    auto r = m->get_const_coo_row_idxs();
    auto n = m->get_ell_num_stored_elements_per_row();
    auto p = m->get_ell_stride();
    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(m->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], value_type{1.0});
    EXPECT_EQ(v[1], value_type{3.0});
    EXPECT_EQ(v[2], value_type{2.0});
    EXPECT_EQ(v[3], value_type{5.0});
}


TYPED_TEST(Hybrid, CanBeReadFromMatrixDataByColumns2)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec,
                         std::make_shared<typename Mtx::column_limit>(2));
    m->read({{2, 3},
             {{0, 0, 1.0},
              {0, 1, 3.0},
              {0, 2, 2.0},
              {1, 0, 0.0},
              {1, 1, 5.0},
              {1, 2, 0.0}}});

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(Hybrid, CanBeReadFromMatrixDataByPercent40)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto m = Mtx::create(this->exec,
                         std::make_shared<typename Mtx::imbalance_limit>(0.4));
    m->read({{2, 3},
             {{0, 0, 1.0},
              {0, 1, 3.0},
              {0, 2, 2.0},
              {1, 0, 0.0},
              {1, 1, 5.0},
              {1, 2, 0.0}}});

    auto v = m->get_const_ell_values();
    auto c = m->get_const_ell_col_idxs();
    auto n = m->get_ell_num_stored_elements_per_row();
    auto p = m->get_ell_stride();
    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_ell_num_stored_elements(), 2);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(v[0], value_type{1.0});
    EXPECT_EQ(v[1], value_type{5.0});

    auto coo_v = m->get_const_coo_values();
    auto coo_c = m->get_const_coo_col_idxs();
    auto coo_r = m->get_const_coo_row_idxs();
    ASSERT_EQ(m->get_coo_num_stored_elements(), 2);
    EXPECT_EQ(coo_v[0], value_type{3.0});
    EXPECT_EQ(coo_v[1], value_type{2.0});
    EXPECT_EQ(coo_c[0], 1);
    EXPECT_EQ(coo_c[1], 2);
    EXPECT_EQ(coo_r[0], 0);
    EXPECT_EQ(coo_r[1], 0);
}


TYPED_TEST(Hybrid, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;

    this->mtx->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, value_type{5.0}));
}


}  // namespace
