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

#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class SparsityCsr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::SparsityCsr<value_type, index_type>;

    SparsityCsr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::SparsityCsr<value_type, index_type>::create(
              exec, gko::dim<2>{2, 3}, 4))
    {
        index_type *c = mtx->get_col_idxs();
        index_type *r = mtx->get_row_ptrs();
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx *m)
    {
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_ptrs();
        auto v = m->get_const_value();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_nonzeros(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 3);
        EXPECT_EQ(r[2], 4);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], value_type{1.0});
    }

    void assert_empty(Mtx *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_nonzeros(), 0);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_NE(m->get_const_row_ptrs(), nullptr);
        ASSERT_NE(m->get_const_value(), nullptr);
        ASSERT_EQ(m->get_col_idxs(), nullptr);
        ASSERT_NE(m->get_row_ptrs(), nullptr);
        ASSERT_NE(m->get_value(), nullptr);
    }
};

TYPED_TEST_CASE(SparsityCsr, gko::test::ValueIndexTypes);


TYPED_TEST(SparsityCsr, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_num_nonzeros(), 4);
}


TYPED_TEST(SparsityCsr, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(SparsityCsr, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(SparsityCsr, SetsCorrectDefaultValue)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = gko::matrix::SparsityCsr<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 2}, static_cast<gko::size_type>(0));

    ASSERT_EQ(mtx->get_const_value()[0], value_type{1.0});
    ASSERT_EQ(mtx->get_value()[0], value_type{1.0});
}


TYPED_TEST(SparsityCsr, CanBeCreatedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    index_type col_idxs[] = {0, 1, 1, 0};
    index_type row_ptrs[] = {0, 2, 3, 4};

    auto mtx = gko::matrix::SparsityCsr<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::Array<index_type>::view(this->exec, 4, col_idxs),
        gko::Array<index_type>::view(this->exec, 4, row_ptrs), 2.0);

    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_ptrs(), row_ptrs);
    ASSERT_EQ(mtx->get_const_value()[0], value_type{2.0});
    ASSERT_EQ(mtx->get_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_row_ptrs(), row_ptrs);
    ASSERT_EQ(mtx->get_value()[0], value_type{2.0});
}


TYPED_TEST(SparsityCsr, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(SparsityCsr, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(this->mtx));

    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(SparsityCsr, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->assert_equal_to_original_mtx(dynamic_cast<Mtx *>(clone.get()));
}


TYPED_TEST(SparsityCsr, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(SparsityCsr, CanBeReadFromMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec);

    m->read({{2, 3},
             {{0, 0, 1.0},
              {0, 1, 3.0},
              {0, 2, 2.0},
              {1, 0, 0.0},
              {1, 1, 5.0},
              {1, 2, 0.0}}});

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(SparsityCsr, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;

    this->mtx->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, value_type{1.0}));
}


}  // namespace
