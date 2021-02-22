/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/matrix/batch_csr.hpp>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class BatchCsr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::BatchCsr<value_type, index_type>;

    BatchCsr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::BatchCsr<value_type, index_type>::create(
              exec, 2, gko::dim<2>{2, 3}, 4))
    {
        value_type *v = mtx->get_values();
        index_type *c = mtx->get_col_idxs();
        index_type *r = mtx->get_row_ptrs();
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 5.0;
        v[4] = 3.0;
        v[5] = 5.0;
        v[6] = 1.0;
        v[7] = 1.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_ptrs();
        ASSERT_EQ(m->get_num_batches(), 2);
        ASSERT_EQ(m->get_sizes()[0], gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 8);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 3);
        EXPECT_EQ(r[2], 4);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{3.0});
        EXPECT_EQ(v[2], value_type{2.0});
        EXPECT_EQ(v[3], value_type{5.0});
        EXPECT_EQ(v[4], value_type{3.0});
        EXPECT_EQ(v[5], value_type{5.0});
        EXPECT_EQ(v[6], value_type{1.0});
        EXPECT_EQ(v[7], value_type{1.0});
    }

    void assert_empty(const Mtx *m)
    {
        ASSERT_EQ(m->get_num_batches(), 0);
        ASSERT_FALSE(m->get_sizes().size());
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_NE(m->get_const_row_ptrs(), nullptr);
    }
};

TYPED_TEST_SUITE(BatchCsr, gko::test::ValueIndexTypes);


TYPED_TEST(BatchCsr, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_sizes()[0], gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_sizes()[1], gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_num_stored_elements(), 8);
}


TYPED_TEST(BatchCsr, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(BatchCsr, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(BatchCsr, CanBeCreatedFromExistingData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type values[] = {1.0, 2.0, 3.0, 4.0, -1.0, 12.0, 13.0, 14.0};
    index_type col_idxs[] = {0, 1, 1, 0};
    index_type row_ptrs[] = {0, 2, 3, 4};

    auto mtx = gko::matrix::BatchCsr<value_type, index_type>::create(
        this->exec, 2, gko::dim<2>{3, 2},
        gko::Array<value_type>::view(this->exec, 8, values),
        gko::Array<index_type>::view(this->exec, 4, col_idxs),
        gko::Array<index_type>::view(this->exec, 4, row_ptrs));

    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_ptrs(), row_ptrs);
}


TYPED_TEST(BatchCsr, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(BatchCsr, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(this->mtx));

    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(BatchCsr, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(dynamic_cast<Mtx *>(clone.get()));
}


TYPED_TEST(BatchCsr, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


// TYPED_TEST(BatchCsr, CanBeReadFromMatrixData)
// {
//     using Mtx = typename TestFixture::Mtx;
//     auto m = Mtx::create(this->exec,
//                          std::make_shared<typename Mtx::load_balance>(2));

//     m->read({{2, 3},
//              {{0, 0, 1.0},
//               {0, 1, 3.0},
//               {0, 2, 2.0},
//               {1, 0, 0.0},
//               {1, 1, 5.0},
//               {1, 2, 0.0}}});

//     this->assert_equal_to_original_mtx(m.get());
// }


// TYPED_TEST(BatchCsr, CanBeReadFromMatrixAssemblyData)
// {
//     using Mtx = typename TestFixture::Mtx;
//     using value_type = typename TestFixture::value_type;
//     using index_type = typename TestFixture::index_type;
//     auto m = Mtx::create(this->exec,
//                          std::make_shared<typename Mtx::load_balance>(2));
//     gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2,
//     3}); data.set_value(0, 0, 1.0); data.set_value(0, 1, 3.0);
//     data.set_value(0, 2, 2.0);
//     data.set_value(1, 0, 0.0);
//     data.set_value(1, 1, 5.0);
//     data.set_value(1, 2, 0.0);

//     m->read(data);

//     this->assert_equal_to_original_mtx(m.get());
// }


// TYPED_TEST(BatchCsr, GeneratesCorrectMatrixData)
// {
//     using value_type = typename TestFixture::value_type;
//     using index_type = typename TestFixture::index_type;
//     using tpl = typename gko::matrix_data<value_type,
//     index_type>::nonzero_type; gko::matrix_data<value_type, index_type> data;

//     this->mtx->write(data);

//     ASSERT_EQ(data.size, gko::dim<2>(2, 3));
//     ASSERT_EQ(data.nonzeros.size(), 4);
//     EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
//     EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{3.0}));
//     EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{2.0}));
//     EXPECT_EQ(data.nonzeros[3], tpl(1, 1, value_type{5.0}));
// }


}  // namespace
