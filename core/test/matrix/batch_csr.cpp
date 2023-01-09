/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
#include "core/test/utils/batch.hpp"


namespace {


template <typename ValueIndexType>
class BatchCsr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::BatchCsr<value_type, index_type>;
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using size_type = gko::size_type;

    BatchCsr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::BatchCsr<value_type, index_type>::create(
              exec, 2, gko::dim<2>{2, 3}, 4))
    {
        value_type* v = mtx->get_values();
        index_type* c = mtx->get_col_idxs();
        index_type* r = mtx->get_row_ptrs();
        /*
         * 1  3  2
         * 0  5  0
         *
         * 3  5  1
         * 0  1  0
         */
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

    void assert_equal_to_original_mtx(const Mtx* m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_ptrs();
        ASSERT_EQ(m->get_num_batch_entries(), 2);
        ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 3));
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

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_NE(m->get_const_row_ptrs(), nullptr);
    }


    template <typename ValueType>
    void assert_equal_data_array(size_type num_elems, const ValueType* data1,
                                 const ValueType* data2)
    {
        for (size_type i = 0; i < num_elems; ++i) {
            EXPECT_EQ(data1[i], data2[i]);
        }
    }
};

using valuetypes =
    ::testing::Types<std::tuple<float, int>, std::tuple<double, gko::int32>,
                     std::tuple<std::complex<float>, gko::int32>,
                     std::tuple<std::complex<double>, gko::int32>>;
TYPED_TEST_SUITE(BatchCsr, valuetypes);


TYPED_TEST(BatchCsr, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size().at(0), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_size().at(1), gko::dim<2>(2, 3));
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


TYPED_TEST(BatchCsr, CanBeDuplicatedFromOneCsrMatrix)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type values[] = {1.0, 2.0, 3.0, 4.0};
    index_type col_idxs[] = {0, 1, 1, 0};
    index_type row_ptrs[] = {0, 2, 3, 4};
    value_type batch_values[] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0,
                                 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};

    auto csr_mat = gko::matrix::Csr<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::array<value_type>::view(this->exec, 4, values),
        gko::array<index_type>::view(this->exec, 4, col_idxs),
        gko::array<index_type>::view(this->exec, 4, row_ptrs));

    auto mtx = gko::matrix::BatchCsr<value_type, index_type>::create(
        this->exec, 3, csr_mat.get());

    ASSERT_EQ(mtx->get_size(), gko::batch_dim<2>(3, gko::dim<2>{3, 2}));
    this->assert_equal_data_array(12, batch_values, mtx->get_values());
    this->assert_equal_data_array(4, col_idxs, mtx->get_col_idxs());
    this->assert_equal_data_array(4, row_ptrs, mtx->get_row_ptrs());
}


TYPED_TEST(BatchCsr, CanBeDuplicatedFromBatchMatrices)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type values[] = {1.0, 2.0, 3.0, 4.0, -1.0, 12.0, 13.0, 14.0};
    index_type col_idxs[] = {0, 1, 1, 0};
    index_type row_ptrs[] = {0, 2, 3, 4};
    value_type bvalues[] = {1.0, 2.0, 3.0, 4.0, -1.0, 12.0, 13.0, 14.0,
                            1.0, 2.0, 3.0, 4.0, -1.0, 12.0, 13.0, 14.0,
                            1.0, 2.0, 3.0, 4.0, -1.0, 12.0, 13.0, 14.0};

    auto batch_mtx = gko::matrix::BatchCsr<value_type, index_type>::create(
        this->exec, 2, gko::dim<2>{3, 2},
        gko::array<value_type>::view(this->exec, 8, values),
        gko::array<index_type>::view(this->exec, 4, col_idxs),
        gko::array<index_type>::view(this->exec, 4, row_ptrs));

    auto mtx = gko::matrix::BatchCsr<value_type, index_type>::create(
        this->exec, 3, batch_mtx.get());

    ASSERT_EQ(mtx->get_size(), gko::batch_dim<2>(6, gko::dim<2>{3, 2}));
    this->assert_equal_data_array(24, bvalues, mtx->get_values());
    this->assert_equal_data_array(4, col_idxs, mtx->get_col_idxs());
    this->assert_equal_data_array(4, row_ptrs, mtx->get_row_ptrs());
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
        this->exec, gko::batch_dim<>(2, gko::dim<2>{3, 2}),
        gko::array<value_type>::view(this->exec, 8, values),
        gko::array<index_type>::view(this->exec, 4, col_idxs),
        gko::array<index_type>::view(this->exec, 4, row_ptrs));

    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_ptrs(), row_ptrs);
}


TYPED_TEST(BatchCsr, CanBeCreatedFromExistingConstData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const value_type values[] = {1.0, 2.0, 3.0, 4.0, -1.0, 12.0, 13.0, 14.0};
    const index_type col_idxs[] = {0, 1, 1, 0};
    index_type row_ptrs[] = {0, 2, 3, 4};

    auto mtx = gko::matrix::BatchCsr<value_type, index_type>::create_const(
        this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 2}},
        gko::array<value_type>::const_view(this->exec, 8, values),
        gko::array<index_type>::const_view(this->exec, 4, col_idxs),
        gko::array<index_type>::const_view(this->exec, 4, row_ptrs));

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
    ASSERT_EQ(this->mtx.get(), nullptr);
}


TYPED_TEST(BatchCsr, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(dynamic_cast<Mtx*>(clone.get()));
}


TYPED_TEST(BatchCsr, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(BatchCsr, CanBeReadFromMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec);

    m->read({{{2, 3},
              {{0, 0, 1.0},
               {0, 1, 3.0},
               {0, 2, 2.0},
               {1, 0, 0.0},
               {1, 1, 5.0},
               {1, 2, 0.0}}},
             {{2, 3},
              {{0, 0, 3.0},
               {0, 1, 5.0},
               {0, 2, 1.0},
               {1, 0, 0.0},
               {1, 1, 1.0},
               {1, 2, 0.0}}}});

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(BatchCsr, CanBeReadFromMatrixAssemblyData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec);
    gko::matrix_assembly_data<value_type, index_type> data1(gko::dim<2>{2, 3});
    gko::matrix_assembly_data<value_type, index_type> data2(gko::dim<2>{2, 3});
    data1.set_value(0, 0, 1.0);
    data1.set_value(0, 1, 3.0);
    data1.set_value(0, 2, 2.0);
    data1.set_value(1, 0, 0.0);
    data1.set_value(1, 1, 5.0);
    data1.set_value(1, 2, 0.0);
    data2.set_value(0, 0, 3.0);
    data2.set_value(0, 1, 5.0);
    data2.set_value(0, 2, 1.0);
    data2.set_value(1, 0, 0.0);
    data2.set_value(1, 1, 1.0);
    data2.set_value(1, 2, 0.0);
    auto data = std::vector<gko::matrix_assembly_data<value_type, index_type>>{
        data1, data2};

    m->read(data);

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(BatchCsr, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    std::vector<gko::matrix_data<value_type, index_type>> data;

    this->mtx->write(data);

    ASSERT_EQ(data[0].size, gko::dim<2>(2, 3));
    ASSERT_EQ(data[0].nonzeros.size(), 4);
    EXPECT_EQ(data[0].nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data[0].nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data[0].nonzeros[2], tpl(0, 2, value_type{2.0}));
    EXPECT_EQ(data[0].nonzeros[3], tpl(1, 1, value_type{5.0}));
    ASSERT_EQ(data[1].size, gko::dim<2>(2, 3));
    ASSERT_EQ(data[1].nonzeros.size(), 4);
    EXPECT_EQ(data[1].nonzeros[0], tpl(0, 0, value_type{3.0}));
    EXPECT_EQ(data[1].nonzeros[1], tpl(0, 1, value_type{5.0}));
    EXPECT_EQ(data[1].nonzeros[2], tpl(0, 2, value_type{1.0}));
    EXPECT_EQ(data[1].nonzeros[3], tpl(1, 1, value_type{1.0}));
}


TYPED_TEST(BatchCsr, NonZeroMatrixReadWriteIsReversible)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    using MtxType = typename TestFixture::Mtx;
    using real_type = typename gko::remove_complex<value_type>;
    const size_t batch_size = 7;
    const int num_rows = 63;
    const int num_cols = 31;
    const int min_nnz_row = 8;
    auto mtx = gko::test::generate_uniform_batch_random_matrix<MtxType>(
        batch_size, num_rows, num_cols,
        std::uniform_int_distribution<>(min_nnz_row, num_cols),
        std::uniform_real_distribution<real_type>(1.0, 2.0), std::ranlux48(42),
        false, this->exec);
    std::vector<gko::matrix_data<value_type, index_type>> data;
    auto test_mtx = MtxType::create(this->exec);

    mtx->write(data);
    test_mtx->read(data);

    GKO_ASSERT_BATCH_MTX_NEAR(mtx, test_mtx, 0.0);
}


}  // namespace
