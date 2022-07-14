/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/matrix/batch_ell.hpp>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class BatchEll : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::BatchEll<value_type, index_type>;
    using EllMtx = gko::matrix::Ell<value_type, index_type>;
    using size_type = gko::size_type;

    BatchEll()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::BatchEll<value_type, index_type>::create(
              exec, gko::batch_dim<2>{2, gko::dim<2>{3, 3}},
              gko::batch_stride{2, 2}))
    {
        value_type* v = mtx->get_values();
        index_type* c = mtx->get_col_idxs();
        /*
         * 1  3  0
         * 2  0  0
         * 0  7  9
         *
         * 4  6  0
         * 8  0  0
         * 0  18  14
         */
        c[0] = 0;
        c[1] = 0;
        c[2] = 1;
        c[3] = 1;
        c[4] = 0;
        c[5] = 2;
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 7.0;
        v[3] = 3.0;
        v[4] = 0.0;
        v[5] = 9.0;
        v[6] = 4.0;
        v[7] = 8.0;
        v[8] = 18.0;
        v[9] = 6.0;
        v[10] = 0.0;
        v[11] = 14.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx* m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        ASSERT_EQ(m->get_num_batch_entries(), 2);
        ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 3));
        ASSERT_EQ(m->get_stride().at(0), 3);
        ASSERT_EQ(m->get_num_stored_elements_per_row().at(0), 2);
        ASSERT_EQ(m->get_num_stored_elements(), 12);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 0);
        EXPECT_EQ(c[2], 1);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(c[4], 0);
        EXPECT_EQ(c[5], 2);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{2.0});
        EXPECT_EQ(v[2], value_type{7.0});
        EXPECT_EQ(v[3], value_type{3.0});
        EXPECT_EQ(v[4], value_type{0.0});
        EXPECT_EQ(v[5], value_type{9.0});
        EXPECT_EQ(v[6], value_type{4.0});
        EXPECT_EQ(v[7], value_type{8.0});
        EXPECT_EQ(v[8], value_type{18.0});
        EXPECT_EQ(v[9], value_type{6.0});
        EXPECT_EQ(v[10], value_type{0.0});
        EXPECT_EQ(v[11], value_type{14.0});
    }

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
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
TYPED_TEST_SUITE(BatchEll, valuetypes);


TYPED_TEST(BatchEll, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size().at(0), gko::dim<2>(3, 3));
    ASSERT_EQ(this->mtx->get_size().at(1), gko::dim<2>(3, 3));
    ASSERT_EQ(this->mtx->get_num_stored_elements(), 12);
    ASSERT_EQ(this->mtx->get_stride().at(0), 3);
    ASSERT_EQ(this->mtx->get_stride().at(1), 3);
}


TYPED_TEST(BatchEll, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(BatchEll, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(BatchEll, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(BatchEll, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(this->mtx));

    this->assert_equal_to_original_mtx(copy.get());
    ASSERT_EQ(this->mtx.get(), nullptr);
}


TYPED_TEST(BatchEll, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(dynamic_cast<Mtx*>(clone.get()));
}


TYPED_TEST(BatchEll, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(BatchEll, CanBeCreatedFromExistingData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type values[] = {1.0,  0.0, 2.0,  3.0,  4.0,  0.0,
                           -1.0, 0.0, 12.0, 13.0, 14.0, 0.0};
    index_type col_idxs[] = {0, 1, 1, 2, 2, 3};

    auto batch_mtx = gko::matrix::BatchEll<value_type, index_type>::create(
        this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 4}},
        gko::batch_stride(2, 2), gko::batch_stride{2, 3},
        gko::array<value_type>::view(this->exec, 12, values),
        gko::array<index_type>::view(this->exec, 6, col_idxs));

    ASSERT_EQ(batch_mtx->get_const_values(), values);
    ASSERT_EQ(batch_mtx->get_const_col_idxs(), col_idxs);
}


TYPED_TEST(BatchEll, CanBeCreatedFromExistingConstData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const value_type values[] = {1.0,  0.0, 2.0,  3.0,  4.0,  0.0,
                                 -1.0, 0.0, 12.0, 13.0, 14.0, 0.0};
    index_type col_idxs[] = {0, 1, 1, 2, 2, 3};

    auto batch_mtx =
        gko::matrix::BatchEll<value_type, index_type>::create_const(
            this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 4}},
            gko::batch_stride(2, 2), gko::batch_stride{2, 3},
            gko::array<value_type>::const_view(this->exec, 12, values),
            gko::array<index_type>::const_view(this->exec, 6, col_idxs));

    ASSERT_EQ(batch_mtx->get_const_values(), values);
    ASSERT_EQ(batch_mtx->get_const_col_idxs(), col_idxs);
}


TYPED_TEST(BatchEll, CanBeDuplicatedFromOneEllMatrix)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type values[] = {1.0, 0.0, 2.0, 3.0, 4.0, 0.0};
    index_type col_idxs[] = {0, 1, 1, 2, 2, 3};
    value_type batch_values[] = {1.0, 0.0, 2.0, 3.0, 4.0, 0.0, 1.0, 0.0, 2.0,
                                 3.0, 4.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 0.0};

    auto ell_mat = gko::matrix::Ell<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 4},
        gko::array<value_type>::view(this->exec, 6, values),
        gko::array<index_type>::view(this->exec, 6, col_idxs), 2, 3);

    auto mtx = gko::matrix::BatchEll<value_type, index_type>::create(
        this->exec, 3, ell_mat.get());

    ASSERT_EQ(mtx->get_size(), gko::batch_dim<2>(3, gko::dim<2>{3, 4}));
    this->assert_equal_data_array(18, batch_values, mtx->get_values());
    this->assert_equal_data_array(6, col_idxs, mtx->get_col_idxs());
}


TYPED_TEST(BatchEll, CanBeDuplicatedFromBatchMatrices)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type values[] = {1.0,  0.0, 2.0,  3.0,  4.0,  0.0,
                           -1.0, 0.0, 12.0, 13.0, 14.0, 0.0};
    index_type col_idxs[] = {0, 1, 1, 2, 2, 3};
    value_type bvalues[] = {
        1.0, 0.0, 2.0, 3.0, 4.0, 0.0, -1.0, 0.0, 12.0, 13.0, 14.0, 0.0,
        1.0, 0.0, 2.0, 3.0, 4.0, 0.0, -1.0, 0.0, 12.0, 13.0, 14.0, 0.0,
        1.0, 0.0, 2.0, 3.0, 4.0, 0.0, -1.0, 0.0, 12.0, 13.0, 14.0, 0.0};

    auto batch_mtx = gko::matrix::BatchEll<value_type, index_type>::create(
        this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 4}},
        gko::batch_stride{2, 2}, gko::batch_stride{2, 3},
        gko::array<value_type>::view(this->exec, 12, values),
        gko::array<index_type>::view(this->exec, 6, col_idxs));

    auto mtx = gko::matrix::BatchEll<value_type, index_type>::create(
        this->exec, 3, batch_mtx.get());

    ASSERT_EQ(mtx->get_size(), gko::batch_dim<2>(6, gko::dim<2>{3, 4}));
    this->assert_equal_data_array(36, bvalues, mtx->get_values());
    this->assert_equal_data_array(6, col_idxs, mtx->get_col_idxs());
}


TYPED_TEST(BatchEll, CanBeReadFromMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec);

    // clang-format off
    m->read({{{3, 3},
              {{0, 0, 1.0},
               {0, 1, 3.0},
               {1, 0, 2.0},
               {2, 1, 7.0},
               {2, 2, 9.0}}},
             {{3, 3},
              {{0, 0, 4.0},
               {0, 1, 6.0},
               {1, 0, 8.0},
               {2, 1, 18.0},
               {2, 2, 14.0}}}});
    // clang-format on

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(BatchEll, CanBeReadFromMatrixAssemblyData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec);
    gko::matrix_assembly_data<value_type, index_type> data1(gko::dim<2>{3, 3});
    gko::matrix_assembly_data<value_type, index_type> data2(gko::dim<2>{3, 3});
    data1.set_value(0, 0, 1.0);
    data1.set_value(0, 1, 3.0);
    data1.set_value(1, 0, 2.0);
    data1.set_value(2, 1, 7.0);
    data1.set_value(2, 2, 9.0);
    data2.set_value(0, 0, 4.0);
    data2.set_value(0, 1, 6.0);
    data2.set_value(1, 0, 8.0);
    data2.set_value(2, 1, 18.0);
    data2.set_value(2, 2, 14.0);
    auto data = std::vector<gko::matrix_assembly_data<value_type, index_type>>{
        data1, data2};

    m->read(data);

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(BatchEll, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    std::vector<gko::matrix_data<value_type, index_type>> data;

    this->mtx->write(data);

    ASSERT_EQ(data[0].size, gko::dim<2>(3, 3));
    ASSERT_EQ(data[0].nonzeros.size(), 5);
    EXPECT_EQ(data[0].nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data[0].nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data[0].nonzeros[2], tpl(1, 0, value_type{2.0}));
    EXPECT_EQ(data[0].nonzeros[3], tpl(2, 1, value_type{7.0}));
    EXPECT_EQ(data[0].nonzeros[4], tpl(2, 2, value_type{9.0}));
    ASSERT_EQ(data[1].size, gko::dim<2>(3, 3));
    ASSERT_EQ(data[1].nonzeros.size(), 5);
    EXPECT_EQ(data[1].nonzeros[0], tpl(0, 0, value_type{4.0}));
    EXPECT_EQ(data[1].nonzeros[1], tpl(0, 1, value_type{6.0}));
    EXPECT_EQ(data[1].nonzeros[2], tpl(1, 0, value_type{8.0}));
    EXPECT_EQ(data[1].nonzeros[3], tpl(2, 1, value_type{18.0}));
    EXPECT_EQ(data[1].nonzeros[4], tpl(2, 2, value_type{14.0}));
}


}  // namespace
