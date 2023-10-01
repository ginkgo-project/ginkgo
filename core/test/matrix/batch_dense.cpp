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

#include <ginkgo/core/matrix/batch_dense.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class BatchDense : public ::testing::Test {
protected:
    using value_type = T;
    using DenseMtx = gko::matrix::Dense<value_type>;
    using size_type = gko::size_type;
    BatchDense()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::batch_initialize<gko::matrix::BatchDense<value_type>>(
              std::vector<size_type>{4, 3},
              {{{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}},
               {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}},
              exec))
    {}


    static void assert_equal_to_original_mtx(
        gko::matrix::BatchDense<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 2);
        ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_stride().at(0), 4);
        ASSERT_EQ(m->get_stride().at(1), 3);
        ASSERT_EQ(m->get_num_stored_elements(), (2 * 4) + (2 * 3));
        ASSERT_EQ(m->get_num_stored_elements(0), 2 * 4);
        ASSERT_EQ(m->get_num_stored_elements(1), 2 * 3);
        EXPECT_EQ(m->at(0, 0, 0), value_type{-1.0});
        EXPECT_EQ(m->at(0, 0, 1), value_type{2.0});
        EXPECT_EQ(m->at(0, 0, 2), value_type{3.0});
        EXPECT_EQ(m->at(0, 1, 0), value_type{-1.5});
        EXPECT_EQ(m->at(0, 1, 1), value_type{2.5});
        ASSERT_EQ(m->at(0, 1, 2), value_type{3.5});
        EXPECT_EQ(m->at(1, 0, 0), value_type{1.0});
        EXPECT_EQ(m->at(1, 0, 1), value_type{2.5});
        EXPECT_EQ(m->at(1, 0, 2), value_type{3.0});
        EXPECT_EQ(m->at(1, 1, 0), value_type{1.0});
        EXPECT_EQ(m->at(1, 1, 1), value_type{2.0});
        ASSERT_EQ(m->at(1, 1, 2), value_type{3.0});
    }

    static void assert_empty(gko::matrix::BatchDense<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::BatchDense<value_type>> mtx;
};

TYPED_TEST_SUITE(BatchDense, gko::test::ValueTypes);


TYPED_TEST(BatchDense, CanBeEmpty)
{
    auto empty = gko::matrix::BatchDense<TypeParam>::create(this->exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(BatchDense, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::matrix::BatchDense<TypeParam>::create(this->exec);
    ASSERT_EQ(empty->get_const_values(), nullptr);
}


TYPED_TEST(BatchDense, CanBeConstructedWithSize)
{
    using size_type = gko::size_type;
    auto m = gko::matrix::BatchDense<TypeParam>::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{2, 4}, gko::dim<2>{2, 3}});

    ASSERT_EQ(m->get_num_batch_entries(), 2);
    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 4));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 3));
    EXPECT_EQ(m->get_stride().at(0), 4);
    EXPECT_EQ(m->get_stride().at(1), 3);
    ASSERT_EQ(m->get_num_stored_elements(), 14);
    ASSERT_EQ(m->get_num_stored_elements(0), 8);
    ASSERT_EQ(m->get_num_stored_elements(1), 6);
}


TYPED_TEST(BatchDense, CanBeConstructedWithSizeAndStride)
{
    using size_type = gko::size_type;
    auto m = gko::matrix::BatchDense<TypeParam>::create(
        this->exec, std::vector<gko::dim<2>>{gko::dim<2>{2, 3}},
        std::vector<size_type>{4});

    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 3));
    EXPECT_EQ(m->get_stride().at(0), 4);
    ASSERT_EQ(m->get_num_stored_elements(), 8);
}


TYPED_TEST(BatchDense, CanBeConstructedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    // clang-format off
    value_type data[] = {
       1.0, 2.0, -1.0,
       3.0, 4.0, -1.0,
       3.0, 5.0, 1.0,
       5.0, 6.0, -3.0};
    // clang-format on

    auto m = gko::matrix::BatchDense<TypeParam>::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{2, 2}, gko::dim<2>{2, 2}},
        gko::array<value_type>::view(this->exec, 12, data),
        std::vector<size_type>{3, 3});

    ASSERT_EQ(m->get_const_values(), data);
    ASSERT_EQ(m->at(0, 0, 1), value_type{2.0});
    ASSERT_EQ(m->at(0, 1, 2), value_type{-1.0});
    ASSERT_EQ(m->at(1, 0, 1), value_type{5.0});
    ASSERT_EQ(m->at(1, 1, 2), value_type{-3.0});
}


TYPED_TEST(BatchDense, CanBeConstructedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    // clang-format off
    const value_type data[] = {
       1.0, 2.0, -1.0,
       3.0, 4.0, -1.0,
       3.0, 5.0, 1.0,
       5.0, 6.0, -3.0};
    // clang-format on

    auto m = gko::matrix::BatchDense<TypeParam>::create_const(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{2, 2}, gko::dim<2>{2, 2}},
        gko::array<value_type>::const_view(this->exec, 12, data),
        std::vector<size_type>{3, 3});

    ASSERT_EQ(m->get_const_values(), data);
    ASSERT_EQ(m->at(0, 0, 1), value_type{2.0});
    ASSERT_EQ(m->at(0, 1, 2), value_type{-1.0});
    ASSERT_EQ(m->at(1, 0, 1), value_type{5.0});
    ASSERT_EQ(m->at(1, 1, 2), value_type{-3.0});
}


TYPED_TEST(BatchDense, CanBeConstructedFromBatchDenseMatrices)
{
    using value_type = typename TestFixture::value_type;
    using DenseMtx = typename TestFixture::DenseMtx;
    using size_type = gko::size_type;
    auto mat1 = gko::initialize<DenseMtx>(
        3, {{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}}, this->exec);
    auto mat2 = gko::initialize<DenseMtx>({{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}},
                                          this->exec);

    auto m = gko::matrix::BatchDense<TypeParam>::create(
        this->exec, std::vector<DenseMtx*>{mat1.get(), mat2.get()});
    auto m_ref = gko::matrix::BatchDense<TypeParam>::create(
        this->exec, std::vector<DenseMtx*>{mat1.get(), mat2.get(), mat1.get(),
                                           mat2.get(), mat1.get(), mat2.get()});
    auto m2 =
        gko::matrix::BatchDense<TypeParam>::create(this->exec, 3, m.get());

    GKO_ASSERT_BATCH_MTX_NEAR(m2.get(), m_ref.get(), 1e-14);
}


TYPED_TEST(BatchDense, CanBeConstructedFromDenseMatricesByDuplication)
{
    using value_type = typename TestFixture::value_type;
    using DenseMtx = typename TestFixture::DenseMtx;
    using size_type = gko::size_type;
    auto mat1 = gko::initialize<DenseMtx>(
        4, {{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}}, this->exec);
    auto mat2 = gko::initialize<DenseMtx>({{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}},
                                          this->exec);

    auto bat_m = gko::matrix::BatchDense<TypeParam>::create(
        this->exec, std::vector<DenseMtx*>{mat1.get(), mat1.get(), mat1.get()});
    auto m =
        gko::matrix::BatchDense<TypeParam>::create(this->exec, 3, mat1.get());

    GKO_ASSERT_BATCH_MTX_NEAR(bat_m.get(), m.get(), 1e-14);
}


TYPED_TEST(BatchDense, CanBeConstructedFromDenseMatrices)
{
    using value_type = typename TestFixture::value_type;
    using DenseMtx = typename TestFixture::DenseMtx;
    using size_type = gko::size_type;
    auto mat1 = gko::initialize<DenseMtx>(
        4, {{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}}, this->exec);
    auto mat2 = gko::initialize<DenseMtx>({{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}},
                                          this->exec);

    auto m = gko::matrix::BatchDense<TypeParam>::create(
        this->exec, std::vector<DenseMtx*>{mat1.get(), mat2.get()});

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(BatchDense, CanBeUnbatchedIntoDenseMatrices)
{
    using value_type = typename TestFixture::value_type;
    using DenseMtx = typename TestFixture::DenseMtx;
    using size_type = gko::size_type;
    auto mat1 = gko::initialize<DenseMtx>(
        4, {{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}}, this->exec);
    auto mat2 = gko::initialize<DenseMtx>({{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}},
                                          this->exec);

    auto dense_mats = this->mtx->unbatch();


    GKO_ASSERT_MTX_NEAR(dense_mats[0].get(), mat1.get(), 0.);
    GKO_ASSERT_MTX_NEAR(dense_mats[1].get(), mat2.get(), 0.);
}


TYPED_TEST(BatchDense, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(BatchDense, CanBeListConstructed)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::batch_initialize<gko::matrix::BatchDense<TypeParam>>(
        {{1.0, 2.0}, {1.0, 3.0}}, this->exec);

    ASSERT_EQ(m->get_num_batch_entries(), 2);
    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0, 0), value_type{1});
    EXPECT_EQ(m->at(0, 1), value_type{2});
    EXPECT_EQ(m->at(1, 0), value_type{1});
    EXPECT_EQ(m->at(1, 1), value_type{3});
}


TYPED_TEST(BatchDense, CanBeListConstructedWithstride)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::batch_initialize<gko::matrix::BatchDense<TypeParam>>(
        std::vector<gko::size_type>{2}, {{1.0, 2.0}}, this->exec);
    ASSERT_EQ(m->get_num_batch_entries(), 1);
    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1), value_type{2.0});
}


TYPED_TEST(BatchDense, CanBeListConstructedByCopies)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::batch_initialize<gko::matrix::BatchDense<TypeParam>>(
        2, I<value_type>({1.0, 2.0}), this->exec);
    ASSERT_EQ(m->get_num_batch_entries(), 2);
    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0, 0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 0, 1), value_type{2.0});
    EXPECT_EQ(m->at(1, 0, 0), value_type{1.0});
    EXPECT_EQ(m->at(1, 0, 1), value_type{2.0});
}


TYPED_TEST(BatchDense, CanBeDoubleListConstructed)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::batch_initialize<gko::matrix::BatchDense<TypeParam>>(
        {{I<T>{1.0, 1.0, 0.0}, I<T>{2.0, 4.0, 3.0}, I<T>{3.0, 6.0, 1.0}},
         {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}},
        this->exec);

    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 3));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_stride().at(0), 3);
    ASSERT_EQ(m->get_stride().at(1), 2);
    EXPECT_EQ(m->get_num_stored_elements(), 15);
    ASSERT_EQ(m->get_num_stored_elements(0), 9);
    ASSERT_EQ(m->get_num_stored_elements(1), 6);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1), value_type{1.0});
    EXPECT_EQ(m->at(0, 2), value_type{0.0});
    ASSERT_EQ(m->at(0, 3), value_type{2.0});
    EXPECT_EQ(m->at(0, 4), value_type{4.0});
    EXPECT_EQ(m->at(1, 0), value_type{1.0});
    EXPECT_EQ(m->at(1, 1), value_type{2.0});
    EXPECT_EQ(m->at(1, 2), value_type{3.0});
    ASSERT_EQ(m->at(1, 3), value_type{4.0});
    EXPECT_EQ(m->at(1, 4), value_type{5.0});
}


TYPED_TEST(BatchDense, CanBeDoubleListConstructedWithstride)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::batch_initialize<gko::matrix::BatchDense<TypeParam>>(
        {4, 3},
        {{I<T>{1.0, 1.0, 0.0}, I<T>{2.0, 4.0, 3.0}, I<T>{3.0, 6.0, 1.0}},
         {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}},
        this->exec);

    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 3));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_stride().at(0), 4);
    ASSERT_EQ(m->get_stride().at(1), 3);
    EXPECT_EQ(m->get_num_stored_elements(), 21);
    ASSERT_EQ(m->get_num_stored_elements(0), 12);
    ASSERT_EQ(m->get_num_stored_elements(1), 9);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1), value_type{1.0});
    EXPECT_EQ(m->at(0, 2), value_type{0.0});
    ASSERT_EQ(m->at(0, 3), value_type{2.0});
    EXPECT_EQ(m->at(0, 4), value_type{4.0});
    EXPECT_EQ(m->at(1, 0), value_type{1.0});
    EXPECT_EQ(m->at(1, 1), value_type{2.0});
    EXPECT_EQ(m->at(1, 2), value_type{3.0});
    ASSERT_EQ(m->at(1, 3), value_type{4.0});
    EXPECT_EQ(m->at(1, 4), value_type{5.0});
}


TYPED_TEST(BatchDense, CanBeCopied)
{
    auto mtx_copy = gko::matrix::BatchDense<TypeParam>::create(this->exec);
    mtx_copy->copy_from(this->mtx.get());
    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->at(0, 0, 0) = 7;
    this->mtx->at(0, 1) = 7;
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(BatchDense, CanBeMoved)
{
    auto mtx_copy = gko::matrix::BatchDense<TypeParam>::create(this->exec);
    mtx_copy->copy_from(std::move(this->mtx));
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(BatchDense, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();
    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(BatchDense, CanBeCleared)
{
    this->mtx->clear();
    this->assert_empty(this->mtx.get());
}


TYPED_TEST(BatchDense, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::BatchDense<TypeParam>::create(this->exec);
    // clang-format off
    m->read({gko::matrix_data<TypeParam>{{2, 3},
                                         {{0, 0, 1.0},
                                          {0, 1, 3.0},
                                          {0, 2, 2.0},
                                          {1, 0, 0.0},
                                          {1, 1, 5.0},
                                          {1, 2, 0.0}}},
             gko::matrix_data<TypeParam>{{2, 2},
                                         {{0, 0, -1.0},
                                          {0, 1, 0.5},
                                          {1, 0, 0.0},
                                          {1, 1, 9.0}}}});
    // clang-format on

    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 10);
    ASSERT_EQ(m->get_num_stored_elements(0), 6);
    ASSERT_EQ(m->get_num_stored_elements(1), 4);
    EXPECT_EQ(m->at(0, 0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1, 0), value_type{0.0});
    EXPECT_EQ(m->at(0, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at(0, 1, 1), value_type{5.0});
    EXPECT_EQ(m->at(0, 0, 2), value_type{2.0});
    EXPECT_EQ(m->at(0, 1, 2), value_type{0.0});
    EXPECT_EQ(m->at(1, 0, 0), value_type{-1.0});
    EXPECT_EQ(m->at(1, 0, 1), value_type{0.5});
    EXPECT_EQ(m->at(1, 1, 0), value_type{0.0});
    EXPECT_EQ(m->at(1, 1, 1), value_type{9.0});
}


TYPED_TEST(BatchDense, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using tpl = typename gko::matrix_data<TypeParam>::nonzero_type;
    std::vector<gko::matrix_data<TypeParam>> data;

    this->mtx->write(data);

    ASSERT_EQ(data[0].size, gko::dim<2>(2, 3));
    ASSERT_EQ(data[0].nonzeros.size(), 6);
    EXPECT_EQ(data[0].nonzeros[0], tpl(0, 0, value_type{-1.0}));
    EXPECT_EQ(data[0].nonzeros[1], tpl(0, 1, value_type{2.0}));
    EXPECT_EQ(data[0].nonzeros[2], tpl(0, 2, value_type{3.0}));
    EXPECT_EQ(data[0].nonzeros[3], tpl(1, 0, value_type{-1.5}));
    EXPECT_EQ(data[0].nonzeros[4], tpl(1, 1, value_type{2.5}));
    EXPECT_EQ(data[0].nonzeros[5], tpl(1, 2, value_type{3.5}));
    ASSERT_EQ(data[1].size, gko::dim<2>(2, 3));
    ASSERT_EQ(data[1].nonzeros.size(), 6);
    EXPECT_EQ(data[1].nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data[1].nonzeros[1], tpl(0, 1, value_type{2.5}));
    EXPECT_EQ(data[1].nonzeros[2], tpl(0, 2, value_type{3.0}));
    EXPECT_EQ(data[1].nonzeros[3], tpl(1, 0, value_type{1.0}));
    EXPECT_EQ(data[1].nonzeros[4], tpl(1, 1, value_type{2.0}));
    EXPECT_EQ(data[1].nonzeros[5], tpl(1, 2, value_type{3.0}));
}


TYPED_TEST(BatchDense, CanBeReadFromMatrixAssemblyData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::BatchDense<TypeParam>::create(this->exec);
    gko::matrix_assembly_data<TypeParam> data1(gko::dim<2>{2, 3});
    data1.set_value(0, 0, 1.0);
    data1.set_value(0, 1, 3.0);
    data1.set_value(0, 2, 2.0);
    data1.set_value(1, 0, 0.0);
    data1.set_value(1, 1, 5.0);
    data1.set_value(1, 2, 0.0);
    gko::matrix_assembly_data<TypeParam> data2(gko::dim<2>{2, 1});
    data2.set_value(0, 0, 2.0);
    data2.set_value(1, 0, 5.0);
    auto data = std::vector<gko::matrix_assembly_data<TypeParam>>{data1, data2};

    m->read(data);

    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 8);
    ASSERT_EQ(m->get_num_stored_elements(0), 6);
    ASSERT_EQ(m->get_num_stored_elements(1), 2);
    EXPECT_EQ(m->at(0, 0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1, 0), value_type{0.0});
    EXPECT_EQ(m->at(0, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at(0, 1, 1), value_type{5.0});
    EXPECT_EQ(m->at(0, 0, 2), value_type{2.0});
    ASSERT_EQ(m->at(0, 1, 2), value_type{0.0});
    EXPECT_EQ(m->at(1, 0, 0), value_type{2.0});
    EXPECT_EQ(m->at(1, 1, 0), value_type{5.0});
}


}  // namespace
