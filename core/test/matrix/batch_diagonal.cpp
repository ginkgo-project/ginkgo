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

#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class BatchDiagonal : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::BatchDiagonal<value_type>;
    using DiagonalMtx = gko::matrix::Diagonal<value_type>;
    using size_type = gko::size_type;
    BatchDiagonal()
        : exec(gko::ReferenceExecutor::create()),
          u_mtx(gko::batch_diagonal_initialize(
              {I<T>({-1.0, 2.0, 3.0}), I<T>({1.0, 2.5, -3.0})}, exec)),
          nu_mtx(generate_nonuniform_batch_diag_matrix())
    {}

    std::unique_ptr<Mtx> generate_nonuniform_batch_diag_matrix()
    {
        const std::vector<gko::dim<2>> sizes{gko::dim<2>{3, 5},
                                             gko::dim<2>{6, 4}};
        const size_type num_entries = sizes.size();
        const size_type stored = 3 + 4;
        gko::array<value_type> vals(exec, stored);
        auto valarr = vals.get_data();
        valarr[0] = 2.5;
        valarr[1] = -1.5;
        valarr[2] = 3.0;
        valarr[3] = 1.5;
        valarr[4] = 5.5;
        valarr[5] = -2.0;
        valarr[6] = -2.2;
        const gko::batch_dim<2> bsize(sizes);
        return Mtx::create(exec, bsize, std::move(vals));
    }

    // compares with nu_mtx
    static void assert_equal_to_original_nonuniform_mtx(const Mtx* const m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 2);
        ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 5));
        ASSERT_EQ(m->get_size().at(0), gko::dim<2>(6, 4));
        ASSERT_EQ(m->get_num_stored_elements(), 7);
        ASSERT_EQ(m->get_num_stored_elements(0), 3);
        ASSERT_EQ(m->get_num_stored_elements(1), 4);
        EXPECT_EQ(m->at(0, 0), value_type{2.5});
        EXPECT_EQ(m->at(0, 1), value_type{-1.5});
        EXPECT_EQ(m->at(0, 2), value_type{3.0});
        EXPECT_EQ(m->at(1, 0), value_type{1.5});
        EXPECT_EQ(m->at(1, 1), value_type{5.5});
        EXPECT_EQ(m->at(1, 2), value_type{-2.0});
        EXPECT_EQ(m->at(1, 3), value_type{-2.2});
    }

    // compares with u_mtx
    static void assert_equal_to_original_uniform_mtx(const Mtx* const m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 2);
        ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 3));
        ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 6);
        ASSERT_EQ(m->get_num_stored_elements(0), 3);
        ASSERT_EQ(m->get_num_stored_elements(1), 3);
        EXPECT_EQ(m->at(0, 0), value_type{-1.0});
        EXPECT_EQ(m->at(0, 1), value_type{2.0});
        EXPECT_EQ(m->at(0, 2), value_type{3.0});
        EXPECT_EQ(m->at(1, 0), value_type{1.0});
        EXPECT_EQ(m->at(1, 1), value_type{2.5});
        EXPECT_EQ(m->at(1, 2), value_type{-3.0});
    }

    std::array<std::unique_ptr<DiagonalMtx>, 2> get_unbatched_uniform_mtx()
        const
    {
        const size_type csize{3};
        gko::array<value_type> vals1(this->exec, 3);
        auto valarr1 = vals1.get_data();
        valarr1[0] = -1.0;
        valarr1[1] = 2.0;
        valarr1[2] = 3.0;
        gko::array<value_type> vals2(this->exec, 3);
        auto valarr2 = vals2.get_data();
        valarr2[0] = 1.0;
        valarr2[1] = 2.5;
        valarr2[2] = -3.0;
        auto diag1 = DiagonalMtx::create(this->exec, csize, std::move(vals1));
        auto diag2 = DiagonalMtx::create(this->exec, csize, std::move(vals2));
        return {std::move(diag1), std::move(diag2)};
    }

    static void assert_empty(const Mtx* const m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> u_mtx;
    std::unique_ptr<Mtx> nu_mtx;
};

TYPED_TEST_SUITE(BatchDiagonal, gko::test::ValueTypes);


TYPED_TEST(BatchDiagonal, CanBeEmpty)
{
    auto empty = gko::matrix::BatchDiagonal<TypeParam>::create(this->exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(BatchDiagonal, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::matrix::BatchDiagonal<TypeParam>::create(this->exec);
    ASSERT_EQ(empty->get_const_values(), nullptr);
}


TYPED_TEST(BatchDiagonal, CanBeConstructedWithSize)
{
    using size_type = gko::size_type;
    auto m = gko::matrix::BatchDiagonal<TypeParam>::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{3, 4}, gko::dim<2>{2, 3}});

    ASSERT_EQ(m->get_num_batch_entries(), 2);
    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 4));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_num_stored_elements(), 5);
    ASSERT_EQ(m->get_num_stored_elements(0), 3);
    ASSERT_EQ(m->get_num_stored_elements(1), 2);
}


TYPED_TEST(BatchDiagonal, CanBeConstructedFromExistingData)
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

    auto m = gko::matrix::BatchDiagonal<TypeParam>::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{4, 2}, gko::dim<2>{4, 5}},
        gko::array<value_type>::view(this->exec, 12, data));

    ASSERT_EQ(m->get_const_values(), data);
    ASSERT_EQ(m->at(0, 0), value_type{1.0});
    ASSERT_EQ(m->at(0, 1), value_type{2.0});
    ASSERT_EQ(m->at(1, 0), value_type{-1.0});
    ASSERT_EQ(m->at(1, 1), value_type{3.0});
    ASSERT_EQ(m->at(1, 2), value_type{4.0});
    ASSERT_EQ(m->at(1, 3), value_type{-1.0});
}


TYPED_TEST(BatchDiagonal, CanBeConstructedFromBatchDiagonalMatrices)
{
    using value_type = typename TestFixture::value_type;
    using DiagonalMtx = typename TestFixture::DiagonalMtx;
    using size_type = gko::size_type;
    auto mat1 = DiagonalMtx::create(this->exec, 3);
    auto m1vals = mat1->get_values();
    m1vals[0] = -1.0;
    m1vals[1] = 2.0;
    m1vals[2] = 3.0;
    auto mat2 = DiagonalMtx::create(this->exec, 2);
    auto m2vals = mat2->get_values();
    m2vals[0] = 1.0;
    m2vals[1] = -2.5;

    auto m = gko::matrix::BatchDiagonal<TypeParam>::create(
        this->exec, std::vector<DiagonalMtx*>{mat1.get(), mat2.get()});
    auto m_ref = gko::matrix::BatchDiagonal<TypeParam>::create(
        this->exec,
        std::vector<DiagonalMtx*>{mat1.get(), mat2.get(), mat1.get(),
                                  mat2.get(), mat1.get(), mat2.get()});
    auto m2 =
        gko::matrix::BatchDiagonal<TypeParam>::create(this->exec, 3, m.get());

    GKO_ASSERT_BATCH_MTX_NEAR(m2.get(), m_ref.get(), 0.0);
}


TYPED_TEST(BatchDiagonal, CanBeConstructedFromDiagonalMatricesByDuplication)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using DiagonalMtx = typename TestFixture::DiagonalMtx;
    using size_type = gko::size_type;
    auto mat1 = DiagonalMtx::create(this->exec, 3);
    auto m1vals = mat1->get_values();
    m1vals[0] = -1.0;
    m1vals[1] = 2.0;
    m1vals[2] = 3.0;

    auto bat_m = Mtx::create(
        this->exec,
        std::vector<DiagonalMtx*>{mat1.get(), mat1.get(), mat1.get()});
    auto m = Mtx::create(this->exec, static_cast<size_type>(3), mat1.get());

    GKO_ASSERT_BATCH_MTX_NEAR(bat_m.get(), m.get(), 0.0);
}


TYPED_TEST(BatchDiagonal,
           CanBeConstructedFromBatchDiagonalMatricesByDuplication)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using DiagonalMtx = typename TestFixture::DiagonalMtx;
    using size_type = gko::size_type;
    auto mat1 = DiagonalMtx::create(this->exec, 3);
    auto m1vals = mat1->get_values();
    m1vals[0] = -1.0;
    m1vals[1] = 2.0;
    m1vals[2] = 3.0;
    auto mat2 = DiagonalMtx::create(this->exec, 3);
    auto m2vals = mat2->get_values();
    m2vals[0] = 1.0;
    m2vals[1] = -2.5;
    m2vals[2] = -4.0;
    auto bat_m = Mtx::create(this->exec,
                             std::vector<DiagonalMtx*>{mat1.get(), mat2.get()});
    auto full_m = Mtx::create(
        this->exec,
        std::vector<DiagonalMtx*>{mat1.get(), mat2.get(), mat1.get(),
                                  mat2.get(), mat1.get(), mat2.get()});

    auto m = Mtx::create(this->exec, static_cast<size_type>(3), bat_m.get());

    GKO_ASSERT_BATCH_MTX_NEAR(full_m.get(), m.get(), 0.0);
}


TYPED_TEST(BatchDiagonal, CanBeConstructedFromDiagonalMatrices)
{
    using value_type = typename TestFixture::value_type;
    using DiagonalMtx = typename TestFixture::DiagonalMtx;
    using size_type = gko::size_type;
    auto diags = this->get_unbatched_uniform_mtx();

    auto m = gko::matrix::BatchDiagonal<TypeParam>::create(
        this->exec, std::vector<DiagonalMtx*>{diags[0].get(), diags[1].get()});

    this->assert_equal_to_original_uniform_mtx(m.get());
}


TYPED_TEST(BatchDiagonal, CanBeUnbatchedIntoDiagonalMatrices)
{
    using value_type = typename TestFixture::value_type;
    using DiagonalMtx = typename TestFixture::DiagonalMtx;
    using size_type = gko::size_type;
    auto diags = this->get_unbatched_uniform_mtx();

    auto check_mats = this->u_mtx->unbatch();


    GKO_ASSERT_MTX_NEAR(diags[0].get(), check_mats[0].get(), 0.);
    GKO_ASSERT_MTX_NEAR(diags[1].get(), check_mats[1].get(), 0.);
}


TYPED_TEST(BatchDiagonal, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_uniform_mtx(this->u_mtx.get());
}


TYPED_TEST(BatchDiagonal, CanBeListConstructed)
{
    using value_type = typename TestFixture::value_type;

    auto m = gko::batch_diagonal_initialize<TypeParam>({{1.0, 2.0}, {1.0, 3.0}},
                                                       this->exec);

    ASSERT_EQ(m->get_num_batch_entries(), 2);
    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 2));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0, 0), value_type{1});
    EXPECT_EQ(m->at(0, 1), value_type{2});
    EXPECT_EQ(m->at(1, 0), value_type{1});
    EXPECT_EQ(m->at(1, 1), value_type{3});
}


TYPED_TEST(BatchDiagonal, NonUniformCanBeListConstructed)
{
    using value_type = typename TestFixture::value_type;

    auto m = gko::batch_diagonal_initialize<value_type>(
        {{1.0, 2.0, 3.1}, {1.0, 3.0}}, this->exec);

    ASSERT_EQ(m->get_num_batch_entries(), 2);
    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 3));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 5);
    EXPECT_EQ(m->at(0, 0), value_type{1});
    EXPECT_EQ(m->at(0, 1), value_type{2});
    EXPECT_EQ(m->at(0, 2), value_type{3.1});
    EXPECT_EQ(m->at(1, 0), value_type{1});
    EXPECT_EQ(m->at(1, 1), value_type{3});
}


TYPED_TEST(BatchDiagonal, CanBeListConstructedByCopies)
{
    using value_type = typename TestFixture::value_type;

    auto m = gko::batch_diagonal_initialize<value_type>(
        2, I<value_type>({1.0, 2.0}), this->exec);

    ASSERT_EQ(m->get_num_batch_entries(), 2);
    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(2, 2));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0, 0), value_type{1});
    EXPECT_EQ(m->at(0, 1), value_type{2});
    EXPECT_EQ(m->at(1, 0), value_type{1});
    EXPECT_EQ(m->at(1, 1), value_type{2});
}


TYPED_TEST(BatchDiagonal, CanBeCopied)
{
    auto mtx_copy = gko::matrix::BatchDiagonal<TypeParam>::create(this->exec);

    mtx_copy->copy_from(this->u_mtx.get());

    this->assert_equal_to_original_uniform_mtx(this->u_mtx.get());
    this->u_mtx->at(0, 0) = 7;
    this->u_mtx->at(0, 1) = 7;
    this->assert_equal_to_original_uniform_mtx(mtx_copy.get());
}


TYPED_TEST(BatchDiagonal, CanBeMoved)
{
    auto mtx_copy = gko::matrix::BatchDiagonal<TypeParam>::create(this->exec);
    mtx_copy->copy_from(std::move(this->u_mtx));
    this->assert_equal_to_original_uniform_mtx(mtx_copy.get());
    ASSERT_FALSE(this->u_mtx);
}


TYPED_TEST(BatchDiagonal, CanBeCloned)
{
    auto mtx_clone = this->u_mtx->clone();
    this->assert_equal_to_original_uniform_mtx(
        dynamic_cast<decltype(this->u_mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(BatchDiagonal, CanBeCleared)
{
    this->u_mtx->clear();
    this->assert_empty(this->u_mtx.get());
}


TYPED_TEST(BatchDiagonal, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::BatchDiagonal<TypeParam>::create(this->exec);

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
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    ASSERT_EQ(m->get_num_stored_elements(0), 2);
    ASSERT_EQ(m->get_num_stored_elements(1), 2);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1), value_type{5.0});
    EXPECT_EQ(m->at(1, 0), value_type{-1.0});
    EXPECT_EQ(m->at(1, 1), value_type{9.0});
}


TYPED_TEST(BatchDiagonal, GeneratesCorrectUniformMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using tpl = typename gko::matrix_data<TypeParam>::nonzero_type;
    std::vector<gko::matrix_data<TypeParam>> data;

    this->u_mtx->write(data);

    ASSERT_EQ(data[0].size, gko::dim<2>(3, 3));
    ASSERT_EQ(data[0].nonzeros.size(), 3);
    EXPECT_EQ(data[0].nonzeros[0], tpl(0, 0, value_type{-1.0}));
    EXPECT_EQ(data[0].nonzeros[1], tpl(1, 1, value_type{2.0}));
    EXPECT_EQ(data[0].nonzeros[2], tpl(2, 2, value_type{3.0}));
    ASSERT_EQ(data[1].size, gko::dim<2>(3, 3));
    ASSERT_EQ(data[1].nonzeros.size(), 3);
    EXPECT_EQ(data[1].nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data[1].nonzeros[1], tpl(1, 1, value_type{2.5}));
    EXPECT_EQ(data[1].nonzeros[2], tpl(2, 2, value_type{-3.0}));
}


TYPED_TEST(BatchDiagonal, GeneratesCorrectNonUniformMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using tpl = typename gko::matrix_data<TypeParam>::nonzero_type;
    std::vector<gko::matrix_data<TypeParam>> data;

    this->nu_mtx->write(data);

    ASSERT_EQ(data[0].size, gko::dim<2>(3, 5));
    ASSERT_EQ(data[0].nonzeros.size(), 3);
    EXPECT_EQ(data[0].nonzeros[0], tpl(0, 0, value_type{2.5}));
    EXPECT_EQ(data[0].nonzeros[1], tpl(1, 1, value_type{-1.5}));
    EXPECT_EQ(data[0].nonzeros[2], tpl(2, 2, value_type{3.0}));
    ASSERT_EQ(data[1].size, gko::dim<2>(6, 4));
    ASSERT_EQ(data[1].nonzeros.size(), 4);
    EXPECT_EQ(data[1].nonzeros[0], tpl(0, 0, value_type{1.5}));
    EXPECT_EQ(data[1].nonzeros[1], tpl(1, 1, value_type{5.5}));
    EXPECT_EQ(data[1].nonzeros[2], tpl(2, 2, value_type{-2.0}));
    EXPECT_EQ(data[1].nonzeros[3], tpl(3, 3, value_type{-2.2}));
}


TYPED_TEST(BatchDiagonal, CanBeReadFromMatrixAssemblyData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::BatchDiagonal<TypeParam>::create(this->exec);
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
    ASSERT_EQ(m->get_num_stored_elements(), 3);
    ASSERT_EQ(m->get_num_stored_elements(0), 2);
    ASSERT_EQ(m->get_num_stored_elements(1), 1);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1), value_type{5.0});
    EXPECT_EQ(m->at(1, 0), value_type{2.0});
}


}  // namespace
