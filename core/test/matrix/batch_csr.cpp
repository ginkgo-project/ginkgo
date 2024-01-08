// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/batch_csr.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"


template <typename T>
class Csr : public ::testing::Test {
protected:
    using value_type = T;
    using index_type = gko::int32;
    using BatchCsrMtx = gko::batch::matrix::Csr<value_type, index_type>;
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using size_type = gko::size_type;
    Csr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::batch::initialize<BatchCsrMtx>(
              // clang-format off
              {{{-1.0, 2.0, 3.0},
                {-1.5, 2.5, 3.5}},
               {{1.0, 2.5, 3.0},
                {1.0, 2.0, 3.0}}},
              // clang-format on
              exec, 6)),
          sp_mtx(gko::batch::initialize<BatchCsrMtx>(
              // clang-format off
              {{{-1.0, 0.0, 0.0},
                {0.0, 2.5, 3.5}},
               {{1.0, 0.0, 0.0},
                {0.0, 2.0, 3.0}}},
              // clang-format on
              exec, 3)),
          csr_mtx(gko::initialize<CsrMtx>(
              // clang-format off
              {{1.0, 2.5, 3.0},
               {1.0, 2.0, 3.0}},
              // clang-format on
              exec, gko::dim<2>(2, 3))),
          sp_csr_mtx(gko::initialize<CsrMtx>(
              // clang-format off
              {{1.0, 0.0, 0.0},
               {0.0, 2.0, 3.0}},
              // clang-format on
              exec, gko::dim<2>(2, 3)))
    {}

    static void assert_equal_to_original_sparse_mtx(const BatchCsrMtx* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 2);
        ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 2 * 3);
        EXPECT_EQ(m->get_const_values()[0], value_type{-1.0});
        EXPECT_EQ(m->get_const_values()[1], value_type{2.5});
        EXPECT_EQ(m->get_const_values()[2], value_type{3.5});
        EXPECT_EQ(m->get_const_values()[3], value_type{1.0});
        EXPECT_EQ(m->get_const_values()[4], value_type{2.0});
        EXPECT_EQ(m->get_const_values()[5], value_type{3.0});
        EXPECT_EQ(m->get_const_col_idxs()[0], index_type{0});
        EXPECT_EQ(m->get_const_col_idxs()[1], index_type{1});
        EXPECT_EQ(m->get_const_col_idxs()[2], index_type{2});
        EXPECT_EQ(m->get_const_row_ptrs()[0], index_type{0});
        EXPECT_EQ(m->get_const_row_ptrs()[1], index_type{1});
        EXPECT_EQ(m->get_const_row_ptrs()[2], index_type{3});
    }

    static void assert_equal_to_original_mtx(const BatchCsrMtx* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 2);
        ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 2 * 6);
        EXPECT_EQ(m->get_const_values()[0], value_type{-1.0});
        EXPECT_EQ(m->get_const_values()[1], value_type{2.0});
        EXPECT_EQ(m->get_const_values()[2], value_type{3.0});
        EXPECT_EQ(m->get_const_values()[3], value_type{-1.5});
        EXPECT_EQ(m->get_const_values()[4], value_type{2.5});
        EXPECT_EQ(m->get_const_values()[5], value_type{3.5});
        EXPECT_EQ(m->get_const_values()[6], value_type{1.0});
        EXPECT_EQ(m->get_const_values()[7], value_type{2.5});
        EXPECT_EQ(m->get_const_values()[8], value_type{3.0});
        EXPECT_EQ(m->get_const_values()[9], value_type{1.0});
        EXPECT_EQ(m->get_const_values()[10], value_type{2.0});
        EXPECT_EQ(m->get_const_values()[11], value_type{3.0});
        EXPECT_EQ(m->get_const_col_idxs()[0], index_type{0});
        EXPECT_EQ(m->get_const_col_idxs()[1], index_type{1});
        EXPECT_EQ(m->get_const_col_idxs()[2], index_type{2});
        EXPECT_EQ(m->get_const_col_idxs()[3], index_type{0});
        EXPECT_EQ(m->get_const_col_idxs()[4], index_type{1});
        EXPECT_EQ(m->get_const_col_idxs()[5], index_type{2});
        EXPECT_EQ(m->get_const_row_ptrs()[0], index_type{0});
        EXPECT_EQ(m->get_const_row_ptrs()[1], index_type{3});
        EXPECT_EQ(m->get_const_row_ptrs()[2], index_type{6});
    }

    static void assert_empty(BatchCsrMtx* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<BatchCsrMtx> mtx;
    std::unique_ptr<BatchCsrMtx> sp_mtx;
    std::unique_ptr<CsrMtx> csr_mtx;
    std::unique_ptr<CsrMtx> sp_csr_mtx;
};

TYPED_TEST_SUITE(Csr, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Csr, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Csr, SparseMtxKnowsItsSizeAndValues)
{
    this->assert_equal_to_original_sparse_mtx(this->sp_mtx.get());
}


TYPED_TEST(Csr, CanBeEmpty)
{
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;

    auto empty = BatchCsrMtx::create(this->exec);

    this->assert_empty(empty.get());
    ASSERT_EQ(empty->get_const_values(), nullptr);
}


TYPED_TEST(Csr, CanGetValuesForEntry)
{
    using value_type = typename TestFixture::value_type;

    ASSERT_EQ(this->mtx->get_values_for_item(1)[0], value_type{1.0});
}


TYPED_TEST(Csr, CanCreateCsrItemView)
{
    GKO_ASSERT_MTX_NEAR(this->mtx->create_view_for_item(1), this->csr_mtx, 0.0);
}


TYPED_TEST(Csr, CanCreateSpCsrItemView)
{
    GKO_ASSERT_MTX_NEAR(this->sp_mtx->create_view_for_item(1), this->sp_csr_mtx,
                        0.0);
}


TYPED_TEST(Csr, CanBeCopied)
{
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;

    auto mtx_copy = BatchCsrMtx::create(this->exec);

    mtx_copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[0] = 7;
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Csr, CanBeMoved)
{
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;

    auto mtx_copy = BatchCsrMtx::create(this->exec);

    this->mtx->move_to(mtx_copy);

    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Csr, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(Csr, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Csr, CanBeConstructedWithSize)
{
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;

    auto m = BatchCsrMtx::create(this->exec,
                                 gko::batch_dim<2>(2, gko::dim<2>{5, 3}));

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(5, 3));
    ASSERT_EQ(m->get_num_stored_elements(), 0);
}


TYPED_TEST(Csr, CanBeConstructedWithSizeAndNnz)
{
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;

    auto m = BatchCsrMtx::create(this->exec,
                                 gko::batch_dim<2>(2, gko::dim<2>{5, 3}), 5);

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(5, 3));
    ASSERT_EQ(m->get_num_stored_elements(), 10);
}


TYPED_TEST(Csr, CanBeConstructedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;
    value_type values[] = {-1.0, 2.5, 3.5, 1.0, 2.0, 3.0};
    index_type col_idxs[] = {0, 1, 2};
    index_type row_ptrs[] = {0, 1, 3};

    auto m = BatchCsrMtx::create(
        this->exec, gko::batch_dim<2>(2, gko::dim<2>(2, 3)),
        gko::array<value_type>::view(this->exec, 6, values),
        gko::array<index_type>::view(this->exec, 3, col_idxs),
        gko::array<index_type>::view(this->exec, 3, row_ptrs));

    this->assert_equal_to_original_sparse_mtx(m.get());
}


TYPED_TEST(Csr, CanBeConstructedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;
    value_type values[] = {-1.0, 2.5, 3.5, 1.0, 2.0, 3.0};
    index_type col_idxs[] = {0, 1, 2};
    index_type row_ptrs[] = {0, 1, 3};

    auto m = BatchCsrMtx::create_const(
        this->exec, gko::batch_dim<2>(2, gko::dim<2>(2, 3)),
        gko::array<value_type>::const_view(this->exec, 6, values),
        gko::array<index_type>::const_view(this->exec, 3, col_idxs),
        gko::array<index_type>::const_view(this->exec, 3, row_ptrs));

    this->assert_equal_to_original_sparse_mtx(m.get());
}


TYPED_TEST(Csr, CanBeConstructedFromCsrMatrices)
{
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;
    using CsrMtx = typename TestFixture::CsrMtx;
    auto mat1 = gko::initialize<CsrMtx>({{-1.0, 0.0, 0.0}, {0.0, 2.5, 3.5}},
                                        this->exec);
    auto mat2 =
        gko::initialize<CsrMtx>({{1.0, 0.0, 0.0}, {0.0, 2.0, 3.0}}, this->exec);

    auto m = gko::batch::create_from_item<BatchCsrMtx>(
        this->exec, std::vector<CsrMtx*>{mat1.get(), mat2.get()}, 3);

    this->assert_equal_to_original_sparse_mtx(m.get());
}


TYPED_TEST(Csr, CanBeConstructedFromCsrMatricesByDuplication)
{
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;
    using CsrMtx = typename TestFixture::CsrMtx;
    auto mat1 =
        gko::initialize<CsrMtx>({{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}}, this->exec);
    auto bat_m = gko::batch::create_from_item<BatchCsrMtx>(
        this->exec, std::vector<CsrMtx*>{mat1.get(), mat1.get(), mat1.get()},
        2);

    auto m =
        gko::batch::create_from_item<BatchCsrMtx>(this->exec, 3, mat1.get(), 2);

    GKO_ASSERT_BATCH_MTX_NEAR(bat_m.get(), m.get(), 0.);
}


TYPED_TEST(Csr, CanBeConstructedByDuplicatingCsrMatrices)
{
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;
    using CsrMtx = typename TestFixture::CsrMtx;
    auto mat1 = gko::initialize<CsrMtx>({{-1.0, 0.0, 0.0}, {0.0, 2.5, 0.0}},
                                        this->exec);
    auto mat2 =
        gko::initialize<CsrMtx>({{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}}, this->exec);

    auto m = gko::batch::create_from_item<BatchCsrMtx>(
        this->exec, std::vector<CsrMtx*>{mat1.get(), mat2.get()}, 2);
    auto m_ref = gko::batch::create_from_item<BatchCsrMtx>(
        this->exec,
        std::vector<CsrMtx*>{mat1.get(), mat2.get(), mat1.get(), mat2.get(),
                             mat1.get(), mat2.get()},
        2);

    auto m2 = gko::batch::duplicate<BatchCsrMtx>(this->exec, 3, m.get(), 2);

    GKO_ASSERT_BATCH_MTX_NEAR(m2.get(), m_ref.get(), 0.);
}


TYPED_TEST(Csr, CanBeUnbatchedIntoCsrMatrices)
{
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;
    using CsrMtx = typename TestFixture::CsrMtx;
    auto mat1 = gko::initialize<CsrMtx>({{-1.0, 0.0, 0.0}, {0.0, 2.5, 3.5}},
                                        this->exec);
    auto mat2 =
        gko::initialize<CsrMtx>({{1.0, 0.0, 0.0}, {0.0, 2.0, 3.0}}, this->exec);

    auto csr_mats = gko::batch::unbatch<BatchCsrMtx>(this->sp_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mats[0].get(), mat1.get(), 0.);
    GKO_ASSERT_MTX_NEAR(csr_mats[1].get(), mat2.get(), 0.);
}


TYPED_TEST(Csr, CanBeListConstructed)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;

    auto m = gko::batch::initialize<BatchCsrMtx>({{0.0, -1.0}, {0.0, -5.0}},
                                                 this->exec, 1);

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 2);
    EXPECT_EQ(m->get_values()[0], value_type{-1.0});
    EXPECT_EQ(m->get_values()[1], value_type{-5.0});
    EXPECT_EQ(m->get_col_idxs()[0], index_type{0});
    EXPECT_EQ(m->get_row_ptrs()[0], index_type{0});
    EXPECT_EQ(m->get_row_ptrs()[1], index_type{0});
    EXPECT_EQ(m->get_row_ptrs()[2], index_type{1});
}


TYPED_TEST(Csr, CanBeListConstructedByCopies)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;

    auto m = gko::batch::initialize<BatchCsrMtx>(2, I<value_type>({0.0, -1.0}),
                                                 this->exec, 1);

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 2);
    EXPECT_EQ(m->get_values()[0], value_type{-1.0});
    EXPECT_EQ(m->get_values()[1], value_type{-1.0});
    EXPECT_EQ(m->get_col_idxs()[0], index_type{0});
    EXPECT_EQ(m->get_row_ptrs()[0], index_type{0});
    EXPECT_EQ(m->get_row_ptrs()[1], index_type{0});
    EXPECT_EQ(m->get_row_ptrs()[2], index_type{1});
}


TYPED_TEST(Csr, CanBeDoubleListConstructed)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;
    using T = value_type;

    auto m = gko::batch::initialize<BatchCsrMtx>(
        // clang-format off
        {{I<T>{1.0, 0.0, 0.0},
          I<T>{2.0, 0.0, 3.0},
          I<T>{3.0, 6.0, 0.0}},
         {I<T>{1.0, 0.0, 0.0},
          I<T>{3.0, 0.0, -2.0},
          I<T>{5.0, 8.0, 0.0}}},
        // clang-format on
        this->exec, 5);

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(m->get_num_stored_elements(), 2 * 5);
    EXPECT_EQ(m->get_values()[0], value_type{1.0});
    EXPECT_EQ(m->get_values()[1], value_type{2.0});
    EXPECT_EQ(m->get_values()[2], value_type{3.0});
    EXPECT_EQ(m->get_values()[3], value_type{3.0});
    EXPECT_EQ(m->get_values()[4], value_type{6.0});
    EXPECT_EQ(m->get_values()[5], value_type{1.0});
    EXPECT_EQ(m->get_values()[6], value_type{3.0});
    EXPECT_EQ(m->get_values()[7], value_type{-2.0});
    EXPECT_EQ(m->get_values()[8], value_type{5.0});
    EXPECT_EQ(m->get_values()[9], value_type{8.0});
    EXPECT_EQ(m->get_col_idxs()[0], index_type{0});
    EXPECT_EQ(m->get_col_idxs()[1], index_type{0});
    EXPECT_EQ(m->get_col_idxs()[2], index_type{2});
    EXPECT_EQ(m->get_col_idxs()[3], index_type{0});
    EXPECT_EQ(m->get_col_idxs()[4], index_type{1});
    EXPECT_EQ(m->get_row_ptrs()[0], index_type{0});
    EXPECT_EQ(m->get_row_ptrs()[1], index_type{1});
    EXPECT_EQ(m->get_row_ptrs()[2], index_type{3});
    EXPECT_EQ(m->get_row_ptrs()[3], index_type{5});
}


TYPED_TEST(Csr, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;
    auto vec_data = std::vector<gko::matrix_data<value_type, index_type>>{};
    vec_data.emplace_back(gko::matrix_data<value_type, index_type>(
        {2, 3}, {{0, 0, -1.0}, {1, 1, 2.5}, {1, 2, 3.5}}));
    vec_data.emplace_back(gko::matrix_data<value_type, index_type>(
        {2, 3}, {{0, 0, 1.0}, {1, 1, 2.0}, {1, 2, 3.0}}));

    auto m = gko::batch::read<value_type, index_type, BatchCsrMtx>(this->exec,
                                                                   vec_data, 3);

    this->assert_equal_to_original_sparse_mtx(m.get());
}


TYPED_TEST(Csr, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchCsrMtx = typename TestFixture::BatchCsrMtx;
    using tpl = typename gko::matrix_data<TypeParam>::nonzero_type;

    auto data = gko::batch::write<value_type, index_type, BatchCsrMtx>(
        this->sp_mtx.get());

    ASSERT_EQ(data[0].size, gko::dim<2>(2, 3));
    ASSERT_EQ(data[0].nonzeros.size(), 3);
    EXPECT_EQ(data[0].nonzeros[0], tpl(0, 0, value_type{-1.0}));
    EXPECT_EQ(data[0].nonzeros[1], tpl(1, 1, value_type{2.5}));
    EXPECT_EQ(data[0].nonzeros[2], tpl(1, 2, value_type{3.5}));
    ASSERT_EQ(data[1].size, gko::dim<2>(2, 3));
    ASSERT_EQ(data[1].nonzeros.size(), 3);
    EXPECT_EQ(data[1].nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data[1].nonzeros[1], tpl(1, 1, value_type{2.0}));
    EXPECT_EQ(data[1].nonzeros[2], tpl(1, 2, value_type{3.0}));
}
