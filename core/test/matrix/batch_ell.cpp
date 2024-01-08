// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/batch_ell.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/ell.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"


template <typename T>
class Ell : public ::testing::Test {
protected:
    using value_type = T;
    using index_type = gko::int32;
    using BatchEllMtx = gko::batch::matrix::Ell<value_type, index_type>;
    using EllMtx = gko::matrix::Ell<value_type, index_type>;
    using size_type = gko::size_type;
    Ell()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::batch::initialize<BatchEllMtx>(
              {{{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}},
               {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}},
              exec, 3)),
          sp_mtx(gko::batch::initialize<BatchEllMtx>(
              {{{-1.0, 0.0, 0.0}, {0.0, 2.5, 3.5}},
               {{1.0, 0.0, 0.0}, {0.0, 2.0, 3.0}}},
              exec, 2)),
          ell_mtx(gko::initialize<EllMtx>({{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}},
                                          exec, gko::dim<2>(2, 3), 3)),
          sp_ell_mtx(gko::initialize<EllMtx>({{1.0, 0.0, 0.0}, {0.0, 2.0, 3.0}},
                                             exec, gko::dim<2>(2, 3), 2))
    {}

    static void assert_equal_to_original_sparse_mtx(const BatchEllMtx* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 2);
        ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 2 * (2 * 2));
        ASSERT_EQ(m->get_num_stored_elements_per_row(), 2);
        EXPECT_EQ(m->get_const_values()[0], value_type{-1.0});
        EXPECT_EQ(m->get_const_values()[1], value_type{2.5});
        EXPECT_EQ(m->get_const_values()[2], value_type{0.0});
        EXPECT_EQ(m->get_const_values()[3], value_type{3.5});
        EXPECT_EQ(m->get_const_values()[4], value_type{1.0});
        EXPECT_EQ(m->get_const_values()[5], value_type{2.0});
        EXPECT_EQ(m->get_const_values()[6], value_type{0.0});
        EXPECT_EQ(m->get_const_values()[7], value_type{3.0});
        EXPECT_EQ(m->get_const_col_idxs()[0], index_type{0});
        EXPECT_EQ(m->get_const_col_idxs()[1], index_type{1});
        EXPECT_EQ(m->get_const_col_idxs()[2], index_type{-1});
        ASSERT_EQ(m->get_const_col_idxs()[3], index_type{2});
    }

    static void assert_equal_to_original_mtx(const BatchEllMtx* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 2);
        ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 2 * (2 * 3));
        ASSERT_EQ(m->get_num_stored_elements_per_row(), 3);
        EXPECT_EQ(m->get_const_values()[0], value_type{-1.0});
        EXPECT_EQ(m->get_const_values()[1], value_type{-1.5});
        EXPECT_EQ(m->get_const_values()[2], value_type{2.0});
        EXPECT_EQ(m->get_const_values()[3], value_type{2.5});
        EXPECT_EQ(m->get_const_values()[4], value_type{3.0});
        EXPECT_EQ(m->get_const_values()[5], value_type{3.5});
        EXPECT_EQ(m->get_const_values()[6], value_type{1.0});
        EXPECT_EQ(m->get_const_values()[7], value_type{1.0});
        EXPECT_EQ(m->get_const_values()[8], value_type{2.5});
        EXPECT_EQ(m->get_const_values()[9], value_type{2.0});
        EXPECT_EQ(m->get_const_values()[10], value_type{3.0});
        ASSERT_EQ(m->get_const_values()[11], value_type{3.0});
    }

    static void assert_empty(BatchEllMtx* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_num_stored_elements_per_row(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<BatchEllMtx> mtx;
    std::unique_ptr<BatchEllMtx> sp_mtx;
    std::unique_ptr<EllMtx> ell_mtx;
    std::unique_ptr<EllMtx> sp_ell_mtx;
};

TYPED_TEST_SUITE(Ell, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Ell, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Ell, SparseMtxKnowsItsSizeAndValues)
{
    this->assert_equal_to_original_sparse_mtx(this->sp_mtx.get());
}


TYPED_TEST(Ell, CanBeEmpty)
{
    using BatchEllMtx = typename TestFixture::BatchEllMtx;

    auto empty = BatchEllMtx::create(this->exec);

    this->assert_empty(empty.get());
    ASSERT_EQ(empty->get_const_values(), nullptr);
}


TYPED_TEST(Ell, CanGetValuesForEntry)
{
    using value_type = typename TestFixture::value_type;

    ASSERT_EQ(this->mtx->get_values_for_item(1)[0], value_type{1.0});
}


TYPED_TEST(Ell, CanCreateEllItemView)
{
    GKO_ASSERT_MTX_NEAR(this->mtx->create_view_for_item(1), this->ell_mtx, 0.0);
}


TYPED_TEST(Ell, CanCreateSpEllItemView)
{
    GKO_ASSERT_MTX_NEAR(this->sp_mtx->create_view_for_item(1), this->sp_ell_mtx,
                        0.0);
}


TYPED_TEST(Ell, CanBeCopied)
{
    using BatchEllMtx = typename TestFixture::BatchEllMtx;

    auto mtx_copy = BatchEllMtx::create(this->exec);

    mtx_copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[0] = 7;
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Ell, CanBeMoved)
{
    using BatchEllMtx = typename TestFixture::BatchEllMtx;

    auto mtx_copy = BatchEllMtx::create(this->exec);

    this->mtx->move_to(mtx_copy);

    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Ell, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(Ell, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Ell, CanBeConstructedWithSize)
{
    using BatchEllMtx = typename TestFixture::BatchEllMtx;

    auto m = BatchEllMtx::create(this->exec,
                                 gko::batch_dim<2>(2, gko::dim<2>{5, 3}), 2);

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(5, 3));
    ASSERT_EQ(m->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(m->get_num_stored_elements(), 20);
}


TYPED_TEST(Ell, CanBeConstructedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    // clang-format off
    value_type values[] = {
       -1.0,  2.5,
        0.0,  3.5,
        1.0,  2.0,
        0.0,  3.0};
    index_type col_idxs[] = {
       0, 1,
      -1, 2};
    // clang-format on

    auto m = BatchEllMtx::create(
        this->exec, gko::batch_dim<2>(2, gko::dim<2>(2, 3)), 2,
        gko::array<value_type>::view(this->exec, 8, values),
        gko::array<index_type>::view(this->exec, 4, col_idxs));

    this->assert_equal_to_original_sparse_mtx(m.get());
}


TYPED_TEST(Ell, CanBeConstructedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    // clang-format off
    value_type values[] = {
       -1.0,  2.5,
        0.0,  3.5,
        1.0,  2.0,
        0.0,  3.0};
    index_type col_idxs[] = {
       0, 1,
      -1, 2};
    // clang-format on

    auto m = BatchEllMtx::create_const(
        this->exec, gko::batch_dim<2>(2, gko::dim<2>(2, 3)), 2,
        gko::array<value_type>::const_view(this->exec, 8, values),
        gko::array<index_type>::const_view(this->exec, 4, col_idxs));

    this->assert_equal_to_original_sparse_mtx(m.get());
}


TYPED_TEST(Ell, CanBeConstructedFromEllMatrices)
{
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    using EllMtx = typename TestFixture::EllMtx;
    auto mat1 = gko::initialize<EllMtx>({{-1.0, 0.0, 0.0}, {0.0, 2.5, 3.5}},
                                        this->exec);
    auto mat2 =
        gko::initialize<EllMtx>({{1.0, 0.0, 0.0}, {0.0, 2.0, 3.0}}, this->exec);

    auto m = gko::batch::create_from_item<BatchEllMtx>(
        this->exec, std::vector<EllMtx*>{mat1.get(), mat2.get()},
        mat1->get_num_stored_elements_per_row());

    this->assert_equal_to_original_sparse_mtx(m.get());
}


TYPED_TEST(Ell, CanBeConstructedFromEllMatricesByDuplication)
{
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    using EllMtx = typename TestFixture::EllMtx;
    auto mat1 =
        gko::initialize<EllMtx>({{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}}, this->exec);
    auto bat_m = gko::batch::create_from_item<BatchEllMtx>(
        this->exec, std::vector<EllMtx*>{mat1.get(), mat1.get(), mat1.get()},
        mat1->get_num_stored_elements_per_row());

    auto m = gko::batch::create_from_item<BatchEllMtx>(
        this->exec, 3, mat1.get(), mat1->get_num_stored_elements_per_row());

    GKO_ASSERT_BATCH_MTX_NEAR(bat_m.get(), m.get(), 1e-14);
}


TYPED_TEST(Ell, CanBeConstructedByDuplicatingEllMatrices)
{
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    using EllMtx = typename TestFixture::EllMtx;
    auto mat1 = gko::initialize<EllMtx>({{-1.0, 0.0, 0.0}, {0.0, 2.5, 0.0}},
                                        this->exec);
    auto mat2 =
        gko::initialize<EllMtx>({{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}}, this->exec);

    auto m = gko::batch::create_from_item<BatchEllMtx>(
        this->exec, std::vector<EllMtx*>{mat1.get(), mat2.get()},
        mat1->get_num_stored_elements_per_row());
    auto m_ref = gko::batch::create_from_item<BatchEllMtx>(
        this->exec,
        std::vector<EllMtx*>{mat1.get(), mat2.get(), mat1.get(), mat2.get(),
                             mat1.get(), mat2.get()},
        mat1->get_num_stored_elements_per_row());

    auto m2 = gko::batch::duplicate<BatchEllMtx>(
        this->exec, 3, m.get(), mat1->get_num_stored_elements_per_row());

    GKO_ASSERT_BATCH_MTX_NEAR(m2.get(), m_ref.get(), 1e-14);
}


TYPED_TEST(Ell, CanBeUnbatchedIntoEllMatrices)
{
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    using EllMtx = typename TestFixture::EllMtx;
    auto mat1 = gko::initialize<EllMtx>({{-1.0, 0.0, 0.0}, {0.0, 2.5, 3.5}},
                                        this->exec);
    auto mat2 =
        gko::initialize<EllMtx>({{1.0, 0.0, 0.0}, {0.0, 2.0, 3.0}}, this->exec);

    auto ell_mats = gko::batch::unbatch<BatchEllMtx>(this->sp_mtx.get());

    GKO_ASSERT_MTX_NEAR(ell_mats[0].get(), mat1.get(), 0.);
    GKO_ASSERT_MTX_NEAR(ell_mats[1].get(), mat2.get(), 0.);
}


TYPED_TEST(Ell, CanBeListConstructed)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    using EllMtx = typename TestFixture::EllMtx;

    auto m = gko::batch::initialize<BatchEllMtx>({{0.0, -1.0}, {0.0, -5.0}},
                                                 this->exec);

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    ASSERT_EQ(m->get_num_stored_elements_per_row(), 1);
    EXPECT_EQ(m->get_values()[0], value_type{0.0});
    EXPECT_EQ(m->get_values()[1], value_type{-1.0});
    EXPECT_EQ(m->get_values()[2], value_type{0.0});
    EXPECT_EQ(m->get_values()[3], value_type{-5.0});
    EXPECT_EQ(m->get_col_idxs()[0], index_type{-1});
    EXPECT_EQ(m->get_col_idxs()[1], index_type{0});
}


TYPED_TEST(Ell, CanBeListConstructedByCopies)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchEllMtx = typename TestFixture::BatchEllMtx;

    auto m = gko::batch::initialize<BatchEllMtx>(2, I<value_type>({0.0, -1.0}),
                                                 this->exec, 1);

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    ASSERT_EQ(m->get_num_stored_elements_per_row(), 1);
    EXPECT_EQ(m->get_values()[0], value_type{0.0});
    EXPECT_EQ(m->get_values()[1], value_type{-1.0});
    EXPECT_EQ(m->get_values()[2], value_type{0.0});
    EXPECT_EQ(m->get_values()[3], value_type{-1.0});
    EXPECT_EQ(m->get_col_idxs()[0], index_type{-1});
    EXPECT_EQ(m->get_col_idxs()[1], index_type{0});
}


TYPED_TEST(Ell, CanBeDoubleListConstructed)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    using T = value_type;

    auto m = gko::batch::initialize<BatchEllMtx>(
        // clang-format off
        {{I<T>{1.0, 0.0, 0.0},
          I<T>{2.0, 0.0, 3.0},
          I<T>{3.0, 6.0, 0.0}},
         {I<T>{1.0, 0.0, 0.0},
          I<T>{3.0, 0.0, -2.0},
          I<T>{5.0, 8.0, 0.0}}},
        // clang-format on
        this->exec, 2);

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(m->get_num_stored_elements(), 2 * (2 * 3));
    ASSERT_EQ(m->get_num_stored_elements_per_row(), 2);
    EXPECT_EQ(m->get_values()[0], value_type{1.0});
    EXPECT_EQ(m->get_values()[1], value_type{2.0});
    EXPECT_EQ(m->get_values()[2], value_type{3.0});
    EXPECT_EQ(m->get_values()[3], value_type{0.0});
    EXPECT_EQ(m->get_values()[4], value_type{3.0});
    EXPECT_EQ(m->get_values()[5], value_type{6.0});
    EXPECT_EQ(m->get_values()[6], value_type{1.0});
    EXPECT_EQ(m->get_values()[7], value_type{3.0});
    EXPECT_EQ(m->get_values()[8], value_type{5.0});
    EXPECT_EQ(m->get_values()[9], value_type{0.0});
    EXPECT_EQ(m->get_values()[10], value_type{-2.0});
    EXPECT_EQ(m->get_values()[11], value_type{8.0});
    EXPECT_EQ(m->get_col_idxs()[0], index_type{0});
    EXPECT_EQ(m->get_col_idxs()[1], index_type{0});
    EXPECT_EQ(m->get_col_idxs()[2], index_type{0});
    EXPECT_EQ(m->get_col_idxs()[3], index_type{-1});
    EXPECT_EQ(m->get_col_idxs()[4], index_type{2});
    EXPECT_EQ(m->get_col_idxs()[5], index_type{1});
}


TYPED_TEST(Ell, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    auto vec_data = std::vector<gko::matrix_data<value_type, index_type>>{};
    vec_data.emplace_back(gko::matrix_data<value_type, index_type>(
        {2, 3}, {{0, 0, -1.0}, {1, 1, 2.5}, {1, 2, 3.5}}));
    vec_data.emplace_back(gko::matrix_data<value_type, index_type>(
        {2, 3}, {{0, 0, 1.0}, {1, 1, 2.0}, {1, 2, 3.0}}));

    auto m = gko::batch::read<value_type, index_type, BatchEllMtx>(this->exec,
                                                                   vec_data, 2);

    this->assert_equal_to_original_sparse_mtx(m.get());
}


TYPED_TEST(Ell, ThrowsForDataWithDifferentNnz)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    auto vec_data = std::vector<gko::matrix_data<value_type, index_type>>{};
    vec_data.emplace_back(
        gko::matrix_data<value_type, index_type>({2, 3}, {
                                                             {0, 0, -1.0},
                                                             {1, 1, 2.5},
                                                             {1, 2, 0.5},
                                                             {2, 2, -3.0},
                                                         }));
    vec_data.emplace_back(gko::matrix_data<value_type, index_type>(
        {2, 3}, {{0, 0, 1.0}, {1, 1, 2.0}, {1, 2, 3.0}}));

    EXPECT_THROW(
        gko::batch::detail::assert_same_sparsity_in_batched_data(vec_data),
        gko::NotImplemented);
}


TYPED_TEST(Ell, ThrowsForDataWithDifferentSparsity)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    auto vec_data = std::vector<gko::matrix_data<value_type, index_type>>{};
    vec_data.emplace_back(
        gko::matrix_data<value_type, index_type>({2, 3}, {
                                                             {0, 0, -1.0},
                                                             {1, 1, 2.5},
                                                             {2, 2, -3.0},
                                                         }));
    vec_data.emplace_back(gko::matrix_data<value_type, index_type>(
        {2, 3}, {{0, 0, 1.0}, {1, 1, 2.0}, {1, 2, 3.0}}));

    EXPECT_THROW(
        gko::batch::detail::assert_same_sparsity_in_batched_data(vec_data),
        gko::NotImplemented);
}


TYPED_TEST(Ell, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using BatchEllMtx = typename TestFixture::BatchEllMtx;
    using tpl = typename gko::matrix_data<TypeParam>::nonzero_type;

    auto data = gko::batch::write<value_type, index_type, BatchEllMtx>(
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
