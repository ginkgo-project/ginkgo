// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/jacobi_kernels.hpp"

#include <algorithm>
#include <type_traits>

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>

#include "core/base/extended_float.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Jacobi : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Bj = gko::preconditioner::Jacobi<value_type, index_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using mdata = gko::matrix_data<value_type, index_type>;

    Jacobi()
        : exec(gko::ReferenceExecutor::create()),
          block_pointers(exec, 3),
          block_precisions(exec, 2),
          mtx(gko::matrix::Csr<value_type, index_type>::create(
              exec, gko::dim<2>{5}, 13))
    {
        block_pointers.get_data()[0] = 0;
        block_pointers.get_data()[1] = 2;
        block_pointers.get_data()[2] = 5;
        block_precisions.get_data()[0] = gko::precision_reduction(0, 1);
        block_precisions.get_data()[1] = gko::precision_reduction(0, 0);
        scalar_j_factory = Bj::build().with_max_block_size(1u).on(exec);
        bj_factory = Bj::build()
                         .with_max_block_size(3u)
                         .with_block_pointers(block_pointers)
                         .on(exec);
        adaptive_bj_factory = Bj::build()
                                  .with_max_block_size(17u)
                                  // make sure group size is 1
                                  .with_block_pointers(block_pointers)
                                  .with_storage_optimization(block_precisions)
                                  .on(exec);
        /* test matrix:
            4  -2 |        -2
           -1   4 |
           -------+----------
                  | 4  -2
                  |-1   4  -2
           -1     |    -1   4
         */
        init_array<index_type>(mtx->get_row_ptrs(), {0, 3, 5, 7, 10, 13});
        init_array<index_type>(mtx->get_col_idxs(),
                               {0, 1, 4, 0, 1, 2, 3, 2, 3, 4, 0, 3, 4});
        init_array<value_type>(mtx->get_values(),
                               {4.0, -2.0, -2.0, -1.0, 4.0, 4.0, -2.0, -1.0,
                                4.0, -2.0, -1.0, -1.0, 4.0});
    }

    template <typename T>
    void init_array(T* arr, std::initializer_list<T> vals)
    {
        for (auto elem : vals) {
            *(arr++) = elem;
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<typename Bj::Factory> bj_factory;
    std::unique_ptr<typename Bj::Factory> scalar_j_factory;
    std::unique_ptr<typename Bj::Factory> adaptive_bj_factory;
    gko::array<index_type> block_pointers;
    gko::array<gko::precision_reduction> block_precisions;
    std::shared_ptr<gko::matrix::Csr<value_type, index_type>> mtx;
};

TYPED_TEST_SUITE(Jacobi, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Jacobi, CanBeGenerated)
{
    auto bj = this->bj_factory->generate(this->mtx);

    ASSERT_NE(bj, nullptr);
    EXPECT_EQ(bj->get_executor(), this->exec);
    EXPECT_EQ(bj->get_parameters().max_block_size, 3);
    ASSERT_EQ(bj->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    ASSERT_EQ(ptrs[2], 5);
}


TYPED_TEST(Jacobi, CanBeGeneratedWithAdaptivePrecision)
{
    auto bj = this->adaptive_bj_factory->generate(this->mtx);

    EXPECT_EQ(bj->get_executor(), this->exec);
    EXPECT_EQ(bj->get_parameters().max_block_size, 17);
    ASSERT_EQ(bj->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    ASSERT_EQ(ptrs[2], 5);
    auto prec =
        bj->get_parameters().storage_optimization.block_wise.get_const_data();
    EXPECT_EQ(prec[0], gko::precision_reduction(0, 1));
    ASSERT_EQ(prec[1], gko::precision_reduction(0, 0));
}


TYPED_TEST(Jacobi, FindsNaturalBlocks)
{
    /* example matrix:
        1   1
        1   1
        1       1
        1       1
     */
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto mtx = Mtx::create(this->exec, gko::dim<2>{4, 4}, 8);
    this->template init_array<index_type>(mtx->get_row_ptrs(), {0, 2, 4, 6, 8});
    this->template init_array<index_type>(mtx->get_col_idxs(),
                                          {0, 1, 0, 1, 0, 2, 0, 2});
    this->template init_array<value_type>(
        mtx->get_values(), {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    auto bj =
        Bj::build().with_max_block_size(3u).on(this->exec)->generate(give(mtx));

    EXPECT_EQ(bj->get_parameters().max_block_size, 3);
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    EXPECT_EQ(ptrs[2], 4);
}


TYPED_TEST(Jacobi, ExecutesSupervariableAgglomeration)
{
    /* example matrix:
        1   1
        1   1
                1   1
                1   1
                        1
     */
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto mtx = Mtx::create(this->exec, gko::dim<2>{5}, 9);
    this->template init_array<index_type>(mtx->get_row_ptrs(),
                                          {0, 2, 4, 6, 8, 9});
    this->template init_array<index_type>(mtx->get_col_idxs(),
                                          {0, 1, 0, 1, 2, 3, 2, 3, 4});
    this->template init_array<value_type>(
        mtx->get_values(), {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    auto bj =
        Bj::build().with_max_block_size(3u).on(this->exec)->generate(give(mtx));

    EXPECT_EQ(bj->get_parameters().max_block_size, 3);
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    EXPECT_EQ(ptrs[2], 5);
}


TYPED_TEST(Jacobi, AdheresToBlockSizeBound)
{
    /* example matrix:
        1
            1
                1
                    1
                        1
                            1
                                1
     */
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto mtx = Mtx::create(this->exec, gko::dim<2>{7}, 7);
    this->template init_array<index_type>(mtx->get_row_ptrs(),
                                          {0, 1, 2, 3, 4, 5, 6, 7});
    this->template init_array<index_type>(mtx->get_col_idxs(),
                                          {0, 1, 2, 3, 4, 5, 6});
    this->template init_array<value_type>(mtx->get_values(),
                                          {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    auto bj =
        Bj::build().with_max_block_size(3u).on(this->exec)->generate(give(mtx));

    EXPECT_EQ(bj->get_parameters().max_block_size, 3);
    ASSERT_EQ(bj->get_num_blocks(), 3);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 3);
    EXPECT_EQ(ptrs[2], 6);
    EXPECT_EQ(ptrs[3], 7);
}


TYPED_TEST(Jacobi, CanBeGeneratedWithUnknownBlockSizes)
{
    using Bj = typename TestFixture::Bj;
    auto bj =
        Bj::build().with_max_block_size(3u).on(this->exec)->generate(this->mtx);

    ASSERT_NE(bj, nullptr);
    EXPECT_EQ(bj->get_executor(), this->exec);
    EXPECT_EQ(bj->get_parameters().max_block_size, 3);
    ASSERT_EQ(bj->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 3);
    ASSERT_EQ(ptrs[2], 5);
}


TYPED_TEST(Jacobi, InvertsDiagonalBlocks)
{
    using T = typename TestFixture::value_type;
    auto bj = this->bj_factory->generate(this->mtx);

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    auto b1 = bj->get_blocks() + scheme.get_global_block_offset(0);
    GKO_EXPECT_NEAR(b1[0 + 0 * p], T{4.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[0 + 1 * p], T{2.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 0 * p], T{1.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 1 * p], T{4.0 / 14.0}, r<T>::value);

    auto b2 = bj->get_blocks() + scheme.get_global_block_offset(1);
    GKO_EXPECT_NEAR(b2[0 + 0 * p], T{14.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 1 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 2 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 0 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 1 * p], T{16.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 2 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 0 * p], T{1.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 1 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 2 * p], T{14.0 / 48.0}, r<T>::value);
}


TYPED_TEST(Jacobi, InvertsDiagonalBlocksWithAdaptivePrecision)
{
    using T = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<T>::value);
    auto bj = this->adaptive_bj_factory->generate(this->mtx);

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    const auto b_prec_bj =
        bj->get_parameters().storage_optimization.block_wise.get_const_data();
    using reduced = ::gko::reduce_precision<T>;
    auto b1 = reinterpret_cast<const reduced*>(
        bj->get_blocks() + scheme.get_global_block_offset(0));
    GKO_EXPECT_NEAR(b1[0 + 0 * p], reduced{4.0 / 14.0}, half_tol);
    GKO_EXPECT_NEAR(b1[0 + 1 * p], reduced{2.0 / 14.0}, half_tol);
    GKO_EXPECT_NEAR(b1[1 + 0 * p], reduced{1.0 / 14.0}, half_tol);
    GKO_EXPECT_NEAR(b1[1 + 1 * p], reduced{4.0 / 14.0}, half_tol);

    auto b2 = bj->get_blocks() + scheme.get_global_block_offset(1);
    GKO_EXPECT_NEAR(b2[0 + 0 * p], T{14.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 1 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 2 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 0 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 1 * p], T{16.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 2 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 0 * p], T{1.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 1 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 2 * p], T{14.0 / 48.0}, r<T>::value);
}


TYPED_TEST(Jacobi, CanTransposeDiagonalBlocks)
{
    using T = typename TestFixture::value_type;
    using Bj = typename TestFixture::Bj;
    auto tmp_bj = this->bj_factory->generate(this->mtx);

    auto bj = gko::as<Bj>(tmp_bj->transpose());

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    auto b1 = bj->get_blocks() + scheme.get_global_block_offset(0);
    GKO_EXPECT_NEAR(b1[0 + 0 * p], T{4.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 0 * p], T{2.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[0 + 1 * p], T{1.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 1 * p], T{4.0 / 14.0}, r<T>::value);
    auto b2 = bj->get_blocks() + scheme.get_global_block_offset(1);
    GKO_EXPECT_NEAR(b2[0 + 0 * p], T{14.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 0 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 0 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 1 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 1 * p], T{16.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 1 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 2 * p], T{1.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 2 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 2 * p], T{14.0 / 48.0}, r<T>::value);
}


TYPED_TEST(Jacobi, CanTransposeDiagonalBlocksWithAdaptivePrecision)
{
    using T = typename TestFixture::value_type;
    using Bj = typename TestFixture::Bj;
    auto half_tol = std::sqrt(r<T>::value);
    auto tmp_bj = this->adaptive_bj_factory->generate(this->mtx);

    auto bj = gko::as<Bj>(tmp_bj->transpose());

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    using reduced = ::gko::reduce_precision<T>;
    auto b1 = reinterpret_cast<const reduced*>(
        bj->get_blocks() + scheme.get_global_block_offset(0));
    GKO_EXPECT_NEAR(b1[0 + 0 * p], reduced{4.0 / 14.0}, half_tol);
    GKO_EXPECT_NEAR(b1[1 + 0 * p], reduced{2.0 / 14.0}, half_tol);
    GKO_EXPECT_NEAR(b1[0 + 1 * p], reduced{1.0 / 14.0}, half_tol);
    GKO_EXPECT_NEAR(b1[1 + 1 * p], reduced{4.0 / 14.0}, half_tol);
    auto b2 = bj->get_blocks() + scheme.get_global_block_offset(1);
    GKO_EXPECT_NEAR(b2[0 + 0 * p], T{14.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 0 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 0 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 1 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 1 * p], T{16.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 1 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 2 * p], T{1.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 2 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 2 * p], T{14.0 / 48.0}, r<T>::value);
}


TYPED_TEST(Jacobi, CanConjTransposeDiagonalBlocks)
{
    using T = typename TestFixture::value_type;
    using Bj = typename TestFixture::Bj;
    auto tmp_bj = this->bj_factory->generate(this->mtx);

    auto bj = gko::as<Bj>(tmp_bj->conj_transpose());

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    auto b1 = bj->get_blocks() + scheme.get_global_block_offset(0);
    GKO_EXPECT_NEAR(b1[0 + 0 * p], T{4.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 0 * p], T{2.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[0 + 1 * p], T{1.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 1 * p], T{4.0 / 14.0}, r<T>::value);
    auto b2 = bj->get_blocks() + scheme.get_global_block_offset(1);
    GKO_EXPECT_NEAR(b2[0 + 0 * p], T{14.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 0 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 0 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 1 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 1 * p], T{16.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 1 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 2 * p], T{1.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 2 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 2 * p], T{14.0 / 48.0}, r<T>::value);
}


TYPED_TEST(Jacobi, CanConjTransposeDiagonalBlocksWithAdaptivePrecision)
{
    using T = typename TestFixture::value_type;
    using Bj = typename TestFixture::Bj;
    auto half_tol = std::sqrt(r<T>::value);
    auto tmp_bj = this->adaptive_bj_factory->generate(this->mtx);

    auto bj = gko::as<Bj>(tmp_bj->conj_transpose());

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    using reduced = ::gko::reduce_precision<T>;
    auto b1 = reinterpret_cast<const reduced*>(
        bj->get_blocks() + scheme.get_global_block_offset(0));
    GKO_EXPECT_NEAR(b1[0 + 0 * p], reduced{4.0 / 14.0}, half_tol);
    GKO_EXPECT_NEAR(b1[1 + 0 * p], reduced{2.0 / 14.0}, half_tol);
    GKO_EXPECT_NEAR(b1[0 + 1 * p], reduced{1.0 / 14.0}, half_tol);
    GKO_EXPECT_NEAR(b1[1 + 1 * p], reduced{4.0 / 14.0}, half_tol);
    auto b2 = bj->get_blocks() + scheme.get_global_block_offset(1);
    GKO_EXPECT_NEAR(b2[0 + 0 * p], T{14.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 0 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 0 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 1 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 1 * p], T{16.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 1 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 2 * p], T{1.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 2 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 2 * p], T{14.0 / 48.0}, r<T>::value);
}


TYPED_TEST(Jacobi, InvertsDiagonalBlocksWithAdaptivePrecisionAndSmallBlocks)
{
    using Bj = typename TestFixture::Bj;
    using T = typename TestFixture::value_type;
    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  // group size will be > 1
                  .with_block_pointers(this->block_pointers)
                  .with_storage_optimization(this->block_precisions)
                  .on(this->exec)
                  ->generate(this->mtx);

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    auto b1 = bj->get_blocks() + scheme.get_global_block_offset(0);
    GKO_EXPECT_NEAR(b1[0 + 0 * p], T{4.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[0 + 1 * p], T{2.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 0 * p], T{1.0 / 14.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 1 * p], T{4.0 / 14.0}, r<T>::value);

    auto b2 = bj->get_blocks() + scheme.get_global_block_offset(1);
    GKO_EXPECT_NEAR(b2[0 + 0 * p], T{14.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 1 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[0 + 2 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 0 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 1 * p], T{16.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[1 + 2 * p], T{8.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 0 * p], T{1.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 1 * p], T{4.0 / 48.0}, r<T>::value);
    GKO_EXPECT_NEAR(b2[2 + 2 * p], T{14.0 / 48.0}, r<T>::value);
}


TYPED_TEST(Jacobi, PivotsWhenInvertingBlocks)
{
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    gko::array<index_type> bp(this->exec, 2);
    this->template init_array<index_type>(bp.get_data(), {0, 3});
    auto mtx = Mtx::create(this->exec, gko::dim<2>{3}, 9);
    /* test matrix:
       0 2 0
       0 0 4
       1 0 0
     */
    this->template init_array<index_type>(mtx->get_row_ptrs(), {0, 3, 6, 9});
    this->template init_array<index_type>(mtx->get_col_idxs(),
                                          {0, 1, 2, 0, 1, 2, 0, 1, 2});
    this->template init_array<T>(mtx->get_values(),
                                 {0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0});

    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  .with_block_pointers(bp)
                  .on(this->exec)
                  ->generate(give(mtx));

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    auto b1 = bj->get_blocks() + scheme.get_global_block_offset(0);
    GKO_EXPECT_NEAR(b1[0 + 0 * p], T{0.0 / 4.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[0 + 1 * p], T{0.0 / 4.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[0 + 2 * p], T{4.0 / 4.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 0 * p], T{2.0 / 4.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 1 * p], T{0.0 / 4.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[1 + 2 * p], T{0.0 / 4.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[2 + 0 * p], T{0.0 / 4.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[2 + 1 * p], T{1.0 / 4.0}, r<T>::value);
    GKO_EXPECT_NEAR(b1[2 + 2 * p], T{0.0 / 4.0}, r<T>::value);
}


TYPED_TEST(Jacobi, PivotsWhenInvertingBlocksWithiAdaptivePrecision)
{
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    using T = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<T>::value);
    gko::array<index_type> bp(this->exec, 2);
    this->template init_array<index_type>(bp.get_data(), {0, 3});
    auto mtx = Mtx::create(this->exec, gko::dim<2>{3}, 9);
    /* test matrix:
       0 2 0
       0 0 4
       1 0 0
     */
    this->template init_array<index_type>(mtx->get_row_ptrs(), {0, 3, 6, 9});
    this->template init_array<index_type>(mtx->get_col_idxs(),
                                          {0, 1, 2, 0, 1, 2, 0, 1, 2});
    this->template init_array<T>(mtx->get_values(),
                                 {0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0});

    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  .with_block_pointers(bp)
                  .with_storage_optimization(this->block_precisions)
                  .on(this->exec)
                  ->generate(give(mtx));

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    using reduced = ::gko::reduce_precision<T>;
    auto b1 = reinterpret_cast<const reduced*>(
        bj->get_blocks() + scheme.get_global_block_offset(0));
    GKO_EXPECT_NEAR(b1[0 + 0 * p], reduced{0.0 / 4.0}, half_tol);
    GKO_EXPECT_NEAR(b1[0 + 1 * p], reduced{0.0 / 4.0}, half_tol);
    GKO_EXPECT_NEAR(b1[0 + 2 * p], reduced{4.0 / 4.0}, half_tol);
    GKO_EXPECT_NEAR(b1[1 + 0 * p], reduced{2.0 / 4.0}, half_tol);
    GKO_EXPECT_NEAR(b1[1 + 1 * p], reduced{0.0 / 4.0}, half_tol);
    GKO_EXPECT_NEAR(b1[1 + 2 * p], reduced{0.0 / 4.0}, half_tol);
    GKO_EXPECT_NEAR(b1[2 + 0 * p], reduced{0.0 / 4.0}, half_tol);
    GKO_EXPECT_NEAR(b1[2 + 1 * p], reduced{1.0 / 4.0}, half_tol);
    GKO_EXPECT_NEAR(b1[2 + 2 * p], reduced{0.0 / 4.0}, half_tol);
}


TYPED_TEST(Jacobi, ComputesConditionNumbersOfBlocks)
{
    using T = typename TestFixture::value_type;
    auto bj = this->adaptive_bj_factory->generate(this->mtx);

    auto cond = bj->get_conditioning();
    GKO_EXPECT_NEAR(cond[0], gko::remove_complex<T>{6.0 * 6.0 / 14.0},
                    r<T>::value * 1e1);
    GKO_ASSERT_NEAR(cond[1], gko::remove_complex<T>{7.0 * 28.0 / 48.0},
                    r<T>::value * 1e1);
}


TYPED_TEST(Jacobi, SelectsCorrectBlockPrecisions)
{
    using Bj = typename TestFixture::Bj;
    using T = typename TestFixture::value_type;
    auto bj =
        Bj::build()
            .with_max_block_size(17u)
            .with_block_pointers(this->block_pointers)
            .with_storage_optimization(gko::precision_reduction::autodetect())
            .with_accuracy(gko::remove_complex<T>{1.5e-3})
            .on(this->exec)
            ->generate(give(this->mtx));

    auto prec =
        bj->get_parameters().storage_optimization.block_wise.get_const_data();
    auto precision1 =
        std::is_same<gko::remove_complex<T>, gko::half>::value
            ? gko::precision_reduction(2, 0)
            : (std::is_same<gko::remove_complex<T>, gko::bfloat16>::value
                   ? gko::precision_reduction(0, 0)
                   : gko::precision_reduction(0, 2));
    auto precision2 =
        std::is_same<gko::remove_complex<T>, double>::value
            ? gko::precision_reduction(0, 1)   // double
            : gko::precision_reduction(0, 0);  // float, half, bfloat16
    EXPECT_EQ(prec[0], precision1);            // u * cond = ~1.2e-3
    ASSERT_EQ(prec[1], precision2);            // u * cond = ~2.0e-3
}


TYPED_TEST(Jacobi, AvoidsPrecisionsThatOverflow)
{
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    using mdata = typename TestFixture::mdata;
    using T = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<T>::value);
    auto mtx = Mtx::create(this->exec);
    // clang-format off
    mtx->read(mdata::diag({
                // perfectly conditioned block, small value difference,
                // can use fp16 (5, 10)
                {{2.0, 1.0},
                 {1.0, 2.0}},
                // perfectly conditioned block (scaled orthogonal),
                // with large value difference, need fp16 (7, 8)
                {{half_tol, -r<T>::value},
                 {r<T>::value,  half_tol}}
    }));
    // clang-format on

    auto bj =
        Bj::build()
            .with_max_block_size(13u)
            .with_block_pointers(gko::array<index_type>(this->exec, {0, 2, 4}))
            .with_storage_optimization(gko::precision_reduction::autodetect())
            .with_accuracy(gko::remove_complex<T>{1e-1})
            .on(this->exec)
            ->generate(give(mtx));

    // both blocks are in the same group, both need (7, 8)
    auto prec =
        bj->get_parameters().storage_optimization.block_wise.get_const_data();
    auto precision = std::is_same<gko::remove_complex<T>, float>::value
                         ? gko::precision_reduction(0, 2)   // float
                         : gko::precision_reduction(1, 1);  // double
    if (std::is_same<gko::remove_complex<T>, gko::float16>::value ||
        std::is_same<gko::remove_complex<T>, gko::bfloat16>::value) {
        precision = gko::precision_reduction(2, 0);
    }
    EXPECT_EQ(prec[0], precision);
    ASSERT_EQ(prec[1], precision);
}


TYPED_TEST(Jacobi, AppliesToVector)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto bj = this->bj_factory->generate(this->mtx);

    bj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 0.0, 0.0, 1.0, 0.0}), r<value_type>::value);
}


TYPED_TEST(Jacobi, ScalarJacobiAppliesToVector)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto sj = this->scalar_j_factory->generate(this->mtx);

    sj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, -0.25, -0.5, 1.0, -0.25}),
                        r<value_type>::value);
}


TYPED_TEST(Jacobi, AppliesToMixedVector)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto bj = this->bj_factory->generate(this->mtx);

    bj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({1.0, 0.0, 0.0, 1.0, 0.0}),
        (r_mixed<value_type, typename TestFixture::value_type>()));
}


TYPED_TEST(Jacobi, AppliesToComplexVector)
{
    using Vec = gko::to_complex<typename TestFixture::Vec>;
    using value_type = typename Vec::value_type;
    auto x = gko::initialize<Vec>(
        {value_type{1.0, 2.0}, value_type{-1.0, -2.0}, value_type{2.0, 4.0},
         value_type{-2.0, -4.0}, value_type{3.0, 6.0}},
        this->exec);
    auto b = gko::initialize<Vec>(
        {value_type{4.0, 8.0}, value_type{-1.0, -2.0}, value_type{-2.0, -4.0},
         value_type{4.0, 8.0}, value_type{-1.0, -2.0}},
        this->exec);
    auto bj = this->bj_factory->generate(this->mtx);

    bj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({value_type{1.0, 2.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0},
           value_type{1.0, 2.0}, value_type{0.0, 0.0}}),
        r<value_type>::value);
}


TYPED_TEST(Jacobi, AppliesToMixedComplexVector)
{
    using value_type =
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>;
    using Vec = gko::matrix::Dense<value_type>;
    auto x = gko::initialize<Vec>(
        {value_type{1.0, 2.0}, value_type{-1.0, -2.0}, value_type{2.0, 4.0},
         value_type{-2.0, -4.0}, value_type{3.0, 6.0}},
        this->exec);
    auto b = gko::initialize<Vec>(
        {value_type{4.0, 8.0}, value_type{-1.0, -2.0}, value_type{-2.0, -4.0},
         value_type{4.0, 8.0}, value_type{-1.0, -2.0}},
        this->exec);
    auto bj = this->bj_factory->generate(this->mtx);

    bj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({value_type{1.0, 2.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0},
           value_type{1.0, 2.0}, value_type{0.0, 0.0}}),
        (r_mixed<value_type, typename TestFixture::value_type>()));
}


TYPED_TEST(Jacobi, AppliesToVectorWithAdaptivePrecision)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto bj = this->adaptive_bj_factory->generate(this->mtx);

    bj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 0.0, 0.0, 1.0, 0.0}), half_tol);
}


TYPED_TEST(Jacobi, AppliesToVectorWithAdaptivePrecisionAndSmallBlocks)
{
    using Bj = typename TestFixture::Bj;
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  // group size will be > 1
                  .with_block_pointers(this->block_pointers)
                  .with_storage_optimization(this->block_precisions)
                  .on(this->exec)
                  ->generate(this->mtx);

    bj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 0.0, 0.0, 1.0, 0.0}), half_tol);
}


TYPED_TEST(Jacobi, AppliesToMultipleVectors)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto x =
        gko::initialize<Vec>(3,
                             {I<T>{1.0, 0.5}, I<T>{-1.0, -0.5}, I<T>{2.0, 1.0},
                              I<T>{-2.0, -1.0}, I<T>{3.0, 1.5}},
                             this->exec);
    auto b =
        gko::initialize<Vec>(3,
                             {I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}, I<T>{-2.0, 0.0},
                              I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}},
                             this->exec);
    auto bj = this->bj_factory->generate(this->mtx);

    bj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}),
        r<value_type>::value);
}


TYPED_TEST(Jacobi, ScalarJacobiAppliesToMultipleVectors)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto x =
        gko::initialize<Vec>(3,
                             {I<T>{1.0, 0.5}, I<T>{-1.0, -0.5}, I<T>{2.0, 1.0},
                              I<T>{-2.0, -1.0}, I<T>{3.0, 1.5}},
                             this->exec);
    auto b =
        gko::initialize<Vec>(3,
                             {I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}, I<T>{-2.0, 0.0},
                              I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}},
                             this->exec);
    auto sj = this->scalar_j_factory->generate(this->mtx);

    sj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{1.0, -0.5}, {-0.25, 1.0}, {-0.5, 0.0}, {1.0, -0.5}, {-0.25, 1.0}}),
        r<value_type>::value);
}


TYPED_TEST(Jacobi, AppliesToMultipleVectorsWithAdaptivePrecision)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto x =
        gko::initialize<Vec>(3,
                             {I<T>{1.0, 0.5}, I<T>{-1.0, -0.5}, I<T>{2.0, 1.0},
                              I<T>{-2.0, -1.0}, I<T>{3.0, 1.5}},
                             this->exec);
    auto b =
        gko::initialize<Vec>(3,
                             {I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}, I<T>{-2.0, 0.0},
                              I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}},
                             this->exec);
    auto bj = this->adaptive_bj_factory->generate(this->mtx);

    bj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}),
        half_tol);
}


TYPED_TEST(Jacobi, AppliesToMultipleVectorsWithAdaptivePrecisionAndSmallBlocks)
{
    using Vec = typename TestFixture::Vec;
    using Bj = typename TestFixture::Bj;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto x =
        gko::initialize<Vec>(3,
                             {I<T>{1.0, 0.5}, I<T>{-1.0, -0.5}, I<T>{2.0, 1.0},
                              I<T>{-2.0, -1.0}, I<T>{3.0, 1.5}},
                             this->exec);
    auto b =
        gko::initialize<Vec>(3,
                             {I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}, I<T>{-2.0, 0.0},
                              I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}},
                             this->exec);
    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  // group size will be > 1
                  .with_block_pointers(this->block_pointers)
                  .with_storage_optimization(this->block_precisions)
                  .on(this->exec)
                  ->generate(this->mtx);

    bj->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}),
        half_tol);
}


TYPED_TEST(Jacobi, AppliesLinearCombinationToVector)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto bj = this->bj_factory->generate(this->mtx);

    bj->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 1.0, -2.0, 4.0, -3.0}),
                        r<value_type>::value);
}


TYPED_TEST(Jacobi, ScalarJacobiAppliesLinearCombinationToVector)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto sj = this->scalar_j_factory->generate(this->mtx);

    sj->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 0.5, -3.0, 4.0, -3.5}),
                        r<value_type>::value);
}


TYPED_TEST(Jacobi, AppliesLinearCombinationToMixedVector)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto bj = this->bj_factory->generate(this->mtx);

    bj->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({1.0, 1.0, -2.0, 4.0, -3.0}),
        (r_mixed<value_type, typename TestFixture::value_type>()));
}


TYPED_TEST(Jacobi, AppliesLinearCombinationToComplexVector)
{
    using Dense = typename TestFixture::Vec;
    using DenseComplex = gko::to_complex<Dense>;
    using value_type = typename TestFixture::value_type;
    using T = gko::to_complex<value_type>;
    auto x = gko::initialize<DenseComplex>(
        {T{1.0, 2.0}, T{-1.0, -2.0}, T{2.0, 4.0}, T{-2.0, -4.0}, T{3.0, 6.0}},
        this->exec);
    auto b = gko::initialize<DenseComplex>(
        {T{4.0, 8.0}, T{-1.0, -2.0}, T{-2.0, -4.0}, T{4.0, 8.0}, T{-1.0, -2.0}},
        this->exec);
    auto alpha = gko::initialize<Dense>({2.0}, this->exec);
    auto beta = gko::initialize<Dense>({-1.0}, this->exec);
    auto bj = this->bj_factory->generate(this->mtx);

    bj->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({T{1.0, 2.0}, T{1.0, 2.0}, T{-2.0, -4.0}, T{4.0, 8.0},
                           T{-3.0, -6.0}}),
                        r<value_type>::value);
}


TYPED_TEST(Jacobi, AppliesLinearCombinationToMixedComplexVector)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using MixedDense = gko::matrix::Dense<value_type>;
    using MixedDenseComplex = gko::to_complex<MixedDense>;
    using T = gko::to_complex<value_type>;
    auto x = gko::initialize<MixedDenseComplex>(
        {T{1.0, 2.0}, T{-1.0, -2.0}, T{2.0, 4.0}, T{-2.0, -4.0}, T{3.0, 6.0}},
        this->exec);
    auto b = gko::initialize<MixedDenseComplex>(
        {T{4.0, 8.0}, T{-1.0, -2.0}, T{-2.0, -4.0}, T{4.0, 8.0}, T{-1.0, -2.0}},
        this->exec);
    auto alpha = gko::initialize<MixedDense>({2.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto bj = this->bj_factory->generate(this->mtx);

    bj->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({T{1.0, 2.0}, T{1.0, 2.0}, T{-2.0, -4.0}, T{4.0, 8.0},
           T{-3.0, -6.0}}),
        (r_mixed<value_type, typename TestFixture::value_type>()));
}


TYPED_TEST(Jacobi, AppliesLinearCombinationToVectorWithAdaptivePrecision)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto bj = this->adaptive_bj_factory->generate(this->mtx);

    bj->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 1.0, -2.0, 4.0, -3.0}), half_tol);
}


TYPED_TEST(Jacobi, AppliesLinearCombinationToMultipleVectors)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto x =
        gko::initialize<Vec>(3,
                             {I<T>{1.0, 0.5}, I<T>{-1.0, -0.5}, I<T>{2.0, 1.0},
                              I<T>{-2.0, -1.0}, I<T>{3.0, 1.5}},
                             this->exec);
    auto b =
        gko::initialize<Vec>(3,
                             {I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}, I<T>{-2.0, 0.0},
                              I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}},
                             this->exec);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto bj = this->bj_factory->generate(this->mtx);

    bj->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, -0.5}, {1.0, 2.5}, {-2.0, -1.0}, {4.0, 1.0}, {-3.0, 0.5}}),
        r<value_type>::value);
}


TYPED_TEST(Jacobi, ScalarAppliesLinearCombinationToMultipleVectors)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto x =
        gko::initialize<Vec>(3,
                             {I<T>{1.0, 0.5}, I<T>{-1.0, -0.5}, I<T>{2.0, 1.0},
                              I<T>{-2.0, -1.0}, I<T>{3.0, 1.5}},
                             this->exec);
    auto b =
        gko::initialize<Vec>(3,
                             {I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}, I<T>{-2.0, 0.0},
                              I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}},
                             this->exec);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto sj = this->scalar_j_factory->generate(this->mtx);

    sj->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, -1.5}, {0.5, 2.5}, {-3.0, -1.0}, {4.0, 0.0}, {-3.5, 0.5}}),
        r<value_type>::value);
}


TYPED_TEST(Jacobi,
           AppliesLinearCombinationToMultipleVectorsWithAdaptivePrecision)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto x =
        gko::initialize<Vec>(3,
                             {I<T>{1.0, 0.5}, I<T>{-1.0, -0.5}, I<T>{2.0, 1.0},
                              I<T>{-2.0, -1.0}, I<T>{3.0, 1.5}},
                             this->exec);
    auto b =
        gko::initialize<Vec>(3,
                             {I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}, I<T>{-2.0, 0.0},
                              I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}},
                             this->exec);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto bj = this->adaptive_bj_factory->generate(this->mtx);

    bj->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, -0.5}, {1.0, 2.5}, {-2.0, -1.0}, {4.0, 1.0}, {-3.0, 0.5}}),
        half_tol);
}


TYPED_TEST(Jacobi, ConvertsToDense)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto dense = Vec::create(this->exec);

    dense->move_from(this->bj_factory->generate(this->mtx));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense,
        l({{4.0 / 14, 2.0 / 14,       0.0,       0.0,       0.0},
           {1.0 / 14, 4.0 / 14,       0.0,       0.0,       0.0},
           {     0.0,      0.0, 14.0 / 48,  8.0 / 48,  4.0 / 48},
           {     0.0,      0.0,  4.0 / 48, 16.0 / 48,  8.0 / 48},
           {     0.0,      0.0,  1.0 / 48,  4.0 / 48, 14.0 / 48}}), r<value_type>::value);
    // clang-format on
}


TYPED_TEST(Jacobi, ConvertsToDenseWithAdaptivePrecision)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto dense = Vec::create(this->exec);

    dense->move_from(this->adaptive_bj_factory->generate(this->mtx));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense,
        l({{4.0 / 14, 2.0 / 14,       0.0,       0.0,       0.0},
           {1.0 / 14, 4.0 / 14,       0.0,       0.0,       0.0},
           {     0.0,      0.0, 14.0 / 48,  8.0 / 48,  4.0 / 48},
           {     0.0,      0.0,  4.0 / 48, 16.0 / 48,  8.0 / 48},
           {     0.0,      0.0,  1.0 / 48,  4.0 / 48, 14.0 / 48}}), half_tol);
    // clang-format on
}


TYPED_TEST(Jacobi, ConvertsEmptyToDense)
{
    using Vec = typename TestFixture::Vec;
    auto empty = gko::share(Vec::create(this->exec));
    auto res = Vec::create(this->exec);

    res->move_from(TestFixture::Bj::build().on(this->exec)->generate(empty));

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Jacobi, ScalarL1Diag)
{
    /* example matrix:
        1   1   0   0
        2   2   0   0
       -5   0   1   0
        1   0  -1   0
     */
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto mtx = Mtx::create(this->exec, gko::dim<2>{4, 4}, 8);
    this->template init_array<index_type>(mtx->get_row_ptrs(), {0, 2, 4, 6, 8});
    this->template init_array<index_type>(mtx->get_col_idxs(),
                                          {0, 1, 0, 1, 0, 2, 0, 2});
    this->template init_array<value_type>(
        mtx->get_values(), {1.0, 1.0, 2.0, 2.0, -5.0, 1.0, 1.0, -1.0});
    auto diag = mtx->extract_diagonal();
    auto diag_val = diag->get_const_values();

    gko::kernels::reference::jacobi::scalar_l1(this->exec, mtx.get(),
                                               diag.get());

    EXPECT_EQ(diag_val[0], value_type{2.0});
    EXPECT_EQ(diag_val[1], value_type{4.0});
    EXPECT_EQ(diag_val[2], value_type{6.0});
    EXPECT_EQ(diag_val[3], value_type{2.0});
}


TYPED_TEST(Jacobi, BlockL1)
{
    /* example matrix:
        1 | 1
       ---+---+
        2 | 2 |
          +---+--------
       -5     |  1
        1     | -1   1
     */
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto mtx = Mtx::create(this->exec, gko::dim<2>{4, 4}, 9);
    this->template init_array<index_type>(mtx->get_row_ptrs(), {0, 2, 4, 6, 9});
    this->template init_array<index_type>(mtx->get_col_idxs(),
                                          {0, 1, 0, 1, 0, 2, 0, 2, 3});
    this->template init_array<value_type>(
        mtx->get_values(), {1.0, 1.0, 2.0, 2.0, -5.0, 1.0, 1.0, -1.0, 1});
    gko::array<index_type> block_ptr(this->exec, {0, 1, 2, 4});

    gko::kernels::reference::jacobi::block_l1(this->exec, 3, block_ptr,
                                              mtx.get());

    GKO_ASSERT_MTX_NEAR(mtx,
                        l({{2.0, 1.0, 0.0, 0.0},
                           {2.0, 4.0, 0.0, 0.0},
                           {-5.0, 0.0, 6.0, 0.0},
                           {1.0, 0.0, -1.0, 2.0}}),
                        0.0);
}


TYPED_TEST(Jacobi, L1BlockJaocbiConvertsToDense)
{
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto dense = Vec::create(this->exec);
    auto bj_factory = Bj::build()
                          .with_max_block_size(3u)
                          .with_block_pointers(this->block_pointers)
                          .with_aggregate_l1(true)
                          .on(this->exec);
    // after L1 modification, it will be the same as the matrix in other tests
    /* test matrix:
        2  -2 |        -2
        -1  4 |
       -------+----------
              | 4  -2
              |-1   4  -2
       -4     |    -1
    */
    auto mtx = gko::share(Mtx::create(this->exec, gko::dim<2>{5}, 12));
    this->template init_array<index_type>(mtx->get_row_ptrs(),
                                          {0, 3, 5, 7, 10, 12});
    this->template init_array<index_type>(mtx->get_col_idxs(),
                                          {0, 1, 4, 0, 1, 2, 3, 2, 3, 4, 0, 3});
    this->template init_array<value_type>(
        mtx->get_values(),
        {2.0, -2.0, -2.0, -1.0, 4.0, 4.0, -2.0, -1.0, 4.0, -2.0, -4.0, -1.0});

    dense->move_from(bj_factory->generate(mtx));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense,
        l({{4.0 / 14, 2.0 / 14,       0.0,       0.0,       0.0},
           {1.0 / 14, 4.0 / 14,       0.0,       0.0,       0.0},
           {     0.0,      0.0, 14.0 / 48,  8.0 / 48,  4.0 / 48},
           {     0.0,      0.0,  4.0 / 48, 16.0 / 48,  8.0 / 48},
           {     0.0,      0.0,  1.0 / 48,  4.0 / 48, 14.0 / 48}}), r<value_type>::value);
    // clang-format on
}


TYPED_TEST(Jacobi, L1BlockJaocbiConvertsToDenseWithAdaptivePrecision)
{
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto dense = Vec::create(this->exec);
    auto bj_factory = Bj::build()
                          .with_max_block_size(17u)
                          // make sure group size is 1
                          .with_block_pointers(this->block_pointers)
                          .with_storage_optimization(this->block_precisions)
                          .with_aggregate_l1(true)
                          .on(this->exec);
    // after L1 modification, it will be the same as the matrix in other tests
    /* test matrix:
        2  -2 |        -2
        -1  4 |
       -------+----------
              | 4  -2
              |-1   4  -2
       -4     |    -1
    */
    auto mtx = gko::share(Mtx::create(this->exec, gko::dim<2>{5}, 12));
    this->template init_array<index_type>(mtx->get_row_ptrs(),
                                          {0, 3, 5, 7, 10, 12});
    this->template init_array<index_type>(mtx->get_col_idxs(),
                                          {0, 1, 4, 0, 1, 2, 3, 2, 3, 4, 0, 3});
    this->template init_array<value_type>(
        mtx->get_values(),
        {2.0, -2.0, -2.0, -1.0, 4.0, 4.0, -2.0, -1.0, 4.0, -2.0, -4.0, -1.0});

    dense->move_from(bj_factory->generate(mtx));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense,
        l({{4.0 / 14, 2.0 / 14,       0.0,       0.0,       0.0},
           {1.0 / 14, 4.0 / 14,       0.0,       0.0,       0.0},
           {     0.0,      0.0, 14.0 / 48,  8.0 / 48,  4.0 / 48},
           {     0.0,      0.0,  4.0 / 48, 16.0 / 48,  8.0 / 48},
           {     0.0,      0.0,  1.0 / 48,  4.0 / 48, 14.0 / 48}}), half_tol);
    // clang-format on
}


}  // namespace
