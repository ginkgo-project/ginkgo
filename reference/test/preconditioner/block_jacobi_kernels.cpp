/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/preconditioner/block_jacobi.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <core/matrix/csr.hpp>
#include <core/matrix/dense.hpp>
#include <core/test/utils.hpp>


namespace {


template <typename ConcreteBlockJacobiFactory>
class BasicBlockJacobiTest : public ::testing::Test {
protected:
    using BjFactory = ConcreteBlockJacobiFactory;
    using Bj = typename ConcreteBlockJacobiFactory::generated_type;
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;

    BasicBlockJacobiTest()
        : exec(gko::ReferenceExecutor::create()),
          bj_factory(BjFactory::create(exec, 3)),
          block_pointers(exec, 3),
          mtx(gko::matrix::Csr<>::create(exec, gko::dim<2>{5}, 13))
    {
        block_pointers.get_data()[0] = 0;
        block_pointers.get_data()[1] = 2;
        block_pointers.get_data()[2] = 5;
        /* test matrix:
            4  -2 |        -2
           -1   4 |
           -------+----------
                  | 4  -2
                  |-1   4  -2
           -1     |    -1   4
         */
        init_array(mtx->get_row_ptrs(), {0, 3, 5, 7, 10, 13});
        init_array(mtx->get_col_idxs(),
                   {0, 1, 4, 0, 1, 2, 3, 2, 3, 4, 0, 3, 4});
        init_array(mtx->get_values(), {4.0, -2.0, -2.0, -1.0, 4.0, 4.0, -2.0,
                                       -1.0, 4.0, -2.0, -1.0, -1.0, 4.0});
    }

    template <typename T>
    void init_array(T *arr, std::initializer_list<T> vals)
    {
        for (auto elem : vals) {
            *(arr++) = elem;
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<BjFactory> bj_factory;
    gko::Array<gko::int32> block_pointers;
    std::shared_ptr<gko::matrix::Csr<>> mtx;
    std::unique_ptr<gko::LinOp> bj_lin_op;
    Bj *bj;
};


class BlockJacobi
    : public BasicBlockJacobiTest<gko::preconditioner::BlockJacobiFactory<>> {};


TEST_F(BlockJacobi, CanBeGenerated)
{
    bj_factory->set_block_pointers(block_pointers);
    bj_lin_op = bj_factory->generate(mtx);
    bj = static_cast<Bj *>(bj_lin_op.get());

    ASSERT_NE(bj, nullptr);
    EXPECT_EQ(bj->get_executor(), exec);
    EXPECT_EQ(bj->get_max_block_size(), 3);
    ASSERT_EQ(bj->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_const_block_pointers();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    ASSERT_EQ(ptrs[2], 5);
}


TEST_F(BlockJacobi, FindsNaturalBlocks)
{
    /* example matrix:
        1   1
        1   1
        1       1
        1       1
     */
    auto mtx = Mtx::create(exec, gko::dim<2>{4}, 8);
    init_array(mtx->get_row_ptrs(), {0, 2, 4, 6, 8});
    init_array(mtx->get_col_idxs(), {0, 1, 0, 1, 0, 2, 0, 2});
    init_array(mtx->get_values(), {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    bj_lin_op = bj_factory->generate(std::move(mtx));
    bj = static_cast<Bj *>(bj_lin_op.get());

    EXPECT_EQ(bj->get_max_block_size(), 3);
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_const_block_pointers();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    EXPECT_EQ(ptrs[2], 4);
}


TEST_F(BlockJacobi, ExecutesSupervariableAgglomeration)
{
    /* example matrix:
        1   1
        1   1
                1   1
                1   1
                        1
     */
    auto mtx = Mtx::create(exec, gko::dim<2>{5}, 9);
    init_array(mtx->get_row_ptrs(), {0, 2, 4, 6, 8, 9});
    init_array(mtx->get_col_idxs(), {0, 1, 0, 1, 2, 3, 2, 3, 4});
    init_array(mtx->get_values(),
               {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    bj_lin_op = bj_factory->generate(std::move(mtx));
    bj = static_cast<Bj *>(bj_lin_op.get());

    EXPECT_EQ(bj->get_max_block_size(), 3);
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_const_block_pointers();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    EXPECT_EQ(ptrs[2], 5);
}


TEST_F(BlockJacobi, AdheresToBlockSizeBound)
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
    auto mtx = Mtx::create(exec, gko::dim<2>{7}, 7);
    init_array(mtx->get_row_ptrs(), {0, 1, 2, 3, 4, 5, 6, 7});
    init_array(mtx->get_col_idxs(), {0, 1, 2, 3, 4, 5, 6});
    init_array(mtx->get_values(), {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    bj_lin_op = bj_factory->generate(std::move(mtx));
    bj = static_cast<Bj *>(bj_lin_op.get());

    EXPECT_EQ(bj->get_max_block_size(), 3);
    ASSERT_EQ(bj->get_num_blocks(), 3);
    auto ptrs = bj->get_const_block_pointers();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 3);
    EXPECT_EQ(ptrs[2], 6);
    EXPECT_EQ(ptrs[3], 7);
}


TEST_F(BlockJacobi, CanBeGeneratedWithUnknownBlockSizes)
{
    bj_lin_op = bj_factory->generate(mtx);
    bj = dynamic_cast<Bj *>(bj_lin_op.get());

    ASSERT_NE(bj, nullptr);
    EXPECT_EQ(bj->get_executor(), exec);
    EXPECT_EQ(bj->get_max_block_size(), 3);
    ASSERT_EQ(bj->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_const_block_pointers();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 3);
    ASSERT_EQ(ptrs[2], 5);
}


TEST_F(BlockJacobi, InvertsDiagonalBlocks)
{
    bj_factory->set_block_pointers(block_pointers);

    bj_lin_op = bj_factory->generate(mtx);

    bj = static_cast<Bj *>(bj_lin_op.get());
    auto p = bj->get_stride();
    auto b1 = bj->get_blocks();
    EXPECT_NEAR(b1[0 * p + 0], 4.0 / 14.0, 1e-14);
    EXPECT_NEAR(b1[0 * p + 1], 2.0 / 14.0, 1e-14);
    EXPECT_NEAR(b1[1 * p + 0], 1.0 / 14.0, 1e-14);
    EXPECT_NEAR(b1[1 * p + 1], 4.0 / 14.0, 1e-14);

    auto b2 = bj->get_blocks() + 2 * p;
    EXPECT_NEAR(b2[0 * p + 0], 14.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[0 * p + 1], 8.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[0 * p + 2], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 * p + 0], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 * p + 1], 16.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 * p + 2], 8.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 * p + 0], 1.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 * p + 1], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 * p + 2], 14.0 / 48.0, 1e-14);
}


TEST_F(BlockJacobi, PivotsWhenInvertingBlock)
{
    gko::Array<gko::int32> bp(exec, 2);
    init_array(bp.get_data(), {0, 3});
    bj_factory->set_block_pointers(bp);
    auto mtx = Mtx::create(exec, gko::dim<2>{3}, 9);
    /* test matrix:
       0 2 0
       0 0 4
       1 0 0
     */
    init_array(mtx->get_row_ptrs(), {0, 3, 6, 9});
    init_array(mtx->get_col_idxs(), {0, 1, 2, 0, 1, 2, 0, 1, 2});
    init_array(mtx->get_values(),
               {0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0});

    bj_lin_op = bj_factory->generate(std::move(mtx));

    bj = static_cast<Bj *>(bj_lin_op.get());
    auto p = bj->get_stride();
    auto b1 = bj->get_blocks();
    EXPECT_NEAR(b1[0 * p + 0], 0.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[0 * p + 1], 0.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[0 * p + 2], 4.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[1 * p + 0], 2.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[1 * p + 1], 0.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[1 * p + 2], 0.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[2 * p + 0], 0.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[2 * p + 1], 1.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[2 * p + 2], 0.0 / 4.0, 1e-14);
}


TEST_F(BlockJacobi, AppliesToVector)
{
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, exec);
    bj_factory->set_block_pointers(block_pointers);
    auto bj = bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({1.0, 0.0, 0.0, 1.0, 0.0}), 1e-14);
}


TEST_F(BlockJacobi, AppliesToMultipleVectors)
{
    auto x = gko::initialize<Vec>(
        3, {{1.0, 0.5}, {-1.0, -0.5}, {2.0, 1.0}, {-2.0, -1.0}, {3.0, 1.5}},
        exec);
    auto b = gko::initialize<Vec>(
        3, {{4.0, -2.0}, {-1.0, 4.0}, {-2.0, 0.0}, {4.0, -2.0}, {-1.0, 4.0}},
        exec);
    bj_factory->set_block_pointers(block_pointers);
    auto bj = bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(
        x, l({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}),
        1e-14);
}


TEST_F(BlockJacobi, AppliesLinearCombinationToVector)
{
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, exec);
    auto alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, exec);
    bj_factory->set_block_pointers(block_pointers);
    auto bj = bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(x, l({1.0, 1.0, -2.0, 4.0, -3.0}), 1e-14);
}


TEST_F(BlockJacobi, AppliesLinearCombinationToMultipleVectors)
{
    auto x = gko::initialize<Vec>(
        3, {{1.0, 0.5}, {-1.0, -0.5}, {2.0, 1.0}, {-2.0, -1.0}, {3.0, 1.5}},
        exec);
    auto b = gko::initialize<Vec>(
        3, {{4.0, -2.0}, {-1.0, 4.0}, {-2.0, 0.0}, {4.0, -2.0}, {-1.0, 4.0}},
        exec);
    auto alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, exec);
    bj_factory->set_block_pointers(block_pointers);
    auto bj = bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(
        x, l({{1.0, -0.5}, {1.0, 2.5}, {-2.0, -1.0}, {4.0, 1.0}, {-3.0, 0.5}}),
        1e-14);
}


TEST_F(BlockJacobi, ConvertsToDense)
{
    auto dense = gko::matrix::Dense<>::create(exec);
    bj_factory->set_block_pointers(block_pointers);

    dense->copy_from(bj_factory->generate(mtx));

    // clang-format off
    ASSERT_MTX_NEAR(dense,
        l({{4.0 / 14, 2.0 / 14,       0.0,       0.0,       0.0},
           {1.0 / 14, 4.0 / 14,       0.0,       0.0,       0.0},
           {     0.0,      0.0, 14.0 / 48,  8.0 / 48,  4.0 / 48},
           {     0.0,      0.0,  4.0 / 48, 16.0 / 48,  8.0 / 48},
           {     0.0,      0.0,  1.0 / 48,  4.0 / 48, 14.0 / 48}}), 1e-14);
    // clang-format on
}


class AdaptiveBlockJacobi
    : public BasicBlockJacobiTest<
          gko::preconditioner::AdaptiveBlockJacobiFactory<>> {
protected:
    AdaptiveBlockJacobi() : block_precisions(exec, 2)
    {
        block_precisions.get_data()[0] = Bj::single_precision;
        block_precisions.get_data()[1] = Bj::double_precision;
    }

    gko::Array<Bj::precision> block_precisions;
};


// TODO: take into account different precisions in the following tests


TEST_F(AdaptiveBlockJacobi, CanBeGenerated)
{
    bj_factory->set_block_pointers(block_pointers);
    bj_factory->set_block_precisions(block_precisions);
    bj_lin_op = bj_factory->generate(mtx);
    bj = static_cast<Bj *>(bj_lin_op.get());

    ASSERT_NE(bj, nullptr);
    EXPECT_EQ(bj->get_executor(), exec);
    EXPECT_EQ(bj->get_max_block_size(), 3);
    ASSERT_EQ(bj->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_const_block_pointers();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    ASSERT_EQ(ptrs[2], 5);
    auto prec = bj->get_const_block_precisions();
    EXPECT_EQ(prec[0], Bj::single_precision);
    ASSERT_EQ(prec[1], Bj::double_precision);
}


TEST_F(AdaptiveBlockJacobi, InvertsDiagonalBlocks)
{
    bj_factory->set_block_pointers(block_pointers);
    bj_factory->set_block_precisions(block_precisions);

    bj_lin_op = bj_factory->generate(mtx);

    bj = static_cast<Bj *>(bj_lin_op.get());
    auto p = bj->get_stride();
    auto b1 = reinterpret_cast<const float *>(bj->get_const_blocks());
    EXPECT_NEAR(b1[0 * p + 0], 4.0 / 14.0, 1e-7);
    EXPECT_NEAR(b1[0 * p + 1], 2.0 / 14.0, 1e-7);
    EXPECT_NEAR(b1[1 * p + 0], 1.0 / 14.0, 1e-7);
    EXPECT_NEAR(b1[1 * p + 1], 4.0 / 14.0, 1e-7);

    auto b2 = bj->get_blocks() + 2 * p;
    EXPECT_NEAR(b2[0 * p + 0], 14.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[0 * p + 1], 8.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[0 * p + 2], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 * p + 0], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 * p + 1], 16.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 * p + 2], 8.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 * p + 0], 1.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 * p + 1], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 * p + 2], 14.0 / 48.0, 1e-14);
}


TEST_F(AdaptiveBlockJacobi, PivotsWhenInvertingBlock)
{
    gko::Array<gko::int32> bp(exec, 2);
    init_array(bp.get_data(), {0, 3});
    bj_factory->set_block_pointers(bp);
    bj_factory->set_block_precisions(block_precisions);
    auto mtx = Mtx::create(exec, gko::dim<2>{3}, 9);
    /* test matrix:
       0 2 0
       0 0 4
       1 0 0
     */
    init_array(mtx->get_row_ptrs(), {0, 3, 6, 9});
    init_array(mtx->get_col_idxs(), {0, 1, 2, 0, 1, 2, 0, 1, 2});
    init_array(mtx->get_values(),
               {0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0});

    bj_lin_op = bj_factory->generate(std::move(mtx));

    bj = static_cast<Bj *>(bj_lin_op.get());
    auto p = bj->get_stride();
    auto b1 = reinterpret_cast<const float *>(bj->get_const_blocks());
    EXPECT_NEAR(b1[0 * p + 0], 0.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[0 * p + 1], 0.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[0 * p + 2], 4.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[1 * p + 0], 2.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[1 * p + 1], 0.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[1 * p + 2], 0.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[2 * p + 0], 0.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[2 * p + 1], 1.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[2 * p + 2], 0.0 / 4.0, 1e-7);
}


TEST_F(AdaptiveBlockJacobi, AppliesToVector)
{
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, exec);
    bj_factory->set_block_pointers(block_pointers);
    bj_factory->set_block_precisions(block_precisions);
    auto bj = bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({1.0, 0.0, 0.0, 1.0, 0.0}), 1e-7);
}


TEST_F(AdaptiveBlockJacobi, AppliesToMultipleVectors)
{
    auto x = gko::initialize<Vec>(
        3, {{1.0, 0.5}, {-1.0, -0.5}, {2.0, 1.0}, {-2.0, -1.0}, {3.0, 1.5}},
        exec);
    auto b = gko::initialize<Vec>(
        3, {{4.0, -2.0}, {-1.0, 4.0}, {-2.0, 0.0}, {4.0, -2.0}, {-1.0, 4.0}},
        exec);
    bj_factory->set_block_pointers(block_pointers);
    bj_factory->set_block_precisions(block_precisions);
    auto bj = bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(
        x, l({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}),
        1e-7);
}


TEST_F(AdaptiveBlockJacobi, AppliesLinearCombinationToVector)
{
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, exec);
    auto alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, exec);
    bj_factory->set_block_pointers(block_pointers);
    bj_factory->set_block_precisions(block_precisions);
    auto bj = bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(x, l({1.0, 1.0, -2.0, 4.0, -3.0}), 1e-7);
}


TEST_F(AdaptiveBlockJacobi, AppliesLinearCombinationToMultipleVectors)
{
    auto x = gko::initialize<Vec>(
        3, {{1.0, 0.5}, {-1.0, -0.5}, {2.0, 1.0}, {-2.0, -1.0}, {3.0, 1.5}},
        exec);
    auto b = gko::initialize<Vec>(
        3, {{4.0, -2.0}, {-1.0, 4.0}, {-2.0, 0.0}, {4.0, -2.0}, {-1.0, 4.0}},
        exec);
    auto alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, exec);
    bj_factory->set_block_pointers(block_pointers);
    bj_factory->set_block_precisions(block_precisions);
    auto bj = bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(
        x, l({{1.0, -0.5}, {1.0, 2.5}, {-2.0, -1.0}, {4.0, 1.0}, {-3.0, 0.5}}),
        1e-7);
}


TEST_F(AdaptiveBlockJacobi, ConvertsToDense)
{
    auto dense = gko::matrix::Dense<>::create(exec);
    bj_factory->set_block_pointers(block_pointers);
    bj_factory->set_block_precisions(block_precisions);

    dense->copy_from(bj_factory->generate(mtx));

    // clang-format off
    ASSERT_MTX_NEAR(dense,
        l({{4.0 / 14, 2.0 / 14,       0.0,       0.0,       0.0},
           {1.0 / 14, 4.0 / 14,       0.0,       0.0,       0.0},
           {     0.0,      0.0, 14.0 / 48,  8.0 / 48,  4.0 / 48},
           {     0.0,      0.0,  4.0 / 48, 16.0 / 48,  8.0 / 48},
           {     0.0,      0.0,  1.0 / 48,  4.0 / 48, 14.0 / 48}}), 1e-7);
    // clang-format on
}


}  // namespace
