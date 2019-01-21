/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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

#include <ginkgo/core/preconditioner/jacobi.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class Jacobi : public ::testing::Test {
protected:
    using Bj = gko::preconditioner::Jacobi<>;
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;
    using mdata = gko::matrix_data<>;

    Jacobi()
        : exec(gko::ReferenceExecutor::create()),
          block_pointers(exec, 3),
          block_precisions(exec, 2),
          mtx(gko::matrix::Csr<>::create(exec, gko::dim<2>{5}, 13))
    {
        block_pointers.get_data()[0] = 0;
        block_pointers.get_data()[1] = 2;
        block_pointers.get_data()[2] = 5;
        block_precisions.get_data()[0] = gko::precision_reduction(0, 1);
        block_precisions.get_data()[1] = gko::precision_reduction(0, 0);
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
    std::unique_ptr<Bj::Factory> bj_factory;
    std::unique_ptr<Bj::Factory> adaptive_bj_factory;
    gko::Array<gko::int32> block_pointers;
    gko::Array<gko::precision_reduction> block_precisions;
    std::shared_ptr<gko::matrix::Csr<>> mtx;
};


TEST_F(Jacobi, CanBeGenerated)
{
    auto bj = bj_factory->generate(mtx);

    ASSERT_NE(bj, nullptr);
    EXPECT_EQ(bj->get_executor(), exec);
    EXPECT_EQ(bj->get_parameters().max_block_size, 3);
    ASSERT_EQ(bj->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    ASSERT_EQ(ptrs[2], 5);
}


TEST_F(Jacobi, CanBeGeneratedWithAdaptivePrecision)
{
    auto bj = adaptive_bj_factory->generate(mtx);

    EXPECT_EQ(bj->get_executor(), exec);
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


TEST_F(Jacobi, FindsNaturalBlocks)
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
    auto bj = Bj::build().with_max_block_size(3u).on(exec)->generate(give(mtx));

    EXPECT_EQ(bj->get_parameters().max_block_size, 3);
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    EXPECT_EQ(ptrs[2], 4);
}


TEST_F(Jacobi, ExecutesSupervariableAgglomeration)
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
    auto bj = Bj::build().with_max_block_size(3u).on(exec)->generate(give(mtx));

    EXPECT_EQ(bj->get_parameters().max_block_size, 3);
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    EXPECT_EQ(ptrs[2], 5);
}


TEST_F(Jacobi, AdheresToBlockSizeBound)
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
    auto bj = Bj::build().with_max_block_size(3u).on(exec)->generate(give(mtx));

    EXPECT_EQ(bj->get_parameters().max_block_size, 3);
    ASSERT_EQ(bj->get_num_blocks(), 3);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 3);
    EXPECT_EQ(ptrs[2], 6);
    EXPECT_EQ(ptrs[3], 7);
}


TEST_F(Jacobi, CanBeGeneratedWithUnknownBlockSizes)
{
    auto bj = Bj::build().with_max_block_size(3u).on(exec)->generate(mtx);

    ASSERT_NE(bj, nullptr);
    EXPECT_EQ(bj->get_executor(), exec);
    EXPECT_EQ(bj->get_parameters().max_block_size, 3);
    ASSERT_EQ(bj->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_parameters().block_pointers.get_const_data();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 3);
    ASSERT_EQ(ptrs[2], 5);
}


TEST_F(Jacobi, InvertsDiagonalBlocks)
{
    auto bj = bj_factory->generate(mtx);

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    auto b1 = bj->get_blocks() + scheme.get_global_block_offset(0);
    EXPECT_NEAR(b1[0 + 0 * p], 4.0 / 14.0, 1e-14);
    EXPECT_NEAR(b1[0 + 1 * p], 2.0 / 14.0, 1e-14);
    EXPECT_NEAR(b1[1 + 0 * p], 1.0 / 14.0, 1e-14);
    EXPECT_NEAR(b1[1 + 1 * p], 4.0 / 14.0, 1e-14);

    auto b2 = bj->get_blocks() + scheme.get_global_block_offset(1);
    EXPECT_NEAR(b2[0 + 0 * p], 14.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[0 + 1 * p], 8.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[0 + 2 * p], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 + 0 * p], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 + 1 * p], 16.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 + 2 * p], 8.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 + 0 * p], 1.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 + 1 * p], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 + 2 * p], 14.0 / 48.0, 1e-14);
}

TEST_F(Jacobi, InvertsDiagonalBlocksWithAdaptivePrecision)
{
    auto bj = adaptive_bj_factory->generate(mtx);

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    auto b1 = reinterpret_cast<const float *>(
        bj->get_blocks() + scheme.get_global_block_offset(0));
    EXPECT_NEAR(b1[0 + 0 * p], 4.0 / 14.0, 1e-7);
    EXPECT_NEAR(b1[0 + 1 * p], 2.0 / 14.0, 1e-7);
    EXPECT_NEAR(b1[1 + 0 * p], 1.0 / 14.0, 1e-7);
    EXPECT_NEAR(b1[1 + 1 * p], 4.0 / 14.0, 1e-7);

    auto b2 = bj->get_blocks() + scheme.get_global_block_offset(1);
    EXPECT_NEAR(b2[0 + 0 * p], 14.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[0 + 1 * p], 8.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[0 + 2 * p], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 + 0 * p], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 + 1 * p], 16.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 + 2 * p], 8.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 + 0 * p], 1.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 + 1 * p], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 + 2 * p], 14.0 / 48.0, 1e-14);
}


TEST_F(Jacobi, InvertsDiagonalBlocksWithAdaptivePrecisionAndSmallBlocks)
{
    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  // group size will be > 1
                  .with_block_pointers(block_pointers)
                  .with_storage_optimization(block_precisions)
                  .on(exec)
                  ->generate(mtx);

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    auto b1 = bj->get_blocks() + scheme.get_global_block_offset(0);
    EXPECT_NEAR(b1[0 + 0 * p], 4.0 / 14.0, 1e-14);
    EXPECT_NEAR(b1[0 + 1 * p], 2.0 / 14.0, 1e-14);
    EXPECT_NEAR(b1[1 + 0 * p], 1.0 / 14.0, 1e-14);
    EXPECT_NEAR(b1[1 + 1 * p], 4.0 / 14.0, 1e-14);

    auto b2 = bj->get_blocks() + scheme.get_global_block_offset(1);
    EXPECT_NEAR(b2[0 + 0 * p], 14.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[0 + 1 * p], 8.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[0 + 2 * p], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 + 0 * p], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 + 1 * p], 16.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[1 + 2 * p], 8.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 + 0 * p], 1.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 + 1 * p], 4.0 / 48.0, 1e-14);
    EXPECT_NEAR(b2[2 + 2 * p], 14.0 / 48.0, 1e-14);
}


TEST_F(Jacobi, PivotsWhenInvertingBlocks)
{
    gko::Array<gko::int32> bp(exec, 2);
    init_array(bp.get_data(), {0, 3});
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

    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  .with_block_pointers(bp)
                  .on(exec)
                  ->generate(give(mtx));

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    auto b1 = bj->get_blocks() + scheme.get_global_block_offset(0);
    EXPECT_NEAR(b1[0 + 0 * p], 0.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[0 + 1 * p], 0.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[0 + 2 * p], 4.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[1 + 0 * p], 2.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[1 + 1 * p], 0.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[1 + 2 * p], 0.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[2 + 0 * p], 0.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[2 + 1 * p], 1.0 / 4.0, 1e-14);
    EXPECT_NEAR(b1[2 + 2 * p], 0.0 / 4.0, 1e-14);
}


TEST_F(Jacobi, PivotsWhenInvertingBlocksWithiAdaptivePrecision)
{
    gko::Array<gko::int32> bp(exec, 2);
    init_array(bp.get_data(), {0, 3});
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

    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  .with_block_pointers(bp)
                  .with_storage_optimization(block_precisions)
                  .on(exec)
                  ->generate(give(mtx));

    auto scheme = bj->get_storage_scheme();
    auto p = scheme.get_stride();
    auto b1 = reinterpret_cast<const float *>(
        bj->get_blocks() + scheme.get_global_block_offset(0));
    EXPECT_NEAR(b1[0 + 0 * p], 0.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[0 + 1 * p], 0.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[0 + 2 * p], 4.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[1 + 0 * p], 2.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[1 + 1 * p], 0.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[1 + 2 * p], 0.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[2 + 0 * p], 0.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[2 + 1 * p], 1.0 / 4.0, 1e-7);
    EXPECT_NEAR(b1[2 + 2 * p], 0.0 / 4.0, 1e-7);
}


TEST_F(Jacobi, ComputesConditionNumbersOfBlocks)
{
    auto bj = adaptive_bj_factory->generate(mtx);

    auto cond = bj->get_conditioning();
    EXPECT_NEAR(cond[0], 6.0 * 6.0 / 14.0, 1e-14);
    ASSERT_NEAR(cond[1], 7.0 * 28.0 / 48.0, 1e-14);
}


TEST_F(Jacobi, SelectsCorrectBlockPrecisions)
{
    auto bj =
        Bj::build()
            .with_max_block_size(17u)
            .with_block_pointers(block_pointers)
            .with_storage_optimization(gko::precision_reduction::autodetect())
            .with_accuracy(1.5e-3)
            .on(exec)
            ->generate(give(mtx));

    auto prec =
        bj->get_parameters().storage_optimization.block_wise.get_const_data();
    EXPECT_EQ(prec[0], gko::precision_reduction(0, 2));  // u * cond = ~1.2e-3
    ASSERT_EQ(prec[1], gko::precision_reduction(0, 1));  // u * cond = ~2.0e-3
}


TEST_F(Jacobi, AvoidsPrecisionsThatOverflow)
{
    auto mtx = gko::matrix::Csr<>::create(exec);
    // clang-format off
    mtx->read(mdata::diag({
        // perfectly conditioned block, small value difference,
        // can use fp16 (5, 10)
        {{2.0, 1.0},
         {1.0, 2.0}},
        // perfectly conditioned block (scaled orthogonal),
        // with large value difference, need fp16 (7, 8)
        {{1e-7, -1e-14},
         {1e-14,  1e-7}}
    }));
    // clang-format on

    auto bj =
        Bj::build()
            .with_max_block_size(13u)
            .with_block_pointers(gko::Array<gko::int32>(exec, {0, 2, 4}))
            .with_storage_optimization(gko::precision_reduction::autodetect())
            .with_accuracy(0.1)
            .on(exec)
            ->generate(give(mtx));

    // both blocks are in the same group, both need (7, 8)
    auto prec =
        bj->get_parameters().storage_optimization.block_wise.get_const_data();
    EXPECT_EQ(prec[0], gko::precision_reduction(1, 1));
    ASSERT_EQ(prec[1], gko::precision_reduction(1, 1));
}


TEST_F(Jacobi, AppliesToVector)
{
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, exec);
    auto bj = bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 0.0, 0.0, 1.0, 0.0}), 1e-14);
}


TEST_F(Jacobi, AppliesToVectorWithAdaptivePrecision)
{
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, exec);
    auto bj = adaptive_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 0.0, 0.0, 1.0, 0.0}), 1e-7);
}


TEST_F(Jacobi, AppliesToVectorWithAdaptivePrecisionAndSmallBlocks)
{
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, exec);
    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  // group size will be > 1
                  .with_block_pointers(block_pointers)
                  .with_storage_optimization(block_precisions)
                  .on(exec)
                  ->generate(mtx);

    bj->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 0.0, 0.0, 1.0, 0.0}), 1e-7);
}


TEST_F(Jacobi, AppliesToMultipleVectors)
{
    auto x = gko::initialize<Vec>(
        3, {{1.0, 0.5}, {-1.0, -0.5}, {2.0, 1.0}, {-2.0, -1.0}, {3.0, 1.5}},
        exec);
    auto b = gko::initialize<Vec>(
        3, {{4.0, -2.0}, {-1.0, 4.0}, {-2.0, 0.0}, {4.0, -2.0}, {-1.0, 4.0}},
        exec);
    auto bj = bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}),
        1e-14);
}


TEST_F(Jacobi, AppliesToMultipleVectorsWithAdaptivePrecision)
{
    auto x = gko::initialize<Vec>(
        3, {{1.0, 0.5}, {-1.0, -0.5}, {2.0, 1.0}, {-2.0, -1.0}, {3.0, 1.5}},
        exec);
    auto b = gko::initialize<Vec>(
        3, {{4.0, -2.0}, {-1.0, 4.0}, {-2.0, 0.0}, {4.0, -2.0}, {-1.0, 4.0}},
        exec);
    auto bj = adaptive_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}),
        1e-7);
}


TEST_F(Jacobi, AppliesToMultipleVectorsWithAdaptivePrecisionAndSmallBlocks)
{
    auto x = gko::initialize<Vec>(
        3, {{1.0, 0.5}, {-1.0, -0.5}, {2.0, 1.0}, {-2.0, -1.0}, {3.0, 1.5}},
        exec);
    auto b = gko::initialize<Vec>(
        3, {{4.0, -2.0}, {-1.0, 4.0}, {-2.0, 0.0}, {4.0, -2.0}, {-1.0, 4.0}},
        exec);
    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  // group size will be > 1
                  .with_block_pointers(block_pointers)
                  .with_storage_optimization(block_precisions)
                  .on(exec)
                  ->generate(mtx);

    bj->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}),
        1e-7);
}


TEST_F(Jacobi, AppliesLinearCombinationToVector)
{
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, exec);
    auto alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, exec);
    auto bj = bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 1.0, -2.0, 4.0, -3.0}), 1e-14);
}


TEST_F(Jacobi, AppliesLinearCombinationToVectorWithAdaptivePrecision)
{
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, exec);
    auto alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, exec);
    auto bj = adaptive_bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 1.0, -2.0, 4.0, -3.0}), 1e-7);
}


TEST_F(Jacobi, AppliesLinearCombinationToMultipleVectors)
{
    auto x = gko::initialize<Vec>(
        3, {{1.0, 0.5}, {-1.0, -0.5}, {2.0, 1.0}, {-2.0, -1.0}, {3.0, 1.5}},
        exec);
    auto b = gko::initialize<Vec>(
        3, {{4.0, -2.0}, {-1.0, 4.0}, {-2.0, 0.0}, {4.0, -2.0}, {-1.0, 4.0}},
        exec);
    auto alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, exec);
    auto bj = bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, -0.5}, {1.0, 2.5}, {-2.0, -1.0}, {4.0, 1.0}, {-3.0, 0.5}}),
        1e-14);
}


TEST_F(Jacobi, AppliesLinearCombinationToMultipleVectorsWithAdaptivePrecision)
{
    auto x = gko::initialize<Vec>(
        3, {{1.0, 0.5}, {-1.0, -0.5}, {2.0, 1.0}, {-2.0, -1.0}, {3.0, 1.5}},
        exec);
    auto b = gko::initialize<Vec>(
        3, {{4.0, -2.0}, {-1.0, 4.0}, {-2.0, 0.0}, {4.0, -2.0}, {-1.0, 4.0}},
        exec);
    auto alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, exec);
    auto bj = adaptive_bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, -0.5}, {1.0, 2.5}, {-2.0, -1.0}, {4.0, 1.0}, {-3.0, 0.5}}),
        1e-7);
}


TEST_F(Jacobi, ConvertsToDense)
{
    auto dense = gko::matrix::Dense<>::create(exec);

    dense->copy_from(bj_factory->generate(mtx));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense,
        l({{4.0 / 14, 2.0 / 14,       0.0,       0.0,       0.0},
           {1.0 / 14, 4.0 / 14,       0.0,       0.0,       0.0},
           {     0.0,      0.0, 14.0 / 48,  8.0 / 48,  4.0 / 48},
           {     0.0,      0.0,  4.0 / 48, 16.0 / 48,  8.0 / 48},
           {     0.0,      0.0,  1.0 / 48,  4.0 / 48, 14.0 / 48}}), 1e-14);
    // clang-format on
}


TEST_F(Jacobi, ConvertsToDenseWithAdaptivePrecision)
{
    auto dense = gko::matrix::Dense<>::create(exec);

    dense->copy_from(adaptive_bj_factory->generate(mtx));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense,
        l({{4.0 / 14, 2.0 / 14,       0.0,       0.0,       0.0},
           {1.0 / 14, 4.0 / 14,       0.0,       0.0,       0.0},
           {     0.0,      0.0, 14.0 / 48,  8.0 / 48,  4.0 / 48},
           {     0.0,      0.0,  4.0 / 48, 16.0 / 48,  8.0 / 48},
           {     0.0,      0.0,  1.0 / 48,  4.0 / 48, 14.0 / 48}}), 1e-7);
    // clang-format on
}


}  // namespace
