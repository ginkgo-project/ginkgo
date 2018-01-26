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


namespace {


class BlockJacobi : public ::testing::Test {
protected:
    using BjFactory = gko::preconditioner::BlockJacobiFactory<>;
    using Bj = gko::preconditioner::BlockJacobi<>;
    using Mtx = gko::matrix::Csr<>;

    BlockJacobi()
        : exec(gko::ReferenceExecutor::create()),
          bj_factory(BjFactory::create(exec, 3)),
          block_pointers(exec, 3),
          mtx(gko::matrix::Csr<>::create(exec, 5, 5, 13))
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


TEST_F(BlockJacobi, CanBeGenerated)
{
    bj_factory->set_block_pointers(block_pointers);
    bj_lin_op = bj_factory->generate(mtx);
    bj = dynamic_cast<Bj *>(bj_lin_op.get());

    ASSERT_NE(bj, nullptr);
    EXPECT_EQ(bj->get_executor(), exec);
    EXPECT_EQ(bj->get_max_block_size(), 3);
    EXPECT_EQ(bj->get_num_rows(), 5);
    EXPECT_EQ(bj->get_num_cols(), 5);
    ASSERT_EQ(bj->get_num_blocks(), 2);
    auto ptrs = bj->get_const_block_pointers();
    EXPECT_EQ(ptrs[0], 0);
    EXPECT_EQ(ptrs[1], 2);
    ASSERT_EQ(ptrs[2], 5);
}


TEST_F(BlockJacobi, InvertsDiagonalBlocks)
{
    bj_factory->set_block_pointers(block_pointers);

    bj_lin_op = bj_factory->generate(mtx);

    bj = dynamic_cast<Bj *>(bj_lin_op.get());
    auto p = bj->get_padding();
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
    auto mtx = Mtx::create(exec, 3, 3, 9);
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

    bj = dynamic_cast<Bj *>(bj_lin_op.get());
    auto p = bj->get_padding();
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


}  // namespace
