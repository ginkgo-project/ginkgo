/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/matrix/coo.hpp>


#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/block_matrix.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/coo_kernels.hpp"
#include "core/test/utils.hpp"


namespace {

template <typename T>
class BlockMatrix : public ::testing::Test {
protected:
    using value_type = T;
    using mtx = gko::matrix::Csr<value_type>;
    using vec = gko::matrix::Dense<value_type>;

    BlockMatrix()
        : exec(gko::ReferenceExecutor::create()),
          size(5, 5),
          spans{{0, 3}, {3, 5}},
          a_block(gko::share(
              gko::initialize<mtx>({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}, exec))),
          b_block(
              gko::share(gko::initialize<mtx>({{1, 2}, {2, 3}, {3, 4}}, exec))),
          c_block(
              gko::share(gko::initialize<mtx>({{1, 2, 3}, {2, 3, 4}}, exec))),
          d_block(gko::share(gko::initialize<mtx>({{1, 2}, {2, 3}}, exec))),
          b_dense(gko::share(gko::initialize<vec>({1, 1, 1, 1, 1}, exec))),
          x_dense(gko::share(
              gko::initialize<vec>({0.5, 0.5, 0.5, 0.5, 0.5}, exec))),
          y_dense(gko::share(gko::initialize<vec>({9, 14, 19, 9, 14}, exec))),
          b0_block(gko::share(gko::initialize<vec>({1, 1, 1}, exec))),
          b1_block(gko::share(gko::initialize<vec>({1, 1}, exec))),
          x0_block(gko::share(gko::initialize<vec>({0.5, 0.5, 0.5}, exec))),
          x1_block(gko::share(gko::initialize<vec>({0.5, 0.5}, exec))),
          y0_block(gko::share(gko::initialize<vec>({9, 14, 19}, exec))),
          y1_block(gko::share(gko::initialize<vec>({9, 14}, exec))),
          alpha(gko::share(gko::initialize<vec>({2.2}, exec))),
          beta(gko::share(gko::initialize<vec>({3.3}, exec)))
    {}

    std::shared_ptr<const gko::Executor> exec;

    gko::dim<2> size;

    std::vector<gko::span> spans;

    std::shared_ptr<mtx> a_block;
    std::shared_ptr<mtx> b_block;
    std::shared_ptr<mtx> c_block;
    std::shared_ptr<mtx> d_block;

    std::shared_ptr<vec> b_dense;
    std::shared_ptr<vec> x_dense;
    std::shared_ptr<vec> y_dense;
    std::shared_ptr<vec> b0_block;
    std::shared_ptr<vec> b1_block;
    std::shared_ptr<vec> x0_block;
    std::shared_ptr<vec> x1_block;
    std::shared_ptr<vec> y0_block;
    std::shared_ptr<vec> y1_block;

    std::shared_ptr<vec> alpha;
    std::shared_ptr<vec> beta;
};


TYPED_TEST_SUITE(BlockMatrix, gko::test::RealValueTypes, TypenameNameGenerator);


TYPED_TEST(BlockMatrix, KnowsBlockSize)
{
    auto block_mtx = gko::matrix::BlockMatrix::create(
        this->exec, this->size,
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{
            {this->a_block, this->b_block}, {this->c_block, this->d_block}},
        this->spans);

    GKO_ASSERT_EQUAL_DIMENSIONS(block_mtx->get_block_size(), gko::dim<2>(2, 2));
}


TYPED_TEST(BlockMatrix, KnowsSizePerBlock)
{
    auto block_mtx = gko::matrix::BlockMatrix::create(
        this->exec, this->size,
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{
            {this->a_block, this->b_block}, {this->c_block, this->d_block}},
        this->spans);

    ASSERT_EQ(block_mtx->get_span_per_block()[0], this->spans[0]);
    ASSERT_EQ(block_mtx->get_span_per_block()[1], this->spans[1]);
}


TYPED_TEST(BlockMatrix, AppliesToVector)
{
    using value_type = typename TestFixture::value_type;
    using mtx = typename TestFixture::mtx;
    using vec = typename TestFixture::vec;
    auto block_mtx = gko::matrix::BlockMatrix::create(
        this->exec, this->size,
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{
            {this->a_block, this->b_block}, {this->c_block, this->d_block}},
        this->spans);
    auto block_vec_b = gko::matrix::BlockMatrix::create(
        this->exec, gko::dim<2>{5, 1},
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{{this->b0_block},
                                                              {this->b1_block}},
        this->spans);
    auto block_vec_x = gko::matrix::BlockMatrix::create(
        this->exec, gko::dim<2>{5, 1},
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{{this->x0_block},
                                                              {this->x1_block}},
        this->spans);
    auto block_vec_y = gko::matrix::BlockMatrix::create(
        this->exec, gko::dim<2>{5, 1},
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{{this->y0_block},
                                                              {this->y1_block}},
        this->spans);

    block_mtx->apply(block_vec_b.get(), block_vec_x.get());

    GKO_ASSERT_MTX_NEAR(gko::as<vec>(block_vec_x->get_block()[0][0]),
                        gko::as<vec>(block_vec_y->get_block()[0][0]),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<vec>(block_vec_x->get_block()[1][0]),
                        gko::as<vec>(block_vec_y->get_block()[1][0]),
                        r<value_type>::value);
}


TYPED_TEST(BlockMatrix, AppliesToVectorWithAlphaBeta)
{
    using value_type = typename TestFixture::value_type;
    using mtx = typename TestFixture::mtx;
    using vec = typename TestFixture::vec;
    auto block_mtx = gko::matrix::BlockMatrix::create(
        this->exec, this->size,
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{
            {this->a_block, this->b_block}, {this->c_block, this->d_block}},
        this->spans);
    auto block_vec_b = gko::matrix::BlockMatrix::create(
        this->exec, gko::dim<2>{5, 1},
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{{this->b0_block},
                                                              {this->b1_block}},
        this->spans);
    auto block_vec_x = gko::matrix::BlockMatrix::create(
        this->exec, gko::dim<2>{5, 1},
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{{this->x0_block},
                                                              {this->x1_block}},
        this->spans);
    auto block_vec_y = gko::matrix::BlockMatrix::create(
        this->exec, gko::dim<2>{5, 1},
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{
            {gko::initialize<vec>(
                {2.2 * 9 + 3.3 / 2, 2.2 * 14 + 3.3 / 2, 2.2 * 19 + 3.3 / 2},
                this->exec)},
            {gko::initialize<vec>({2.2 * 9 + 3.3 / 2, 2.2 * 14 + 3.3 / 2},
                                  this->exec)}},
        this->spans);

    block_mtx->apply(this->alpha.get(), block_vec_b.get(), this->beta.get(),
                     block_vec_x.get());

    GKO_ASSERT_MTX_NEAR(gko::as<vec>(block_vec_x->get_block()[0][0]),
                        gko::as<vec>(block_vec_y->get_block()[0][0]),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<vec>(block_vec_x->get_block()[1][0]),
                        gko::as<vec>(block_vec_y->get_block()[1][0]),
                        r<value_type>::value);
}

TYPED_TEST(BlockMatrix, AppliesToDenseVector)
{
    using value_type = typename TestFixture::value_type;
    using mtx = typename TestFixture::mtx;
    using vec = typename TestFixture::vec;
    auto block_mtx = gko::matrix::BlockMatrix::create(
        this->exec, this->size,
        std::vector<std::vector<std::shared_ptr<gko::LinOp>>>{
            {this->a_block, this->b_block}, {this->c_block, this->d_block}},
        this->spans);

    block_mtx->apply(this->b_dense.get(), this->x_dense.get());

    GKO_ASSERT_MTX_NEAR(this->y_dense, this->x_dense, r<value_type>::value);
}

}  // namespace
