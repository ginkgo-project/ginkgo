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

#include <ginkgo/core/matrix/block_approx.hpp>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class BlockApprox : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using T = typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Mtx = gko::matrix::BlockApprox<CsrMtx>;

    BlockApprox()
        : exec(gko::OmpExecutor::create()),
          csr_mtx(gko::initialize<CsrMtx>({{1.0, 2.0, 0.0, 0.0, 3.0},
                                           {0.0, 3.0, 0.0, 0.0, 0.0},
                                           {0.0, 3.0, 2.5, 1.5, 0.0},
                                           {1.0, 0.0, 1.0, 2.0, 4.0},
                                           {0.0, 1.0, 2.0, 1.5, 3.0}},
                                          exec)),
          csr_mtx0(gko::initialize<CsrMtx>({I<T>({1.0, 2.0}), I<T>({0.0, 3.0})},
                                           exec)),
          csr_mtx1(gko::initialize<CsrMtx>(
              {{2.5, 1.5, 0.0}, {1.0, 2.0, 4.0}, {2.0, 1.5, 3.0}}, exec)),
          ov_csr_mtx0(gko::initialize<CsrMtx>(
              {I<T>({1.0, 2.0, 0.0}), I<T>({0.0, 3.0, 0.0}),
               I<T>({0.0, 3.0, 2.5})},
              exec)),
          ov_csr_mtx1(gko::initialize<CsrMtx>({{3.0, 0.0, 0.0, 0.0},
                                               {3.0, 2.5, 1.5, 0.0},
                                               {0.0, 1.0, 2.0, 4.0},
                                               {1.0, 2.0, 1.5, 3.0}},
                                              exec)),
          b(gko::initialize<Dense>({2.0, 1.0, -1.0, 3.0, 0.0}, exec)),
          ov_b0(gko::initialize<Dense>({2.0, 1.0, -1.0}, exec)),
          ov_b1(gko::initialize<Dense>({1.0, -1.0, 3.0, 0.0}, exec)),
          b0(gko::initialize<Dense>({2.0, 1.0}, exec)),
          b1(gko::initialize<Dense>({-1.0, 3.0, 0.0}, exec)),
          x(gko::initialize<Dense>({2.0, 1.0, -1.0, 3.0, 0.0}, exec)),
          ov_x0(gko::initialize<Dense>({2.0, 1.0, -1.0}, exec)),
          ov_x1(gko::initialize<Dense>({1.0, -1.0, 3.0, 0.0}, exec)),
          x0(gko::initialize<Dense>({2.0, 1.0}, exec)),
          x1(gko::initialize<Dense>({-1.0, 3.0, 0.0}, exec)),
          mtx(Mtx::create(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<CsrMtx> csr_mtx;
    std::unique_ptr<CsrMtx> csr_mtx0;
    std::unique_ptr<CsrMtx> csr_mtx1;
    std::unique_ptr<CsrMtx> ov_csr_mtx0;
    std::unique_ptr<CsrMtx> ov_csr_mtx1;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Dense> b;
    std::unique_ptr<Dense> b0;
    std::unique_ptr<Dense> b1;
    std::unique_ptr<Dense> ov_b0;
    std::unique_ptr<Dense> ov_b1;
    std::unique_ptr<Dense> x;
    std::unique_ptr<Dense> x0;
    std::unique_ptr<Dense> x1;
    std::unique_ptr<Dense> ov_x0;
    std::unique_ptr<Dense> ov_x1;
};

TYPED_TEST_SUITE(BlockApprox, gko::test::ValueIndexTypes);


TYPED_TEST(BlockApprox, CanApplyToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto block_sizes = gko::Array<gko::size_type>(this->exec, {2, 3});
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), block_sizes);

    mtx->apply(this->b.get(), this->x.get());
    this->csr_mtx0->apply(this->b0.get(), this->x0.get());
    this->csr_mtx1->apply(this->b1.get(), this->x1.get());

    ASSERT_EQ(mtx->get_num_blocks(), 2);
    ASSERT_EQ(mtx->get_block_dimensions()[0], gko::dim<2>(2));
    ASSERT_EQ(mtx->get_block_dimensions()[1], gko::dim<2>(3));
    ASSERT_EQ(mtx->get_block_nonzeros()[0], 3);
    ASSERT_EQ(mtx->get_block_nonzeros()[1], 8);
    GKO_EXPECT_MTX_NEAR(mtx->get_block_mtxs()[0], this->csr_mtx0,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_block_mtxs()[1], this->csr_mtx1,
                        r<value_type>::value);
    EXPECT_EQ(this->x->get_values()[0], this->x0->get_values()[0]);
    EXPECT_EQ(this->x->get_values()[1], this->x0->get_values()[1]);
    EXPECT_EQ(this->x->get_values()[2], this->x1->get_values()[0]);
    EXPECT_EQ(this->x->get_values()[3], this->x1->get_values()[1]);
    EXPECT_EQ(this->x->get_values()[4], this->x1->get_values()[2]);
}


TYPED_TEST(BlockApprox, CanApplyToDenseWithOverlap)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto block_sizes = gko::Array<gko::size_type>(this->exec, {2, 3});
    auto block_overlaps = gko::Overlap<gko::size_type>(
        this->exec, gko::Array<gko::size_type>{this->exec, {1, 1}},
        gko::Array<bool>{this->exec, {true, true}},
        gko::Array<bool>{this->exec, {false, true}});
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), block_sizes,
                           block_overlaps);

    mtx->apply(this->b.get(), this->x.get());
    this->ov_csr_mtx0->apply(this->ov_b0.get(), this->ov_x0.get());
    this->ov_csr_mtx1->apply(this->ov_b1.get(), this->ov_x1.get());

    ASSERT_EQ(mtx->get_num_blocks(), 2);
    ASSERT_EQ(mtx->get_block_dimensions()[0], gko::dim<2>(3));
    ASSERT_EQ(mtx->get_block_dimensions()[1], gko::dim<2>(4));
    ASSERT_EQ(mtx->get_block_nonzeros()[0], 5);
    ASSERT_EQ(mtx->get_block_nonzeros()[1], 11);
    GKO_EXPECT_MTX_NEAR(mtx->get_block_mtxs()[0], this->ov_csr_mtx0,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_block_mtxs()[1], this->ov_csr_mtx1,
                        r<value_type>::value);
    ASSERT_EQ(this->x->get_num_stored_elements(), 5);
    EXPECT_EQ(this->x->get_values()[0], this->ov_x0->get_values()[0]);
    EXPECT_EQ(this->x->get_values()[1], this->ov_x0->get_values()[1]);
    EXPECT_EQ(this->x->get_values()[2], this->ov_x1->get_values()[0]);
    EXPECT_EQ(this->x->get_values()[3], this->ov_x1->get_values()[1]);
    EXPECT_EQ(this->x->get_values()[4], this->ov_x1->get_values()[2]);
}


TYPED_TEST(BlockApprox, CanAdvancedApplyToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto block_sizes = gko::Array<gko::size_type>(this->exec, {2, 3});
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), block_sizes);
    auto alpha = gko::initialize<Dense>({2.0}, this->exec);
    auto beta = gko::initialize<Dense>({-1.0}, this->exec);

    mtx->apply(alpha.get(), this->b.get(), beta.get(), this->x.get());
    this->csr_mtx0->apply(alpha.get(), this->b0.get(), beta.get(),
                          this->x0.get());
    this->csr_mtx1->apply(alpha.get(), this->b1.get(), beta.get(),
                          this->x1.get());

    ASSERT_EQ(mtx->get_num_blocks(), 2);
    ASSERT_EQ(mtx->get_block_dimensions()[0], gko::dim<2>(2));
    ASSERT_EQ(mtx->get_block_dimensions()[1], gko::dim<2>(3));
    ASSERT_EQ(mtx->get_block_nonzeros()[0], 3);
    ASSERT_EQ(mtx->get_block_nonzeros()[1], 8);
    GKO_EXPECT_MTX_NEAR(mtx->get_block_mtxs()[0], this->csr_mtx0,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_block_mtxs()[1], this->csr_mtx1,
                        r<value_type>::value);
    ASSERT_EQ(this->x->get_values()[0], this->x0->get_values()[0]);
    ASSERT_EQ(this->x->get_values()[1], this->x0->get_values()[1]);
    ASSERT_EQ(this->x->get_values()[2], this->x1->get_values()[0]);
    ASSERT_EQ(this->x->get_values()[3], this->x1->get_values()[1]);
    ASSERT_EQ(this->x->get_values()[4], this->x1->get_values()[2]);
}


TYPED_TEST(BlockApprox, CanAdvancedApplyToDenseWithOverlap)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto block_sizes = gko::Array<gko::size_type>(this->exec, {2, 3});
    auto block_overlaps = gko::Overlap<gko::size_type>(
        this->exec, gko::Array<gko::size_type>{this->exec, {1, 1}},
        gko::Array<bool>{this->exec, {true, true}},
        gko::Array<bool>{this->exec, {false, true}});
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), block_sizes,
                           block_overlaps);
    auto alpha = gko::initialize<Dense>({2.0}, this->exec);
    auto beta = gko::initialize<Dense>({-1.0}, this->exec);

    mtx->apply(alpha.get(), this->b.get(), beta.get(), this->x.get());
    this->ov_csr_mtx0->apply(alpha.get(), this->ov_b0.get(), beta.get(),
                             this->ov_x0.get());
    this->ov_csr_mtx1->apply(alpha.get(), this->ov_b1.get(), beta.get(),
                             this->ov_x1.get());

    ASSERT_EQ(mtx->get_num_blocks(), 2);
    ASSERT_EQ(mtx->get_block_dimensions()[0], gko::dim<2>(3));
    ASSERT_EQ(mtx->get_block_dimensions()[1], gko::dim<2>(4));
    ASSERT_EQ(mtx->get_block_nonzeros()[0], 5);
    ASSERT_EQ(mtx->get_block_nonzeros()[1], 11);
    GKO_EXPECT_MTX_NEAR(mtx->get_block_mtxs()[0], this->ov_csr_mtx0,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_block_mtxs()[1], this->ov_csr_mtx1,
                        r<value_type>::value);
    EXPECT_EQ(this->x->get_values()[0], this->ov_x0->get_values()[0]);
    EXPECT_EQ(this->x->get_values()[1], this->ov_x0->get_values()[1]);
    EXPECT_EQ(this->x->get_values()[2], this->ov_x1->get_values()[0]);
    EXPECT_EQ(this->x->get_values()[3], this->ov_x1->get_values()[1]);
    EXPECT_EQ(this->x->get_values()[4], this->ov_x1->get_values()[2]);
}


}  // namespace
