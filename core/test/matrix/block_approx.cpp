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


#include <gtest/gtest.h>

#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

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
    using Mtx = gko::matrix::BlockApprox<CsrMtx>;

    BlockApprox()
        : exec(gko::ReferenceExecutor::create()),
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
          mtx(Mtx::create(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<CsrMtx> csr_mtx;
    std::unique_ptr<CsrMtx> csr_mtx0;
    std::unique_ptr<CsrMtx> csr_mtx1;
    std::unique_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(BlockApprox, gko::test::ValueIndexTypes);


TYPED_TEST(BlockApprox, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    ASSERT_EQ(mtx->get_num_blocks(), 0);
}


TYPED_TEST(BlockApprox, CanBeCreatedFromCsr)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto block_sizes = gko::Array<gko::size_type>{this->exec, {2, 3}};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), block_sizes);

    ASSERT_EQ(mtx->get_num_blocks(), 2);
    GKO_EXPECT_MTX_NEAR(mtx->get_block_mtxs()[0]->get_sub_matrix(),
                        this->csr_mtx0, r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_block_mtxs()[1]->get_sub_matrix(),
                        this->csr_mtx1, r<value_type>::value);
}


TYPED_TEST(BlockApprox, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto block_sizes = gko::Array<gko::size_type>{this->exec, {2, 3}};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), block_sizes);
    auto copy = Mtx::create(this->exec);

    copy->copy_from(mtx.get());

    ASSERT_EQ(copy->get_num_blocks(), 2);
    GKO_EXPECT_MTX_NEAR(copy->get_block_mtxs()[0]->get_sub_matrix(),
                        this->csr_mtx0, r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(copy->get_block_mtxs()[1]->get_sub_matrix(),
                        this->csr_mtx1, r<value_type>::value);
}


TYPED_TEST(BlockApprox, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto block_sizes = gko::Array<gko::size_type>{this->exec, {2, 3}};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), block_sizes);
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(mtx.get()));

    ASSERT_EQ(copy->get_num_blocks(), 2);
    GKO_EXPECT_MTX_NEAR(copy->get_block_mtxs()[0]->get_sub_matrix(),
                        this->csr_mtx0, r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(copy->get_block_mtxs()[1]->get_sub_matrix(),
                        this->csr_mtx1, r<value_type>::value);
}


TYPED_TEST(BlockApprox, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto block_sizes = gko::Array<gko::size_type>{this->exec, {2, 3}};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), block_sizes);
    auto clone = mtx->clone();

    ASSERT_EQ(clone->get_num_blocks(), 2);
    GKO_EXPECT_MTX_NEAR(clone->get_block_mtxs()[0]->get_sub_matrix(),
                        this->csr_mtx0, r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(clone->get_block_mtxs()[1]->get_sub_matrix(),
                        this->csr_mtx1, r<value_type>::value);
}


TYPED_TEST(BlockApprox, CanBeCleared)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto block_sizes = gko::Array<gko::size_type>{this->exec, {2, 3}};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), block_sizes);
    ASSERT_NE(mtx->get_num_blocks(), 0);

    mtx->clear();

    ASSERT_EQ(mtx->get_num_blocks(), 0);
}


}  // namespace
