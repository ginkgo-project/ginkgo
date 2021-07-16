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

#include <ginkgo/core/matrix/sub_matrix.hpp>


#include <gtest/gtest.h>

#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class SubMatrix : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using T = typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using Mtx = gko::matrix::SubMatrix<CsrMtx>;

    SubMatrix()
        : exec(gko::ReferenceExecutor::create()),
          csr_mtx(gko::initialize<CsrMtx>({{1.0, 2.0, 0.0, 0.0, 3.0},
                                           {0.0, 3.0, 0.0, 0.0, 0.0},
                                           {0.0, 3.0, 2.5, 1.5, 0.0},
                                           {1.0, 0.0, 1.0, 2.0, 4.0},
                                           {0.0, 1.0, 2.0, 1.5, 3.0}},
                                          exec)),
          csr_mtx0(gko::initialize<CsrMtx>({I<T>({1.0, 2.0}), I<T>({0.0, 3.0})},
                                           exec)),
          csr_mtx11(gko::initialize<CsrMtx>(
              {I<T>({3.0, 0.0}), I<T>({3.0, 2.5})}, exec)),
          csr_mtx01(gko::initialize<CsrMtx>({I<T>({0.0}), I<T>({0.0})}, exec)),
          csr_mtx12(gko::initialize<CsrMtx>({I<T>({0.0}), I<T>({1.5})}, exec)),
          csr_mtx1(gko::initialize<CsrMtx>(
              {{2.5, 1.5, 0.0}, {1.0, 2.0, 4.0}, {2.0, 1.5, 3.0}}, exec)),
          csr_mtx2(gko::initialize<CsrMtx>(
              {I<T>({0.0, 0.0, 0.0}), I<T>({2.5, 1.5, 0.0})}, exec)),
          csr_mtx3(gko::initialize<CsrMtx>({I<T>({1.0, 0.0}), I<T>({0.0, 1.0})},
                                           exec)),
          mtx(Mtx::create(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<CsrMtx> csr_mtx;
    std::unique_ptr<CsrMtx> csr_mtx0;
    std::unique_ptr<CsrMtx> csr_mtx01;
    std::unique_ptr<CsrMtx> csr_mtx12;
    std::unique_ptr<CsrMtx> csr_mtx11;
    std::unique_ptr<CsrMtx> csr_mtx1;
    std::unique_ptr<CsrMtx> csr_mtx2;
    std::unique_ptr<CsrMtx> csr_mtx3;
    std::unique_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(SubMatrix, gko::test::ValueIndexTypes);


TYPED_TEST(SubMatrix, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    ASSERT_EQ(mtx->get_size(), gko::dim<2>{});
    ASSERT_EQ(mtx->get_overlap_mtxs().size(), 0);
}


TYPED_TEST(SubMatrix, CanBeCreatedFromCsr)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto rspan = gko::span(0, 2);
    auto cspan = gko::span(0, 2);
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan);
    auto rspan1 = gko::span(2, 5);
    auto cspan1 = gko::span(2, 5);
    auto mtx1 = Mtx::create(this->exec, this->csr_mtx.get(), rspan1, cspan1);
    auto rspan2 = gko::span(1, 3);
    auto cspan2 = gko::span(2, 5);
    auto mtx2 = Mtx::create(this->exec, this->csr_mtx.get(), rspan2, cspan2);
    auto rspan3 = gko::span(1, 3);
    auto cspan3 = gko::span(0, 1);
    auto mtx3 = Mtx::create(this->exec, this->csr_mtx.get(), rspan3, cspan3);
    auto rspan4 = gko::span(1, 3);
    auto cspan4 = gko::span(3, 4);
    auto mtx4 = Mtx::create(this->exec, this->csr_mtx.get(), rspan4, cspan4);

    GKO_EXPECT_MTX_NEAR(mtx->get_sub_matrix(), this->csr_mtx0,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx1->get_sub_matrix(), this->csr_mtx1,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx2->get_sub_matrix(), this->csr_mtx2,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx3->get_sub_matrix(), this->csr_mtx01,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx4->get_sub_matrix(), this->csr_mtx12,
                        r<value_type>::value);
}


TYPED_TEST(SubMatrix, CanBeCreatedFromCsrWithOverlaps)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto rspan = gko::span(1, 3);
    auto cspan = gko::span(1, 3);
    auto ov_rspan = std::vector<gko::span>{gko::span(1, 3), gko::span(1, 3)};
    auto ov_cspan = std::vector<gko::span>{gko::span(0, 1), gko::span(3, 4)};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan,
                           ov_rspan, ov_cspan);

    GKO_EXPECT_MTX_NEAR(mtx->get_sub_matrix(), this->csr_mtx11,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[0], this->csr_mtx01,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[1], this->csr_mtx12,
                        r<value_type>::value);
}


TYPED_TEST(SubMatrix, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto rspan = gko::span(1, 3);
    auto cspan = gko::span(1, 3);
    auto ov_rspan = std::vector<gko::span>{gko::span(1, 3), gko::span(1, 3)};
    auto ov_cspan = std::vector<gko::span>{gko::span(0, 1), gko::span(3, 4)};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan,
                           ov_rspan, ov_cspan);
    auto copy = Mtx::create(this->exec);

    copy->copy_from(mtx.get());

    EXPECT_NE(copy->get_sub_matrix(), mtx->get_sub_matrix());
    EXPECT_NE(copy->get_overlap_mtxs()[0], mtx->get_overlap_mtxs()[0]);
    EXPECT_NE(copy->get_overlap_mtxs()[1], mtx->get_overlap_mtxs()[1]);
    GKO_EXPECT_MTX_NEAR(copy->get_sub_matrix(), this->csr_mtx11,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(copy->get_overlap_mtxs()[0], this->csr_mtx01,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(copy->get_overlap_mtxs()[1], this->csr_mtx12,
                        r<value_type>::value);
}


TYPED_TEST(SubMatrix, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto rspan = gko::span(1, 3);
    auto cspan = gko::span(1, 3);
    auto ov_rspan = std::vector<gko::span>{gko::span(1, 3), gko::span(1, 3)};
    auto ov_cspan = std::vector<gko::span>{gko::span(0, 1), gko::span(3, 4)};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan,
                           ov_rspan, ov_cspan);
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(mtx.get()));

    EXPECT_NE(copy->get_sub_matrix(), mtx->get_sub_matrix());
    EXPECT_NE(copy->get_overlap_mtxs()[0], mtx->get_overlap_mtxs()[0]);
    EXPECT_NE(copy->get_overlap_mtxs()[1], mtx->get_overlap_mtxs()[1]);
    GKO_EXPECT_MTX_NEAR(copy->get_sub_matrix(), this->csr_mtx11,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(copy->get_overlap_mtxs()[0], this->csr_mtx01,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(copy->get_overlap_mtxs()[1], this->csr_mtx12,
                        r<value_type>::value);
}


TYPED_TEST(SubMatrix, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto rspan = gko::span(1, 3);
    auto cspan = gko::span(1, 3);
    auto ov_rspan = std::vector<gko::span>{gko::span(1, 3), gko::span(1, 3)};
    auto ov_cspan = std::vector<gko::span>{gko::span(0, 1), gko::span(3, 4)};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan,
                           ov_rspan, ov_cspan);
    auto copy = Mtx::create(this->exec);

    auto clone = mtx->clone();

    EXPECT_NE(clone->get_sub_matrix(), mtx->get_sub_matrix());
    EXPECT_NE(clone->get_overlap_mtxs()[0], mtx->get_overlap_mtxs()[0]);
    EXPECT_NE(clone->get_overlap_mtxs()[1], mtx->get_overlap_mtxs()[1]);
    GKO_EXPECT_MTX_NEAR(clone->get_sub_matrix(), this->csr_mtx11,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(clone->get_overlap_mtxs()[0], this->csr_mtx01,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(clone->get_overlap_mtxs()[1], this->csr_mtx12,
                        r<value_type>::value);
}


TYPED_TEST(SubMatrix, CanBeCleared)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto rspan = gko::span(0, 2);
    auto cspan = gko::span(0, 2);
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan);
    auto copy = Mtx::create(this->exec);

    mtx->clear();

    ASSERT_EQ(mtx->get_size(), gko::dim<2>{});
    ASSERT_EQ(mtx->get_overlap_mtxs().size(), 0);
}


}  // namespace
