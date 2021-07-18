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


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <gtest/gtest.h>


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
    using Dense = gko::matrix::Dense<value_type>;
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
          csr_mtx1(gko::initialize<CsrMtx>(
              {{3.0, 2.5, 1.5}, {0.0, 1.0, 2.0}, {1.0, 2.0, 1.5}}, exec)),
          csr_mtx10(gko::initialize<CsrMtx>(
              {I<T>({0.0}), I<T>({1.0}), I<T>({0.0})}, exec)),
          csr_mtx11(gko::initialize<CsrMtx>(
              {I<T>({0.0}), I<T>({4.0}), I<T>({3.0})}, exec)),
          csr_mtx111(gko::initialize<CsrMtx>({{0.0, 3.0, 2.5, 1.5, 0.0},
                                              {1.0, 0.0, 1.0, 2.0, 4.0},
                                              {0.0, 1.0, 2.0, 1.5, 3.0}},
                                             exec)),
          csr_mtx2(gko::initialize<CsrMtx>(
              {I<T>({3.0, 0.0}), I<T>({3.0, 2.5}), I<T>({0.0, 1.0})}, exec)),
          csr_mtx20(gko::initialize<CsrMtx>(
              {I<T>({0.0}), I<T>({0.0}), I<T>({1.0})}, exec)),
          csr_mtx21(gko::initialize<CsrMtx>(
              {I<T>({0.0}), I<T>({1.5}), I<T>({2.0})}, exec)),
          csr_mtx22(gko::initialize<CsrMtx>(
              {I<T>({0.0}), I<T>({0.0}), I<T>({4.0})}, exec)),
          csr_mtx222(gko::initialize<CsrMtx>({{0.0, 3.0, 0.0, 0.0, 0.0},
                                              {0.0, 3.0, 2.5, 1.5, 0.0},
                                              {1.0, 0.0, 1.0, 2.0, 4.0}},
                                             exec)),
          csr_mtx3(gko::initialize<CsrMtx>(
              {I<T>({0.0, 0.0}), I<T>({2.5, 1.5}), I<T>({1.0, 2.0})}, exec)),
          csr_mtx30(gko::initialize<CsrMtx>(
              {I<T>({0.0}), I<T>({0.0}), I<T>({1.0})}, exec)),
          csr_mtx31(gko::initialize<CsrMtx>(
              {I<T>({3.0}), I<T>({3.0}), I<T>({0.0})}, exec)),
          csr_mtx32(gko::initialize<CsrMtx>(
              {I<T>({0.0}), I<T>({0.0}), I<T>({4.0})}, exec)),
          b(gko::initialize<Dense>({2.0, 1.0, -1.0, 3.0, 0.0}, exec)),
          b0(gko::initialize<Dense>({2.0, 1.0}, exec)),
          b1(gko::initialize<Dense>({-1.0, 3.0, 0.0}, exec)),
          x(gko::initialize<Dense>({2.0, 1.0, -1.0}, exec)),
          x0(gko::initialize<Dense>({2.0, 1.0}, exec)),
          x1(gko::initialize<Dense>({-1.0, 3.0, 0.0}, exec)),
          mtx(Mtx::create(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<CsrMtx> csr_mtx;
    std::unique_ptr<CsrMtx> csr_mtx0;
    std::unique_ptr<CsrMtx> csr_mtx10;
    std::unique_ptr<CsrMtx> csr_mtx11;
    std::unique_ptr<CsrMtx> csr_mtx111;
    std::unique_ptr<CsrMtx> csr_mtx1;
    std::unique_ptr<CsrMtx> csr_mtx2;
    std::unique_ptr<CsrMtx> csr_mtx20;
    std::unique_ptr<CsrMtx> csr_mtx21;
    std::unique_ptr<CsrMtx> csr_mtx22;
    std::unique_ptr<CsrMtx> csr_mtx222;
    std::unique_ptr<CsrMtx> csr_mtx3;
    std::unique_ptr<CsrMtx> csr_mtx30;
    std::unique_ptr<CsrMtx> csr_mtx31;
    std::unique_ptr<CsrMtx> csr_mtx32;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Dense> b;
    std::unique_ptr<Dense> b0;
    std::unique_ptr<Dense> b1;
    std::unique_ptr<Dense> x;
    std::unique_ptr<Dense> x0;
    std::unique_ptr<Dense> x1;
};

TYPED_TEST_SUITE(SubMatrix, gko::test::ValueIndexTypes);


TYPED_TEST(SubMatrix, CanApplyToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto rspan = gko::span(0, 2);
    auto cspan = gko::span(0, 2);
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan);
    auto s_x = Dense::create(this->exec);
    s_x->copy_from(this->x0.get());

    mtx->apply(this->b0.get(), s_x.get());
    this->csr_mtx0->apply(this->b0.get(), this->x0.get());

    GKO_EXPECT_MTX_NEAR(mtx->get_sub_matrix(), this->csr_mtx0,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(this->x0, s_x, r<value_type>::value);
}


TYPED_TEST(SubMatrix, CanApplyToDenseWithOverlap)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto rspan = gko::span(2, 5);
    auto cspan = gko::span(1, 4);
    auto ov_rspan = std::vector<gko::span>{gko::span(2, 5), gko::span(2, 5)};
    auto ov_cspan = std::vector<gko::span>{gko::span(0, 1), gko::span(4, 5)};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan,
                           ov_rspan, ov_cspan);
    auto s_x = Dense::create(this->exec);
    s_x->copy_from(this->x.get());

    mtx->apply(this->b.get(), s_x.get());
    this->csr_mtx111->apply(this->b.get(), this->x.get());

    GKO_EXPECT_MTX_NEAR(mtx->get_sub_matrix(), this->csr_mtx1,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[0], this->csr_mtx10,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[1], this->csr_mtx11,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(this->x, s_x, r<value_type>::value);
}


TYPED_TEST(SubMatrix, CanApplyToDenseWithNonUnitOverlap)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto rspan = gko::span(1, 4);
    auto cspan = gko::span(1, 3);
    auto ov_rspan = std::vector<gko::span>{gko::span(1, 4), gko::span(1, 4),
                                           gko::span(1, 4)};
    auto ov_cspan = std::vector<gko::span>{gko::span(0, 1), gko::span(3, 4),
                                           gko::span(4, 5)};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan,
                           ov_rspan, ov_cspan);
    auto s_x = Dense::create(this->exec);
    s_x->copy_from(this->x.get());

    mtx->apply(this->b.get(), s_x.get());
    this->csr_mtx222->apply(this->b.get(), this->x.get());

    GKO_EXPECT_MTX_NEAR(mtx->get_sub_matrix(), this->csr_mtx2,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[0], this->csr_mtx20,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[1], this->csr_mtx21,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[2], this->csr_mtx22,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(this->x, s_x, r<value_type>::value);
}


TYPED_TEST(SubMatrix, CanApplyToDenseWithNonUnitLeftOverlap)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto rspan = gko::span(1, 4);
    auto cspan = gko::span(2, 4);
    auto ov_rspan = std::vector<gko::span>{gko::span(1, 4), gko::span(1, 4),
                                           gko::span(1, 4)};
    auto ov_cspan = std::vector<gko::span>{gko::span(0, 1), gko::span(1, 2),
                                           gko::span(4, 5)};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan,
                           ov_rspan, ov_cspan);
    auto s_x = Dense::create(this->exec);
    s_x->copy_from(this->x.get());

    mtx->apply(this->b.get(), s_x.get());
    this->csr_mtx222->apply(this->b.get(), this->x.get());

    GKO_EXPECT_MTX_NEAR(mtx->get_sub_matrix(), this->csr_mtx3,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[0], this->csr_mtx30,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[1], this->csr_mtx31,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[2], this->csr_mtx32,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(this->x, s_x, r<value_type>::value);
}


TYPED_TEST(SubMatrix, CanAdvancedApplyToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto rspan = gko::span(1, 4);
    auto cspan = gko::span(2, 4);
    auto ov_rspan = std::vector<gko::span>{gko::span(1, 4), gko::span(1, 4),
                                           gko::span(1, 4)};
    auto ov_cspan = std::vector<gko::span>{gko::span(0, 1), gko::span(1, 2),
                                           gko::span(4, 5)};
    auto mtx = Mtx::create(this->exec, this->csr_mtx.get(), rspan, cspan,
                           ov_rspan, ov_cspan);
    auto s_x = Dense::create(this->exec);
    auto alpha = gko::initialize<Dense>({2.0}, this->exec);
    auto beta = gko::initialize<Dense>({-1.0}, this->exec);
    s_x->copy_from(this->x.get());

    mtx->apply(alpha.get(), this->b.get(), beta.get(), s_x.get());

    this->csr_mtx222->apply(alpha.get(), this->b.get(), beta.get(),
                            this->x.get());

    GKO_EXPECT_MTX_NEAR(mtx->get_sub_matrix(), this->csr_mtx3,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[0], this->csr_mtx30,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[1], this->csr_mtx31,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(mtx->get_overlap_mtxs()[2], this->csr_mtx32,
                        r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(this->x, s_x, r<value_type>::value);
}


}  // namespace
