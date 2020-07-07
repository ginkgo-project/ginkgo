/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/factorization/par_ilut.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/par_ilut_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class DummyLinOp : public gko::EnableLinOp<DummyLinOp>,
                   public gko::EnableCreateMethod<DummyLinOp> {
public:
    DummyLinOp(std::shared_ptr<const gko::Executor> exec,
               gko::dim<2> size = gko::dim<2>{})
        : EnableLinOp<DummyLinOp>(exec, size)
    {}

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override {}

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {}
};


template <typename ValueIndexType>
class ParIlut : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using factorization_type =
        gko::factorization::ParIlut<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using ComplexCsr =
        gko::matrix::Csr<std::complex<gko::remove_complex<value_type>>,
                         index_type>;

    ParIlut()
        : ref(gko::ReferenceExecutor::create()),
          exec(std::static_pointer_cast<const gko::Executor>(ref)),

          mtx1(gko::initialize<Csr>({{.1, 0., 0., 0.},
                                     {.1, .1, 0., 0.},
                                     {-1., -2., -1., 0.},
                                     {-2., -3., -1., 1.}},
                                    ref)),
          mtx1_expect_thrm2(gko::initialize<Csr>({{.1, 0., 0., 0.},
                                                  {0., .1, 0., 0.},
                                                  {0., -2., -1., 0.},
                                                  {-2., -3., 0., 1.}},
                                                 ref)),
          mtx1_expect_thrm3(gko::initialize<Csr>({{.1, 0., 0., 0.},
                                                  {0., .1, 0., 0.},
                                                  {0., 0., -1., 0.},
                                                  {0., -3., 0., 1.}},
                                                 ref)),
          mtx1_complex(gko::initialize<ComplexCsr>(
              {{{.1, 0.}, {0., 0.}, {0., 0.}, {0., 0.}},
               {{-1., .1}, {.1, -1.}, {0., 0.}, {0., 0.}},
               {{-1., 1.}, {-2., .2}, {-1., -.3}, {0., 0.}},
               {{1., -2.}, {-3., -.1}, {-1., .1}, {.1, 2.}}},
              ref)),
          mtx1_expect_complex_thrm(gko::initialize<ComplexCsr>(
              {{{.1, 0.}, {0., 0.}, {0., 0.}, {0., 0.}},
               {{0., 0.}, {.1, -1.}, {0., 0.}, {0., 0.}},
               {{-1., 1.}, {-2., .2}, {-1., -.3}, {0., 0.}},
               {{1., -2.}, {-3., -.1}, {0., 0.}, {.1, 2.}}},
              ref)),
          identity(gko::initialize<Csr>(
              {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}, ref)),
          lower_tri(gko::initialize<Csr>(
              {{1., 0., 0.}, {1., 1., 0.}, {1., 1., 1.}}, ref)),
          upper_tri(gko::initialize<Csr>(
              {{2., 1., 1.}, {0., -3., 1.}, {0., 0., 4.}}, ref)),
          mtx_system(gko::initialize<Csr>({{1., 6., 4., 7.},
                                           {2., -5., 0., 8.},
                                           {.5, -3., 6., 0.},
                                           {.2, -.5, -9., 0.}},
                                          ref)),
          mtx_l_system(gko::initialize<Csr>({{1., 0., 0., 0.},
                                             {2., 1., 0., 0.},
                                             {.5, -3., 1., 0.},
                                             {.2, -.5, -9., 1.}},
                                            ref)),
          mtx_u_system(gko::initialize<Csr>({{1., 6., 4., 7.},
                                             {0., 1., 0., 8.},
                                             {0., 0., 6., 0.},
                                             {0., 0., 0., 1.}},
                                            ref)),
          mtx_l(gko::initialize<Csr>({{1., 0., 0., 0.},
                                      {4., 1., 0., 0.},
                                      {-1., 0., 1., 0.},
                                      {0., -3., -1., 1.}},
                                     ref)),
          mtx_u(gko::initialize<Csr>({{2., 0., 1., 1.},
                                      {0., 3., 0., 2.},
                                      {0., 0., .5, 0.},
                                      {0., 0., 0., 4.}},
                                     ref)),
          mtx_lu(gko::initialize<Csr>({{1., 2., 3., 4.},
                                       {0., 6., 7., 8.},
                                       {9., .1, .2, 0.},
                                       {.3, .4, .5, .6}},
                                      ref)),
          mtx_l_add_expect(gko::initialize<Csr>({{1., 0., 0., 0.},
                                                 {4., 1., 0., 0.},
                                                 {-1., -3.1 / 3., 1., 0.},
                                                 {-.05, -3., -1., 1.}},
                                                ref)),
          mtx_u_add_expect(gko::initialize<Csr>({{2., 4., 1., 1.},
                                                 {0., 3., -7., 2.},
                                                 {0., 0., .5, 0.},
                                                 {0., 0., 0., 4.}},
                                                ref)),
          mtx_l_it_expect(gko::initialize<Csr>({{1., 0., 0., 0.},
                                                {2., 1., 0., 0.},
                                                {.5, 6. / 17., 1., 0.},
                                                {.2, .1, -2.45, 1.}},
                                               ref)),
          mtx_u_it_expect(gko::initialize<Csr>({{1., 0., 0., 0.},
                                                {6., -17., 0., 0.},
                                                {4., 0., 4., 0.},
                                                {7., -6., 0., -.8}},
                                               ref)),
          mtx_l_small_expect(gko::initialize<Csr>({{1., 0., 0., 0.},
                                                   {2., 1., 0., 0.},
                                                   {.5, 6. / 17., 1., 0.},
                                                   {0., 0., -153. / 116., 1.}},
                                                  ref)),
          mtx_u_small_expect(gko::initialize<Csr>({{1., 6., 4., 7.},
                                                   {0., -17., -8., -6.},
                                                   {0., 0., 116. / 17., 0.},
                                                   {0., 0., 0., .0}},
                                                  ref)),
          mtx_l_large_expect(
              gko::initialize<Csr>({{1., 0., 0., 0.},
                                    {2., 1., 0., 0.},
                                    {.5, 6. / 17., 1., 0.},
                                    {0.2, 0.1, -153. / 116., 1.}},
                                   ref)),
          mtx_u_large_expect(
              gko::initialize<Csr>({{1., 6., 4., 7.},
                                    {0., -17., -8., -6.},
                                    {0., 0., 116. / 17., -47. / 34.},
                                    {0., 0., 0., -3043. / 1160.}},
                                   ref)),
          fact_fact(factorization_type::build().on(exec)),
          tol{r<value_type>::value}
    {}

    template <typename Mtx>
    void test_select(const std::unique_ptr<Mtx> &mtx, index_type rank,
                     gko::remove_complex<value_type> expected,
                     gko::remove_complex<value_type> tolerance = 0.0)
    {
        using ValueType = typename Mtx::value_type;
        gko::remove_complex<ValueType> result{};

        gko::remove_complex<ValueType> res{};
        gko::remove_complex<ValueType> dres{};
        gko::Array<ValueType> tmp(ref);
        gko::Array<gko::remove_complex<ValueType>> tmp2(ref);
        gko::kernels::reference::par_ilut_factorization::threshold_select(
            ref, mtx.get(), rank, tmp, tmp2, result);

        ASSERT_NEAR(result, expected, tolerance);
    }

    template <typename Mtx,
              typename Coo = gko::matrix::Coo<typename Mtx::value_type,
                                              typename Mtx::index_type>>
    void test_filter(const std::unique_ptr<Mtx> &mtx,
                     gko::remove_complex<value_type> threshold,
                     const std::unique_ptr<Mtx> &expected, bool lower)
    {
        auto res_mtx = Mtx::create(exec, mtx->get_size());
        auto res_mtx_coo = Coo::create(exec, mtx->get_size());

        auto local_mtx = gko::as<Mtx>(lower ? mtx->clone() : mtx->transpose());
        auto local_expected =
            gko::as<Mtx>(lower ? expected->clone() : expected->transpose());

        gko::kernels::reference::par_ilut_factorization::threshold_filter(
            ref, local_mtx.get(), threshold, res_mtx.get(), res_mtx_coo.get(),
            lower);

        GKO_ASSERT_MTX_EQ_SPARSITY(local_expected, res_mtx);
        GKO_ASSERT_MTX_NEAR(local_expected, res_mtx, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx, res_mtx_coo);
        GKO_ASSERT_MTX_NEAR(res_mtx, res_mtx_coo, 0);
    }

    template <typename Mtx,
              typename Coo = gko::matrix::Coo<typename Mtx::value_type,
                                              typename Mtx::index_type>>
    void test_filter_approx(const std::unique_ptr<Mtx> &mtx, index_type rank,
                            const std::unique_ptr<Mtx> &expected)
    {
        auto res_mtx = Mtx::create(exec, mtx->get_size());
        auto res_mtx_coo = Coo::create(exec, mtx->get_size());
        auto res_mtx2 = Mtx::create(exec, mtx->get_size());
        auto res_mtx_coo2 = Coo::create(exec, mtx->get_size());

        auto tmp = gko::Array<typename Mtx::value_type>{exec};
        gko::remove_complex<typename Mtx::value_type> threshold{};
        gko::kernels::reference::par_ilut_factorization::
            threshold_filter_approx(ref, mtx.get(), rank, tmp, threshold,
                                    res_mtx.get(), res_mtx_coo.get());
        gko::kernels::reference::par_ilut_factorization::threshold_filter(
            ref, mtx.get(), threshold, res_mtx2.get(), res_mtx_coo2.get(),
            true);

        GKO_ASSERT_MTX_EQ_SPARSITY(expected, res_mtx);
        GKO_ASSERT_MTX_EQ_SPARSITY(expected, res_mtx2);
        GKO_ASSERT_MTX_NEAR(expected, res_mtx, 0);
        GKO_ASSERT_MTX_NEAR(expected, res_mtx2, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx, res_mtx_coo);
        GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx, res_mtx_coo2);
        GKO_ASSERT_MTX_NEAR(res_mtx, res_mtx_coo, 0);
        GKO_ASSERT_MTX_NEAR(res_mtx, res_mtx_coo2, 0);
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Csr> mtx1;
    std::unique_ptr<Csr> mtx1_expect_thrm2;
    std::unique_ptr<Csr> mtx1_expect_thrm3;
    std::unique_ptr<ComplexCsr> mtx1_complex;
    std::unique_ptr<ComplexCsr> mtx1_expect_complex_thrm;
    std::shared_ptr<Csr> identity;
    std::shared_ptr<Csr> lower_tri;
    std::shared_ptr<Csr> upper_tri;
    std::shared_ptr<Csr> mtx_system;
    std::unique_ptr<Csr> mtx_l_system;
    std::unique_ptr<Csr> mtx_u_system;
    std::unique_ptr<Csr> mtx_l;
    std::unique_ptr<Csr> mtx_u;
    std::unique_ptr<Csr> mtx_lu;
    std::unique_ptr<Csr> mtx_l_add_expect;
    std::unique_ptr<Csr> mtx_u_add_expect;
    std::unique_ptr<Csr> mtx_l_it_expect;
    std::unique_ptr<Csr> mtx_u_it_expect;
    std::unique_ptr<Csr> mtx_l_small_expect;
    std::unique_ptr<Csr> mtx_u_small_expect;
    std::unique_ptr<Csr> mtx_l_large_expect;
    std::unique_ptr<Csr> mtx_u_large_expect;
    std::unique_ptr<typename factorization_type::Factory> fact_fact;
    gko::remove_complex<value_type> tol;
};  // namespace

TYPED_TEST_CASE(ParIlut, gko::test::ValueIndexTypes);


TYPED_TEST(ParIlut, KernelThresholdSelect)
{
    this->test_select(this->mtx1, 7, 2.0);
}


TYPED_TEST(ParIlut, KernelThresholdSelectMin)
{
    this->test_select(this->mtx1, 0, 0.1);
}


TYPED_TEST(ParIlut, KernelThresholdSelectMax)
{
    this->test_select(this->mtx1, 9, 3.0);
}


TYPED_TEST(ParIlut, KernelComplexThresholdSelect)
{
    using value_type = typename TestFixture::value_type;
    this->test_select(this->mtx1_complex, 5, sqrt(2), this->tol);
}


TYPED_TEST(ParIlut, KernelComplexThresholdSelectMin)
{
    using value_type = typename TestFixture::value_type;
    this->test_select(this->mtx1_complex, 0, 0.1, this->tol);
}


TYPED_TEST(ParIlut, KernelComplexThresholdSelectMax)
{
    using value_type = typename TestFixture::value_type;
    this->test_select(this->mtx1_complex, 9, sqrt(9.01), this->tol);
}


TYPED_TEST(ParIlut, KernelThresholdFilterNullptrCoo)
{
    using Csr = typename TestFixture::Csr;
    using Coo = typename TestFixture::Coo;
    auto res_mtx = Csr::create(this->exec, this->mtx1->get_size());
    Coo *null_coo = nullptr;

    gko::kernels::reference::par_ilut_factorization::threshold_filter(
        this->ref, this->mtx1.get(), 0.0, res_mtx.get(), null_coo, true);

    GKO_ASSERT_MTX_EQ_SPARSITY(this->mtx1, res_mtx);
    GKO_ASSERT_MTX_NEAR(this->mtx1, res_mtx, 0);
}


TYPED_TEST(ParIlut, KernelThresholdFilterNoneLower)
{
    this->test_filter(this->mtx1, 0.0, this->mtx1, true);
}


TYPED_TEST(ParIlut, KernelThresholdFilterNoneUpper)
{
    this->test_filter(this->mtx1, 0.0, this->mtx1, false);
}


TYPED_TEST(ParIlut, KernelThresholdFilterSomeAtThresholdLower)
{
    this->test_filter(this->mtx1, 2.0, this->mtx1_expect_thrm2, true);
}


TYPED_TEST(ParIlut, KernelThresholdFilterSomeAtThresholdUpper)
{
    this->test_filter(this->mtx1, 2.0, this->mtx1_expect_thrm2, false);
}


TYPED_TEST(ParIlut, KernelThresholdFilterSomeAboveThresholdLower)
{
    this->test_filter(this->mtx1, 3.0, this->mtx1_expect_thrm3, true);
}


TYPED_TEST(ParIlut, KernelThresholdFilterSomeAboveThresholdUpper)
{
    this->test_filter(this->mtx1, 3.0, this->mtx1_expect_thrm3, false);
}


TYPED_TEST(ParIlut, KernelComplexThresholdFilterNoneLower)
{
    this->test_filter(this->mtx1_complex, 0.0, this->mtx1_complex, true);
}


TYPED_TEST(ParIlut, KernelComplexThresholdFilterNoneUpper)
{
    this->test_filter(this->mtx1_complex, 0.0, this->mtx1_complex, false);
}


TYPED_TEST(ParIlut, KernelComplexThresholdFilterSomeAtThresholdLower)
{
    this->test_filter(this->mtx1_complex, 1.01, this->mtx1_expect_complex_thrm,
                      true);
}


TYPED_TEST(ParIlut, KernelComplexThresholdFilterSomeAtThresholdUpper)
{
    this->test_filter(this->mtx1_complex, 1.01, this->mtx1_expect_complex_thrm,
                      false);
}


TYPED_TEST(ParIlut, KernelThresholdFilterApproxNullptrCoo)
{
    using Csr = typename TestFixture::Csr;
    using Coo = typename TestFixture::Coo;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto res_mtx = Csr::create(this->exec, this->mtx1->get_size());
    auto tmp = gko::Array<value_type>{this->ref};
    gko::remove_complex<value_type> threshold{};
    Coo *null_coo = nullptr;
    index_type rank{};

    gko::kernels::reference::par_ilut_factorization::threshold_filter_approx(
        this->ref, this->mtx1.get(), rank, tmp, threshold, res_mtx.get(),
        null_coo);

    GKO_ASSERT_MTX_EQ_SPARSITY(this->mtx1, res_mtx);
    GKO_ASSERT_MTX_NEAR(this->mtx1, res_mtx, 0);
}


TYPED_TEST(ParIlut, KernelThresholdFilterSomeApprox1)
{
    this->test_filter_approx(this->mtx1, 7, this->mtx1_expect_thrm2);
}


TYPED_TEST(ParIlut, KernelThresholdFilterSomeApprox2)
{
    this->test_filter_approx(this->mtx1, 8, this->mtx1_expect_thrm2);
}


TYPED_TEST(ParIlut, KernelThresholdFilterNoneApprox)
{
    this->test_filter_approx(this->mtx1, 0, this->mtx1);
}


TYPED_TEST(ParIlut, KernelComplexThresholdFilterSomeApprox)
{
    this->test_filter_approx(this->mtx1_complex, 4,
                             this->mtx1_expect_complex_thrm);
}


TYPED_TEST(ParIlut, KernelComplexThresholdFilterNoneApprox)
{
    this->test_filter_approx(this->mtx1_complex, 0, this->mtx1_complex);
}


TYPED_TEST(ParIlut, KernelAddCandidates)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto res_mtx_l = Csr::create(this->exec, this->mtx_system->get_size());
    auto res_mtx_u = Csr::create(this->exec, this->mtx_system->get_size());

    gko::kernels::reference::par_ilut_factorization::add_candidates(
        this->ref, this->mtx_lu.get(), this->mtx_system.get(),
        this->mtx_l.get(), this->mtx_u.get(), res_mtx_l.get(), res_mtx_u.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_l, this->mtx_l_add_expect);
    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_u, this->mtx_u_add_expect);
    GKO_ASSERT_MTX_NEAR(res_mtx_l, this->mtx_l_add_expect, this->tol);
    GKO_ASSERT_MTX_NEAR(res_mtx_u, this->mtx_u_add_expect, this->tol);
}


TYPED_TEST(ParIlut, KernelComputeLU)
{
    using Csr = typename TestFixture::Csr;
    using Coo = typename TestFixture::Coo;
    using value_type = typename TestFixture::value_type;
    auto mtx_l_coo = Coo::create(this->exec, this->mtx_system->get_size());
    this->mtx_l_system->convert_to(mtx_l_coo.get());
    auto mtx_u_transp = this->mtx_u_system->transpose();
    auto mtx_u_coo = Coo::create(this->exec, this->mtx_system->get_size());
    this->mtx_u_system->convert_to(mtx_u_coo.get());
    auto mtx_u_csc = gko::as<Csr>(mtx_u_transp.get());

    gko::kernels::reference::par_ilut_factorization::compute_l_u_factors(
        this->ref, this->mtx_system.get(), this->mtx_l_system.get(),
        mtx_l_coo.get(), this->mtx_u_system.get(), mtx_u_coo.get(), mtx_u_csc);
    auto mtx_utt = gko::as<Csr>(mtx_u_csc->transpose());

    GKO_ASSERT_MTX_NEAR(this->mtx_l_system, this->mtx_l_it_expect, this->tol);
    GKO_ASSERT_MTX_NEAR(mtx_u_csc, this->mtx_u_it_expect, this->tol);
    GKO_ASSERT_MTX_NEAR(this->mtx_u_system, mtx_utt, 0);
}


TYPED_TEST(ParIlut, ThrowNotSupportedForWrongLinOp)
{
    auto lin_op = DummyLinOp::create(this->ref);

    ASSERT_THROW(this->fact_fact->generate(gko::share(lin_op)),
                 gko::NotSupported);
}


TYPED_TEST(ParIlut, ThrowDimensionMismatch)
{
    using Csr = typename TestFixture::Csr;
    auto matrix = Csr::create(this->ref, gko::dim<2>{2, 3}, 4);

    ASSERT_THROW(this->fact_fact->generate(gko::share(matrix)),
                 gko::DimensionMismatch);
}


TYPED_TEST(ParIlut, SetStrategies)
{
    using Csr = typename TestFixture::Csr;
    using factorization_type = typename TestFixture::factorization_type;
    auto l_strategy = std::make_shared<typename Csr::merge_path>();
    auto u_strategy = std::make_shared<typename Csr::classical>();

    auto factory = factorization_type::build()
                       .with_l_strategy(l_strategy)
                       .with_u_strategy(u_strategy)
                       .on(this->ref);
    auto fact = factory->generate(this->mtx_system);

    ASSERT_EQ(factory->get_parameters().l_strategy, l_strategy);
    ASSERT_EQ(fact->get_l_factor()->get_strategy()->get_name(),
              l_strategy->get_name());
    ASSERT_EQ(factory->get_parameters().u_strategy, u_strategy);
    ASSERT_EQ(fact->get_u_factor()->get_strategy()->get_name(),
              u_strategy->get_name());
}


TYPED_TEST(ParIlut, IsConsistentWithComposition)
{
    auto fact = this->fact_fact->generate(this->mtx_system);

    auto lin_op_l_factor =
        static_cast<const gko::LinOp *>(gko::lend(fact->get_l_factor()));
    auto lin_op_u_factor =
        static_cast<const gko::LinOp *>(gko::lend(fact->get_u_factor()));
    auto first_operator = gko::lend(fact->get_operators()[0]);
    auto second_operator = gko::lend(fact->get_operators()[1]);

    ASSERT_EQ(lin_op_l_factor, first_operator);
    ASSERT_EQ(lin_op_u_factor, second_operator);
}


TYPED_TEST(ParIlut, GenerateIdentity)
{
    auto fact = this->fact_fact->generate(this->identity);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->identity, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), this->identity, this->tol);
}


TYPED_TEST(ParIlut, GenerateDenseIdentity)
{
    using Dense = typename TestFixture::Dense;
    auto dense_id = Dense::create(this->exec, this->identity->get_size());
    this->identity->convert_to(dense_id.get());
    auto fact = this->fact_fact->generate(gko::share(dense_id));

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->identity, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), this->identity, this->tol);
}


TYPED_TEST(ParIlut, GenerateLowerTri)
{
    auto fact = this->fact_fact->generate(this->lower_tri);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->lower_tri, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), this->identity, this->tol);
}


TYPED_TEST(ParIlut, GenerateUpperTri)
{
    auto fact = this->fact_fact->generate(this->upper_tri);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->identity, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), this->upper_tri, this->tol);
}


TYPED_TEST(ParIlut, GenerateWithExactSmallLimit)
{
    using factorization_type = typename TestFixture::factorization_type;
    auto fact = factorization_type::build()
                    .with_approximate_select(false)
                    .with_fill_in_limit(0.75)
                    .on(this->exec)
                    ->generate(this->mtx_system);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->mtx_l_small_expect,
                        this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), this->mtx_u_small_expect,
                        this->tol);
}


TYPED_TEST(ParIlut, GenerateWithApproxSmallLimit)
{
    using factorization_type = typename TestFixture::factorization_type;
    auto fact = factorization_type::build()
                    .with_approximate_select(true)
                    .with_fill_in_limit(0.75)
                    .on(this->exec)
                    ->generate(this->mtx_system);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->mtx_l_small_expect,
                        this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), this->mtx_u_small_expect,
                        this->tol);
}


TYPED_TEST(ParIlut, GenerateWithExactLargeLimit)
{
    using factorization_type = typename TestFixture::factorization_type;
    auto fact = factorization_type::build()
                    .with_approximate_select(false)
                    .with_fill_in_limit(1.2)
                    .on(this->exec)
                    ->generate(this->mtx_system);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->mtx_l_large_expect,
                        this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), this->mtx_u_large_expect,
                        this->tol);
}


TYPED_TEST(ParIlut, GenerateWithApproxLargeLimit)
{
    using factorization_type = typename TestFixture::factorization_type;
    auto fact = factorization_type::build()
                    .with_approximate_select(true)
                    .with_fill_in_limit(1.2)
                    .on(this->exec)
                    ->generate(this->mtx_system);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->mtx_l_large_expect,
                        this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), this->mtx_u_large_expect,
                        this->tol);
}


}  // namespace
