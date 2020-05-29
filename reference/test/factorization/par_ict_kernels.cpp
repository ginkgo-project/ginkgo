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

#include <ginkgo/core/factorization/par_ict.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/par_ict_kernels.hpp"
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
class ParIct : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using factorization_type =
        gko::factorization::ParIct<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;

    ParIct()
        : ref(gko::ReferenceExecutor::create()),
          exec(std::static_pointer_cast<const gko::Executor>(ref)),
          identity(gko::initialize<Csr>(
              {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}, ref)),
          lower_tri(gko::initialize<Csr>(
              {{1., 0., 0.}, {1., 1., 0.}, {1., 1., 1.}}, ref)),
          upper_tri(gko::initialize<Csr>(
              {{2., 1., 1.}, {0., -3., 1.}, {0., 0., 4.}}, ref)),
          mtx_system(gko::initialize<Csr>({{9., 0., -6., 3.},
                                           {0., 36., 18., 24.},
                                           {-6., 18., 17., 14.},
                                           {-3., 24., 14., 18.}},
                                          ref)),
          mtx_init(gko::initialize<Csr>({{9., 0., -6., 3.},
                                         {0., 0., 18., 24.},
                                         {-6., 18., 17., 14.},
                                         {-3., 24., 14., 18.}},
                                        ref)),
          mtx_l_system(gko::initialize<Csr>({{1., 0., 0., 0.},
                                             {0., 1., 0., 0.},
                                             {1., 1., 1., 0.},
                                             {1., 1., 0., 1.}},
                                            ref)),
          mtx_l(gko::initialize<Csr>({{1., 0., 0., 0.},
                                      {1., 2., 0., 0.},
                                      {0., 0., 3., 0.},
                                      {-2., 0., -3., 4.}},
                                     ref)),
          mtx_llt(gko::initialize<Csr>({{1., 1., 0., -2.},
                                        {1., 5., 0., -2.},
                                        {0., 0., 9., -9.},
                                        {-2., -2., -9., 29.}},
                                       ref)),
          mtx_l_init_expect(gko::initialize<Csr>(
              {{3., 0., 0., 0.},
               {0., 1., 0., 0.},
               {-6., 18., static_cast<value_type>(sqrt(17.)), 0.},
               {-3., 24., 14., static_cast<value_type>(sqrt(18.))}},
              ref)),
          mtx_l_add_expect(gko::initialize<Csr>({{1., 0., 0., 0.},
                                                 {1., 2., 0., 0.},
                                                 {-6., 9., 3., 0.},
                                                 {-2., 13., -3., 4.}},
                                                ref)),
          mtx_l_it_expect(gko::initialize<Csr>({{3., 0., 0., 0.},
                                                {0., 6., 0., 0.},
                                                {-2., 3., 2., 0.},
                                                {-1., 4., 0., 1.}},
                                               ref)),
          mtx_l_small_expect(gko::initialize<Csr>(
              {{3., 0., 0., 0.},
               {0., 6., 0., 0.},
               {-2., 3., 2., 0.},
               {0., 4., 0., static_cast<value_type>(sqrt(2.))}},
              ref)),
          mtx_l_large_expect(gko::initialize<Csr>({{3., 0., 0., 0.},
                                                   {0., 6., 0., 0.},
                                                   {-2., 3., 2., 0.},
                                                   {-1., 4., 0., 1.}},
                                                  ref)),
          fact_fact(factorization_type::build().on(exec)),
          tol{r<value_type>::value}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Csr> identity;
    std::shared_ptr<Csr> lower_tri;
    std::shared_ptr<Csr> upper_tri;
    std::shared_ptr<Csr> mtx_system;
    std::unique_ptr<Csr> mtx_l_system;
    std::unique_ptr<Csr> mtx_init;
    std::unique_ptr<Csr> mtx_l;
    std::unique_ptr<Csr> mtx_llt;
    std::unique_ptr<Csr> mtx_l_init_expect;
    std::unique_ptr<Csr> mtx_l_add_expect;
    std::unique_ptr<Csr> mtx_l_it_expect;
    std::unique_ptr<Csr> mtx_l_small_expect;
    std::unique_ptr<Csr> mtx_l_large_expect;
    std::unique_ptr<typename factorization_type::Factory> fact_fact;
    gko::remove_complex<value_type> tol;
};

TYPED_TEST_CASE(ParIct, gko::test::ValueIndexTypes);


TYPED_TEST(ParIct, KernelInitializeRowPtrsL)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto res_mtx_l = Csr::create(this->exec, this->mtx_system->get_size());
    auto row_ptrs = res_mtx_l->get_const_row_ptrs();

    gko::kernels::reference::factorization::initialize_row_ptrs_l(
        this->ref, this->mtx_system.get(), res_mtx_l->get_row_ptrs());

    ASSERT_EQ(row_ptrs[0], 0);
    ASSERT_EQ(row_ptrs[1], 1);
    ASSERT_EQ(row_ptrs[2], 2);
    ASSERT_EQ(row_ptrs[3], 5);
    ASSERT_EQ(row_ptrs[4], 9);
}


TYPED_TEST(ParIct, KernelInitializeL)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto res_mtx_l = Csr::create(this->exec, this->mtx_system->get_size(), 9);
    auto row_ptrs = res_mtx_l->get_const_row_ptrs();

    gko::kernels::reference::factorization::initialize_row_ptrs_l(
        this->ref, this->mtx_init.get(), res_mtx_l->get_row_ptrs());
    gko::kernels::reference::factorization::initialize_l(
        this->ref, this->mtx_init.get(), res_mtx_l.get(), true);

    GKO_ASSERT_MTX_NEAR(res_mtx_l, this->mtx_l_init_expect, this->tol);
    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_l, this->mtx_l_init_expect);
}


TYPED_TEST(ParIct, KernelAddCandidates)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto res_mtx_l = Csr::create(this->exec, this->mtx_system->get_size());

    gko::kernels::reference::par_ict_factorization::add_candidates(
        this->ref, this->mtx_llt.get(), this->mtx_system.get(),
        this->mtx_l.get(), res_mtx_l.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_l, this->mtx_l_add_expect);
    GKO_ASSERT_MTX_NEAR(res_mtx_l, this->mtx_l_add_expect, this->tol);
}


TYPED_TEST(ParIct, KernelComputeLU)
{
    using Csr = typename TestFixture::Csr;
    using Coo = typename TestFixture::Coo;
    using value_type = typename TestFixture::value_type;
    auto mtx_l_coo = Coo::create(this->exec, this->mtx_system->get_size());
    this->mtx_l_system->convert_to(mtx_l_coo.get());

    gko::kernels::reference::par_ict_factorization::compute_factor(
        this->ref, this->mtx_system.get(), this->mtx_l_system.get(),
        mtx_l_coo.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_l_system, this->mtx_l_it_expect, this->tol);
}


TYPED_TEST(ParIct, ThrowNotSupportedForWrongLinOp)
{
    auto lin_op = DummyLinOp::create(this->ref);

    ASSERT_THROW(this->fact_fact->generate(gko::share(lin_op)),
                 gko::NotSupported);
}


TYPED_TEST(ParIct, ThrowDimensionMismatch)
{
    using Csr = typename TestFixture::Csr;
    auto matrix = Csr::create(this->ref, gko::dim<2>{2, 3}, 4);

    ASSERT_THROW(this->fact_fact->generate(gko::share(matrix)),
                 gko::DimensionMismatch);
}


TYPED_TEST(ParIct, SetStrategies)
{
    using Csr = typename TestFixture::Csr;
    using factorization_type = typename TestFixture::factorization_type;
    auto l_strategy = std::make_shared<typename Csr::merge_path>();
    auto lt_strategy = std::make_shared<typename Csr::classical>();

    auto factory = factorization_type::build()
                       .with_l_strategy(l_strategy)
                       .with_lt_strategy(lt_strategy)
                       .on(this->ref);
    auto fact = factory->generate(this->mtx_system);

    ASSERT_EQ(factory->get_parameters().l_strategy, l_strategy);
    ASSERT_EQ(fact->get_l_factor()->get_strategy(), l_strategy);
    ASSERT_EQ(factory->get_parameters().lt_strategy, lt_strategy);
    ASSERT_EQ(fact->get_lt_factor()->get_strategy(), lt_strategy);
}


TYPED_TEST(ParIct, IsConsistentWithComposition)
{
    auto fact = this->fact_fact->generate(this->mtx_system);

    auto lin_op_l_factor =
        static_cast<const gko::LinOp *>(gko::lend(fact->get_l_factor()));
    auto lin_op_lt_factor =
        static_cast<const gko::LinOp *>(gko::lend(fact->get_lt_factor()));
    auto first_operator = gko::lend(fact->get_operators()[0]);
    auto second_operator = gko::lend(fact->get_operators()[1]);

    ASSERT_EQ(lin_op_l_factor, first_operator);
    ASSERT_EQ(lin_op_lt_factor, second_operator);
}


TYPED_TEST(ParIct, GenerateIdentity)
{
    auto fact = this->fact_fact->generate(this->identity);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->identity, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(), this->identity, this->tol);
}


TYPED_TEST(ParIct, GenerateDenseIdentity)
{
    using Dense = typename TestFixture::Dense;
    auto dense_id = Dense::create(this->exec, this->identity->get_size());
    this->identity->convert_to(dense_id.get());
    auto fact = this->fact_fact->generate(gko::share(dense_id));

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->identity, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(), this->identity, this->tol);
}


TYPED_TEST(ParIct, GenerateWithExactSmallLimit)
{
    using factorization_type = typename TestFixture::factorization_type;
    using Csr = typename TestFixture::Csr;
    auto fact = factorization_type::build()
                    .with_approximate_select(false)
                    .with_fill_in_limit(0.6)
                    .on(this->exec)
                    ->generate(this->mtx_system);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->mtx_l_small_expect,
                        this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(),
                        gko::as<Csr>(this->mtx_l_small_expect->transpose()),
                        this->tol);
}


TYPED_TEST(ParIct, GenerateWithApproxSmallLimit)
{
    using factorization_type = typename TestFixture::factorization_type;
    using Csr = typename TestFixture::Csr;
    auto fact = factorization_type::build()
                    .with_approximate_select(true)
                    .with_fill_in_limit(0.6)
                    .on(this->exec)
                    ->generate(this->mtx_system);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->mtx_l_small_expect,
                        this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(),
                        gko::as<Csr>(this->mtx_l_small_expect->transpose()),
                        this->tol);
}


TYPED_TEST(ParIct, GenerateWithExactLargeLimit)
{
    using factorization_type = typename TestFixture::factorization_type;
    using Csr = typename TestFixture::Csr;
    auto fact = factorization_type::build()
                    .with_approximate_select(false)
                    .with_fill_in_limit(1.2)
                    .on(this->exec)
                    ->generate(this->mtx_system);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->mtx_l_large_expect,
                        this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(),
                        gko::as<Csr>(this->mtx_l_large_expect->transpose()),
                        this->tol);
}


TYPED_TEST(ParIct, GenerateWithApproxLargeLimit)
{
    using factorization_type = typename TestFixture::factorization_type;
    using Csr = typename TestFixture::Csr;
    auto fact = factorization_type::build()
                    .with_approximate_select(true)
                    .with_fill_in_limit(1.2)
                    .on(this->exec)
                    ->generate(this->mtx_system);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->mtx_l_large_expect,
                        this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(),
                        gko::as<Csr>(this->mtx_l_large_expect->transpose()),
                        this->tol);
}


}  // namespace