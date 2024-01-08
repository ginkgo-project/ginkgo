// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/par_ic.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/par_ic_kernels.hpp"
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
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}
};


template <typename ValueIndexType>
class ParIc : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using factorization_type =
        gko::factorization::ParIc<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;

    ParIc()
        : ref(gko::ReferenceExecutor::create()),
          exec(std::static_pointer_cast<const gko::Executor>(ref)),
          identity(gko::initialize<Csr>(
              {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}, ref)),
          banded(gko::initialize<Csr>(
              {{1., 1., 0.}, {1., 2., 1.}, {0., 1., 2.}}, ref)),
          banded_l_expect(gko::initialize<Csr>(
              {{1., 0., 0.}, {1., 1., 0.}, {0., 1., 1.}}, ref)),
          mtx_system(gko::initialize<Csr>({{9., 0., -6., 3.},
                                           {0., 36., 18., 24.},
                                           {-6., 18., 17., 14.},
                                           {-3., 24., 14., 18.}},
                                          ref)),
          mtx_l_system(gko::initialize<Csr>({{9., 0., 0., 0.},
                                             {0., 36., 0., 0.},
                                             {-6., 18., 17., 0.},
                                             {-3., 24., 14., 18.}},
                                            ref)),
          mtx_l_system_coo(Coo::create(exec)),
          mtx_l_init_expect(gko::initialize<Csr>(
              {{3., 0., 0., 0.},
               {0., 6., 0., 0.},
               {-6., 18., static_cast<value_type>(sqrt(17.)), 0.},
               {-3., 24., 14., static_cast<value_type>(sqrt(18.))}},
              ref)),
          mtx_l_it_expect(gko::initialize<Csr>({{3., 0., 0., 0.},
                                                {0., 6., 0., 0.},
                                                {-2., 3., 2., 0.},
                                                {-1., 4., 0., 1.}},
                                               ref)),
          fact_fact(factorization_type::build().on(exec)),
          tol{r<value_type>::value}
    {
        mtx_l_system->convert_to(mtx_l_system_coo);
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Csr> identity;
    std::shared_ptr<Csr> banded;
    std::shared_ptr<Csr> banded_l_expect;
    std::shared_ptr<Csr> mtx_system;
    std::unique_ptr<Csr> mtx_l_system;
    std::unique_ptr<Coo> mtx_l_system_coo;
    std::unique_ptr<Csr> mtx_l_init_expect;
    std::unique_ptr<Csr> mtx_l_it_expect;
    std::unique_ptr<typename factorization_type::Factory> fact_fact;
    gko::remove_complex<value_type> tol;
};

TYPED_TEST_SUITE(ParIc, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(ParIc, KernelCompute)
{
    gko::kernels::reference::par_ic_factorization::compute_factor(
        this->ref, 1, this->mtx_l_system_coo.get(), this->mtx_l_system.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_l_system, this->mtx_l_it_expect, this->tol);
}


TYPED_TEST(ParIc, KernelInit)
{
    gko::kernels::reference::par_ic_factorization::init_factor(
        this->ref, this->mtx_l_system.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_l_system, this->mtx_l_init_expect, this->tol);
}


TYPED_TEST(ParIc, ThrowNotSupportedForWrongLinOp)
{
    auto lin_op = gko::share(DummyLinOp::create(this->ref));

    ASSERT_THROW(this->fact_fact->generate(lin_op), gko::NotSupported);
}


TYPED_TEST(ParIc, ThrowDimensionMismatch)
{
    using Csr = typename TestFixture::Csr;
    auto matrix = gko::share(Csr::create(this->ref, gko::dim<2>{2, 3}, 4));

    ASSERT_THROW(this->fact_fact->generate(matrix), gko::DimensionMismatch);
}


TYPED_TEST(ParIc, SetStrategy)
{
    using Csr = typename TestFixture::Csr;
    using factorization_type = typename TestFixture::factorization_type;
    auto l_strategy = std::make_shared<typename Csr::merge_path>();

    auto factory =
        factorization_type::build().with_l_strategy(l_strategy).on(this->ref);
    auto fact = factory->generate(this->mtx_system);

    ASSERT_EQ(factory->get_parameters().l_strategy, l_strategy);
    ASSERT_EQ(fact->get_l_factor()->get_strategy()->get_name(),
              l_strategy->get_name());
    ASSERT_EQ(fact->get_lt_factor()->get_strategy()->get_name(),
              l_strategy->get_name());
}


TYPED_TEST(ParIc, IsConsistentWithComposition)
{
    auto fact = this->fact_fact->generate(this->mtx_system);

    auto lin_op_l_factor = gko::as<gko::LinOp>(fact->get_l_factor());
    auto lin_op_lt_factor = gko::as<gko::LinOp>(fact->get_lt_factor());
    auto first_operator = fact->get_operators()[0];
    auto second_operator = fact->get_operators()[1];

    ASSERT_EQ(lin_op_l_factor, first_operator);
    ASSERT_EQ(lin_op_lt_factor, second_operator);
}


TYPED_TEST(ParIc, GenerateSingleFactor)
{
    using factorization_type = typename TestFixture::factorization_type;
    using Csr = typename TestFixture::Csr;
    auto factory =
        factorization_type::build().with_both_factors(false).on(this->ref);
    auto fact = factory->generate(this->mtx_system);

    auto lin_op_l_factor = fact->get_l_factor();
    auto lin_op_lt_factor = fact->get_lt_factor();
    auto first_operator = gko::as<Csr>(fact->get_operators()[0]);
    auto first_operator_h = gko::as<Csr>(first_operator->conj_transpose());

    ASSERT_EQ(fact->get_operators().size(), 1);
    ASSERT_EQ(lin_op_l_factor, first_operator);
    GKO_ASSERT_MTX_NEAR(lin_op_lt_factor, first_operator_h, this->tol);
}


TYPED_TEST(ParIc, GenerateIdentity)
{
    auto fact = this->fact_fact->generate(this->identity);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->identity, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(), this->identity, this->tol);
}


TYPED_TEST(ParIc, GenerateDenseIdentity)
{
    using Dense = typename TestFixture::Dense;
    auto dense_id =
        gko::share(Dense::create(this->exec, this->identity->get_size()));
    this->identity->convert_to(dense_id);

    auto fact = this->fact_fact->generate(dense_id);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->identity, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(), this->identity, this->tol);
}


TYPED_TEST(ParIc, GenerateBanded)
{
    using factorization_type = typename TestFixture::factorization_type;
    using Csr = typename TestFixture::Csr;

    auto fact =
        factorization_type::build().on(this->exec)->generate(this->banded);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->banded_l_expect, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(),
                        gko::as<Csr>(this->banded_l_expect->conj_transpose()),
                        this->tol);
}


TYPED_TEST(ParIc, GenerateGeneral)
{
    using factorization_type = typename TestFixture::factorization_type;
    using Csr = typename TestFixture::Csr;

    auto fact =
        factorization_type::build().on(this->exec)->generate(this->mtx_system);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->mtx_l_it_expect, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(),
                        gko::as<Csr>(this->mtx_l_it_expect->conj_transpose()),
                        this->tol);
}


}  // namespace
