// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"


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
class Ic : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using factorization_type = gko::factorization::Ic<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;

    Ic()
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
          mtx_l_it_expect(gko::initialize<Csr>({{3., 0., 0., 0.},
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
    std::shared_ptr<Csr> banded;
    std::shared_ptr<Csr> banded_l_expect;
    std::shared_ptr<Csr> mtx_system;
    std::unique_ptr<Csr> mtx_l_it_expect;
    std::unique_ptr<typename factorization_type::Factory> fact_fact;
    gko::remove_complex<value_type> tol;
};

TYPED_TEST_SUITE(Ic, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Ic, ThrowNotSupportedForWrongLinOp)
{
    auto lin_op = gko::share(DummyLinOp::create(this->ref));

    ASSERT_THROW(this->fact_fact->generate(lin_op), gko::NotSupported);
}


TYPED_TEST(Ic, ThrowDimensionMismatch)
{
    using Csr = typename TestFixture::Csr;
    auto matrix = gko::share(Csr::create(this->ref, gko::dim<2>{2, 3}, 4));

    ASSERT_THROW(this->fact_fact->generate(matrix), gko::DimensionMismatch);
}


TYPED_TEST(Ic, SetStrategy)
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


TYPED_TEST(Ic, IsConsistentWithComposition)
{
    auto fact = this->fact_fact->generate(this->mtx_system);

    auto lin_op_l_factor = gko::as<gko::LinOp>(fact->get_l_factor());
    auto lin_op_lt_factor = gko::as<gko::LinOp>(fact->get_lt_factor());
    auto first_operator = fact->get_operators()[0];
    auto second_operator = fact->get_operators()[1];

    ASSERT_EQ(lin_op_l_factor, first_operator);
    ASSERT_EQ(lin_op_lt_factor, second_operator);
}


TYPED_TEST(Ic, GenerateIdentity)
{
    auto fact = this->fact_fact->generate(this->identity);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->identity, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(), this->identity, this->tol);
}


TYPED_TEST(Ic, GenerateDenseIdentity)
{
    using Dense = typename TestFixture::Dense;
    auto dense_id =
        gko::share(Dense::create(this->exec, this->identity->get_size()));
    this->identity->convert_to(dense_id);

    auto fact = this->fact_fact->generate(dense_id);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->identity, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(), this->identity, this->tol);
}


TYPED_TEST(Ic, GenerateBanded)
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


TYPED_TEST(Ic, GenerateGeneral)
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


TYPED_TEST(Ic, GenerateGeneralBySyncfree)
{
    using factorization_type = typename TestFixture::factorization_type;
    using Csr = typename TestFixture::Csr;

    auto fact =
        factorization_type::build()
            .with_algorithm(
                gko::factorization::incomplete_factorize_algorithm::syncfree)
            .on(this->exec)
            ->generate(this->mtx_system);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), this->mtx_l_it_expect, this->tol);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(),
                        gko::as<Csr>(this->mtx_l_it_expect->conj_transpose()),
                        this->tol);
}


TYPED_TEST(Ic, GenerateIcWithBitmapIsEquivalentToRefBySyncfree)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename TestFixture::Csr;
    // diag + full first row and column
    // the third and forth row use bitmap for lookup table
    auto mtx = gko::share(gko::initialize<Csr>({{1.0, 1.0, 1.0, 1.0},
                                                {1.0, 2.0, 0.0, 0.0},
                                                {1.0, 0.0, 2.0, 0.0},
                                                {1.0, 0.0, 0.0, 2.0}},
                                               this->ref));
    auto result_l = gko::initialize<Csr>({{1.0, 0.0, 0.0, 0.0},
                                          {1.0, 1.0, 0.0, 0.0},
                                          {1.0, 0.0, 1.0, 0.0},
                                          {1.0, 0.0, 0.0, 1.0}},
                                         this->ref);
    auto result_lt = gko::as<Csr>(result_l->conj_transpose());
    auto factory =
        gko::factorization::Ic<value_type, index_type>::build()
            .with_algorithm(
                gko::factorization::incomplete_factorize_algorithm::syncfree)
            .on(this->ref);

    auto ic = factory->generate(mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(ic->get_l_factor(), result_l);
    GKO_ASSERT_MTX_NEAR(ic->get_l_factor(), result_l, this->tol);
    GKO_ASSERT_MTX_EQ_SPARSITY(ic->get_lt_factor(), result_lt);
    GKO_ASSERT_MTX_NEAR(ic->get_lt_factor(), result_lt, this->tol);
}


TYPED_TEST(Ic, GenerateIcWithHashmapIsEquivalentToRefBySyncfree)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename TestFixture::Csr;
    int n = 68;
    gko::matrix_data<value_type, index_type> data(gko::dim<2>(n, n));
    gko::matrix_data<value_type, index_type> result(gko::dim<2>(n, n));
    for (int i = 0; i < n; i++) {
        if (i == n - 2 || i == n - 3) {
            data.nonzeros.emplace_back(i, i, value_type{2});
        } else {
            data.nonzeros.emplace_back(i, i, gko::one<value_type>());
        }
        result.nonzeros.emplace_back(i, i, gko::one<value_type>());
    }
    // the following rows use hashmap for lookup table
    // add dependence
    data.nonzeros.emplace_back(n - 3, 0, gko::one<value_type>());
    data.nonzeros.emplace_back(0, n - 3, gko::one<value_type>());
    // add a entry whose col idx is not shown in the above row
    data.nonzeros.emplace_back(0, n - 2, gko::one<value_type>());
    data.nonzeros.emplace_back(n - 2, 0, gko::one<value_type>());
    data.sort_row_major();
    auto mtx = gko::share(Csr::create(this->ref));
    mtx->read(data);
    // prepare result (lower triangular part)
    result.nonzeros.emplace_back(n - 3, 0, gko::one<value_type>());
    result.nonzeros.emplace_back(n - 2, 0, gko::one<value_type>());
    result.sort_row_major();
    auto result_l = gko::share(Csr::create(this->ref));
    result_l->read(result);
    auto result_lt = gko::as<Csr>(result_l->conj_transpose());
    auto factory =
        gko::factorization::Ic<value_type, index_type>::build()
            .with_algorithm(
                gko::factorization::incomplete_factorize_algorithm::syncfree)
            .on(this->ref);

    auto ic = factory->generate(mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(ic->get_l_factor(), result_l);
    GKO_ASSERT_MTX_NEAR(ic->get_l_factor(), result_l, this->tol);
    GKO_ASSERT_MTX_EQ_SPARSITY(ic->get_lt_factor(), result_lt);
    GKO_ASSERT_MTX_NEAR(ic->get_lt_factor(), result_lt, this->tol);
}
