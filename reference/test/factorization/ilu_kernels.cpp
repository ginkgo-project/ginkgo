// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/ilu.hpp>


#include <algorithm>
#include <initializer_list>
#include <memory>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


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
class Ilu : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using ilu_type = gko::factorization::Ilu<value_type, index_type>;
    Ilu()
        : ref(gko::ReferenceExecutor::create()),
          exec(std::static_pointer_cast<const gko::Executor>(ref)),
          // clang-format off
          identity(gko::initialize<Dense>(
              {{1., 0., 0.},
               {0., 1., 0.},
               {0., 0., 1.}}, exec)),
          lower_triangular(gko::initialize<Dense>(
              {{1., 0., 0.},
               {1., 1., 0.},
               {1., 1., 1.}}, exec)),
          upper_triangular(gko::initialize<Dense>(
              {{1., 1., 1.},
               {0., 1., 1.},
               {0., 0., 1.}}, exec)),
          mtx_small(gko::initialize<Dense>(
              {{4., 6., 8.},
               {2., 2., 5.},
               {1., 1., 1.}}, exec)),
          mtx_csr_small(nullptr),
          small_l_expected(gko::initialize<Dense>(
              {{1., 0., 0.},
               {0.5, 1., 0.},
               {0.25, 0.5, 1.}}, exec)),
          small_u_expected(gko::initialize<Dense>(
              {{4., 6., 8.},
               {0., -1., 1.},
               {0., 0., -1.5}}, exec)),
          mtx_small2(gko::initialize<Dense>(
              {{8., 8., 0},
              {2., 0., 5.},
              {1., 1., 1}}, exec)),
          mtx_csr_small2(nullptr),
          small2_l_expected(gko::initialize<Dense>(
              {{1., 0., 0},
              {.25, 1., 0.},
              {.125, 0., 1}}, exec)),
          small2_u_expected(gko::initialize<Dense>(
              {{8., 8., 0},
              {0., -2., 5.},
              {0., 0., 1}}, exec)),
          mtx_big(gko::initialize<Dense>({{1., 1., 1., 0., 1., 3.},
                                          {1., 2., 2., 0., 2., 0.},
                                          {0., 2., 3., 3., 3., 5.},
                                          {1., 0., 3., 4., 4., 4.},
                                          {1., 2., 0., 4., 5., 6.},
                                          {0., 2., 3., 4., 5., 8.}},
                                         exec)),
          big_l_expected(gko::initialize<Dense>({{1., 0., 0., 0., 0., 0.},
                                                 {1., 1., 0., 0., 0., 0.},
                                                 {0., 2., 1., 0., 0., 0.},
                                                 {1., 0., 2., 1., 0., 0.},
                                                 {1., 1., 0., -2., 1., 0.},
                                                 {0., 2., 1., -0.5, 0.5, 1.}},
                                                exec)),
          big_u_expected(gko::initialize<Dense>({{1., 1., 1., 0., 1., 3.},
                                                 {0., 1., 1., 0., 1., 0.},
                                                 {0., 0., 1., 3., 1., 5.},
                                                 {0., 0., 0., -2., 1., -9.},
                                                 {0., 0., 0., 0., 5., -15.},
                                                 {0., 0., 0., 0., 0., 6.}},
                                                exec)),
          mtx_big_nodiag(gko::initialize<Csr>({{1., 1., 1., 0., 1., 3.},
                                               {1., 2., 2., 0., 2., 0.},
                                               {0., 2., 0., 3., 3., 5.},
                                               {1., 0., 3., 4., 4., 4.},
                                               {1., 2., 0., 4., 1., 6.},
                                               {0., 2., 3., 4., 5., 8.}},
                                         exec)),
          big_nodiag_l_expected(gko::initialize<Dense>(
            {{1., 0., 0., 0., 0., 0.},
             {1., 1., 0., 0., 0., 0.},
             {0., 2., 1., 0., 0., 0.},
             {1., 0., -1., 1., 0., 0.},
             {1., 1., 0., 0.571428571428571, 1., 0.},
             {0., 2., -0.5, 0.785714285714286, -0.108695652173913, 1.}},
            exec)),
          big_nodiag_u_expected(gko::initialize<Dense>(
            {{1., 1., 1., 0., 1., 3.},
             {0., 1., 1., 0., 1., 0.},
             {0., 0., -2., 3., 1., 5.},
             {0., 0., 0., 7., 4., 6.},
             {0., 0., 0., 0., -3.28571428571429, -0.428571428571429},
             {0., 0., 0., 0., 0., 5.73913043478261}},
            exec)),
          // clang-format on
          ilu_factory_skip(ilu_type::build().with_skip_sorting(true).on(exec)),
          ilu_factory_sort(ilu_type::build().with_skip_sorting(false).on(exec))
    {
        auto tmp_csr = Csr::create(exec);
        mtx_small->convert_to(tmp_csr);
        mtx_csr_small = std::move(tmp_csr);
        auto tmp_csr2 = Csr::create(exec);
        mtx_small2->convert_to(tmp_csr2);
        mtx_csr_small2 = std::move(tmp_csr2);
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<const Dense> identity;
    std::shared_ptr<const Dense> lower_triangular;
    std::shared_ptr<const Dense> upper_triangular;
    std::shared_ptr<const Dense> mtx_small;
    std::shared_ptr<const Csr> mtx_csr_small;
    std::shared_ptr<const Dense> small_l_expected;
    std::shared_ptr<const Dense> small_u_expected;
    std::shared_ptr<const Dense> mtx_small2;
    std::shared_ptr<const Csr> mtx_csr_small2;
    std::shared_ptr<const Dense> small2_l_expected;
    std::shared_ptr<const Dense> small2_u_expected;
    std::shared_ptr<const Dense> mtx_big;
    std::shared_ptr<const Dense> big_l_expected;
    std::shared_ptr<const Dense> big_u_expected;
    std::shared_ptr<const Csr> mtx_big_nodiag;
    std::shared_ptr<const Dense> big_nodiag_l_expected;
    std::shared_ptr<const Dense> big_nodiag_u_expected;
    std::unique_ptr<typename ilu_type::Factory> ilu_factory_skip;
    std::unique_ptr<typename ilu_type::Factory> ilu_factory_sort;
};

TYPED_TEST_SUITE(Ilu, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Ilu, ThrowNotSupportedForWrongLinOp1)
{
    auto linOp = gko::share(DummyLinOp::create(this->ref));

    ASSERT_THROW(this->ilu_factory_skip->generate(linOp), gko::NotSupported);
}


TYPED_TEST(Ilu, ThrowNotSupportedForWrongLinOp2)
{
    auto linOp = gko::share(DummyLinOp::create(this->ref));

    ASSERT_THROW(this->ilu_factory_sort->generate(linOp), gko::NotSupported);
}


TYPED_TEST(Ilu, ThrowDimensionMismatch)
{
    using Csr = typename TestFixture::Csr;
    auto matrix = gko::share(Csr::create(this->ref, gko::dim<2>{2, 3}, 4));

    ASSERT_THROW(this->ilu_factory_sort->generate(matrix),
                 gko::DimensionMismatch);
}


TYPED_TEST(Ilu, SetLStrategy)
{
    using Csr = typename TestFixture::Csr;
    using ilu_type = typename TestFixture::ilu_type;
    auto l_strategy = std::make_shared<typename Csr::classical>();

    auto factory = ilu_type::build().with_l_strategy(l_strategy).on(this->ref);
    auto ilu = factory->generate(this->mtx_small);

    ASSERT_EQ(factory->get_parameters().l_strategy, l_strategy);
    ASSERT_EQ(ilu->get_l_factor()->get_strategy()->get_name(),
              l_strategy->get_name());
}


TYPED_TEST(Ilu, SetUStrategy)
{
    using Csr = typename TestFixture::Csr;
    using ilu_type = typename TestFixture::ilu_type;
    auto u_strategy = std::make_shared<typename Csr::classical>();

    auto factory = ilu_type::build().with_u_strategy(u_strategy).on(this->ref);
    auto ilu = factory->generate(this->mtx_small);

    ASSERT_EQ(factory->get_parameters().u_strategy, u_strategy);
    ASSERT_EQ(ilu->get_u_factor()->get_strategy()->get_name(),
              u_strategy->get_name());
}


TYPED_TEST(Ilu, LUFactorFunctionsSetProperly)
{
    auto factors = this->ilu_factory_skip->generate(this->mtx_small);

    auto lin_op_l_factor = gko::as<gko::LinOp>(factors->get_l_factor());
    auto lin_op_u_factor = gko::as<gko::LinOp>(factors->get_u_factor());
    auto first_operator = factors->get_operators()[0];
    auto second_operator = factors->get_operators()[1];

    ASSERT_EQ(lin_op_l_factor, first_operator);
    ASSERT_EQ(lin_op_u_factor, second_operator);
}


TYPED_TEST(Ilu, GenerateForCooIdentity)
{
    using Coo = typename TestFixture::Coo;
    using value_type = typename TestFixture::value_type;
    auto coo_mtx = gko::share(Coo::create(this->exec));
    this->identity->convert_to(coo_mtx);

    auto factors = this->ilu_factory_skip->generate(coo_mtx);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->identity, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->identity, r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForCsrIdentity)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto csr_mtx = gko::share(Csr::create(this->exec));
    this->identity->convert_to(csr_mtx);

    auto factors = this->ilu_factory_skip->generate(csr_mtx);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->identity, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->identity, r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForDenseIdentity)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->identity);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->identity, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->identity, r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForDenseLowerTriangular)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->lower_triangular);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->lower_triangular, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->identity, r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForDenseUpperTriangular)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->upper_triangular);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->identity, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->upper_triangular, r<value_type>::value);
}


TYPED_TEST(Ilu, ApplyMethodDenseSmall)
{
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    const auto x = gko::initialize<Dense>({1., 2., 3.}, this->exec);
    auto b_lu = Dense::create_with_config_of(x);
    auto b_ref = Dense::create_with_config_of(x);

    auto factors = this->ilu_factory_skip->generate(this->mtx_small);
    factors->apply(x, b_lu);
    this->mtx_small->apply(x, b_ref);

    GKO_ASSERT_MTX_NEAR(b_lu, b_ref, r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForDenseSmall)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->mtx_small);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->small_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->small_u_expected, r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForCsrSmall)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->mtx_csr_small);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->small_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->small_u_expected, r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForCsrSmall2ZeroDiagonal)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->mtx_csr_small2);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->small2_l_expected,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->small2_u_expected,
                        r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForCsrBigWithDiagonalZeros)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->mtx_big_nodiag);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->big_nodiag_l_expected,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->big_nodiag_u_expected,
                        r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForDenseBig)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->mtx_big);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->big_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->big_u_expected, r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForDenseBigSort)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_sort->generate(this->mtx_big);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->big_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->big_u_expected, r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForReverseCooSmall)
{
    using value_type = typename TestFixture::value_type;
    using Coo = typename TestFixture::Coo;
    const auto size = this->mtx_small->get_size();
    const auto nnz = size[0] * size[1];
    auto reverse_coo = gko::share(Coo::create(this->exec, size, nnz));
    // Fill the Coo matrix in reversed row order (right to left)
    for (size_t i = 0; i < size[0]; ++i) {
        for (size_t j = 0; j < size[1]; ++j) {
            const auto coo_idx = i * size[1] + (size[1] - 1 - j);
            reverse_coo->get_row_idxs()[coo_idx] = i;
            reverse_coo->get_col_idxs()[coo_idx] = j;
            reverse_coo->get_values()[coo_idx] = this->mtx_small->at(i, j);
        }
    }

    auto factors = this->ilu_factory_sort->generate(reverse_coo);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(reverse_coo, this->mtx_small, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(l_factor, this->small_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->small_u_expected, r<value_type>::value);
}


TYPED_TEST(Ilu, GenerateForReverseCsrSmall)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    const auto size = this->mtx_csr_small->get_size();
    const auto nnz = size[0] * size[1];
    auto reverse_csr = gko::share(gko::clone(this->exec, this->mtx_csr_small));
    // Fill the Csr matrix rows in reverse order
    for (size_t i = 0; i < size[0]; ++i) {
        const auto row_start = reverse_csr->get_row_ptrs()[i];
        const auto row_end = reverse_csr->get_row_ptrs()[i + 1];
        for (size_t j = row_start; j < row_end; ++j) {
            const auto reverse_j = row_end - 1 - (j - row_start);
            reverse_csr->get_values()[reverse_j] =
                this->mtx_csr_small->get_const_values()[j];
            reverse_csr->get_col_idxs()[reverse_j] =
                this->mtx_csr_small->get_const_col_idxs()[j];
        }
    }

    auto factors = this->ilu_factory_sort->generate(reverse_csr);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->small_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->small_u_expected, r<value_type>::value);
}


}  // namespace
