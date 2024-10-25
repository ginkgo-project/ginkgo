// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/sor_kernels.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/sor.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "core/test/utils.hpp"


template <typename ValueIndexType>
class Sor : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using csr_type = gko::matrix::Csr<value_type, index_type>;
    using sor_type = gko::preconditioner::Sor<value_type, index_type>;

    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
    gko::remove_complex<value_type> diag_value =
        static_cast<gko::remove_complex<value_type>>(1.5);
    std::shared_ptr<csr_type> mtx =
        gko::initialize<csr_type>({{diag_value, 2, 0, 3, 4},
                                   {-2, diag_value, 5, 0, 0},
                                   {0, -5, diag_value, 0, 6},
                                   {-3, 0, 0, diag_value, 7},
                                   {-4, 0, -6, -7, diag_value}},
                                  exec);
    std::shared_ptr<csr_type> expected_l =
        gko::initialize<csr_type>({{diag_value, 0, 0, 0, 0},
                                   {-2, diag_value, 0, 0, 0},
                                   {0, -5, diag_value, 0, 0},
                                   {-3, 0, 0, diag_value, 0},
                                   {-4, 0, -6, -7, diag_value}},
                                  exec);
    std::shared_ptr<csr_type> expected_u =
        gko::initialize<csr_type>({{diag_value, 2, 0, 3, 4},
                                   {0, diag_value, 5, 0, 0},
                                   {0, 0, diag_value, 0, 6},
                                   {0, 0, 0, diag_value, 7},
                                   {0, 0, 0, 0, diag_value}},
                                  exec);
};

TYPED_TEST_SUITE(Sor, gko::test::ValueIndexTypesWithHalf,
                 PairTypenameNameGenerator);


TYPED_TEST(Sor, CanInitializeLFactor)
{
    using value_type = typename TestFixture::value_type;
    auto result = gko::clone(this->expected_l);
    result->scale(
        gko::initialize<gko::matrix::Dense<value_type>>({0.0}, this->exec));

    gko::kernels::reference::sor::initialize_weighted_l(
        this->exec, this->mtx.get(), 1.0, result.get());

    GKO_ASSERT_MTX_NEAR(result, this->expected_l, 0.0);
}


TYPED_TEST(Sor, CanInitializeLFactorWithWeight)
{
    using value_type = typename TestFixture::value_type;
    using csr_type = typename TestFixture::csr_type;
    auto result = gko::clone(this->expected_l);
    result->scale(
        gko::initialize<gko::matrix::Dense<value_type>>({0.0}, this->exec));
    std::shared_ptr<csr_type> expected_l =
        gko::initialize<csr_type>({{2 * this->diag_value, 0, 0, 0, 0},
                                   {-2, 2 * this->diag_value, 0, 0, 0},
                                   {0, -5, 2 * this->diag_value, 0, 0},
                                   {-3, 0, 0, 2 * this->diag_value, 0},
                                   {-4, 0, -6, -7, 2 * this->diag_value}},
                                  this->exec);

    gko::kernels::reference::sor::initialize_weighted_l(
        this->exec, this->mtx.get(), 0.5f, result.get());

    GKO_ASSERT_MTX_NEAR(result, expected_l, r<value_type>::value);
}


TYPED_TEST(Sor, CanInitializeLAndUFactor)
{
    using value_type = typename TestFixture::value_type;
    auto result_l = gko::clone(this->expected_l);
    auto result_u = gko::clone(this->expected_u);
    result_l->scale(
        gko::initialize<gko::matrix::Dense<value_type>>({0.0}, this->exec));
    result_u->scale(
        gko::initialize<gko::matrix::Dense<value_type>>({0.0}, this->exec));

    gko::kernels::reference::sor::initialize_weighted_l_u(
        this->exec, this->mtx.get(), 1.0, result_l.get(), result_u.get());

    GKO_ASSERT_MTX_NEAR(result_l, this->expected_l, 0.0);
    GKO_ASSERT_MTX_NEAR(result_u, this->expected_u, 0.0);
}


TYPED_TEST(Sor, CanInitializeLAndUFactorWithWeight)
{
    using value_type = typename TestFixture::value_type;
    using csr_type = typename TestFixture::csr_type;
    auto result_l = gko::clone(this->expected_l);
    auto result_u = gko::clone(this->expected_u);
    result_l->scale(
        gko::initialize<gko::matrix::Dense<value_type>>({0.0}, this->exec));
    result_u->scale(
        gko::initialize<gko::matrix::Dense<value_type>>({0.0}, this->exec));
    auto factor = static_cast<gko::remove_complex<value_type>>(0.5);
    auto diag_weight =
        static_cast<gko::remove_complex<value_type>>(1.0 / (2 - factor));
    auto off_diag_weight = factor * diag_weight;
    std::shared_ptr<csr_type> expected_l =
        gko::initialize<csr_type>({{2 * this->diag_value, 0, 0, 0, 0},
                                   {-2, 2 * this->diag_value, 0, 0, 0},
                                   {0, -5, 2 * this->diag_value, 0, 0},
                                   {-3, 0, 0, 2 * this->diag_value, 0},
                                   {-4, 0, -6, -7, 2 * this->diag_value}},
                                  this->exec);
    std::shared_ptr<csr_type> expected_u = gko::initialize<csr_type>(
        {{this->diag_value * diag_weight, 2 * off_diag_weight, 0,
          3 * off_diag_weight, 4 * off_diag_weight},
         {0, this->diag_value * diag_weight, 5 * off_diag_weight, 0, 0},
         {0, 0, this->diag_value * diag_weight, 0, 6 * off_diag_weight},
         {0, 0, 0, this->diag_value * diag_weight, 7 * off_diag_weight},
         {0, 0, 0, 0, this->diag_value * diag_weight}},
        this->exec);

    gko::kernels::reference::sor::initialize_weighted_l_u(
        this->exec, this->mtx.get(), factor, result_l.get(), result_u.get());

    GKO_ASSERT_MTX_NEAR(result_l, expected_l, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(result_u, expected_u, r<value_type>::value);
}


TYPED_TEST(Sor, CanGenerateNonSymmetric)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using sor_type = typename TestFixture::sor_type;
    using composition_type = typename sor_type::composition_type;
    using trs_type = gko::solver::LowerTrs<value_type, index_type>;

    auto sor_pre = sor_type::build()
                       .with_relaxation_factor(1.0f)
                       .on(this->exec)
                       ->generate(this->mtx);

    testing::StaticAssertTypeEq<decltype(sor_pre),
                                std::unique_ptr<composition_type>>();
    const auto& ops = sor_pre->get_operators();
    ASSERT_EQ(ops.size(), 1);
    GKO_ASSERT_DYNAMIC_TYPE(ops[0], trs_type);
    auto result_l = gko::as<trs_type>(ops[0])->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(result_l, this->expected_l, 0.0);
}


TYPED_TEST(Sor, CanGenerateSymmetric)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using sor_type = typename TestFixture::sor_type;
    using composition_type = typename sor_type::composition_type;
    using l_trs_type = gko::solver::LowerTrs<value_type, index_type>;
    using u_trs_type = gko::solver::UpperTrs<value_type, index_type>;

    auto sor_pre = sor_type::build()
                       .with_symmetric(true)
                       .with_relaxation_factor(1.0f)
                       .on(this->exec)
                       ->generate(this->mtx);

    testing::StaticAssertTypeEq<decltype(sor_pre),
                                std::unique_ptr<composition_type>>();
    const auto& ops = sor_pre->get_operators();
    ASSERT_EQ(ops.size(), 2);
    GKO_ASSERT_DYNAMIC_TYPE(ops[0], u_trs_type);
    GKO_ASSERT_DYNAMIC_TYPE(ops[1], l_trs_type);
    auto result_u = gko::as<u_trs_type>(ops[0])->get_system_matrix();
    auto result_l = gko::as<l_trs_type>(ops[1])->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(result_l, this->expected_l, 0.0);
    auto expected_u = gko::clone(this->expected_u);
    expected_u->inv_scale(gko::initialize<gko::matrix::Dense<value_type>>(
        {this->diag_value}, this->exec));
    GKO_ASSERT_MTX_NEAR(result_u, expected_u, r<value_type>::value);
}
