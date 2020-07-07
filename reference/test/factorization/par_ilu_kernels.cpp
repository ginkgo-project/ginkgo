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

#include <ginkgo/core/factorization/par_ilu.hpp>


#include <algorithm>
#include <initializer_list>
#include <memory>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/par_ilu_kernels.hpp"
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
class ParIlu : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using par_ilu_type = gko::factorization::ParIlu<value_type, index_type>;
    ParIlu()
        : ref(gko::ReferenceExecutor::create()),
          exec(std::static_pointer_cast<const gko::Executor>(ref)),
          // clang-format off
          empty_csr(gko::initialize<Csr>(
              {{0., 0., 0.},
               {0., 0., 0.},
               {0., 0., 0.}}, exec)),
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
          ilu_factory_skip(
              par_ilu_type::build().with_skip_sorting(true).on(exec)),
          ilu_factory_sort(
              par_ilu_type::build().with_skip_sorting(false).on(exec))
    {
        auto tmp_csr = Csr::create(exec);
        mtx_small->convert_to(gko::lend(tmp_csr));
        mtx_csr_small = std::move(tmp_csr);
        auto tmp_csr2 = Csr::create(exec);
        mtx_small2->convert_to(gko::lend(tmp_csr2));
        mtx_csr_small2 = std::move(tmp_csr2);
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<const Csr> empty_csr;
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
    std::unique_ptr<typename par_ilu_type::Factory> ilu_factory_skip;
    std::unique_ptr<typename par_ilu_type::Factory> ilu_factory_sort;
};

TYPED_TEST_CASE(ParIlu, gko::test::ValueIndexTypes);


TYPED_TEST(ParIlu, KernelAddDiagonalElementsEmpty)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    auto expected_mtx =
        Csr::create(this->ref, this->empty_csr->get_size(),
                    std::initializer_list<value_type>{0., 0., 0.},
                    std::initializer_list<index_type>{0, 1, 2},
                    std::initializer_list<index_type>{0, 1, 2, 3});
    auto empty_mtx = this->empty_csr->clone();

    gko::kernels::reference::factorization::add_diagonal_elements(
        this->ref, gko::lend(empty_mtx), true);

    GKO_ASSERT_MTX_NEAR(empty_mtx, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(empty_mtx, expected_mtx);
}


TYPED_TEST(ParIlu, KernelAddDiagonalElementsNonSquare)
{
    using Csr = typename TestFixture::Csr;
    auto matrix = gko::initialize<Csr>(
        {{0., 0., 0.}, {1., 0., 0.}, {1., 1., 1.}, {1., 1., 1.}}, this->ref);
    auto exp_values = {0., 1., 0., 1., 1., 1., 1., 1., 1.};
    auto exp_col_idxs = {0, 0, 1, 0, 1, 2, 0, 1, 2};
    auto exp_row_ptrs = {0, 1, 3, 6, 9};
    auto expected_mtx =
        Csr::create(this->ref, matrix->get_size(), std::move(exp_values),
                    std::move(exp_col_idxs), std::move(exp_row_ptrs));

    gko::kernels::reference::factorization::add_diagonal_elements(
        this->ref, gko::lend(matrix), true);

    GKO_ASSERT_MTX_NEAR(matrix, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(matrix, expected_mtx);
}


TYPED_TEST(ParIlu, KernelAddDiagonalElementsNonSquare2)
{
    using Csr = typename TestFixture::Csr;
    auto matrix = gko::initialize<Csr>({{1., 0., 0.}, {1., 0., 0.}}, this->ref);
    auto exp_values = {1., 1., 0.};
    auto exp_col_idxs = {0, 0, 1};
    auto exp_row_ptrs = {0, 1, 3};
    auto expected_mtx =
        Csr::create(this->ref, matrix->get_size(), std::move(exp_values),
                    std::move(exp_col_idxs), std::move(exp_row_ptrs));

    gko::kernels::reference::factorization::add_diagonal_elements(
        this->ref, gko::lend(matrix), true);

    GKO_ASSERT_MTX_NEAR(matrix, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(matrix, expected_mtx);
}


TYPED_TEST(ParIlu, KernelAddDiagonalElementsUnsorted)
{
    using Csr = typename TestFixture::Csr;
    auto size = gko::dim<2>{3, 3};
    /* matrix:
    1 2 3
    1 0 3
    1 2 0
    */
    auto mtx_values = {3., 2., 1., 3., 1., 2., 1.};
    auto mtx_col_idxs = {2, 1, 0, 2, 0, 1, 0};
    auto mtx_row_ptrs = {0, 3, 5, 7};
    auto matrix = Csr::create(this->ref, size, std::move(mtx_values),
                              std::move(mtx_col_idxs), std::move(mtx_row_ptrs));
    auto exp_values = {1., 2., 3., 1., 0., 3., 1., 2., 0.};
    auto exp_col_idxs = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    auto exp_row_ptrs = {0, 3, 6, 9};
    auto expected_mtx =
        Csr::create(this->ref, size, std::move(exp_values),
                    std::move(exp_col_idxs), std::move(exp_row_ptrs));

    gko::kernels::reference::factorization::add_diagonal_elements(
        this->ref, gko::lend(matrix), false);

    GKO_ASSERT_MTX_NEAR(matrix, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(matrix, expected_mtx);
}


TYPED_TEST(ParIlu, KernelInitializeRowPtrsLU)
{
    using Csr = typename TestFixture::Csr;
    using index_type = typename TestFixture::index_type;
    auto small_csr_l_expected = Csr::create(this->ref);
    this->small_l_expected->convert_to(gko::lend(small_csr_l_expected));
    auto small_csr_u_expected = Csr::create(this->ref);
    this->small_u_expected->convert_to(gko::lend(small_csr_u_expected));
    auto num_row_ptrs = this->mtx_csr_small->get_size()[0] + 1;
    std::vector<index_type> l_row_ptrs_vector(num_row_ptrs);
    std::vector<index_type> u_row_ptrs_vector(num_row_ptrs);
    auto l_row_ptrs = l_row_ptrs_vector.data();
    auto u_row_ptrs = u_row_ptrs_vector.data();

    gko::kernels::reference::factorization::initialize_row_ptrs_l_u(
        this->ref, gko::lend(this->mtx_csr_small), l_row_ptrs, u_row_ptrs);

    ASSERT_TRUE(std::equal(l_row_ptrs, l_row_ptrs + num_row_ptrs,
                           small_csr_l_expected->get_const_row_ptrs()));
    ASSERT_TRUE(std::equal(u_row_ptrs, u_row_ptrs + num_row_ptrs,
                           small_csr_u_expected->get_const_row_ptrs()));
}


TYPED_TEST(ParIlu, KernelInitializeRowPtrsLUZeroMatrix)
{
    using index_type = typename TestFixture::index_type;
    using Csr = typename TestFixture::Csr;
    auto empty_mtx = this->empty_csr->clone();
    gko::kernels::reference::factorization::add_diagonal_elements(
        this->ref, gko::lend(empty_mtx), true);
    auto empty_mtx_l_expected = Csr::create(this->ref);
    this->identity->convert_to(gko::lend(empty_mtx_l_expected));
    auto empty_mtx_u_expected = Csr::create(this->ref);
    this->identity->convert_to(gko::lend(empty_mtx_u_expected));
    auto num_row_ptrs = empty_mtx->get_size()[0] + 1;
    std::vector<index_type> l_row_ptrs_vector(num_row_ptrs);
    std::vector<index_type> u_row_ptrs_vector(num_row_ptrs);
    auto l_row_ptrs = l_row_ptrs_vector.data();
    auto u_row_ptrs = u_row_ptrs_vector.data();

    gko::kernels::reference::factorization::initialize_row_ptrs_l_u(
        this->ref, gko::lend(empty_mtx), l_row_ptrs, u_row_ptrs);

    ASSERT_TRUE(std::equal(l_row_ptrs, l_row_ptrs + num_row_ptrs,
                           empty_mtx_l_expected->get_const_row_ptrs()));
    ASSERT_TRUE(std::equal(u_row_ptrs, u_row_ptrs + num_row_ptrs,
                           empty_mtx_u_expected->get_const_row_ptrs()));
}


TYPED_TEST(ParIlu, KernelInitializeLU)
{
    using Dense = typename TestFixture::Dense;
    using Csr = typename TestFixture::Csr;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    // clang-format off
    auto expected_l =
        gko::initialize<Dense>({{1., 0., 0.},
                                {2., 1., 0.},
                                {1., 1., 1.}}, this->ref);
    auto expected_u =
        gko::initialize<Dense>({{4., 6., 8.},
                                {0., 2., 5.},
                                {0., 0., 1.}}, this->ref);
    // clang-format on
    auto actual_l = Csr::create(this->ref, this->mtx_csr_small->get_size(), 6);
    auto actual_u = Csr::create(this->ref, this->mtx_csr_small->get_size(), 6);
    // Copy row_ptrs into matrices, which usually come from the
    // `initialize_row_ptrs_l_u` kernel
    std::vector<index_type> l_row_ptrs{0, 1, 3, 6};
    std::vector<index_type> u_row_ptrs{0, 3, 5, 6};
    std::copy(l_row_ptrs.begin(), l_row_ptrs.end(), actual_l->get_row_ptrs());
    std::copy(u_row_ptrs.begin(), u_row_ptrs.end(), actual_u->get_row_ptrs());

    gko::kernels::reference::factorization::initialize_l_u(
        this->ref, gko::lend(this->mtx_csr_small), gko::lend(actual_l),
        gko::lend(actual_u));

    GKO_ASSERT_MTX_NEAR(actual_l, expected_l, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(actual_u, expected_u, r<value_type>::value);
}


TYPED_TEST(ParIlu, KernelInitializeLUZeroMatrix)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    auto actual_l = Csr::create(this->ref);
    auto actual_u = Csr::create(this->ref);
    actual_l->copy_from(gko::lend(this->identity));
    actual_u->copy_from(gko::lend(this->identity));

    gko::kernels::reference::factorization::initialize_l_u(
        this->ref, gko::lend(this->empty_csr), gko::lend(actual_l),
        gko::lend(actual_u));

    GKO_ASSERT_MTX_NEAR(actual_l, this->identity, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(actual_u, this->identity, r<value_type>::value);
}


TYPED_TEST(ParIlu, KernelComputeLU)
{
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    using Coo = typename TestFixture::Coo;
    using Csr = typename TestFixture::Csr;
    // clang-format off
    auto l_dense =
        gko::initialize<Dense>({{1., 0., 0.},
                                {2., 1., 0.},
                                {1., 1., 1.}}, this->ref);
    // U must be transposed before calling the kernel, so we simply create it
    // transposed
    auto u_dense =
        gko::initialize<Dense>({{4., 0., 0.},
                                {6., 2., 0.},
                                {8., 5., 1.}}, this->ref);
    // clang-format on
    auto l_csr = Csr::create(this->ref);
    auto u_csr = Csr::create(this->ref);
    auto mtx_coo = Coo::create(this->ref);
    constexpr unsigned int iterations = 1;
    l_dense->convert_to(gko::lend(l_csr));
    u_dense->convert_to(gko::lend(u_csr));
    this->mtx_small->convert_to(gko::lend(mtx_coo));
    // The expected result of U also needs to be transposed
    auto u_expected_lin_op = this->small_u_expected->transpose();
    auto u_expected = std::unique_ptr<Dense>(
        static_cast<Dense *>(u_expected_lin_op.release()));

    gko::kernels::reference::par_ilu_factorization::compute_l_u_factors(
        this->ref, iterations, gko::lend(mtx_coo), gko::lend(l_csr),
        gko::lend(u_csr));

    GKO_ASSERT_MTX_NEAR(l_csr, this->small_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_csr, u_expected, r<value_type>::value);
}


TYPED_TEST(ParIlu, ThrowNotSupportedForWrongLinOp1)
{
    auto linOp = DummyLinOp::create(this->ref);

    ASSERT_THROW(this->ilu_factory_skip->generate(gko::share(linOp)),
                 gko::NotSupported);
}


TYPED_TEST(ParIlu, ThrowNotSupportedForWrongLinOp2)
{
    auto linOp = DummyLinOp::create(this->ref);

    ASSERT_THROW(this->ilu_factory_sort->generate(gko::share(linOp)),
                 gko::NotSupported);
}


TYPED_TEST(ParIlu, ThrowDimensionMismatch)
{
    using Csr = typename TestFixture::Csr;
    auto matrix = Csr::create(this->ref, gko::dim<2>{2, 3}, 4);

    ASSERT_THROW(this->ilu_factory_sort->generate(gko::share(matrix)),
                 gko::DimensionMismatch);
}


TYPED_TEST(ParIlu, SetLStrategy)
{
    using Csr = typename TestFixture::Csr;
    using par_ilu_type = typename TestFixture::par_ilu_type;
    auto l_strategy = std::make_shared<typename Csr::classical>();

    auto factory =
        par_ilu_type::build().with_l_strategy(l_strategy).on(this->ref);
    auto par_ilu = factory->generate(this->mtx_small);

    ASSERT_EQ(factory->get_parameters().l_strategy, l_strategy);
    ASSERT_EQ(par_ilu->get_l_factor()->get_strategy()->get_name(),
              l_strategy->get_name());
}


TYPED_TEST(ParIlu, SetUStrategy)
{
    using Csr = typename TestFixture::Csr;
    using par_ilu_type = typename TestFixture::par_ilu_type;
    auto u_strategy = std::make_shared<typename Csr::classical>();

    auto factory =
        par_ilu_type::build().with_u_strategy(u_strategy).on(this->ref);
    auto par_ilu = factory->generate(this->mtx_small);

    ASSERT_EQ(factory->get_parameters().u_strategy, u_strategy);
    ASSERT_EQ(par_ilu->get_u_factor()->get_strategy()->get_name(),
              u_strategy->get_name());
}


TYPED_TEST(ParIlu, LUFactorFunctionsSetProperly)
{
    auto factors = this->ilu_factory_skip->generate(this->mtx_small);

    auto lin_op_l_factor =
        static_cast<const gko::LinOp *>(gko::lend(factors->get_l_factor()));
    auto lin_op_u_factor =
        static_cast<const gko::LinOp *>(gko::lend(factors->get_u_factor()));
    auto first_operator = gko::lend(factors->get_operators()[0]);
    auto second_operator = gko::lend(factors->get_operators()[1]);

    ASSERT_EQ(lin_op_l_factor, first_operator);
    ASSERT_EQ(lin_op_u_factor, second_operator);
}


TYPED_TEST(ParIlu, GenerateForCooIdentity)
{
    using Coo = typename TestFixture::Coo;
    using value_type = typename TestFixture::value_type;
    auto coo_mtx = gko::share(Coo::create(this->exec));
    this->identity->convert_to(gko::lend(coo_mtx));

    auto factors = this->ilu_factory_skip->generate(coo_mtx);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->identity, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->identity, r<value_type>::value);
}


TYPED_TEST(ParIlu, GenerateForCsrIdentity)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto csr_mtx = gko::share(Csr::create(this->exec));
    this->identity->convert_to(gko::lend(csr_mtx));

    auto factors = this->ilu_factory_skip->generate(csr_mtx);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->identity, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->identity, r<value_type>::value);
}


TYPED_TEST(ParIlu, GenerateForDenseIdentity)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->identity);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->identity, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->identity, r<value_type>::value);
}


TYPED_TEST(ParIlu, GenerateForDenseLowerTriangular)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->lower_triangular);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->lower_triangular, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->identity, r<value_type>::value);
}


TYPED_TEST(ParIlu, GenerateForDenseUpperTriangular)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->upper_triangular);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->identity, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->upper_triangular, r<value_type>::value);
}


TYPED_TEST(ParIlu, ApplyMethodDenseSmall)
{
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    const auto x = gko::initialize<Dense>({1., 2., 3.}, this->exec);
    auto b_lu = Dense::create_with_config_of(gko::lend(x));
    auto b_ref = Dense::create_with_config_of(gko::lend(x));

    auto factors = this->ilu_factory_skip->generate(this->mtx_small);
    factors->apply(gko::lend(x), gko::lend(b_lu));
    this->mtx_small->apply(gko::lend(x), gko::lend(b_ref));

    GKO_ASSERT_MTX_NEAR(b_lu, b_ref, r<value_type>::value);
}


TYPED_TEST(ParIlu, GenerateForDenseSmall)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->mtx_small);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->small_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->small_u_expected, r<value_type>::value);
}


TYPED_TEST(ParIlu, GenerateForCsrSmall)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->mtx_csr_small);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->small_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->small_u_expected, r<value_type>::value);
}


TYPED_TEST(ParIlu, GenerateForCsrSmall2ZeroDiagonal)
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


TYPED_TEST(ParIlu, GenerateForCsrBigWithDiagonalZeros)
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


TYPED_TEST(ParIlu, GenerateForDenseSmallWithMultipleIterations)
{
    using value_type = typename TestFixture::value_type;
    using par_ilu_type = typename TestFixture::par_ilu_type;
    auto multiple_iter_factory =
        par_ilu_type::build().with_iterations(5u).with_skip_sorting(true).on(
            this->exec);
    auto factors = multiple_iter_factory->generate(this->mtx_small);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->small_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->small_u_expected, r<value_type>::value);
}


TYPED_TEST(ParIlu, GenerateForDenseBig)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->mtx_big);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->big_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->big_u_expected, r<value_type>::value);
}


TYPED_TEST(ParIlu, GenerateForDenseBigSort)
{
    using value_type = typename TestFixture::value_type;
    auto factors = this->ilu_factory_skip->generate(this->mtx_big);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, this->big_l_expected, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_factor, this->big_u_expected, r<value_type>::value);
}


TYPED_TEST(ParIlu, GenerateForReverseCooSmall)
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


TYPED_TEST(ParIlu, GenerateForReverseCsrSmall)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    const auto size = this->mtx_csr_small->get_size();
    const auto nnz = size[0] * size[1];
    auto reverse_csr = gko::share(Csr::create(this->exec));
    reverse_csr->copy_from(gko::lend(this->mtx_csr_small));
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
