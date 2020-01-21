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


#include "core/factorization/par_ilu_kernels.hpp"
#include "core/test/utils/assertions.hpp"


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


class ParIlu : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
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
          big_nodiag_l_expected(gko::initialize<Dense>({{1., 0., 0., 0., 0., 0.},
                                                        {1., 1., 0., 0., 0., 0.},
                                                        {0., 2., 1., 0., 0., 0.},
                                                        {1., 0., 2., 1., 0., 0.},
                                                        {1., 1., 0., -2., 1., 0.},
                                                        {0., 2., 1., -0.5, 2.5, 1.}},
                                                exec)),
          big_nodiag_u_expected(gko::initialize<Dense>({{1., 1., 1., 0., 1., 3.},
                                                        {0., 1., 1., 0., 1., 0.},
                                                        {0., 0., 1., 3., 1., 5.},
                                                        {0., 0., 0., -2., 1., -9.},
                                                        {0., 0., 0., 0., 1., -15.},
                                                        {0., 0., 0., 0., 0., 36.}},
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
    std::shared_ptr<const Dense> mtx_big;
    std::shared_ptr<const Dense> big_l_expected;
    std::shared_ptr<const Dense> big_u_expected;
    std::shared_ptr<const Csr> mtx_big_nodiag;
    std::shared_ptr<const Dense> big_nodiag_l_expected;
    std::shared_ptr<const Dense> big_nodiag_u_expected;
    std::unique_ptr<par_ilu_type::Factory> ilu_factory_skip;
    std::unique_ptr<par_ilu_type::Factory> ilu_factory_sort;
};


TEST_F(ParIlu, KernelAddDiagonalElementsEmpty)
{
    auto expected_mtx =
        Csr::create(ref, empty_csr->get_size(),
                    std::initializer_list<value_type>{0., 0., 0.},
                    std::initializer_list<index_type>{0, 1, 2},
                    std::initializer_list<index_type>{0, 1, 2, 3});
    auto empty_mtx = empty_csr->clone();

    gko::kernels::reference::par_ilu_factorization::add_diagonal_elements(
        ref, empty_mtx.get(), true);

    GKO_ASSERT_MTX_NEAR(empty_mtx, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(empty_mtx, expected_mtx);
}


TEST_F(ParIlu, KernelAddDiagonalElementsAsymetric)
{
    auto matrix = gko::initialize<Csr>(
        {{0., 0., 0.}, {1., 0., 0.}, {1., 1., 1.}, {1., 1., 1.}}, ref);
    auto exp_values = {0., 1., 0., 1., 1., 1., 1., 1., 1.};
    auto exp_col_idxs = {0, 0, 1, 0, 1, 2, 0, 1, 2};
    auto exp_row_ptrs = {0, 1, 3, 6, 9};
    auto expected_mtx =
        Csr::create(ref, matrix->get_size(), std::move(exp_values),
                    std::move(exp_col_idxs), std::move(exp_row_ptrs));

    gko::kernels::reference::par_ilu_factorization::add_diagonal_elements(
        ref, matrix.get(), true);

    GKO_ASSERT_MTX_NEAR(matrix, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(matrix, expected_mtx);
}


TEST_F(ParIlu, KernelAddDiagonalElementsAsymetric2)
{
    auto matrix = gko::initialize<Csr>({{1., 0., 0.}, {1., 0., 0.}}, ref);
    auto exp_values = {1., 1., 0.};
    auto exp_col_idxs = {0, 0, 1};
    auto exp_row_ptrs = {0, 1, 3};
    auto expected_mtx =
        Csr::create(ref, matrix->get_size(), std::move(exp_values),
                    std::move(exp_col_idxs), std::move(exp_row_ptrs));

    gko::kernels::reference::par_ilu_factorization::add_diagonal_elements(
        ref, matrix.get(), true);

    GKO_ASSERT_MTX_NEAR(matrix, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(matrix, expected_mtx);
}


TEST_F(ParIlu, KernelAddDiagonalElementsUnsorted)
{
    auto size = gko::dim<2>{3, 3};
    /* matrix:
    1 2 3
    1 0 3
    1 2 0
    */
    auto mtx_values = {3., 2., 1., 3., 1., 2., 1.};
    auto mtx_col_idxs = {2, 1, 0, 2, 0, 1, 0};
    auto mtx_row_ptrs = {0, 3, 5, 7};
    auto matrix = Csr::create(ref, size, std::move(mtx_values),
                              std::move(mtx_col_idxs), std::move(mtx_row_ptrs));
    auto exp_values = {1., 2., 3., 1., 0., 3., 1., 2., 0.};
    auto exp_col_idxs = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    auto exp_row_ptrs = {0, 3, 6, 9};
    auto expected_mtx =
        Csr::create(ref, size, std::move(exp_values), std::move(exp_col_idxs),
                    std::move(exp_row_ptrs));

    gko::kernels::reference::par_ilu_factorization::add_diagonal_elements(
        ref, matrix.get(), false);

    GKO_ASSERT_MTX_NEAR(matrix, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(matrix, expected_mtx);
}


TEST_F(ParIlu, KernelInitializeRowPtrsLU)
{
    auto small_csr_l_expected = Csr::create(ref);
    small_l_expected->convert_to(gko::lend(small_csr_l_expected));
    auto small_csr_u_expected = Csr::create(ref);
    small_u_expected->convert_to(gko::lend(small_csr_u_expected));
    auto num_row_ptrs = mtx_csr_small->get_size()[0] + 1;
    std::vector<index_type> l_row_ptrs_vector(num_row_ptrs);
    std::vector<index_type> u_row_ptrs_vector(num_row_ptrs);
    auto l_row_ptrs = l_row_ptrs_vector.data();
    auto u_row_ptrs = u_row_ptrs_vector.data();

    gko::kernels::reference::par_ilu_factorization::initialize_row_ptrs_l_u(
        ref, gko::lend(mtx_csr_small), l_row_ptrs, u_row_ptrs);

    ASSERT_TRUE(std::equal(l_row_ptrs, l_row_ptrs + num_row_ptrs,
                           small_csr_l_expected->get_const_row_ptrs()));
    ASSERT_TRUE(std::equal(u_row_ptrs, u_row_ptrs + num_row_ptrs,
                           small_csr_u_expected->get_const_row_ptrs()));
}


TEST_F(ParIlu, KernelInitializeRowPtrsLUZeroMatrix)
{
    auto empty_csr_l_expected = Csr::create(ref);
    identity->convert_to(gko::lend(empty_csr_l_expected));
    auto empty_csr_u_expected = Csr::create(ref);
    identity->convert_to(gko::lend(empty_csr_u_expected));
    auto num_row_ptrs = empty_csr->get_size()[0] + 1;
    std::vector<index_type> l_row_ptrs_vector(num_row_ptrs);
    std::vector<index_type> u_row_ptrs_vector(num_row_ptrs);
    auto l_row_ptrs = l_row_ptrs_vector.data();
    auto u_row_ptrs = u_row_ptrs_vector.data();

    gko::kernels::reference::par_ilu_factorization::initialize_row_ptrs_l_u(
        ref, gko::lend(empty_csr), l_row_ptrs, u_row_ptrs);

    ASSERT_TRUE(std::equal(l_row_ptrs, l_row_ptrs + num_row_ptrs,
                           empty_csr_l_expected->get_const_row_ptrs()));
    ASSERT_TRUE(std::equal(u_row_ptrs, u_row_ptrs + num_row_ptrs,
                           empty_csr_u_expected->get_const_row_ptrs()));
}


TEST_F(ParIlu, KernelInitializeLU)
{
    // clang-format off
    auto expected_l =
        gko::initialize<Dense>({{1., 0., 0.},
                                {2., 1., 0.},
                                {1., 1., 1.}}, ref);
    auto expected_u =
        gko::initialize<Dense>({{4., 6., 8.},
                                {0., 2., 5.},
                                {0., 0., 1.}}, ref);
    // clang-format on
    auto actual_l = Csr::create(ref, mtx_csr_small->get_size(), 6);
    auto actual_u = Csr::create(ref, mtx_csr_small->get_size(), 6);
    // Copy row_ptrs into matrices, which usually come from the
    // `initialize_row_ptrs_l_u` kernel
    std::vector<index_type> l_row_ptrs{0, 1, 3, 6};
    std::vector<index_type> u_row_ptrs{0, 3, 5, 6};
    std::copy(l_row_ptrs.begin(), l_row_ptrs.end(), actual_l->get_row_ptrs());
    std::copy(u_row_ptrs.begin(), u_row_ptrs.end(), actual_u->get_row_ptrs());

    gko::kernels::reference::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(mtx_csr_small), gko::lend(actual_l),
        gko::lend(actual_u));

    GKO_ASSERT_MTX_NEAR(actual_l, expected_l, 1e-14);
    GKO_ASSERT_MTX_NEAR(actual_u, expected_u, 1e-14);
}


TEST_F(ParIlu, KernelInitializeLUZeroMatrix)
{
    auto actual_l = Csr::create(ref);
    auto actual_u = Csr::create(ref);
    actual_l->copy_from(identity.get());
    actual_u->copy_from(identity.get());

    gko::kernels::reference::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(empty_csr), gko::lend(actual_l), gko::lend(actual_u));

    GKO_ASSERT_MTX_NEAR(actual_l, identity, 1e-14);
    GKO_ASSERT_MTX_NEAR(actual_u, identity, 1e-14);
}


TEST_F(ParIlu, KernelComputeLU)
{
    // clang-format off
    auto l_dense =
        gko::initialize<Dense>({{1., 0., 0.},
                                {2., 1., 0.},
                                {1., 1., 1.}}, ref);
    // U must be transposed before calling the kernel, so we simply create it
    // transposed
    auto u_dense =
        gko::initialize<Dense>({{4., 0., 0.},
                                {6., 2., 0.},
                                {8., 5., 1.}}, ref);
    // clang-format on
    auto l_csr = Csr::create(ref);
    auto u_csr = Csr::create(ref);
    auto mtx_coo = Coo::create(ref);
    constexpr unsigned int iterations = 1;
    l_dense->convert_to(gko::lend(l_csr));
    u_dense->convert_to(gko::lend(u_csr));
    mtx_small->convert_to(gko::lend(mtx_coo));
    // The expected result of U also needs to be transposed
    auto u_expected_lin_op = small_u_expected->transpose();
    auto u_expected = std::unique_ptr<Dense>(
        static_cast<Dense *>(u_expected_lin_op.release()));

    gko::kernels::reference::par_ilu_factorization::compute_l_u_factors(
        ref, iterations, gko::lend(mtx_coo), gko::lend(l_csr),
        gko::lend(u_csr));

    GKO_ASSERT_MTX_NEAR(l_csr, small_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_csr, u_expected, 1e-14);
}


TEST_F(ParIlu, ThrowNotSupportedForWrongLinOp1)
{
    auto linOp = DummyLinOp::create(ref);

    ASSERT_THROW(ilu_factory_skip->generate(gko::share(linOp)),
                 gko::NotSupported);
}


TEST_F(ParIlu, ThrowNotSupportedForWrongLinOp2)
{
    auto linOp = DummyLinOp::create(ref);

    ASSERT_THROW(ilu_factory_sort->generate(gko::share(linOp)),
                 gko::NotSupported);
}


TEST_F(ParIlu, ThrowDimensionMismatch)
{
    auto matrix = Csr::create(ref, gko::dim<2>{2, 3}, 4);

    ASSERT_THROW(ilu_factory_sort->generate(gko::share(matrix)),
                 gko::DimensionMismatch);
}


TEST_F(ParIlu, SetLStrategy)
{
    auto l_strategy = std::make_shared<typename Csr::automatical>(0, 0);

    auto factory = par_ilu_type::build().with_l_strategy(l_strategy).on(ref);
    auto par_ilu = factory->generate(mtx_small);

    ASSERT_EQ(factory->get_parameters().l_strategy, l_strategy);
    ASSERT_EQ(par_ilu->get_l_factor()->get_strategy(), l_strategy);
}


TEST_F(ParIlu, SetUStrategy)
{
    auto u_strategy = std::make_shared<typename Csr::classical>();

    auto factory = par_ilu_type::build().with_u_strategy(u_strategy).on(ref);
    auto par_ilu = factory->generate(mtx_small);

    ASSERT_EQ(factory->get_parameters().u_strategy, u_strategy);
    ASSERT_EQ(par_ilu->get_u_factor()->get_strategy(), u_strategy);
}


TEST_F(ParIlu, LUFactorFunctionsSetProperly)
{
    auto factors = ilu_factory_skip->generate(mtx_small);

    auto lin_op_l_factor =
        static_cast<const gko::LinOp *>(factors->get_l_factor().get());
    auto lin_op_u_factor =
        static_cast<const gko::LinOp *>(factors->get_u_factor().get());
    auto first_operator = factors->get_operators()[0].get();
    auto second_operator = factors->get_operators()[1].get();

    ASSERT_EQ(lin_op_l_factor, first_operator);
    ASSERT_EQ(lin_op_u_factor, second_operator);
}


TEST_F(ParIlu, GenerateForCooIdentity)
{
    auto coo_mtx = gko::share(Coo::create(exec));
    identity->convert_to(coo_mtx.get());

    auto factors = ilu_factory_skip->generate(coo_mtx);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity, 1e-14);
}


TEST_F(ParIlu, GenerateForCsrIdentity)
{
    auto csr_mtx = gko::share(Csr::create(exec));
    identity->convert_to(csr_mtx.get());

    auto factors = ilu_factory_skip->generate(csr_mtx);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseIdentity)
{
    auto factors = ilu_factory_skip->generate(identity);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseLowerTriangular)
{
    auto factors = ilu_factory_skip->generate(lower_triangular);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, lower_triangular, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseUpperTriangular)
{
    auto factors = ilu_factory_skip->generate(upper_triangular);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, upper_triangular, 1e-14);
}


TEST_F(ParIlu, ApplyMethodDenseSmall)
{
    const auto x = gko::initialize<Dense>({1., 2., 3.}, exec);
    auto b_lu = Dense::create_with_config_of(gko::lend(x));
    auto b_ref = Dense::create_with_config_of(gko::lend(x));

    auto factors = ilu_factory_skip->generate(mtx_small);
    factors->apply(gko::lend(x), gko::lend(b_lu));
    mtx_small->apply(gko::lend(x), gko::lend(b_ref));

    GKO_ASSERT_MTX_NEAR(b_lu, b_ref, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseSmall)
{
    auto factors = ilu_factory_skip->generate(mtx_small);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, small_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, small_u_expected, 1e-14);
}


TEST_F(ParIlu, GenerateForCsrSmall)
{
    auto factors = ilu_factory_skip->generate(mtx_csr_small);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, small_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, small_u_expected, 1e-14);
}


TEST_F(ParIlu, GenerateForCsrBigWithDiagonalZeros)
{
    auto factors = ilu_factory_skip->generate(mtx_big_nodiag);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, big_nodiag_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, big_nodiag_u_expected, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseSmallWithMultipleIterations)
{
    auto multiple_iter_factory =
        par_ilu_type::build().with_iterations(5u).with_skip_sorting(true).on(
            exec);
    auto factors = multiple_iter_factory->generate(mtx_small);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, small_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, small_u_expected, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseBig)
{
    auto factors = ilu_factory_skip->generate(mtx_big);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, big_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, big_u_expected, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseBigSort)
{
    auto factors = ilu_factory_skip->generate(mtx_big);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, big_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, big_u_expected, 1e-14);
}


TEST_F(ParIlu, GenerateForReverseCooSmall)
{
    const auto size = mtx_small->get_size();
    const auto nnz = size[0] * size[1];
    auto reverse_coo = gko::share(Coo::create(exec, size, nnz));
    // Fill the Coo matrix in reversed row order (right to left)
    for (size_t i = 0; i < size[0]; ++i) {
        for (size_t j = 0; j < size[1]; ++j) {
            const auto coo_idx = i * size[1] + (size[1] - 1 - j);
            reverse_coo->get_row_idxs()[coo_idx] = i;
            reverse_coo->get_col_idxs()[coo_idx] = j;
            reverse_coo->get_values()[coo_idx] = mtx_small->at(i, j);
        }
    }

    auto factors = ilu_factory_sort->generate(reverse_coo);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(reverse_coo, mtx_small, 1e-14);
    GKO_ASSERT_MTX_NEAR(l_factor, small_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, small_u_expected, 1e-14);
}


TEST_F(ParIlu, GenerateForReverseCsrSmall)
{
    const auto size = mtx_csr_small->get_size();
    const auto nnz = size[0] * size[1];
    auto reverse_csr = gko::share(Csr::create(exec));
    reverse_csr->copy_from(mtx_csr_small.get());
    // Fill the Csr matrix rows in reverse order
    for (size_t i = 0; i < size[0]; ++i) {
        const auto row_start = reverse_csr->get_row_ptrs()[i];
        const auto row_end = reverse_csr->get_row_ptrs()[i + 1];
        for (size_t j = row_start; j < row_end; ++j) {
            const auto reverse_j = row_end - 1 - (j - row_start);
            reverse_csr->get_values()[reverse_j] =
                mtx_csr_small->get_const_values()[j];
            reverse_csr->get_col_idxs()[reverse_j] =
                mtx_csr_small->get_const_col_idxs()[j];
        }
    }

    auto factors = ilu_factory_sort->generate(reverse_csr);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, small_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, small_u_expected, 1e-14);
}


}  // namespace
