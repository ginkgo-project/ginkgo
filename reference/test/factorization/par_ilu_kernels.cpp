/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/par_ilu_kernels.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


class ParIlu : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    ParIlu()
        : ref(gko::ReferenceExecutor::create()),
          exec(std::static_pointer_cast<const gko::Executor>(ref)),
          identity(gko::initialize<Dense>(
              {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}, exec)),
          lower_triangular(gko::initialize<Dense>(
              {{1., 0., 0.}, {1., 1., 0.}, {1., 1., 1.}}, exec)),
          upper_triangular(gko::initialize<Dense>(
              {{1., 1., 1.}, {0., 1., 1.}, {0., 0., 1.}}, exec)),
          mtx_small(gko::initialize<Dense>(
              {{4., 6., 8.}, {2., 2., 5.}, {1., 1., 1.}}, exec)),
          mtx_csr_small(nullptr),
          small_l_expected(gko::initialize<Dense>(
              {{1., 0., 0.}, {0.5, 1., 0.}, {0.25, 0.5, 1.}}, exec)),
          small_u_expected(gko::initialize<Dense>(
              {{4., 6., 8.}, {0., -1., 1.}, {0., 0., -1.5}}, exec)),
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
          ilu_factory(gko::factorization::ParIlu<>::build().on(exec))
    {
        auto tmp_csr = Csr::create(exec);
        mtx_small->convert_to(gko::lend(tmp_csr));
        mtx_csr_small = std::move(tmp_csr);
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
    std::shared_ptr<const Dense> mtx_big;
    std::shared_ptr<const Dense> big_l_expected;
    std::shared_ptr<const Dense> big_u_expected;
    std::unique_ptr<gko::factorization::ParIlu<>::Factory> ilu_factory;
};


TEST_F(ParIlu, KernelComputeNnzLU)
{
    gko::size_type l_nnz;
    gko::size_type u_nnz;

    gko::kernels::reference::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(mtx_csr_small), &l_nnz, &u_nnz);

    ASSERT_EQ(l_nnz, 6);
    ASSERT_EQ(u_nnz, 6);
}


TEST_F(ParIlu, KernelInitializeLU)
{
    auto expected_l =
        gko::initialize<Dense>({{1., 0., 0.}, {2., 1., 0.}, {1., 1., 1.}}, ref);
    auto expected_u =
        gko::initialize<Dense>({{4., 6., 8.}, {0., 2., 5.}, {0., 0., 1.}}, ref);
    auto actual_l = Csr::create(ref, mtx_csr_small->get_size(), 6);
    auto actual_u = Csr::create(ref, mtx_csr_small->get_size(), 6);

    gko::kernels::reference::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(mtx_csr_small), gko::lend(actual_l),
        gko::lend(actual_u));

    GKO_ASSERT_MTX_NEAR(actual_l, expected_l, 1e-14);
    GKO_ASSERT_MTX_NEAR(actual_u, expected_u, 1e-14);
}


TEST_F(ParIlu, KernelComputeLU)
{
    auto l_dense =
        gko::initialize<Dense>({{1., 0., 0.}, {2., 1., 0.}, {1., 1., 1.}}, ref);
    // U must be transposed before calling the kernel, so we simply create it
    // transposed
    auto u_dense =
        gko::initialize<Dense>({{4., 0., 0.}, {6., 2., 0.}, {8., 5., 1.}}, ref);
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


TEST_F(ParIlu, GenerateForCooIdentity)
{
    auto coo_mtx = Coo::create(exec);
    identity->convert_to(coo_mtx.get());

    auto factors = ilu_factory->generate(identity);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity, 1e-14);
}


TEST_F(ParIlu, GenerateForCsrIdentity)
{
    auto csr_mtx = Csr::create(exec);
    identity->convert_to(csr_mtx.get());

    auto factors = ilu_factory->generate(identity);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseIdentity)
{
    auto factors = ilu_factory->generate(identity);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseLowerTriangular)
{
    auto factors = ilu_factory->generate(lower_triangular);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, lower_triangular, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseUpperTriangular)
{
    auto factors = ilu_factory->generate(upper_triangular);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, upper_triangular, 1e-14);
}


TEST_F(ParIlu, ApplyMethodDenseSmall)
{
    const auto x = gko::initialize<const Dense>({1., 2., 3.}, exec);
    auto b_lu = Dense::create_with_config_of(gko::lend(x));
    auto b_ref = Dense::create_with_config_of(gko::lend(x));

    auto factors = ilu_factory->generate(mtx_small);
    factors->apply(gko::lend(x), gko::lend(b_lu));
    mtx_small->apply(gko::lend(x), gko::lend(b_ref));

    GKO_ASSERT_MTX_NEAR(b_lu, b_ref, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseSmall)
{
    auto factors = ilu_factory->generate(mtx_small);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, small_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, small_u_expected, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseBig)
{
    auto factors = ilu_factory->generate(mtx_big);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, big_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, big_u_expected, 1e-14);
}


// TODO add test for `with_skip_sorting`
TEST_F(ParIlu, GenerateForReverseCooSmall)
{
    const auto size = mtx_small->get_size();
    const auto nnz = size[0] * size[1];
    std::shared_ptr<Coo> reverse_coo = Coo::create(exec, size, nnz);
    // Fill the Coo matrix in reversed row order (right to left)
    for (size_t i = 0; i < size[0]; ++i) {
        for (size_t j = 0; j < size[1]; ++j) {
            const auto coo_idx = i * size[1] + (size[1] - 1 - j);
            reverse_coo->get_row_idxs()[coo_idx] = i;
            reverse_coo->get_col_idxs()[coo_idx] = j;
            reverse_coo->get_values()[coo_idx] = mtx_small->at(i, j);
        }
    }

    auto factors = ilu_factory->generate(reverse_coo);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, small_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, small_u_expected, 1e-14);
}


TEST_F(ParIlu, GenerateForReverseCsrSmall)
{
    const auto size = mtx_csr_small->get_size();
    const auto nnz = size[0] * size[1];
    std::shared_ptr<Csr> reverse_csr = Csr::create(exec);
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

    auto factors = ilu_factory->generate(reverse_csr);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, small_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, small_u_expected, 1e-14);
}


}  // namespace
