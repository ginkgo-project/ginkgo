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

#include <ginkgo/core/factorization/bilu.hpp>
#include <ginkgo/core/factorization/par_bilu.hpp>


#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>


#include "core/factorization/par_bilu_kernels.hpp"
#include "core/test/factorization/block_factorization_test_utils.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"


namespace {


template <typename ValueIndexType>
class ParBilu : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using par_bilu_t = gko::factorization::ParBilu<value_type, index_type>;
    using bilu_t = gko::factorization::Bilu<value_type, index_type>;

    const real_type tol = std::numeric_limits<real_type>::epsilon();

    ParBilu()
        : refexec(gko::ReferenceExecutor::create()),
          ompexec{gko::OmpExecutor::create()}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> refexec;
    std::shared_ptr<const gko::OmpExecutor> ompexec;

    void test_parbilu(const bool skip_sorting, const int block_size,
                      const int iterations)
    {
        const int nbrows = 50;
        const bool diag_dominant = true;
        const bool unsort = !skip_sorting;
        std::shared_ptr<const Fbcsr> rmtx =
            gko::test::generate_random_fbcsr<value_type, index_type>(
                this->refexec, std::ranlux48(42), nbrows, nbrows, block_size,
                diag_dominant, unsort);
        std::shared_ptr<Fbcsr> mtx = Fbcsr::create(this->ompexec);
        mtx->copy_from(rmtx.get());
        auto parbilu_factory = par_bilu_t::build()
                                   .with_iterations(iterations)
                                   .with_skip_sorting(skip_sorting)
                                   .on(this->ompexec);
        auto bilu_factory = par_bilu_t::build()
                                .with_iterations(1)
                                .with_skip_sorting(skip_sorting)
                                .on(this->refexec);

        auto bfacts = bilu_factory->generate(rmtx);
        auto parbfacts = parbilu_factory->generate(mtx);

        auto bfL = bfacts->get_l_factor();
        auto pbfL = parbfacts->get_l_factor();
        auto bfU = bfacts->get_u_factor();
        auto pbfU = parbfacts->get_u_factor();
        const auto eps = 10.0 * this->tol;
        GKO_ASSERT_MTX_EQ_SPARSITY(bfL, pbfL);
        GKO_ASSERT_MTX_EQ_SPARSITY(bfU, pbfU);
        GKO_ASSERT_MTX_NEAR(bfL, pbfL, eps);
        GKO_ASSERT_MTX_NEAR(bfU, pbfU, eps);
    }

    void test_jac_parbilu(const bool skip_sorting, const int nbrows,
                          const int block_size, const int iterations)
    {
        const bool diag_dominant = true;
        const bool unsort = !skip_sorting;
        std::shared_ptr<const Fbcsr> rmtx =
            gko::test::generate_random_fbcsr<value_type, index_type>(
                this->refexec, std::ranlux48(42), nbrows, nbrows, block_size,
                diag_dominant, unsort);
        std::shared_ptr<Fbcsr> mtx = Fbcsr::create(this->ompexec);
        mtx->copy_from(rmtx.get());

        std::shared_ptr<Fbcsr> rl_fact, ru_fact_init;
        gko::test::initialize_bilu(rmtx.get(), &rl_fact, &ru_fact_init);
        auto ru_transpose = gko::as<Fbcsr>(ru_fact_init->transpose());
        auto l_fact = Fbcsr::create(this->ompexec);
        auto u_fact_init = Fbcsr::create(this->ompexec);
        l_fact->copy_from(rl_fact.get());
        u_fact_init->copy_from(ru_fact_init.get());

        gko::kernels::reference::par_bilu_factorization::
            compute_bilu_factors_jacobi(this->refexec, iterations,
                                        gko::lend(rmtx), gko::lend(rl_fact),
                                        gko::lend(ru_transpose));
        auto ru_fact = gko::as<Fbcsr>(ru_transpose->transpose());

        auto u_transpose = gko::as<Fbcsr>(u_fact_init->transpose());

        gko::kernels::omp::par_bilu_factorization::compute_bilu_factors_jacobi(
            this->ompexec, iterations, gko::lend(mtx), gko::lend(l_fact),
            gko::lend(u_transpose));
        auto u_fact = gko::as<Fbcsr>(u_transpose->transpose());

        const auto eps = 10.0 * this->tol;
        GKO_ASSERT_MTX_EQ_SPARSITY(rl_fact, l_fact);
        GKO_ASSERT_MTX_EQ_SPARSITY(ru_fact, u_fact);
        GKO_ASSERT_MTX_NEAR(rl_fact, l_fact, eps);
        GKO_ASSERT_MTX_NEAR(ru_fact, u_fact, eps);
    }
};

using SomeTypes = ::testing::Types<std::tuple<float, gko::int32>,
                                   std::tuple<std::complex<float>, gko::int32>>;

TYPED_TEST_SUITE(ParBilu, SomeTypes);


TYPED_TEST(ParBilu, FactorizationSortedBS4) { this->test_parbilu(true, 4, 20); }

TYPED_TEST(ParBilu, FactorizationSortedBS7) { this->test_parbilu(true, 7, 20); }

TYPED_TEST(ParBilu, FactorizationUnsortedBS3)
{
    this->test_parbilu(false, 3, 20);
}

TYPED_TEST(ParBilu, JacobiFactorizationSortedBS4)
{
    this->test_jac_parbilu(true, 15, 4, 1);
}

TYPED_TEST(ParBilu, JacobiFactorizationSortedBS7)
{
    this->test_jac_parbilu(true, 11, 7, 1);
}

}  // namespace
