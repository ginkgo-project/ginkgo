/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/preconditioner/batch_ilu_kernels.hpp"


#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/preconditioner/batch_ilu.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "test/utils/executor.hpp"


namespace {


class BatchIlu : public CommonTestFixture {
protected:
    using value_type = double;
    using index_type = int;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using prec_type = gko::preconditioner::BatchIlu<value_type>;

    BatchIlu()
        : mtx(gko::share(gko::test::generate_uniform_batch_random_matrix<Mtx>(
              nbatch, nrows, nrows,
              std::uniform_int_distribution<>(min_nnz_row, nrows),
              std::normal_distribution<real_type>(0.0, 1.0), rand_engine, true,
              ref)))
    {}

    std::ranlux48 rand_engine;

    const size_t nbatch = 9;
    const index_type nrows = 15;
    const int min_nnz_row = 3;
    std::shared_ptr<const Mtx> mtx;

    // TODO: Add tests for non-sorted input matrix
    void test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type ilu_type,
                                   const int num_sweeps = 30)
    {
        auto d_mtx = gko::share(gko::clone(exec, mtx.get()));
        auto prec_fact = prec_type::build()
                             .with_skip_sorting(true)
                             .with_ilu_type(ilu_type)
                             .with_parilu_num_sweeps(num_sweeps)
                             .on(ref);
        auto d_prec_fact = prec_type::build()
                               .with_skip_sorting(true)
                               .with_ilu_type(ilu_type)
                               .with_parilu_num_sweeps(num_sweeps)
                               .on(exec);

        auto prec = prec_fact->generate(mtx);
        auto d_prec = d_prec_fact->generate(d_mtx);

        const auto factorized_mat = prec->get_const_factorized_matrix().get();
        const auto d_factorized_mat =
            d_prec->get_const_factorized_matrix().get();
        const auto tol = 5000 * r<value_type>::value;
        GKO_ASSERT_BATCH_MTX_NEAR(factorized_mat, d_factorized_mat, tol);
    }

    void test_apply_eqvt_to_ref(gko::preconditioner::batch_ilu_type ilu_type,
                                const int num_sweeps = 30)
    {
        using BDense = gko::matrix::BatchDense<value_type>;

        auto prec_fact = prec_type::build()
                             .with_skip_sorting(true)
                             .with_ilu_type(ilu_type)
                             .with_parilu_num_sweeps(num_sweeps)
                             .on(ref);
        auto prec = prec_fact->generate(mtx);

        const auto factorized_mat = prec->get_const_factorized_matrix().get();
        const auto diag_locs = prec->get_const_diag_locations();

        auto d_mtx = gko::share(gko::clone(exec, mtx.get()));
        auto d_prec_fact = prec_type::build()
                               .with_skip_sorting(true)
                               .with_ilu_type(ilu_type)
                               .with_parilu_num_sweeps(num_sweeps)
                               .on(exec);
        auto d_prec = d_prec_fact->generate(d_mtx);

        const auto d_factorized_mat =
            d_prec->get_const_factorized_matrix().get();
        const auto d_diag_locs = d_prec->get_const_diag_locations();

        auto rv = gko::test::generate_uniform_batch_random_matrix<BDense>(
            nbatch, nrows, 1, std::uniform_int_distribution<>(1, 1),
            std::normal_distribution<real_type>(0.0, 1.0), rand_engine, false,
            ref);
        auto z = BDense::create(ref, rv->get_size());
        auto d_rv = gko::as<BDense>(gko::clone(exec, rv));
        auto d_z = BDense::create(exec, rv->get_size());

        gko::kernels::reference::batch_ilu::apply_ilu(
            ref, mtx.get(), factorized_mat, diag_locs, rv.get(), z.get());
        gko::kernels::EXEC_NAMESPACE::batch_ilu::apply_ilu(
            exec, d_mtx.get(), d_factorized_mat, d_diag_locs, d_rv.get(),
            d_z.get());

        const auto tol = 5000 * r<value_type>::value;
        GKO_ASSERT_BATCH_MTX_NEAR(z, d_z, tol);
    }
};


TEST_F(BatchIlu, ExactIluGenerateIsEquivalentToReference)
{
    test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type::exact_ilu);
}


TEST_F(BatchIlu, ParIluGenerateIsEquivalentToReference)
{
    // Note: Since ref. parilu is basically eqvt. to exact ilu (even for a
    // single sweep), we use very high number of sweeps to compare the cuda/hip
    // parilu with the reference one. Such a high number of sweeps for cuda/hip
    // parilu would ensure that generated incomplete L and U factors are close
    // to the exact ILU factors.
    test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type::parilu, 30);
}


TEST_F(BatchIlu, ExactIluApplyIsEquivalentToReference)
{
    test_apply_eqvt_to_ref(gko::preconditioner::batch_ilu_type::exact_ilu);
}


TEST_F(BatchIlu, ParIluApplyIsEquivalentToReference)
{
    test_apply_eqvt_to_ref(gko::preconditioner::batch_ilu_type::parilu, 30);
}

TEST_F(BatchIlu, GenerateSplitFactorsIsEquivalentToReference)
{
    auto d_mtx = gko::share(gko::clone(exec, mtx.get()));
    auto prec_fact = prec_type::build().with_skip_sorting(true).on(ref);
    auto d_prec_fact = prec_type::build().with_skip_sorting(true).on(exec);

    auto prec = prec_fact->generate(mtx);
    auto d_prec = d_prec_fact->generate(d_mtx);

    const auto l_and_u_pair =
        prec->generate_split_factors_from_factored_matrix();
    const auto d_l_and_u_pair =
        d_prec->generate_split_factors_from_factored_matrix();
    const auto tol = 5000 * r<value_type>::value;
    GKO_ASSERT_BATCH_MTX_NEAR(l_and_u_pair.first.get(),
                              d_l_and_u_pair.first.get(), tol);
    GKO_ASSERT_BATCH_MTX_NEAR(l_and_u_pair.second.get(),
                              d_l_and_u_pair.second.get(), tol);
}

}  // namespace
