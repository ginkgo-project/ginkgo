/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/batch_par_ilu.hpp>


#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/preconditioner/batch_par_ilu_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


class BatchParIlu : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using prec_type = gko::preconditioner::BatchParIlu<value_type>;

    BatchParIlu()
        : ref(gko::ReferenceExecutor::create()),
          d_exec(gko::OmpExecutor::create()),
          mtx(gko::share(gko::test::generate_uniform_batch_random_matrix<Mtx>(
              nbatch, nrows, nrows,
              std::uniform_int_distribution<>(min_nnz_row, nrows),
              std::normal_distribution<real_type>(0.0, 1.0), rand_engine, true,
              ref)))
    {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> d_exec;
    std::ranlux48 rand_engine;

    const size_t nbatch = 9;
    const index_type nrows = 33;
    const int min_nnz_row = 5;
    std::shared_ptr<const Mtx> mtx;
};


TEST_F(BatchParIlu, GenerateIsEquivalentToReference)
{
    auto d_mtx = gko::share(gko::clone(d_exec, mtx.get()));
    auto prec_fact =
        prec_type::build().with_skip_sorting(true).with_num_sweeps(20).on(ref);
    auto d_prec_fact =
        prec_type::build().with_skip_sorting(true).with_num_sweeps(20).on(
            d_exec);

    auto prec = prec_fact->generate(mtx);
    auto d_prec = d_prec_fact->generate(d_mtx);

    const auto lower_factor = prec->get_const_l_factor();
    const auto upper_factor = prec->get_const_u_factor();
    const auto d_lower_factor = d_prec->get_const_l_factor();
    const auto d_upper_factor = d_prec->get_const_u_factor();
    const auto tol = 1000 * r<value_type>::value;
    GKO_ASSERT_BATCH_MTX_NEAR(lower_factor, d_lower_factor, tol);
    GKO_ASSERT_BATCH_MTX_NEAR(upper_factor, d_upper_factor, tol);
}


TEST_F(BatchParIlu, ApplyIsEquivalentToReference)
{
    using BDense = gko::matrix::BatchDense<value_type>;

    auto prec_fact =
        prec_type::build().with_skip_sorting(true).with_num_sweeps(20).on(ref);
    auto prec = prec_fact->generate(mtx);

    const auto l_factor = prec->get_const_l_factor();
    const auto u_factor = prec->get_const_u_factor();

    auto d_mtx = gko::share(gko::clone(d_exec, mtx.get()));
    auto d_prec_fact =
        prec_type::build().with_skip_sorting(true).with_num_sweeps(20).on(
            d_exec);
    auto d_prec = d_prec_fact->generate(d_mtx);

    const auto d_l_factor = d_prec->get_const_l_factor();
    const auto d_u_factor = d_prec->get_const_u_factor();

    auto rv = gko::test::generate_uniform_batch_random_matrix<BDense>(
        nbatch, nrows, 1, std::uniform_int_distribution<>(1, 1),
        std::normal_distribution<real_type>(0.0, 1.0), rand_engine, false, ref);
    auto z = BDense::create(ref, rv->get_size());
    auto d_rv = gko::as<BDense>(gko::clone(d_exec, rv));
    auto d_z = BDense::create(d_exec, rv->get_size());

    gko::kernels::reference::batch_par_ilu::apply_par_ilu0(
        ref, l_factor, u_factor, rv.get(), z.get());
    gko::kernels::omp::batch_par_ilu::apply_par_ilu0(
        d_exec, d_l_factor, d_u_factor, d_rv.get(), d_z.get());

    const auto tol = 5000 * r<value_type>::value;
    GKO_ASSERT_BATCH_MTX_NEAR(z, d_z, tol);
}


}  // namespace
