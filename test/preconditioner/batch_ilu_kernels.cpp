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

#include <ginkgo/core/preconditioner/batch_ilu.hpp>


#include <limits>


#include <gtest/gtest.h>


#include "core/preconditioner/batch_ilu_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


class BatchIlu : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using prec_type = gko::preconditioner::BatchIlu<value_type>;

    BatchIlu()
        : rand_engine(42),
          ref(gko::ReferenceExecutor::create()),
          d_exec(gko::HipExecutor::create(0, ref)),
          mtx(gko::share(gko::test::generate_uniform_batch_random_matrix<Mtx>(
              nbatch, nrows, nrows,
              std::uniform_int_distribution<>(min_nnz_row, nrows),
              std::normal_distribution<real_type>(0.0, 1.0), rand_engine, true,
              ref)))
    {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> d_exec;
    std::ranlux48 rand_engine;

    const size_t nbatch = 9;
    const index_type nrows = 33;
    const int min_nnz_row = 5;
    std::shared_ptr<const Mtx> mtx;
};


TEST_F(BatchIlu, GenerateIsEquivalentToReference)
{
    auto d_mtx = gko::share(gko::clone(d_exec, mtx.get()));
    auto prec_fact = prec_type::build().on(ref);
    auto d_prec_fact = prec_type::build().on(d_exec);

    auto prec = prec_fact->generate(mtx);
    auto d_prec = d_prec_fact->generate(d_mtx);

    auto l_factor = prec->get_const_lower_factor();
    auto u_factor = prec->get_const_upper_factor();
    auto d_l_factor = d_prec->get_const_lower_factor();
    auto d_u_factor = d_prec->get_const_upper_factor();
    const auto tol = 500 * r<value_type>::value;
    GKO_ASSERT_BATCH_MTX_NEAR(l_factor, d_l_factor, tol);
    GKO_ASSERT_BATCH_MTX_NEAR(u_factor, d_u_factor, tol);
}


TEST_F(BatchIlu, ApplyIsEquivalentToReference)
{
    using BDense = gko::matrix::BatchDense<value_type>;
    auto prec_fact = prec_type::build().on(ref);
    auto prec = prec_fact->generate(mtx);
    auto l_factor = prec->get_const_lower_factor();
    auto u_factor = prec->get_const_upper_factor();
    auto d_l_factor = gko::clone(d_exec, l_factor);
    auto d_u_factor = gko::clone(d_exec, u_factor);
    auto rv = gko::test::generate_uniform_batch_random_matrix<BDense>(
        nbatch, nrows, 1, std::uniform_int_distribution<>(1, 1),
        std::normal_distribution<real_type>(0.0, 1.0), rand_engine, false, ref);
    auto z = BDense::create(ref, rv->get_size());
    auto d_rv = gko::as<BDense>(gko::clone(d_exec, rv));
    auto d_z = BDense::create(d_exec, rv->get_size());

    gko::kernels::reference::batch_ilu::apply_split(ref, l_factor, u_factor,
                                                    rv.get(), z.get());
    gko::kernels::hip::batch_ilu::apply_split(
        d_exec, d_l_factor.get(), d_u_factor.get(), d_rv.get(), d_z.get());

    const auto tol = 500 * r<value_type>::value;
    GKO_ASSERT_BATCH_MTX_NEAR(z, d_z, tol);
}


}  // namespace
