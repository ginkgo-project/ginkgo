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


#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/preconditioner/batch_ilu.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
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
        : ref(gko::ReferenceExecutor::create()),
          d_exec(gko::CudaExecutor::create(0, ref)),
          mtx(gko::share(gko::test::generate_uniform_batch_random_matrix<Mtx>(
              nbatch, nrows, nrows,
              std::uniform_int_distribution<>(min_nnz_row, nrows),
              std::normal_distribution<real_type>(0.0, 1.0), rand_engine, true,
              ref)))
    {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> d_exec;
    std::ranlux48 rand_engine;

    const size_t nbatch = 9;
    const index_type nrows = 33;
    const int min_nnz_row = 5;
    std::shared_ptr<const Mtx> mtx;
};


TEST_F(BatchIlu, GenerateIsEquivalentToReference)
{
    auto d_mtx = gko::clone(d_exec, mtx.get());
    auto prec_fact = prec_type::build().on(ref);
    auto d_prec_fact = prec_type::build().on(d_exec);

    auto prec = prec_fact->generate(mtx);
    auto d_prec = d_prec_fact->generate(gko::share(d_mtx));

    auto l_factor = prec->get_const_lower_factor();
    auto u_factor = prec->get_const_upper_factor();
    auto d_l_factor = d_prec->get_const_lower_factor();
    auto d_u_factor = d_prec->get_const_upper_factor();
    const auto tol = 10 * r<value_type>::value;
    GKO_ASSERT_BATCH_MTX_NEAR(l_factor, d_l_factor, tol);
    GKO_ASSERT_BATCH_MTX_NEAR(u_factor, d_u_factor, tol);
}


}  // namespace
