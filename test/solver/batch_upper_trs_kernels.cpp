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

#include "core/solver/batch_upper_trs_kernels.hpp"


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/solver/batch_upper_trs.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"
#include "test/utils/executor.hpp"


#ifndef GKO_COMPILING_DPCPP

//TODO: Add tests for non-sorted input matrix
//TODO: Add tests for matrix in ell format
class BatchUpperTrs : public CommonTestFixture {
protected:
    using real_type = gko::remove_complex<value_type>;
    using BCsr = gko::matrix::BatchCsr<value_type, int>;
    using Csr = BCsr::unbatch_type;
    using BDense = gko::matrix::BatchDense<value_type>;
    using Dense = BDense::unbatch_type;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using solver_type = gko::solver::BatchUpperTrs<value_type>;

    BatchUpperTrs() : csr_upper_mat(get_csr_upper_matrix()) , dense_upper_mat(get_dense_upper_matrix()) 
    {
        set_up_data();
    }

    const real_type eps = r<value_type>::value;

    std::ranlux48 rand_engine;

    const size_t nbatch = 9;
    const index_type nrows = 300;
    
    std::shared_ptr<BCsr> csr_upper_mat;
    std::shared_ptr<BDense> dense_upper_mat;
    std::shared_ptr<BDense> b;
    std::shared_ptr<BDense> x;
    std::shared_ptr<BDiag> left_scale;
    std::shared_ptr<BDiag> right_scale;

    std::unique_ptr<BCsr> get_csr_upper_matrix()
    {
        auto unbatch_mat =
            gko::test::generate_random_triangular_matrix<Csr>(
                nrows, false, false,
                std::uniform_int_distribution<>(nrows, nrows),
                std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                ref);

        return BCsr::create(ref, nbatch, unbatch_mat.get());
    }

    std::unique_ptr<BDense> get_dense_upper_matrix()
    {
        auto unbatch_mat =
            gko::test::generate_random_triangular_matrix<Dense>(
                nrows, false, false,
                std::uniform_int_distribution<>(nrows, nrows),
                std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                ref);

        return BDense::create(ref, nbatch, unbatch_mat.get());
    }

    void set_up_data()
    {
        b = gko::share(gko::test::generate_uniform_batch_random_matrix<BDense>(
                  nbatch, nrows, 1,
                  std::uniform_int_distribution<>(nrows, nrows),
                  std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                  true, ref));
        x = gko::share(BDense::create(
            ref, gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, 1))));

        left_scale = gko::share(gko::test::generate_uniform_batch_random_matrix<BDiag>(
                  nbatch, nrows, nrows,
                  std::uniform_int_distribution<>(nrows, nrows),
                  std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                  true, ref));

        
        right_scale = gko::share(gko::test::generate_uniform_batch_random_matrix<BDiag>(
                  nbatch, nrows, nrows,
                  std::uniform_int_distribution<>(nrows, nrows),
                  std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                  true, ref));

    }


};


TEST_F(BatchUpperTrs, CsrSolveIsEquivalentToReference)
{
    using solver_type = gko::solver::BatchUpperTrs<value_type>;
    auto d_csr_upper_mat = gko::share(gko::clone(exec, csr_upper_mat.get()));
    auto d_b = gko::share(gko::clone(exec, b.get()));
    auto d_x = gko::share(gko::clone(exec, x.get()));

    auto upper_trs = solver_type::build().with_skip_sorting(true).on(ref)->generate(csr_upper_mat);
    auto d_upper_trs = solver_type::build().with_skip_sorting(true).on(exec)->generate(d_csr_upper_mat);

    upper_trs->apply(b.get(), x.get());
    d_upper_trs->apply(d_b.get(), d_x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(x, d_x, 100* eps);

}


TEST_F(BatchUpperTrs, CsrSolveWithScalingIsEquivalentToReference)
{
    using solver_type = gko::solver::BatchUpperTrs<value_type>;
    auto d_csr_upper_mat = gko::share(gko::clone(exec, csr_upper_mat.get()));
    auto d_b = gko::share(gko::clone(exec, b.get()));
    auto d_x = gko::share(gko::clone(exec, x.get()));
    auto d_left_scale =  gko::share(gko::clone(exec, left_scale.get()));
    auto d_right_scale = gko::share(gko::clone(exec, right_scale.get()));

    auto upper_trs = solver_type::build().with_skip_sorting(true).with_left_scaling_op(left_scale)
    .with_right_scaling_op(right_scale).on(ref)->generate(csr_upper_mat);
    auto d_upper_trs = solver_type::build().with_skip_sorting(true)
    .with_left_scaling_op(d_left_scale).with_right_scaling_op(d_right_scale).on(exec)->generate(d_csr_upper_mat);

    upper_trs->apply(b.get(), x.get());
    d_upper_trs->apply(d_b.get(), d_x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(x, d_x, 100 * eps);

}


TEST_F(BatchUpperTrs, DenseSolveIsEquivalentToReference)
{
    using solver_type = gko::solver::BatchUpperTrs<value_type>;
    auto d_dense_upper_mat = gko::share(gko::clone(exec, dense_upper_mat.get()));
    auto d_b = gko::share(gko::clone(exec, b.get()));
    auto d_x = gko::share(gko::clone(exec, x.get()));

    auto upper_trs = solver_type::build().with_skip_sorting(true).on(ref)->generate(dense_upper_mat);
    auto d_upper_trs = solver_type::build().with_skip_sorting(true).on(exec)->generate(d_dense_upper_mat);

    upper_trs->apply(b.get(), x.get());
    d_upper_trs->apply(d_b.get(), d_x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(x, d_x, 100* eps);

}


TEST_F(BatchUpperTrs, DenseSolveWithScalingIsEquivalentToReference)
{
    using solver_type = gko::solver::BatchUpperTrs<value_type>;
    auto d_dense_upper_mat = gko::share(gko::clone(exec, dense_upper_mat.get()));
    auto d_b = gko::share(gko::clone(exec, b.get()));
    auto d_x = gko::share(gko::clone(exec, x.get()));
    auto d_left_scale =  gko::share(gko::clone(exec, left_scale.get()));
    auto d_right_scale = gko::share(gko::clone(exec, right_scale.get()));

    auto upper_trs = solver_type::build().with_skip_sorting(true).with_left_scaling_op(left_scale)
    .with_right_scaling_op(right_scale).on(ref)->generate(dense_upper_mat);
    auto d_upper_trs = solver_type::build().with_skip_sorting(true)
    .with_left_scaling_op(d_left_scale).with_right_scaling_op(d_right_scale).on(exec)->generate(d_dense_upper_mat);

    upper_trs->apply(b.get(), x.get());
    d_upper_trs->apply(d_b.get(), d_x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(x, d_x, 100 * eps);

}


#endif
