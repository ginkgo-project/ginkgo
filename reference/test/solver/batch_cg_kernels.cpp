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

#include <ginkgo/core/solver/batch_cg.hpp>

#include <gtest/gtest.h>

#include <ginkgo/core/log/batch_convergence.hpp>

#include "core/solver/batch_cg_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"


namespace {


template <typename T>
class BatchCg : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using Options = gko::kernels::batch_cg::BatchCgOptions<real_type>;

    BatchCg()
        : exec(gko::ReferenceExecutor::create()),
          xex_1(gko::batch_initialize<BDense>(nbatch, {1.0, 3.0, 2.0}, exec)),
          b_1(gko::batch_initialize<BDense>(nbatch, {-1.0, 3.0, 1.0}, exec)),
          xex_m(gko::batch_initialize<BDense>(
              nbatch,
              std::initializer_list<std::initializer_list<value_type>>{
                  {1.0, 1.0}, {3.0, 0.0}, {2.0, 0.0}},
              exec)),
          b_m(gko::batch_initialize<BDense>(
              nbatch,
              std::initializer_list<std::initializer_list<value_type>>{
                  {-1.0, 2.0}, {3.0, -1.0}, {1.0, 0.0}},
              exec))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    const real_type eps = r<value_type>::value;

    const size_t nbatch = 2;
    const int nrows = 3;
    std::shared_ptr<const BDense> b_1;
    std::shared_ptr<const BDense> xex_1;
    std::shared_ptr<RBDense> bnorm_1;
    const Options opts_1{gko::preconditioner::batch::type::none, 500,
                         static_cast<real_type>(1e3) * eps,
                         gko::stop::batch::ToleranceType::relative};

    const int nrhs = 2;
    std::shared_ptr<const BDense> b_m;
    std::shared_ptr<const BDense> xex_m;
    std::shared_ptr<RBDense> bnorm_m;
    const Options opts_m{gko::preconditioner::batch::type::none, 500, eps,
                         gko::stop::batch::ToleranceType::absolute};

    struct Result {
        std::shared_ptr<BDense> x;
        std::shared_ptr<RBDense> resnorm;
        gko::log::BatchLogData<value_type> logdata;
        std::shared_ptr<BDense> residual;
    };
    Result r_1;
    Result r_m;

    Result solve_poisson_uniform_1(const Options opts,
                                   const BDense *const left_scale = nullptr,
                                   const BDense *const right_scale = nullptr)
    {
        bnorm_1 = gko::batch_initialize<RBDense>(nbatch, {0.0}, exec);
        b_1->compute_norm2(bnorm_1.get());

        const int nrhs_1 = 1;
        auto mtx = gko::test::create_poisson1d_batch<value_type>(this->exec,
                                                                 nrows, nbatch);
        auto orig_mtx = gko::test::create_poisson1d_batch<value_type>(
            this->exec, nrows, nbatch);
        Result result;
        // Initialize r = b before b is potentially modified
        result.residual = b_1->clone();
        result.x =
            gko::batch_initialize<BDense>(nbatch, {0.0, 0.0, 0.0}, this->exec);

        std::vector<gko::dim<2>> sizes(nbatch, gko::dim<2>(1, nrhs_1));
        result.logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->exec, sizes);
        result.logdata.iter_counts.set_executor(this->exec);
        result.logdata.iter_counts.resize_and_reset(nrhs_1 * nbatch);

        gko::kernels::reference::batch_cg::apply<value_type>(
            this->exec, opts, mtx.get(), left_scale, right_scale, b_1.get(),
            result.x.get(), result.logdata);

        result.resnorm =
            gko::batch_initialize<RBDense>(nbatch, {0.0}, this->exec);
        auto alpha = gko::batch_initialize<BDense>(nbatch, {-1.0}, this->exec);
        auto beta = gko::batch_initialize<BDense>(nbatch, {1.0}, this->exec);
        orig_mtx->apply(alpha.get(), result.x.get(), beta.get(),
                        result.residual.get());
        result.residual->compute_norm2(result.resnorm.get());
        return result;
    }


    int single_iters_regression() const
    {
        if (std::is_same<real_type, float>::value) {
            return 3;
        } else if (std::is_same<real_type, double>::value) {
            return 3;
        } else {
            return -1;
        }
    }

    Result solve_poisson_uniform_mult()
    {
        bnorm_m = gko::batch_initialize<RBDense>(nbatch, {{0.0, 0.0}}, exec);
        b_m->compute_norm2(bnorm_m.get());

        const int nrows = 3;
        auto mtx = gko::test::create_poisson1d_batch<value_type>(this->exec,
                                                                 nrows, nbatch);
        Result result;
        result.x = gko::batch_initialize<BDense>(
            nbatch,
            std::initializer_list<std::initializer_list<value_type>>{
                {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
            this->exec);

        std::vector<gko::dim<2>> sizes(nbatch, gko::dim<2>(1, nrhs));
        result.logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->exec, sizes);
        result.logdata.iter_counts.set_executor(this->exec);
        result.logdata.iter_counts.resize_and_reset(nrhs * nbatch);

        gko::kernels::reference::batch_cg::apply<value_type>(
            this->exec, opts_m, mtx.get(), nullptr, nullptr, b_m.get(),
            result.x.get(), result.logdata);

        result.residual = b_m->clone();
        result.resnorm =
            gko::batch_initialize<RBDense>(nbatch, {{0.0, 0.0}}, this->exec);
        auto alpha = gko::batch_initialize<BDense>(nbatch, {-1.0}, this->exec);
        auto beta = gko::batch_initialize<BDense>(nbatch, {1.0}, this->exec);
        mtx->apply(alpha.get(), result.x.get(), beta.get(),
                   result.residual.get());
        result.residual->compute_norm2(result.resnorm.get());
        return result;
    }


    std::vector<int> multiple_iters_regression() const
    {
        std::vector<int> iters(2);
        if (std::is_same<real_type, float>::value) {
            iters[0] = 3;
            iters[1] = 3;
        } else if (std::is_same<real_type, double>::value) {
            iters[0] = 3;
            iters[1] = 3;
        } else {
            iters[0] = -1;
            iters[1] = -1;
        }
        return iters;
    }
};

TYPED_TEST_SUITE(BatchCg, gko::test::ValueTypes);


TYPED_TEST(BatchCg, SolvesStencilSystem)
{
    using value_type = typename TestFixture::value_type;

    this->r_1 = this->solve_poisson_uniform_1(this->opts_1);

    GKO_ASSERT_BATCH_MTX_NEAR(this->r_1.x, this->xex_1, this->eps);
}

TYPED_TEST(BatchCg, StencilSystemLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    this->r_1 = this->solve_poisson_uniform_1(this->opts_1);

    const int ref_iters = this->single_iters_regression();

    const int *const iter_array =
        this->r_1.logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->r_1.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger

        GKO_ASSERT((iter_array[i] <= ref_iters + 1) &&
                   (iter_array[i] >= ref_iters - 1));
        ASSERT_LE(res_log_array[i] / this->bnorm_1->at(i, 0, 0),
                  this->opts_1.residual_tol);
        ASSERT_NEAR(res_log_array[i], this->r_1.resnorm->get_const_values()[i],
                    10 * this->eps);
    }
}


TYPED_TEST(BatchCg, SolvesStencilMultipleSystem)
{
    using value_type = typename TestFixture::value_type;

    this->r_m = this->solve_poisson_uniform_mult();

    GKO_ASSERT_BATCH_MTX_NEAR(this->r_m.x, this->xex_m, this->eps);
}


TYPED_TEST(BatchCg, StencilMultipleSystemLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    this->r_m = this->solve_poisson_uniform_mult();

    const std::vector<int> ref_iters = this->multiple_iters_regression();

    const int *const iter_array =
        this->r_m.logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->r_m.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        for (size_t j = 0; j < this->nrhs; j++) {
            GKO_ASSERT((iter_array[i * this->nrhs + j] <= ref_iters[j] + 1) &&
                       (iter_array[i * this->nrhs + j] >= ref_iters[j] - 1));

            ASSERT_LE(res_log_array[i * this->nrhs + j],
                      this->opts_m.residual_tol);

            ASSERT_NEAR(
                res_log_array[i * this->nrhs + j],
                this->r_m.resnorm->get_const_values()[i * this->nrhs + j],
                10 * this->eps);
        }
    }
}


TYPED_TEST(BatchCg, UnitScalingDoesNotChangeResult)
{
    using value_type = typename TestFixture::value_type;

    using Result = typename TestFixture::Result;
    using BDense = typename TestFixture::BDense;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);

    Result result = this->solve_poisson_uniform_1(
        this->opts_1, left_scale.get(), right_scale.get());


    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->xex_1, this->eps);
}


TYPED_TEST(BatchCg, GeneralScalingDoesNotChangeResult)
{
    using value_type = typename TestFixture::value_type;

    using Result = typename TestFixture::Result;
    using BDense = typename TestFixture::BDense;
    using Options = typename TestFixture::Options;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {0.8, 0.9, 0.95}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.5, 1.05}, this->exec);


    Result result = this->solve_poisson_uniform_1(
        this->opts_1, left_scale.get(), right_scale.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->xex_1, 1e3 * this->eps);
}


TEST(BatchCg, CanSolveWithoutScaling)
{
    using T = std::complex<double>;
    using RT = typename gko::remove_complex<T>;
    using Solver = gko::solver::BatchCg<T>;
    using Dense = gko::matrix::BatchDense<T>;
    using RDense = gko::matrix::BatchDense<RT>;
    using Mtx = typename gko::matrix::BatchCsr<T>;
    const RT tol = 1e-9;
    const int maxits = 1000;
    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
    auto batchcg_factory =
        Solver::build()
            .with_max_iterations(maxits)
            .with_rel_residual_tol(tol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .on(exec);
    const int nrows = 40;
    const size_t nbatch = 3;
    const int nrhs = 6;
    gko::test::test_solve_without_scaling<Solver>(
        exec, nbatch, nrows, nrhs, tol, maxits, batchcg_factory.get(), 1.01);
}


}  // namespace
