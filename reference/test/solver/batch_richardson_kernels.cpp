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

#include <ginkgo/core/solver/batch_richardson.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/log/batch_convergence.hpp>


#include "core/solver/batch_richardson_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


namespace gpb = gko::preconditioner::batch;


template <typename T>
class BatchRich : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using Options = gko::kernels::batch_rich::BatchRichardsonOptions<real_type>;

    BatchRich()
        : exec(gko::ReferenceExecutor::create()),
          xex_1(gko::batch_initialize<BDense>(nbatch, {1.0, 3.0, 2.0}, exec)),
          b_1(gko::batch_initialize<BDense>(nbatch, {-1.0, 3.0, 1.0}, exec)),
          xex_m(gko::batch_initialize<BDense>(
              nbatch,
              std::initializer_list<std::initializer_list<value_type>>{
                  {1.0, 1.0}, {3.0, -2.0}, {2.0, 2.5}},
              exec)),
          b_m(gko::batch_initialize<BDense>(
              nbatch,
              std::initializer_list<std::initializer_list<value_type>>{
                  {-1.0, 4.0}, {3.0, -7.5}, {1.0, 7.0}},
              exec))
    {
        bnorm_1 = gko::batch_initialize<RBDense>(nbatch, {0.0}, exec);
        b_1->compute_norm2(bnorm_1.get());
        bnorm_m = gko::batch_initialize<RBDense>(nbatch, {{0.0, 0.0}}, exec);
        b_m->compute_norm2(bnorm_m.get());
        r_1 = solve_poisson_uniform_1(opts_1);
        r_m = solve_poisson_uniform_mult();
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    const size_t nbatch = 2;
    const int nrows = 3;
    std::shared_ptr<const BDense> b_1;
    std::shared_ptr<const BDense> xex_1;
    std::shared_ptr<RBDense> bnorm_1;
    const Options opts_1{gpb::type::jacobi, 500, r<real_type>::value, 1.0};

    const int nrhs = 2;
    std::shared_ptr<const BDense> b_m;
    std::shared_ptr<const BDense> xex_m;
    std::shared_ptr<RBDense> bnorm_m;
    const Options opts_m{gpb::type::jacobi, 1000, r<real_type>::value, 1.0};

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

        gko::kernels::reference::batch_rich::apply<value_type>(
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
            return 40;
        } else if (std::is_same<real_type, double>::value) {
            return 98;
        } else {
            return -1;
        }
    }

    Result solve_poisson_uniform_mult()
    {
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

        gko::kernels::reference::batch_rich::apply<value_type>(
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
            iters[0] = 40;
            iters[1] = 40;
        } else if (std::is_same<real_type, double>::value) {
            iters[0] = 98;
            iters[1] = 98;
        } else {
            iters[0] = -1;
            iters[1] = -1;
        }
        return iters;
    }
};

TYPED_TEST_SUITE(BatchRich, gko::test::ValueTypes);


TYPED_TEST(BatchRich, SolvesStencilSystemJacobi)
{
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(this->r_1.resnorm->get_const_values()[i] /
                      this->bnorm_1->get_const_values()[i],
                  this->opts_1.rel_residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(this->r_1.x, this->xex_1,
                              1e-6 /*r<value_type>::value*/);
}

TYPED_TEST(BatchRich, StencilSystemJacobiLoggerIsSameAsBefore)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    const int ref_iters = this->single_iters_regression();

    const int *const iter_array =
        this->r_1.logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->r_1.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        ASSERT_EQ(iter_array[i], ref_iters);
        ASSERT_LE(res_log_array[i] / this->bnorm_1->get_const_values()[i],
                  this->opts_1.rel_residual_tol);
        ASSERT_NEAR(res_log_array[i] / this->bnorm_1->get_const_values()[i],
                    this->r_1.resnorm->get_const_values()[i] /
                        this->bnorm_1->get_const_values()[i],
                    10 * r<value_type>::value);
    }
}


TYPED_TEST(BatchRich, SolvesStencilMultipleSystemJacobi)
{
    for (size_t i = 0; i < this->nbatch; i++) {
        for (size_t j = 0; j < this->nrhs; j++) {
            ASSERT_LE(
                this->r_m.resnorm->get_const_values()[i * this->nrhs + j] /
                    this->bnorm_m->get_const_values()[i * this->nrhs + j],
                this->opts_m.rel_residual_tol);
        }
    }
    GKO_ASSERT_BATCH_MTX_NEAR(this->r_m.x, this->xex_m,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, StencilMultipleSystemJacobiLoggerIsSameAsBefore)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    const std::vector<int> ref_iters = this->multiple_iters_regression();

    const int *const iter_array =
        this->r_m.logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->r_m.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        for (size_t j = 0; j < this->nrhs; j++) {
            ASSERT_EQ(iter_array[i * this->nrhs + j], ref_iters[j]);
            ASSERT_LE(res_log_array[i * this->nrhs + j] /
                          this->bnorm_m->get_const_values()[i * this->nrhs + j],
                      this->opts_m.rel_residual_tol);
            ASSERT_NEAR(res_log_array[i] / this->bnorm_m->get_const_values()[i],
                        this->r_m.resnorm->get_const_values()[i] /
                            this->bnorm_m->get_const_values()[i],
                        10 * r<value_type>::value);
        }
    }
}


TYPED_TEST(BatchRich, BetterRelaxationFactorGivesBetterConvergence)
{
    using Result = typename TestFixture::Result;
    using BDense = typename TestFixture::BDense;
    using Options = typename TestFixture::Options;
    const Options opts{gpb::type::jacobi, 1000, 1e-8, 1.0};
    const Options opts_slower{gpb::type::jacobi, 1000, 1e-8, 0.8};

    Result result1 = this->solve_poisson_uniform_1(opts);
    Result result2 = this->solve_poisson_uniform_1(opts_slower);

    const int *const iter_arr1 = result1.logdata.iter_counts.get_const_data();
    const int *const iter_arr2 = result2.logdata.iter_counts.get_const_data();
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(iter_arr1[i], iter_arr2[i]);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result2.x, this->xex_1,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, UnitScalingDoesNotChangeResult)
{
    using Result = typename TestFixture::Result;
    using BDense = typename TestFixture::BDense;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);

    Result result = this->solve_poisson_uniform_1(
        this->opts_1, left_scale.get(), right_scale.get());

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(result.resnorm->get_const_values()[i] /
                      this->bnorm_1->get_const_values()[i],
                  this->opts_1.rel_residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->xex_1,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, GeneralScalingDoesNotChangeResult)
{
    using Result = typename TestFixture::Result;
    using BDense = typename TestFixture::BDense;
    using Options = typename TestFixture::Options;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {0.8, 0.9, 0.95}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.5, 1.05}, this->exec);
    const Options opts = this->opts_1;

    Result result = this->solve_poisson_uniform_1(opts, left_scale.get(),
                                                  right_scale.get());

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(result.resnorm->get_const_values()[i] /
                      this->bnorm_1->get_const_values()[i],
                  opts.rel_residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->xex_1,
                              1e-5 /*r<value_type>::value*/);
}


TEST(BatchRich, CanSolveWithoutScaling)
{
    using T = std::complex<float>;
    using RT = typename gko::remove_complex<T>;
    using Solver = gko::solver::BatchRichardson<T>;
    using Dense = gko::matrix::BatchDense<T>;
    using RDense = gko::matrix::BatchDense<RT>;
    using Mtx = typename gko::matrix::BatchCsr<T>;
    const RT tol = std::numeric_limits<RT>::epsilon();
    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
    const int maxits = 10000;
    auto batchrich_factory = Solver::build()
                                 .with_max_iterations(maxits)
                                 .with_rel_residual_tol(tol * 200)
                                 .with_relaxation_factor(RT{0.98})
                                 .on(exec);
    const int nrows = 40;
    const size_t nbatch = 3;
    std::shared_ptr<Mtx> mtx =
        gko::test::create_poisson1d_batch<T>(exec, nrows, nbatch);
    auto solver = batchrich_factory->generate(mtx);
    std::shared_ptr<const gko::log::BatchConvergence<T>> logger =
        gko::log::BatchConvergence<T>::create(exec);
    solver->add_logger(logger);
    const int nrhs = 7;
    auto b =
        Dense::create(exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrhs)));
    auto x = Dense::create_with_config_of(b.get());
    auto res = Dense::create_with_config_of(b.get());
    auto alpha = gko::batch_initialize<Dense>(nbatch, {-1.0}, exec);
    auto beta = gko::batch_initialize<Dense>(nbatch, {1.0}, exec);
    auto bnorm =
        RDense::create(exec, gko::batch_dim<>(nbatch, gko::dim<2>(1, nrhs)));
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int j = 0; j < nrhs; j++) {
            bnorm->at(ib, 0, j) = gko::zero<RT>();
            const T val = 1.0 + std::cos(ib / 2.0 - j / 4.0);
            for (int i = 0; i < nrows; i++) {
                b->at(ib, i, j) = val;
                x->at(ib, i, j) = 0.0;
                res->at(ib, i, j) = val;
                bnorm->at(ib, 0, j) += gko::squared_norm(val);
            }
            bnorm->at(ib, 0, j) = std::sqrt(bnorm->at(ib, 0, j));
        }
    }

    solver->apply(b.get(), x.get());

    mtx->apply(alpha.get(), x.get(), beta.get(), res.get());
    auto rnorm =
        RDense::create(exec, gko::batch_dim<>(nbatch, gko::dim<2>(1, nrhs)));
    res->compute_norm2(rnorm.get());
    const auto iter_array = logger->get_num_iterations();
    const auto logged_res = logger->get_residual_norm();
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int j = 0; j < nrhs; j++) {
            ASSERT_LE(rnorm->at(ib, 0, j) / bnorm->at(ib, 0, j), tol * 200);
            ASSERT_GT(iter_array.get_const_data()[ib * nrhs + j], 0);
            ASSERT_LE(iter_array.get_const_data()[ib * nrhs + j], maxits);
        }
    }
    GKO_ASSERT_BATCH_MTX_NEAR(logged_res, rnorm, tol);
}


}  // namespace
