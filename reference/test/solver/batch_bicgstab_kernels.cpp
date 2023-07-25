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

#include <ginkgo/core/solver/batch_bicgstab.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/log/batch_convergence.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"


namespace {


template <typename T>
class BatchBicgstab : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using Options =
        gko::kernels::batch_bicgstab::BatchBicgstabOptions<real_type>;
    using LogData = gko::log::BatchLogData<value_type>;

    BatchBicgstab() : exec(gko::ReferenceExecutor::create())
    {
        sys_1.xex =
            gko::batch_initialize<BDense>(nbatch, {1.0, 3.0, 2.0}, exec);
        sys_1.b = gko::batch_initialize<BDense>(nbatch, {-1.0, 3.0, 1.0}, exec);
        sys_1.mtx =
            gko::test::create_poisson1d_batch<value_type>(exec, nrows, nbatch);
        sys_1.bnorm = gko::batch_initialize<RBDense>(nbatch, {0.0}, exec);
        sys_1.b->compute_norm2(sys_1.bnorm.get());

        sys_m.xex = gko::batch_initialize<BDense>(
            nbatch,
            std::initializer_list<std::initializer_list<value_type>>{
                {1.0, 1.0}, {3.0, 0.0}, {2.0, 0.0}},
            exec);
        sys_m.b = gko::batch_initialize<BDense>(
            nbatch,
            std::initializer_list<std::initializer_list<value_type>>{
                {-1.0, 2.0}, {3.0, -1.0}, {1.0, 0.0}},
            exec);
        sys_m.mtx =
            gko::test::create_poisson1d_batch<value_type>(exec, nrows, nbatch);
        sys_m.bnorm =
            gko::batch_initialize<RBDense>(nbatch, {{0.0, 0.0}}, exec);
        sys_m.b->compute_norm2(sys_m.bnorm.get());

        auto execp = this->exec;
        solve_fn = [execp](const Options opts, const Mtx *mtx, const BDense *b,
                           BDense *x, LogData &logdata) {
            gko::kernels::reference::batch_bicgstab::apply<value_type>(
                execp, opts, mtx, b, x, logdata);
        };
        scale_mat = [execp](const BDense *const left, const BDense *const right,
                            Mtx *const mat) {
            gko::kernels::reference::batch_csr::batch_scale<value_type>(
                execp, left, right, mat);
        };
        scale_vecs = [execp](const BDense *const scale, BDense *const mat) {
            gko::kernels::reference::batch_dense::batch_scale<value_type>(
                execp, scale, mat);
        };
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    const real_type eps = r<value_type>::value;

    const size_t nbatch = 2;
    const int nrows = 3;
    const Options opts_1{gko::preconditioner::batch::type::none, 500,
                         static_cast<real_type>(1e3) * eps,
                         gko::stop::batch::ToleranceType::relative};

    const int nrhs = 2;
    const Options opts_m{gko::preconditioner::batch::type::none, 500, eps,
                         gko::stop::batch::ToleranceType::absolute};

    gko::test::LinSys<value_type> sys_1;
    gko::test::LinSys<value_type> sys_m;

    std::function<void(Options, const Mtx *, const BDense *, BDense *,
                       LogData &)>
        solve_fn;
    std::function<void(const BDense *, const BDense *, Mtx *)> scale_mat;
    std::function<void(const BDense *, BDense *)> scale_vecs;

    int single_iters_regression() const
    {
        if (std::is_same<real_type, float>::value) {
            return 2;
        } else if (std::is_same<real_type, double>::value) {
            return 2;
        } else {
            return -1;
        }
    }

    std::vector<int> multiple_iters_regression() const
    {
        std::vector<int> iters(2);
        if (std::is_same<real_type, float>::value) {
            iters[0] = 2;
            iters[1] = 2;
        } else if (std::is_same<real_type, double>::value) {
            iters[0] = 2;
            iters[1] = 2;
        } else {
            iters[0] = -1;
            iters[1] = -1;
        }
        return iters;
    }
};

TYPED_TEST_SUITE(BatchBicgstab, gko::test::ValueTypes);


TYPED_TEST(BatchBicgstab, SolvesStencilSystem)
{
    auto r_1 = gko::test::solve_poisson_uniform(
        this->exec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts_1, this->sys_1, 1);

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(r_1.resnorm->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  this->opts_1.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(r_1.x, this->sys_1.xex, this->eps);
}

TYPED_TEST(BatchBicgstab, StencilSystemLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto r_1 = gko::test::solve_poisson_uniform<value_type>(
        this->exec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts_1, this->sys_1, 1);

    const int ref_iters = this->single_iters_regression();
    const int *const iter_array = r_1.logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        r_1.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        GKO_ASSERT((iter_array[i] <= ref_iters + 1) &&
                   (iter_array[i] >= ref_iters - 1));
        ASSERT_LE(res_log_array[i] / this->sys_1.bnorm->at(i, 0, 0),
                  this->opts_1.residual_tol);
        ASSERT_NEAR(res_log_array[i], r_1.resnorm->get_const_values()[i],
                    10 * this->eps);
    }
}


TYPED_TEST(BatchBicgstab, SolvesStencilMultipleSystem)
{
    auto r_m = gko::test::solve_poisson_uniform(
        this->exec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts_m, this->sys_m, this->nrhs);

    GKO_ASSERT_BATCH_MTX_NEAR(r_m.x, this->sys_m.xex, this->eps);
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(r_m.resnorm->get_const_values()[i],
                  this->opts_m.residual_tol);
    }
}


TYPED_TEST(BatchBicgstab, StencilMultipleSystemLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto r_m = gko::test::solve_poisson_uniform(
        this->exec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts_m, this->sys_m, this->nrhs);

    const std::vector<int> ref_iters = this->multiple_iters_regression();
    const int *const iter_array = r_m.logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        r_m.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        for (size_t j = 0; j < this->nrhs; j++) {
            GKO_ASSERT((iter_array[i * this->nrhs + j] <= ref_iters[j] + 1) &&
                       (iter_array[i * this->nrhs + j] >= ref_iters[j] - 1));
            ASSERT_LE(res_log_array[i * this->nrhs + j],
                      this->opts_m.residual_tol);
            ASSERT_NEAR(res_log_array[i * this->nrhs + j],
                        r_m.resnorm->get_const_values()[i * this->nrhs + j],
                        10 * this->eps);
        }
    }
}


TYPED_TEST(BatchBicgstab, UnitScalingDoesNotChangeResult)
{
    using BDense = typename TestFixture::BDense;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);

    auto result = gko::test::solve_poisson_uniform(
        this->exec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts_1, this->sys_1, 1, left_scale.get(), right_scale.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, this->eps);
}


TYPED_TEST(BatchBicgstab, GeneralScalingDoesNotChangeResult)
{
    using BDense = typename TestFixture::BDense;
    using Options = typename TestFixture::Options;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {0.8, 0.9, 0.95}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.5, 1.05}, this->exec);

    auto result = gko::test::solve_poisson_uniform(
        this->exec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts_1, this->sys_1, 1, left_scale.get(), right_scale.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, this->eps);
}


TEST(BatchBicgstab, CanSolveWithoutScaling)
{
    using T = std::complex<float>;
    using RT = typename gko::remove_complex<T>;
    using Solver = gko::solver::BatchBicgstab<T>;
    const RT tol = 1e-5;
    const int maxits = 1000;
    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
    auto batchbicgstab_factory =
        Solver::build()
            .with_max_iterations(maxits)
            .with_rel_residual_tol(tol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .on(exec);
    const int nrows = 40;
    const size_t nbatch = 3;
    const int nrhs = 5;

    gko::test::test_solve_without_scaling<Solver>(
        exec, nbatch, nrows, nrhs, tol, maxits, batchbicgstab_factory.get(),
        10);
}


}  // namespace
