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

#include <ginkgo/core/solver/batch_direct.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/batch_convergence.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_direct_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"


namespace {


struct DummyOptions {};


template <typename T>
class BatchDirect : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using solver_type = gko::solver::BatchDirect<value_type>;
    using Options = DummyOptions;
    using LogData = gko::log::BatchLogData<value_type>;

    BatchDirect()
        : exec(gko::ReferenceExecutor::create()),
          cuexec(gko::CudaExecutor::create(0, exec)),
          opts{},
          sys_1(gko::test::get_poisson_problem<T>(exec, 1, nbatch)),
          sys_m(gko::test::get_poisson_problem<T>(exec, nrhs, nbatch))
    {
        auto execp = cuexec;
        solve_fn = [execp](const Options opts, const Mtx *mtx, const BDense *b,
                           BDense *x, LogData &logdata) {
            auto btemp =
                std::dynamic_pointer_cast<BDense>(gko::share(b->transpose()));
            auto a = BDense::create(execp, mtx->get_size());
            mtx->convert_to(a.get());
            auto atrans =
                std::dynamic_pointer_cast<BDense>(gko::share(a->transpose()));
            gko::kernels::cuda::batch_direct::apply<value_type>(
                execp, atrans.get(), btemp.get(), logdata);
            auto xtemp = std::dynamic_pointer_cast<BDense>(
                gko::share(btemp->transpose()));
            x->copy_from(xtemp.get());
        };
        scale_mat = [execp](const BDense *const left, const BDense *const right,
                            Mtx *const mat) {
            gko::kernels::cuda::batch_csr::batch_scale<value_type>(execp, left,
                                                                   right, mat);
        };
        scale_vecs = [execp](const BDense *const scale, BDense *const mat) {
            gko::kernels::cuda::batch_dense::batch_scale<value_type>(
                execp, scale, mat);
        };
    }

    void TearDown()
    {
        if (cuexec != nullptr) {
            ASSERT_NO_THROW(cuexec->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::CudaExecutor> cuexec;

    const real_type eps = r<value_type>::value;

    const size_t nbatch = 2;
    const int nrows = 3;
    const int nrhs = 2;
    const Options opts;
    gko::test::LinSys<T> sys_1;
    gko::test::LinSys<T> sys_m;

    std::function<void(Options, const Mtx *, const BDense *, BDense *,
                       LogData &)>
        solve_fn;
    std::function<void(const BDense *, const BDense *, Mtx *)> scale_mat;
    std::function<void(const BDense *, BDense *)> scale_vecs;

    std::unique_ptr<BDense> ref_scaling_vec;

    void setup_ref_scaling_test()
    {
        ref_scaling_vec = BDense::create(
            exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, 1)));
        ref_scaling_vec->at(0, 0, 0) = 2.0;
        ref_scaling_vec->at(0, 1, 0) = 3.0;
        ref_scaling_vec->at(0, 2, 0) = -1.0;
        ref_scaling_vec->at(1, 0, 0) = 1.0;
        ref_scaling_vec->at(1, 1, 0) = -2.0;
        ref_scaling_vec->at(1, 2, 0) = -4.0;
    }
};

TYPED_TEST_SUITE(BatchDirect, gko::test::ValueTypes);


TYPED_TEST(BatchDirect, TransposeScaleCopyWorks)
{
    using T = typename TestFixture::value_type;
    using BDense = typename TestFixture::BDense;
    this->setup_ref_scaling_test();
    auto ref_b_orig = gko::batch_initialize<BDense>(
        {{I<T>({1.0, -1.0, 1.5}), I<T>({-2.0, 2.0, 3.0})},
         {{1.0, -2.0, -0.5}, {1.0, -2.5, 4.0}}},
        this->exec);
    auto ref_b_scaled = gko::batch_initialize<BDense>(
        {{I<T>({2.0, -4.0}), I<T>({-3.0, 6.0}), I<T>({-1.5, -3.0})},
         {{1.0, 1.0}, {4.0, 5.0}, {2.0, -16.0}}},
        this->exec);
    auto scaling_vec = BDense::create(this->cuexec);
    scaling_vec->copy_from(this->ref_scaling_vec.get());
    auto b_orig = BDense::create(this->cuexec);
    b_orig->copy_from(ref_b_orig.get());
    auto b_scaled = BDense::create(
        this->cuexec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, this->nrhs)));

    gko::kernels::cuda::batch_direct::transpose_scale_copy(
        this->cuexec, scaling_vec.get(), b_orig.get(), b_scaled.get());

    auto scaled_res = BDense::create(this->exec);
    scaled_res->copy_from(b_scaled.get());
    GKO_ASSERT_BATCH_MTX_NEAR(scaled_res, ref_b_scaled, this->eps);
}


TYPED_TEST(BatchDirect, SystemLeftScaleTransposeIsEquivalentToReference)
{
    using T = typename TestFixture::value_type;
    using BDense = typename TestFixture::BDense;
    const int ncols = 5;
    this->setup_ref_scaling_test();
    auto ref_b_orig = gko::batch_initialize<BDense>(
        {{I<T>({1.0, -1.0}), I<T>({-2.0, 2.0}), I<T>({1.5, 4.0})},
         {{1.0, -2.0}, {1.0, -2.5}, {-3.0, 0.5}}},
        this->exec);
    auto ref_b_scaled = BDense::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrhs, this->nrows)));
    auto refmat = BDense::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, ncols)));
    for (size_t ib = 0; ib < this->nbatch; ib++) {
        for (int i = 0; i < this->nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                const T val = (ib + 1.0) * (i * ncols + j);
                refmat->at(ib, i, j) = val;
            }
        }
    }
    auto refscaledmat = BDense::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(ncols, this->nrows)));
    auto scaling_vec = BDense::create(this->cuexec);
    scaling_vec->copy_from(this->ref_scaling_vec.get());
    auto orig = BDense::create(this->cuexec);
    orig->copy_from(ref_b_orig.get());
    auto mat = BDense::create(this->cuexec);
    mat->copy_from(refmat.get());
    auto scaledmat = BDense::create(
        this->cuexec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(ncols, this->nrows)));
    auto scaled = BDense::create(
        this->cuexec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrhs, this->nrows)));

    gko::kernels::reference::batch_direct::left_scale_system_transpose(
        this->exec, refmat.get(), ref_b_orig.get(), this->ref_scaling_vec.get(),
        refscaledmat.get(), ref_b_scaled.get());
    gko::kernels::cuda::batch_direct::left_scale_system_transpose(
        this->cuexec, mat.get(), orig.get(), scaling_vec.get(), scaledmat.get(),
        scaled.get());

    auto scaled_b = BDense::create(this->exec);
    scaled_b->copy_from(scaled.get());
    auto ref_mat_ans = BDense::create(this->exec);
    ref_mat_ans->copy_from(scaledmat.get());
    GKO_ASSERT_BATCH_MTX_NEAR(scaled_b, ref_b_scaled, this->eps);
    GKO_ASSERT_BATCH_MTX_NEAR(ref_mat_ans, refscaledmat, this->eps);
}


TYPED_TEST(BatchDirect, SolvesStencilSystem)
{
    auto r_1 = gko::test::solve_poisson_uniform(
        this->cuexec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts, this->sys_1, 1);

    GKO_ASSERT_BATCH_MTX_NEAR(r_1.x, this->sys_1.xex, this->eps);
}


TYPED_TEST(BatchDirect, SolvesStencilMultipleSystem)
{
    auto r_m = gko::test::solve_poisson_uniform(
        this->cuexec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts, this->sys_m, this->nrhs);

    GKO_ASSERT_BATCH_MTX_NEAR(r_m.x, this->sys_m.xex, this->eps);
}


TYPED_TEST(BatchDirect, UnitScalingDoesNotChangeResult)
{
    using BDense = typename TestFixture::BDense;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto factory = Solver::build().on(this->cuexec);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->cuexec, factory.get(), this->sys_1, 1, left_scale.get(),
        right_scale.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, this->eps);
}


TYPED_TEST(BatchDirect, GeneralScalingDoesNotChangeResult)
{
    using BDense = typename TestFixture::BDense;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {0.8, 0.9, 0.95}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.5, 1.05}, this->exec);
    auto factory = Solver::build().on(this->cuexec);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->cuexec, factory.get(), this->sys_1, 1, left_scale.get(),
        right_scale.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, this->eps);
}


TEST(BatchDirect, CanSolveWithoutScaling)
{
    using T = std::complex<float>;
    using RT = typename gko::remove_complex<T>;
    using Solver = gko::solver::BatchDirect<T>;
    const RT tol = 1e-5;
    std::shared_ptr<gko::ReferenceExecutor> refexec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<const gko::CudaExecutor> exec =
        gko::CudaExecutor::create(0, refexec);
    const int maxits = 5000;
    auto batchdirect_factory = Solver::build().on(exec);
    const int nrows = 29;
    const size_t nbatch = 3;
    const int nrhs = 5;

    gko::test::test_solve<Solver>(exec, nbatch, nrows, nrhs, tol, maxits,
                                  batchdirect_factory.get(), 1.0, false, false);
}

}  // namespace
