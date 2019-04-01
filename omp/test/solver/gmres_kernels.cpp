/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/solver/gmres.hpp>


#include <gtest/gtest.h>


#include <random>


#include <core/solver/gmres_kernels.hpp>
#include <core/test/utils.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>


namespace {


class Gmres : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Gmres() : rand_engine(30) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown()
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data()
    {
        int m = 597;
        int n = 43;
        preconditioner = gen_mtx(m, m);
        x = gen_mtx(m, n);
        y = gen_mtx(gko::solver::default_krylov_dim, n);
        before_preconditioner = Mtx::create_with_config_of(x.get());
        b = gen_mtx(m, n);
        b_norm = gen_mtx(1, n);
        krylov_bases = gen_mtx(m, (gko::solver::default_krylov_dim + 1) * n);
        next_krylov_basis = gen_mtx(m, n);
        hessenberg = gen_mtx(gko::solver::default_krylov_dim + 1,
                             gko::solver::default_krylov_dim * n);
        hessenberg_iter = gen_mtx(gko::solver::default_krylov_dim + 1, n);
        residual = gen_mtx(m, n);
        residual_norm = gen_mtx(1, n);
        residual_norm_collection =
            gen_mtx(gko::solver::default_krylov_dim + 1, n);
        givens_sin = gen_mtx(gko::solver::default_krylov_dim, n);
        givens_cos = gen_mtx(gko::solver::default_krylov_dim, n);
        stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(ref, n));
        for (size_t i = 0; i < stop_status->get_num_elems(); ++i) {
            stop_status->get_data()[i].reset();
        }
        final_iter_nums = std::unique_ptr<gko::Array<gko::size_type>>(
            new gko::Array<gko::size_type>(ref, n));
        for (size_t i = 0; i < final_iter_nums->get_num_elems(); ++i) {
            final_iter_nums->get_data()[i] = 5;
        }

        d_preconditioner = Mtx::create(omp);
        d_preconditioner->copy_from(preconditioner.get());
        d_x = Mtx::create(omp);
        d_x->copy_from(x.get());
        d_before_preconditioner = Mtx::create_with_config_of(d_x.get());
        d_y = Mtx::create(omp);
        d_y->copy_from(y.get());
        d_b = Mtx::create(omp);
        d_b->copy_from(b.get());
        d_b_norm = Mtx::create(omp);
        d_b_norm->copy_from(b_norm.get());
        d_krylov_bases = Mtx::create(omp);
        d_krylov_bases->copy_from(krylov_bases.get());
        d_next_krylov_basis = Mtx::create(omp);
        d_next_krylov_basis->copy_from(next_krylov_basis.get());
        d_hessenberg = Mtx::create(omp);
        d_hessenberg->copy_from(hessenberg.get());
        d_hessenberg_iter = Mtx::create(omp);
        d_hessenberg_iter->copy_from(hessenberg_iter.get());
        d_residual = Mtx::create(omp);
        d_residual->copy_from(residual.get());
        d_residual_norm = Mtx::create(omp);
        d_residual_norm->copy_from(residual_norm.get());
        d_residual_norm_collection = Mtx::create(omp);
        d_residual_norm_collection->copy_from(residual_norm_collection.get());
        d_givens_sin = Mtx::create(omp);
        d_givens_sin->copy_from(givens_sin.get());
        d_givens_cos = Mtx::create(omp);
        d_givens_cos->copy_from(givens_cos.get());
        d_stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(omp, n));
        *d_stop_status = *stop_status;
        d_final_iter_nums = std::unique_ptr<gko::Array<gko::size_type>>(
            new gko::Array<gko::size_type>(omp, n));
        *d_final_iter_nums = *final_iter_nums;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> preconditioner;
    std::unique_ptr<Mtx> before_preconditioner;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> b_norm;
    std::unique_ptr<Mtx> krylov_bases;
    std::unique_ptr<Mtx> next_krylov_basis;
    std::unique_ptr<Mtx> hessenberg;
    std::unique_ptr<Mtx> hessenberg_iter;
    std::unique_ptr<Mtx> residual;
    std::unique_ptr<Mtx> residual_norm;
    std::unique_ptr<Mtx> residual_norm_collection;
    std::unique_ptr<Mtx> givens_sin;
    std::unique_ptr<Mtx> givens_cos;
    std::unique_ptr<gko::Array<gko::stopping_status>> stop_status;
    std::unique_ptr<gko::Array<gko::size_type>> final_iter_nums;

    std::unique_ptr<Mtx> d_preconditioner;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_before_preconditioner;
    std::unique_ptr<Mtx> d_y;
    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_b_norm;
    std::unique_ptr<Mtx> d_krylov_bases;
    std::unique_ptr<Mtx> d_next_krylov_basis;
    std::unique_ptr<Mtx> d_hessenberg;
    std::unique_ptr<Mtx> d_hessenberg_iter;
    std::unique_ptr<Mtx> d_residual;
    std::unique_ptr<Mtx> d_residual_norm;
    std::unique_ptr<Mtx> d_residual_norm_collection;
    std::unique_ptr<Mtx> d_givens_sin;
    std::unique_ptr<Mtx> d_givens_cos;
    std::unique_ptr<gko::Array<gko::stopping_status>> d_stop_status;
    std::unique_ptr<gko::Array<gko::size_type>> d_final_iter_nums;
};


TEST_F(Gmres, OmpGmresInitialize1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres::initialize_1(
        ref, b.get(), b_norm.get(), residual.get(), givens_sin.get(),
        givens_cos.get(), stop_status.get(), gko::solver::default_krylov_dim);
    gko::kernels::omp::gmres::initialize_1(
        omp, d_b.get(), d_b_norm.get(), d_residual.get(), d_givens_sin.get(),
        d_givens_cos.get(), d_stop_status.get(),
        gko::solver::default_krylov_dim);

    GKO_ASSERT_MTX_NEAR(d_b_norm, b_norm, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_residual, residual, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, 1e-14);
    GKO_ASSERT_ARRAY_EQ(d_stop_status, stop_status);
}


TEST_F(Gmres, OmpGmresInitialize2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres::initialize_2(
        ref, residual.get(), residual_norm.get(),
        residual_norm_collection.get(), krylov_bases.get(),
        final_iter_nums.get(), gko::solver::default_krylov_dim);
    gko::kernels::omp::gmres::initialize_2(
        omp, d_residual.get(), d_residual_norm.get(),
        d_residual_norm_collection.get(), d_krylov_bases.get(),
        d_final_iter_nums.get(), gko::solver::default_krylov_dim);

    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        1e-14);
    GKO_ASSERT_MTX_NEAR(d_krylov_bases, krylov_bases, 1e-14);
    GKO_ASSERT_ARRAY_EQ(d_final_iter_nums, final_iter_nums);
}


TEST_F(Gmres, OmpGmresStep1IsEquivalentToRef)
{
    initialize_data();
    int iter = 5;

    gko::kernels::reference::gmres::step_1(
        ref, next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
        residual_norm.get(), residual_norm_collection.get(), krylov_bases.get(),
        hessenberg_iter.get(), b_norm.get(), iter, final_iter_nums.get(),
        stop_status.get());
    gko::kernels::omp::gmres::step_1(
        omp, d_next_krylov_basis.get(), d_givens_sin.get(), d_givens_cos.get(),
        d_residual_norm.get(), d_residual_norm_collection.get(),
        d_krylov_bases.get(), d_hessenberg_iter.get(), d_b_norm.get(), iter,
        d_final_iter_nums.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_next_krylov_basis, next_krylov_basis, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        1e-14);
    GKO_ASSERT_MTX_NEAR(d_hessenberg_iter, hessenberg_iter, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_krylov_bases, krylov_bases, 1e-14);
    GKO_ASSERT_ARRAY_EQ(d_final_iter_nums, final_iter_nums);
}


TEST_F(Gmres, OmpGmresStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres::step_2(ref, residual_norm_collection.get(),
                                           krylov_bases.get(), hessenberg.get(),
                                           y.get(), before_preconditioner.get(),
                                           final_iter_nums.get());
    gko::kernels::omp::gmres::step_2(omp, d_residual_norm_collection.get(),
                                     d_krylov_bases.get(), d_hessenberg.get(),
                                     d_y.get(), d_before_preconditioner.get(),
                                     d_final_iter_nums.get());

    GKO_ASSERT_MTX_NEAR(d_y, y, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace
