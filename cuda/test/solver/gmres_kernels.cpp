/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/solver/gmres.hpp>


#include <gtest/gtest.h>


#include <random>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/solver/gmres_kernels.hpp>
#include <core/stop/combined.hpp>
#include <core/stop/iteration.hpp>
#include <core/stop/residual_norm_reduction.hpp>
#include <core/test/utils.hpp>

namespace {


class Gmres : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Gmres() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
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
        int m = 10;
        int n = 5;
        int iter = 5;
        b = gen_mtx(m, n);
        b_norm = gen_mtx(1, n);
        krylov_bases = gen_mtx(m, (gko::solver::default_krylov_dim + 1) * n);
        next_krylov_basis = gen_mtx(m, n);
        hessenberg_iter = gen_mtx(iter + 2, n);
        residual = gen_mtx(m, n);
        residual_norm = gen_mtx(1, n);
        residual_norms = gen_mtx(gko::solver::default_krylov_dim + 1, n);
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
            final_iter_nums->get_data()[i] = 0;
        }

        d_b = Mtx::create(cuda);
        d_b->copy_from(b.get());
        d_b_norm = Mtx::create(cuda);
        d_b_norm->copy_from(b_norm.get());
        d_krylov_bases = Mtx::create(cuda);
        d_krylov_bases->copy_from(krylov_bases.get());
        d_next_krylov_basis = Mtx::create(cuda);
        d_next_krylov_basis->copy_from(next_krylov_basis.get());
        d_hessenberg_iter = Mtx::create(cuda);
        d_hessenberg_iter->copy_from(hessenberg_iter.get());
        d_residual = Mtx::create(cuda);
        d_residual->copy_from(residual.get());
        d_residual_norm = Mtx::create(cuda);
        d_residual_norm->copy_from(residual_norm.get());
        d_residual_norms = Mtx::create(cuda);
        d_residual_norms->copy_from(residual_norms.get());
        d_givens_sin = Mtx::create(cuda);
        d_givens_sin->copy_from(givens_sin.get());
        d_givens_cos = Mtx::create(cuda);
        d_givens_cos->copy_from(givens_cos.get());
        d_stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(cuda, n));
        *d_stop_status = *stop_status;
        d_final_iter_nums = std::unique_ptr<gko::Array<gko::size_type>>(
            new gko::Array<gko::size_type>(cuda, n));
        *d_final_iter_nums = *final_iter_nums;
    }

    void make_symetric(Mtx *mtx)
    {
        for (int i = 0; i < mtx->get_size()[0]; ++i) {
            for (int j = i + 1; j < mtx->get_size()[1]; ++j) {
                mtx->at(i, j) = mtx->at(j, i);
            }
        }
    }

    void make_diag_dominant(Mtx *mtx)
    {
        using std::abs;
        for (int i = 0; i < mtx->get_size()[0]; ++i) {
            auto sum = gko::zero<Mtx::value_type>();
            for (int j = 0; j < mtx->get_size()[1]; ++j) {
                sum += abs(mtx->at(i, j));
            }
            mtx->at(i, i) = sum;
        }
    }

    void make_spd(Mtx *mtx)
    {
        make_symetric(mtx);
        make_diag_dominant(mtx);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> b_norm;
    std::unique_ptr<Mtx> krylov_bases;
    std::unique_ptr<Mtx> next_krylov_basis;
    std::unique_ptr<Mtx> hessenberg_iter;
    std::unique_ptr<Mtx> residual;
    std::unique_ptr<Mtx> residual_norm;
    std::unique_ptr<Mtx> residual_norms;
    std::unique_ptr<Mtx> givens_sin;
    std::unique_ptr<Mtx> givens_cos;
    std::unique_ptr<gko::Array<gko::stopping_status>> stop_status;
    std::unique_ptr<gko::Array<gko::size_type>> final_iter_nums;

    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_b_norm;
    std::unique_ptr<Mtx> d_krylov_bases;
    std::unique_ptr<Mtx> d_next_krylov_basis;
    std::unique_ptr<Mtx> d_hessenberg_iter;
    std::unique_ptr<Mtx> d_residual;
    std::unique_ptr<Mtx> d_residual_norm;
    std::unique_ptr<Mtx> d_residual_norms;
    std::unique_ptr<Mtx> d_givens_sin;
    std::unique_ptr<Mtx> d_givens_cos;
    std::unique_ptr<gko::Array<gko::stopping_status>> d_stop_status;
    std::unique_ptr<gko::Array<gko::size_type>> d_final_iter_nums;
};


TEST_F(Gmres, CudaGmresInitialize1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres::initialize_1(
        ref, b.get(), b_norm.get(), residual.get(), givens_sin.get(),
        givens_cos.get(), stop_status.get(), gko::solver::default_krylov_dim);
    gko::kernels::cuda::gmres::initialize_1(
        cuda, d_b.get(), d_b_norm.get(), d_residual.get(), d_givens_sin.get(),
        d_givens_cos.get(), d_stop_status.get(),
        gko::solver::default_krylov_dim);

    ASSERT_MTX_NEAR(d_b_norm, b_norm, 1e-14);
    ASSERT_MTX_NEAR(d_residual, residual, 1e-14);
    ASSERT_MTX_NEAR(d_givens_sin, givens_sin, 1e-14);
    ASSERT_MTX_NEAR(d_givens_cos, givens_cos, 1e-14);
}


TEST_F(Gmres, CudaGmresInitialize2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres::initialize_2(
        ref, residual.get(), residual_norm.get(), residual_norms.get(),
        krylov_bases.get(), final_iter_nums.get(),
        gko::solver::default_krylov_dim);
    gko::kernels::cuda::gmres::initialize_2(
        cuda, d_residual.get(), d_residual_norm.get(), d_residual_norms.get(),
        d_krylov_bases.get(), d_final_iter_nums.get(),
        gko::solver::default_krylov_dim);

    ASSERT_MTX_NEAR(d_residual_norm, residual_norm, 1e-14);
    ASSERT_MTX_NEAR(d_residual_norms, residual_norms, 1e-14);
    ASSERT_MTX_NEAR(d_krylov_bases, krylov_bases, 1e-14);
}


TEST_F(Gmres, CudaGmresStep1IsEquivalentToRef)
{
    initialize_data();
    int iter = 5;

    gko::kernels::reference::gmres::step_1(
        ref, next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
        residual_norm.get(), residual_norms.get(), krylov_bases.get(),
        hessenberg_iter.get(), b_norm.get(), iter, stop_status.get());
    gko::kernels::cuda::gmres::step_1(
        cuda, d_next_krylov_basis.get(), d_givens_sin.get(), d_givens_cos.get(),
        d_residual_norm.get(), d_residual_norms.get(), d_krylov_bases.get(),
        d_hessenberg_iter.get(), d_b_norm.get(), iter, d_stop_status.get());

    ASSERT_MTX_NEAR(d_next_krylov_basis, next_krylov_basis, 1e-14);
    ASSERT_MTX_NEAR(d_givens_sin, givens_sin, 1e-14);
    ASSERT_MTX_NEAR(d_givens_cos, givens_cos, 1e-14);
    ASSERT_MTX_NEAR(d_hessenberg_iter, hessenberg_iter, 1e-14);
    ASSERT_MTX_NEAR(d_residual_norm, residual_norm, 1e-14);
    ASSERT_MTX_NEAR(d_residual_norms, residual_norms, 1e-14);
}


}  // namespace
