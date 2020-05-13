/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/solver/gmres_mixed_kernels.hpp"


#include <algorithm>
#include <cmath>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/gmres_mixed.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>


#include "core/components/accessor.hpp"
#include "core/test/utils.hpp"


namespace {


class GmresMixed : public ::testing::Test {
protected:
    using value_type = double;
    using krylov_type = float;
    using index_type = int;
    using Accessor2d = gko::Accessor2d<krylov_type, value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Mtx = Dense;


    GmresMixed() : rand_engine(30) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
        krylov_bases.set_executor(ref);
        d_krylov_bases.set_executor(omp);
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
            std::uniform_int_distribution<index_type>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
    }

    gko::Array<krylov_type> generate_krylov_bases(int num_rows, int num_cols)
    {
        gko::Array<krylov_type> result{
            ref, static_cast<gko::size_type>(num_rows * num_cols)};
        auto temp_krylov_bases = gko::test::generate_random_matrix<Dense>(
            num_rows, num_cols,
            std::uniform_int_distribution<index_type>(num_cols, num_cols),
            std::normal_distribution<krylov_type>(-1.0, 1.0), rand_engine, ref);
        std::copy_n(temp_krylov_bases->get_const_values(),
                    result.get_num_elems(), result.get_data());
        return result;
    }

    void initialize_data()
    {
        int m = 597;
        int n = 43;
        x = gen_mtx(m, n);
        y = gen_mtx(gko::solver::default_krylov_dim_mixed, n);
        before_preconditioner = Mtx::create_with_config_of(x.get());
        b = gen_mtx(m, n);
        b_norm = gen_mtx(1, n);
        arnoldi_norm = gen_mtx(2, n);
        gko::dim<2> krylov_bases_dim(
            m, (gko::solver::default_krylov_dim_mixed + 1) * n);
        krylov_bases =
            generate_krylov_bases(krylov_bases_dim[0], krylov_bases_dim[1]);
        krylov_bases_accessor =
            Accessor2d{krylov_bases.get_data(), krylov_bases_dim[1]};

        next_krylov_basis = gen_mtx(m, n);
        hessenberg = gen_mtx(gko::solver::default_krylov_dim_mixed + 1,
                             gko::solver::default_krylov_dim_mixed * n);
        hessenberg_iter = gen_mtx(gko::solver::default_krylov_dim_mixed + 1, n);
        buffer_iter = gen_mtx(gko::solver::default_krylov_dim_mixed + 1, n);
        residual = gen_mtx(m, n);
        residual_norm = gen_mtx(1, n);
        residual_norm_collection =
            gen_mtx(gko::solver::default_krylov_dim_mixed + 1, n);
        givens_sin = gen_mtx(gko::solver::default_krylov_dim_mixed, n);
        givens_cos = gen_mtx(gko::solver::default_krylov_dim_mixed, n);
        stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(ref, n));
        for (size_t i = 0; i < stop_status->get_num_elems(); ++i) {
            stop_status->get_data()[i].reset();
        }
        reorth_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(ref, n));
        for (size_t i = 0; i < reorth_status->get_num_elems(); ++i) {
            reorth_status->get_data()[i].reset();
        }
        final_iter_nums = std::unique_ptr<gko::Array<gko::size_type>>(
            new gko::Array<gko::size_type>(ref, n));
        for (size_t i = 0; i < final_iter_nums->get_num_elems(); ++i) {
            final_iter_nums->get_data()[i] = 5;
        }
        num_reorth = std::unique_ptr<gko::Array<gko::size_type>>(
            new gko::Array<gko::size_type>(ref, n));
        for (size_t i = 0; i < num_reorth->get_num_elems(); ++i) {
            num_reorth->get_data()[i] = 0;
        }

        d_x = Mtx::create(omp);
        d_x->copy_from(x.get());
        d_before_preconditioner = Mtx::create_with_config_of(d_x.get());
        d_y = Mtx::create(omp);
        d_y->copy_from(y.get());
        d_b = Mtx::create(omp);
        d_b->copy_from(b.get());
        d_b_norm = Mtx::create(omp);
        d_b_norm->copy_from(b_norm.get());
        d_arnoldi_norm = Mtx::create(omp);
        d_arnoldi_norm->copy_from(arnoldi_norm.get());
        d_krylov_bases = krylov_bases;
        d_krylov_bases_accessor =
            Accessor2d{d_krylov_bases.get_data(), krylov_bases_dim[1]};
        d_next_krylov_basis = Mtx::create(omp);
        d_next_krylov_basis->copy_from(next_krylov_basis.get());
        d_hessenberg = Mtx::create(omp);
        d_hessenberg->copy_from(hessenberg.get());
        d_hessenberg_iter = Mtx::create(omp);
        d_hessenberg_iter->copy_from(hessenberg_iter.get());
        d_buffer_iter = Mtx::create(omp);
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
        d_reorth_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(omp, n));
        *d_reorth_status = *reorth_status;
        d_final_iter_nums = std::unique_ptr<gko::Array<gko::size_type>>(
            new gko::Array<gko::size_type>(omp, n));
        *d_final_iter_nums = *final_iter_nums;
        d_num_reorth = std::unique_ptr<gko::Array<gko::size_type>>(
            new gko::Array<gko::size_type>(omp, n));
        *d_num_reorth = *num_reorth;
    }

    void assert_krylov_bases_near()
    {
        gko::Array<krylov_type> d_to_host{ref};
        d_to_host = d_krylov_bases;
        const auto tolerance = r<krylov_type>::value;
        using std::abs;
        for (gko::size_type i = 0; i < krylov_bases.get_num_elems(); ++i) {
            const auto ref_value = krylov_bases.get_const_data()[i];
            const auto dev_value = d_to_host.get_const_data()[i];
            ASSERT_LE(abs(dev_value - ref_value), tolerance);
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> before_preconditioner;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> b_norm;
    std::unique_ptr<Mtx> arnoldi_norm;
    gko::Array<krylov_type> krylov_bases;
    Accessor2d krylov_bases_accessor;
    std::unique_ptr<Mtx> next_krylov_basis;
    std::unique_ptr<Mtx> hessenberg;
    std::unique_ptr<Mtx> hessenberg_iter;
    std::unique_ptr<Mtx> buffer_iter;
    std::unique_ptr<Mtx> residual;
    std::unique_ptr<Mtx> residual_norm;
    std::unique_ptr<Mtx> residual_norm_collection;
    std::unique_ptr<Mtx> givens_sin;
    std::unique_ptr<Mtx> givens_cos;
    std::unique_ptr<gko::Array<gko::stopping_status>> stop_status;
    std::unique_ptr<gko::Array<gko::stopping_status>> reorth_status;
    std::unique_ptr<gko::Array<gko::size_type>> final_iter_nums;
    std::unique_ptr<gko::Array<gko::size_type>> num_reorth;

    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_before_preconditioner;
    std::unique_ptr<Mtx> d_y;
    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_b_norm;
    std::unique_ptr<Mtx> d_arnoldi_norm;
    gko::Array<krylov_type> d_krylov_bases;
    Accessor2d d_krylov_bases_accessor;
    std::unique_ptr<Mtx> d_next_krylov_basis;
    std::unique_ptr<Mtx> d_hessenberg;
    std::unique_ptr<Mtx> d_hessenberg_iter;
    std::unique_ptr<Mtx> d_buffer_iter;
    std::unique_ptr<Mtx> d_residual;
    std::unique_ptr<Mtx> d_residual_norm;
    std::unique_ptr<Mtx> d_residual_norm_collection;
    std::unique_ptr<Mtx> d_givens_sin;
    std::unique_ptr<Mtx> d_givens_cos;
    std::unique_ptr<gko::Array<gko::stopping_status>> d_stop_status;
    std::unique_ptr<gko::Array<gko::stopping_status>> d_reorth_status;
    std::unique_ptr<gko::Array<gko::size_type>> d_final_iter_nums;
    std::unique_ptr<gko::Array<gko::size_type>> d_num_reorth;
};


TEST_F(GmresMixed, OmpGmresMixedInitialize1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres_mixed::initialize_1(
        ref, b.get(), b_norm.get(), residual.get(), givens_sin.get(),
        givens_cos.get(), stop_status.get(),
        gko::solver::default_krylov_dim_mixed);
    gko::kernels::omp::gmres_mixed::initialize_1(
        omp, d_b.get(), d_b_norm.get(), d_residual.get(), d_givens_sin.get(),
        d_givens_cos.get(), d_stop_status.get(),
        gko::solver::default_krylov_dim_mixed);

    GKO_ASSERT_MTX_NEAR(d_b_norm, b_norm, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_residual, residual, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, 1e-14);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}

TEST_F(GmresMixed, OmpGmresMixedInitialize2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres_mixed::initialize_2(
        ref, residual.get(), residual_norm.get(),
        residual_norm_collection.get(), krylov_bases_accessor,
        next_krylov_basis.get(), final_iter_nums.get(),
        gko::solver::default_krylov_dim_mixed);
    gko::kernels::omp::gmres_mixed::initialize_2(
        omp, d_residual.get(), d_residual_norm.get(),
        d_residual_norm_collection.get(), d_krylov_bases_accessor,
        d_next_krylov_basis.get(), d_final_iter_nums.get(),
        gko::solver::default_krylov_dim_mixed);

    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        1e-14);
    assert_krylov_bases_near();
    GKO_ASSERT_ARRAY_EQ(*d_final_iter_nums, *final_iter_nums);
}

TEST_F(GmresMixed, OmpGmresMixedStep1IsEquivalentToRef)
{
    initialize_data();
    int iter = 5;
    int num_reorth_steps = 0, num_reorth_vectors = 0;
    int d_num_reorth_steps = 0, d_num_reorth_vectors = 0;

    gko::kernels::reference::gmres_mixed::step_1(
        ref, next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
        residual_norm.get(), residual_norm_collection.get(),
        krylov_bases_accessor, hessenberg_iter.get(), buffer_iter.get(),
        b_norm.get(), arnoldi_norm.get(), iter, final_iter_nums.get(),
        stop_status.get(), reorth_status.get(), num_reorth.get(),
        &num_reorth_steps, &num_reorth_vectors);
    gko::kernels::omp::gmres_mixed::step_1(
        omp, d_next_krylov_basis.get(), d_givens_sin.get(), d_givens_cos.get(),
        d_residual_norm.get(), d_residual_norm_collection.get(),
        d_krylov_bases_accessor, d_hessenberg_iter.get(), d_buffer_iter.get(),
        d_b_norm.get(), d_arnoldi_norm.get(), iter, d_final_iter_nums.get(),
        d_stop_status.get(), d_reorth_status.get(), d_num_reorth.get(),
        &num_reorth_steps, &num_reorth_vectors);

    GKO_ASSERT_MTX_NEAR(d_next_krylov_basis, next_krylov_basis, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        1e-14);
    GKO_ASSERT_MTX_NEAR(d_hessenberg_iter, hessenberg_iter, 1e-14);
    assert_krylov_bases_near();
    GKO_ASSERT_ARRAY_EQ(*d_final_iter_nums, *final_iter_nums);
}

TEST_F(GmresMixed, OmpGmresMixedStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres_mixed::step_2(
        ref, residual_norm_collection.get(), krylov_bases_accessor.to_const(),
        hessenberg.get(), y.get(), before_preconditioner.get(),
        final_iter_nums.get());
    gko::kernels::omp::gmres_mixed::step_2(
        omp, d_residual_norm_collection.get(),
        d_krylov_bases_accessor.to_const(), d_hessenberg.get(), d_y.get(),
        d_before_preconditioner.get(), d_final_iter_nums.get());

    GKO_ASSERT_MTX_NEAR(d_y, y, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace
