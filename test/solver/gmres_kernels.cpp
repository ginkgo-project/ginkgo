// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gmres_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/solver/common_gmres_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class Gmres : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Gmres<value_type>;
    using norm_type = gko::remove_complex<value_type>;
    using NormVector = gko::matrix::Dense<norm_type>;
    template <typename T>
    using Dense = typename gko::matrix::Dense<T>;

    Gmres() : rand_engine(30)
    {
        mtx = gen_mtx(123, 123);
        d_mtx = gko::clone(exec, mtx);
        exec_gmres_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(246u),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(value_type{1e-15}))
                .on(exec);

        ref_gmres_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(246u),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(value_type{1e-15}))
                .on(ref);
    }

    template <typename ValueType = value_type, typename IndexType = index_type>
    std::unique_ptr<Dense<ValueType>> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Dense<ValueType>>(
            num_rows, num_cols,
            std::uniform_int_distribution<IndexType>(num_cols, num_cols),
            std::normal_distribution<ValueType>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(int nrhs = 43)
    {
#ifdef GINKGO_FAST_TESTS
        int m = 123;
#else
        int m = 597;
#endif
        x = gen_mtx(m, nrhs);
        y = gen_mtx(gko::solver::gmres_default_krylov_dim, nrhs);
        before_preconditioner = Mtx::create_with_config_of(x);
        b = gen_mtx(m, nrhs);
        krylov_bases =
            gen_mtx(m * (gko::solver::gmres_default_krylov_dim + 1), nrhs);
        hessenberg = gen_mtx(gko::solver::gmres_default_krylov_dim + 1,
                             gko::solver::gmres_default_krylov_dim * nrhs);
        hessenberg_iter =
            gen_mtx(gko::solver::gmres_default_krylov_dim + 1, nrhs);
        residual = gen_mtx(m, nrhs);
        residual_norm = gen_mtx<norm_type>(1, nrhs);
        residual_norm_collection =
            gen_mtx(gko::solver::gmres_default_krylov_dim + 1, nrhs);
        givens_sin = gen_mtx(gko::solver::gmres_default_krylov_dim, nrhs);
        givens_cos = gen_mtx(gko::solver::gmres_default_krylov_dim, nrhs);
        stop_status = gko::array<gko::stopping_status>(ref, nrhs);
        for (size_t i = 0; i < stop_status.get_size(); ++i) {
            stop_status.get_data()[i].reset();
        }
        final_iter_nums = gko::array<gko::size_type>(ref, nrhs);
        for (size_t i = 0; i < final_iter_nums.get_size(); ++i) {
            final_iter_nums.get_data()[i] = 5;
        }

        d_x = gko::clone(exec, x);
        d_before_preconditioner = Mtx::create_with_config_of(d_x);
        d_y = gko::clone(exec, y);
        d_b = gko::clone(exec, b);
        d_krylov_bases = gko::clone(exec, krylov_bases);
        d_hessenberg = gko::clone(exec, hessenberg);
        d_hessenberg_iter = gko::clone(exec, hessenberg_iter);
        d_residual = gko::clone(exec, residual);
        d_residual_norm = gko::clone(exec, residual_norm);
        d_residual_norm_collection = gko::clone(exec, residual_norm_collection);
        d_givens_sin = gko::clone(exec, givens_sin);
        d_givens_cos = gko::clone(exec, givens_cos);
        d_stop_status = gko::array<gko::stopping_status>(exec, stop_status);
        d_final_iter_nums = gko::array<gko::size_type>(exec, final_iter_nums);
    }

    std::default_random_engine rand_engine;

    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> d_mtx;
    std::unique_ptr<Solver::Factory> exec_gmres_factory;
    std::unique_ptr<Solver::Factory> ref_gmres_factory;

    std::unique_ptr<Mtx> before_preconditioner;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> krylov_bases;
    std::unique_ptr<Mtx> hessenberg;
    std::unique_ptr<Mtx> hessenberg_iter;
    std::unique_ptr<Mtx> residual;
    std::unique_ptr<NormVector> residual_norm;
    std::unique_ptr<Mtx> residual_norm_collection;
    std::unique_ptr<Mtx> givens_sin;
    std::unique_ptr<Mtx> givens_cos;
    gko::array<gko::stopping_status> stop_status;
    gko::array<gko::size_type> final_iter_nums;

    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_before_preconditioner;
    std::unique_ptr<Mtx> d_y;
    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_krylov_bases;
    std::unique_ptr<Mtx> d_hessenberg;
    std::unique_ptr<Mtx> d_hessenberg_iter;
    std::unique_ptr<Mtx> d_residual;
    std::unique_ptr<NormVector> d_residual_norm;
    std::unique_ptr<Mtx> d_residual_norm_collection;
    std::unique_ptr<Mtx> d_givens_sin;
    std::unique_ptr<Mtx> d_givens_cos;
    gko::array<gko::stopping_status> d_stop_status;
    gko::array<gko::size_type> d_final_iter_nums;
};


TEST_F(Gmres, GmresKernelInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::common_gmres::initialize(
        ref, b.get(), residual.get(), givens_sin.get(), givens_cos.get(),
        stop_status.get_data());
    gko::kernels::EXEC_NAMESPACE::common_gmres::initialize(
        exec, d_b.get(), d_residual.get(), d_givens_sin.get(),
        d_givens_cos.get(), d_stop_status.get_data());

    GKO_ASSERT_MTX_NEAR(d_residual, residual, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(d_stop_status, stop_status);
}


TEST_F(Gmres, GmresKernelRestartIsEquivalentToRef)
{
    initialize_data();
    residual->compute_norm2(residual_norm);
    d_residual_norm->copy_from(residual_norm);

    gko::kernels::reference::gmres::restart(
        ref, residual.get(), residual_norm.get(),
        residual_norm_collection.get(), krylov_bases.get(),
        final_iter_nums.get_data());
    gko::kernels::EXEC_NAMESPACE::gmres::restart(
        exec, d_residual.get(), d_residual_norm.get(),
        d_residual_norm_collection.get(), d_krylov_bases.get(),
        d_final_iter_nums.get_data());

    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_krylov_bases, krylov_bases, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(d_final_iter_nums, final_iter_nums);
}


TEST_F(Gmres, GmresKernelHessenbergQRIsEquivalentToRef)
{
    initialize_data();
    int iter = 5;

    gko::kernels::reference::common_gmres::hessenberg_qr(
        ref, givens_sin.get(), givens_cos.get(), residual_norm.get(),
        residual_norm_collection.get(), hessenberg_iter.get(), iter,
        final_iter_nums.get_data(), stop_status.get_const_data());
    gko::kernels::EXEC_NAMESPACE::common_gmres::hessenberg_qr(
        exec, d_givens_sin.get(), d_givens_cos.get(), d_residual_norm.get(),
        d_residual_norm_collection.get(), d_hessenberg_iter.get(), iter,
        d_final_iter_nums.get_data(), d_stop_status.get_const_data());

    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_hessenberg_iter, hessenberg_iter,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_krylov_bases, krylov_bases, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(d_final_iter_nums, final_iter_nums);
}


TEST_F(Gmres, GmresKernelHessenbergQROnSingleRHSIsEquivalentToRef)
{
    initialize_data(1);
    int iter = 5;

    gko::kernels::reference::common_gmres::hessenberg_qr(
        ref, givens_sin.get(), givens_cos.get(), residual_norm.get(),
        residual_norm_collection.get(), hessenberg_iter.get(), iter,
        final_iter_nums.get_data(), stop_status.get_const_data());
    gko::kernels::EXEC_NAMESPACE::common_gmres::hessenberg_qr(
        exec, d_givens_sin.get(), d_givens_cos.get(), d_residual_norm.get(),
        d_residual_norm_collection.get(), d_hessenberg_iter.get(), iter,
        d_final_iter_nums.get_data(), d_stop_status.get_const_data());

    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_hessenberg_iter, hessenberg_iter,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_krylov_bases, krylov_bases, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(d_final_iter_nums, final_iter_nums);
}


TEST_F(Gmres, GmresKernelSolveKrylovIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::common_gmres::solve_krylov(
        ref, residual_norm_collection.get(), hessenberg.get(), y.get(),
        final_iter_nums.get_const_data(), stop_status.get_const_data());
    gko::kernels::EXEC_NAMESPACE::common_gmres::solve_krylov(
        exec, d_residual_norm_collection.get(), d_hessenberg.get(), d_y.get(),
        d_final_iter_nums.get_const_data(), d_stop_status.get_const_data());

    GKO_ASSERT_MTX_NEAR(d_y, y, r<value_type>::value);
}


TEST_F(Gmres, GmresKernelMultiAxpyIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres::multi_axpy(
        ref, krylov_bases.get(), y.get(), before_preconditioner.get(),
        final_iter_nums.get_const_data(), stop_status.get_data());
    gko::kernels::EXEC_NAMESPACE::gmres::multi_axpy(
        exec, d_krylov_bases.get(), d_y.get(), d_before_preconditioner.get(),
        d_final_iter_nums.get_const_data(), d_stop_status.get_data());

    GKO_ASSERT_MTX_NEAR(d_before_preconditioner, before_preconditioner,
                        r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(stop_status, d_stop_status);
}


TEST_F(Gmres, GmresApplyOneRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 1;
    auto ref_solver = ref_gmres_factory->generate(mtx);
    auto exec_solver = exec_gmres_factory->generate(d_mtx);
    auto b = gen_mtx(m, n);
    auto x = gen_mtx(m, n);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);

    ref_solver->apply(b, x);
    exec_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_b, b, 0);
    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value * 1e2);
}


TEST_F(Gmres, GmresApplyMultipleRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 5;
    auto ref_solver = ref_gmres_factory->generate(mtx);
    auto exec_solver = exec_gmres_factory->generate(d_mtx);
    auto b = gen_mtx(m, n);
    auto x = gen_mtx(m, n);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);

    ref_solver->apply(b, x);
    exec_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_b, b, 0);
    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value * 1e3);
}
