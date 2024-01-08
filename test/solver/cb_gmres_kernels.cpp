// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/cb_gmres_kernels.hpp"


#include <algorithm>
#include <cmath>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/solver/cb_gmres_accessor.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class CbGmres : public CommonTestFixture {
protected:
    using storage_type = float;
    using size_type = gko::size_type;
    using Range3dHelper =
        gko::cb_gmres::Range3dHelper<value_type, storage_type>;
    using Range3d = typename Range3dHelper::Range;
    using Dense = gko::matrix::Dense<value_type>;
    using Mtx = Dense;
    static constexpr unsigned int default_krylov_dim_mixed{100};

    CbGmres() : rand_engine(30) {}

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<index_type>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
    }

    Range3dHelper generate_krylov_helper(gko::dim<3> size)
    {
        auto helper = Range3dHelper{ref, size};
        auto& bases = helper.get_bases();
        const auto num_rows = size[0] * size[1];
        const auto num_cols = size[2];
        auto temp_krylov_bases = gko::test::generate_random_matrix<Dense>(
            num_rows, num_cols,
            std::uniform_int_distribution<index_type>(num_cols, num_cols),
            std::normal_distribution<storage_type>(-1.0, 1.0), rand_engine,
            ref);
        std::copy_n(temp_krylov_bases->get_const_values(), bases.get_size(),
                    bases.get_data());
        // Only useful when the Accessor actually has a scale
        auto range = helper.get_range();
        auto dist = std::normal_distribution<value_type>(-1, 1);
        for (size_type k = 0; k < size[0]; ++k) {
            for (size_type i = 0; i < size[2]; ++i) {
                gko::cb_gmres::helper_functions_accessor<Range3d>::write_scalar(
                    range, k, i, dist(rand_engine));
            }
        }
        return helper;
    }

    void initialize_data()
    {
#ifdef GINKGO_FAST_TESTS
        int m = 123;
#else
        int m = 597;
#endif
        int n = 43;
        x = gen_mtx(m, n);
        y = gen_mtx(default_krylov_dim_mixed, n);
        before_preconditioner = Mtx::create_with_config_of(x);
        b = gen_mtx(m, n);
        arnoldi_norm = gen_mtx(3, n);
        gko::dim<3> krylov_bases_dim(default_krylov_dim_mixed + 1, m, n);
        range_helper = generate_krylov_helper(krylov_bases_dim);

        next_krylov_basis = gen_mtx(m, n);
        hessenberg =
            gen_mtx(default_krylov_dim_mixed + 1, default_krylov_dim_mixed * n);
        hessenberg_iter = gen_mtx(default_krylov_dim_mixed + 1, n);
        buffer_iter = gen_mtx(default_krylov_dim_mixed + 1, n);
        residual = gen_mtx(m, n);
        residual_norm = gen_mtx(1, n);
        residual_norm_collection = gen_mtx(default_krylov_dim_mixed + 1, n);
        givens_sin = gen_mtx(default_krylov_dim_mixed, n);
        givens_cos = gen_mtx(default_krylov_dim_mixed, n);
        stop_status =
            std::make_unique<gko::array<gko::stopping_status>>(ref, n);
        for (size_t i = 0; i < stop_status->get_size(); ++i) {
            stop_status->get_data()[i].reset();
        }
        reorth_status =
            std::make_unique<gko::array<gko::stopping_status>>(ref, n);
        for (size_t i = 0; i < reorth_status->get_size(); ++i) {
            reorth_status->get_data()[i].reset();
        }
        final_iter_nums = std::make_unique<gko::array<gko::size_type>>(ref, n);
        for (size_t i = 0; i < final_iter_nums->get_size(); ++i) {
            final_iter_nums->get_data()[i] = 5;
        }
        num_reorth = std::make_unique<gko::array<gko::size_type>>(ref, n);
        for (size_t i = 0; i < num_reorth->get_size(); ++i) {
            num_reorth->get_data()[i] = 5;
        }

        d_x = gko::clone(exec, x);
        d_before_preconditioner = Mtx::create_with_config_of(d_x);
        d_y = gko::clone(exec, y);
        d_b = gko::clone(exec, b);
        d_arnoldi_norm = gko::clone(exec, arnoldi_norm);
        d_range_helper = Range3dHelper{exec, {}};
        d_range_helper = range_helper;
        d_next_krylov_basis = gko::clone(exec, next_krylov_basis);
        d_hessenberg = gko::clone(exec, hessenberg);
        d_hessenberg_iter = gko::clone(exec, hessenberg_iter);
        d_buffer_iter = Mtx::create(exec);
        d_residual = gko::clone(exec, residual);
        d_residual_norm = gko::clone(exec, residual_norm);
        d_residual_norm_collection = gko::clone(exec, residual_norm_collection);
        d_givens_sin = gko::clone(exec, givens_sin);
        d_givens_cos = gko::clone(exec, givens_cos);
        d_stop_status = std::make_unique<gko::array<gko::stopping_status>>(
            exec, *stop_status);
        d_reorth_status = std::make_unique<gko::array<gko::stopping_status>>(
            exec, *reorth_status);
        d_final_iter_nums = std::make_unique<gko::array<gko::size_type>>(
            exec, *final_iter_nums);
        d_num_reorth =
            std::make_unique<gko::array<gko::size_type>>(exec, *num_reorth);
    }

    void assert_krylov_bases_near()
    {
        gko::array<storage_type> d_to_host{ref};
        auto& krylov_bases = range_helper.get_bases();
        d_to_host = d_range_helper.get_bases();
        const auto tolerance = r<storage_type>::value;
        using std::abs;
        for (gko::size_type i = 0; i < krylov_bases.get_size(); ++i) {
            const auto ref_value = krylov_bases.get_const_data()[i];
            const auto dev_value = d_to_host.get_const_data()[i];
            ASSERT_LE(abs(dev_value - ref_value), tolerance);
        }
    }

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> before_preconditioner;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> arnoldi_norm;
    Range3dHelper range_helper;
    std::unique_ptr<Mtx> next_krylov_basis;
    std::unique_ptr<Mtx> hessenberg;
    std::unique_ptr<Mtx> hessenberg_iter;
    std::unique_ptr<Mtx> buffer_iter;
    std::unique_ptr<Mtx> residual;
    std::unique_ptr<Mtx> residual_norm;
    std::unique_ptr<Mtx> residual_norm_collection;
    std::unique_ptr<Mtx> givens_sin;
    std::unique_ptr<Mtx> givens_cos;
    std::unique_ptr<gko::array<gko::stopping_status>> stop_status;
    std::unique_ptr<gko::array<gko::stopping_status>> reorth_status;
    std::unique_ptr<gko::array<gko::size_type>> final_iter_nums;
    std::unique_ptr<gko::array<gko::size_type>> num_reorth;

    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_before_preconditioner;
    std::unique_ptr<Mtx> d_y;
    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_arnoldi_norm;
    Range3dHelper d_range_helper;
    std::unique_ptr<Mtx> d_next_krylov_basis;
    std::unique_ptr<Mtx> d_hessenberg;
    std::unique_ptr<Mtx> d_hessenberg_iter;
    std::unique_ptr<Mtx> d_buffer_iter;
    std::unique_ptr<Mtx> d_residual;
    std::unique_ptr<Mtx> d_residual_norm;
    std::unique_ptr<Mtx> d_residual_norm_collection;
    std::unique_ptr<Mtx> d_givens_sin;
    std::unique_ptr<Mtx> d_givens_cos;
    std::unique_ptr<gko::array<gko::stopping_status>> d_stop_status;
    std::unique_ptr<gko::array<gko::stopping_status>> d_reorth_status;
    std::unique_ptr<gko::array<gko::size_type>> d_final_iter_nums;
    std::unique_ptr<gko::array<gko::size_type>> d_num_reorth;
};


TEST_F(CbGmres, CbGmresInitialize1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::cb_gmres::initialize(
        ref, b.get(), residual.get(), givens_sin.get(), givens_cos.get(),
        stop_status.get(), default_krylov_dim_mixed);
    gko::kernels::EXEC_NAMESPACE::cb_gmres::initialize(
        exec, d_b.get(), d_residual.get(), d_givens_sin.get(),
        d_givens_cos.get(), d_stop_status.get(), default_krylov_dim_mixed);

    GKO_ASSERT_MTX_NEAR(d_residual, residual, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}

TEST_F(CbGmres, CbGmresInitialize2IsEquivalentToRef)
{
    initialize_data();
    gko::array<char> tmp{ref};
    gko::array<char> dtmp{exec};

    gko::kernels::reference::cb_gmres::restart(
        ref, residual.get(), residual_norm.get(),
        residual_norm_collection.get(), arnoldi_norm.get(),
        range_helper.get_range(), next_krylov_basis.get(),
        final_iter_nums.get(), tmp, default_krylov_dim_mixed);
    gko::kernels::EXEC_NAMESPACE::cb_gmres::restart(
        exec, d_residual.get(), d_residual_norm.get(),
        d_residual_norm_collection.get(), d_arnoldi_norm.get(),
        d_range_helper.get_range(), d_next_krylov_basis.get(),
        d_final_iter_nums.get(), dtmp, default_krylov_dim_mixed);

    GKO_ASSERT_MTX_NEAR(d_arnoldi_norm, arnoldi_norm, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        r<value_type>::value);
    assert_krylov_bases_near();
    GKO_ASSERT_ARRAY_EQ(*d_final_iter_nums, *final_iter_nums);
}

TEST_F(CbGmres, CbGmresStep1IsEquivalentToRef)
{
    initialize_data();
    int iter = 5;

    gko::kernels::reference::cb_gmres::arnoldi(
        ref, next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
        residual_norm.get(), residual_norm_collection.get(),
        range_helper.get_range(), hessenberg_iter.get(), buffer_iter.get(),
        arnoldi_norm.get(), iter, final_iter_nums.get(), stop_status.get(),
        reorth_status.get(), num_reorth.get());
    gko::kernels::EXEC_NAMESPACE::cb_gmres::arnoldi(
        exec, d_next_krylov_basis.get(), d_givens_sin.get(), d_givens_cos.get(),
        d_residual_norm.get(), d_residual_norm_collection.get(),
        d_range_helper.get_range(), d_hessenberg_iter.get(),
        d_buffer_iter.get(), d_arnoldi_norm.get(), iter,
        d_final_iter_nums.get(), d_stop_status.get(), d_reorth_status.get(),
        d_num_reorth.get());

    GKO_ASSERT_MTX_NEAR(d_arnoldi_norm, arnoldi_norm, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_next_krylov_basis, next_krylov_basis,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_hessenberg_iter, hessenberg_iter,
                        r<value_type>::value);
    assert_krylov_bases_near();
    GKO_ASSERT_ARRAY_EQ(*d_final_iter_nums, *final_iter_nums);
}

TEST_F(CbGmres, CbGmresStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::cb_gmres::solve_krylov(
        ref, residual_norm_collection.get(),
        range_helper.get_range().get_accessor().to_const(), hessenberg.get(),
        y.get(), before_preconditioner.get(), final_iter_nums.get());
    gko::kernels::EXEC_NAMESPACE::cb_gmres::solve_krylov(
        exec, d_residual_norm_collection.get(),
        d_range_helper.get_range().get_accessor().to_const(),
        d_hessenberg.get(), d_y.get(), d_before_preconditioner.get(),
        d_final_iter_nums.get());

    GKO_ASSERT_MTX_NEAR(d_y, y, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}
