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

#include <ginkgo/core/solver/gmres.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/solver/gmres_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class Gmres : public ::testing::Test {
protected:
#if GINKGO_DPCPP_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using index_type = gko::int32;
    using Mtx = gko::matrix::Dense<value_type>;
    using norm_type = gko::remove_complex<value_type>;
    using NormVector = gko::matrix::Dense<norm_type>;
    template <typename T>
    using Dense = typename gko::matrix::Dense<T>;

    Gmres() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::DpcppExecutor::get_num_devices("all"), 0);
        ref = gko::ReferenceExecutor::create();
        dpcpp = gko::DpcppExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (dpcpp != nullptr) {
            ASSERT_NO_THROW(dpcpp->synchronize());
        }
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
        y = gen_mtx(gko::solver::default_krylov_dim, nrhs);
        before_preconditioner = Mtx::create_with_config_of(x.get());
        b = gen_mtx(m, nrhs);
        krylov_bases = gen_mtx(m * (gko::solver::default_krylov_dim + 1), nrhs);
        hessenberg = gen_mtx(gko::solver::default_krylov_dim + 1,
                             gko::solver::default_krylov_dim * nrhs);
        hessenberg_iter = gen_mtx(gko::solver::default_krylov_dim + 1, nrhs);
        residual = gen_mtx(m, nrhs);
        residual_norm = gen_mtx<norm_type>(1, nrhs);
        residual_norm_collection =
            gen_mtx(gko::solver::default_krylov_dim + 1, nrhs);
        givens_sin = gen_mtx(gko::solver::default_krylov_dim, nrhs);
        givens_cos = gen_mtx(gko::solver::default_krylov_dim, nrhs);
        stop_status =
            std::make_unique<gko::array<gko::stopping_status>>(ref, nrhs);
        for (size_t i = 0; i < stop_status->get_num_elems(); ++i) {
            stop_status->get_data()[i].reset();
        }
        final_iter_nums =
            std::make_unique<gko::array<gko::size_type>>(ref, nrhs);
        for (size_t i = 0; i < final_iter_nums->get_num_elems(); ++i) {
            final_iter_nums->get_data()[i] = 5;
        }

        d_x = gko::clone(dpcpp, x);
        d_before_preconditioner = Mtx::create_with_config_of(d_x.get());
        d_y = gko::clone(dpcpp, y);
        d_b = gko::clone(dpcpp, b);
        d_krylov_bases = gko::clone(dpcpp, krylov_bases);
        d_hessenberg = gko::clone(dpcpp, hessenberg);
        d_hessenberg_iter = gko::clone(dpcpp, hessenberg_iter);
        d_residual = gko::clone(dpcpp, residual);
        d_residual_norm = gko::clone(dpcpp, residual_norm);
        d_residual_norm_collection =
            gko::clone(dpcpp, residual_norm_collection);
        d_givens_sin = gko::clone(dpcpp, givens_sin);
        d_givens_cos = gko::clone(dpcpp, givens_cos);
        d_stop_status = std::make_unique<gko::array<gko::stopping_status>>(
            dpcpp, *stop_status);
        d_final_iter_nums = std::make_unique<gko::array<gko::size_type>>(
            dpcpp, *final_iter_nums);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::DpcppExecutor> dpcpp;

    std::default_random_engine rand_engine;

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
    std::unique_ptr<gko::array<gko::stopping_status>> stop_status;
    std::unique_ptr<gko::array<gko::size_type>> final_iter_nums;

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
    std::unique_ptr<gko::array<gko::stopping_status>> d_stop_status;
    std::unique_ptr<gko::array<gko::size_type>> d_final_iter_nums;
};


TEST_F(Gmres, DpcppGmresInitialize1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres::initialize_1(
        ref, b.get(), residual.get(), givens_sin.get(), givens_cos.get(),
        stop_status.get(), gko::solver::default_krylov_dim);
    gko::kernels::dpcpp::gmres::initialize_1(
        dpcpp, d_b.get(), d_residual.get(), d_givens_sin.get(),
        d_givens_cos.get(), d_stop_status.get(),
        gko::solver::default_krylov_dim);

    GKO_ASSERT_MTX_NEAR(d_residual, residual, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}


TEST_F(Gmres, DpcppGmresInitialize2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres::initialize_2(
        ref, residual.get(), residual_norm.get(),
        residual_norm_collection.get(), krylov_bases.get(),
        final_iter_nums.get(), gko::solver::default_krylov_dim);
    gko::kernels::dpcpp::gmres::initialize_2(
        dpcpp, d_residual.get(), d_residual_norm.get(),
        d_residual_norm_collection.get(), d_krylov_bases.get(),
        d_final_iter_nums.get(), gko::solver::default_krylov_dim);

    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_krylov_bases, krylov_bases, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_final_iter_nums, *final_iter_nums);
}


TEST_F(Gmres, DpcppGmresStep1IsEquivalentToRef)
{
    initialize_data();
    int iter = 5;

    gko::kernels::reference::gmres::step_1(
        ref, x->get_size()[0], givens_sin.get(), givens_cos.get(),
        residual_norm.get(), residual_norm_collection.get(), krylov_bases.get(),
        hessenberg_iter.get(), iter, final_iter_nums.get(), stop_status.get());
    gko::kernels::dpcpp::gmres::step_1(
        dpcpp, d_x->get_size()[0], d_givens_sin.get(), d_givens_cos.get(),
        d_residual_norm.get(), d_residual_norm_collection.get(),
        d_krylov_bases.get(), d_hessenberg_iter.get(), iter,
        d_final_iter_nums.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_hessenberg_iter, hessenberg_iter,
                        2 * r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_krylov_bases, krylov_bases, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_final_iter_nums, *final_iter_nums);
}


TEST_F(Gmres, DpcppGmresStep1OnSingleRHSIsEquivalentToRef)
{
    initialize_data(1);
    int iter = 5;

    gko::kernels::reference::gmres::step_1(
        ref, x->get_size()[0], givens_sin.get(), givens_cos.get(),
        residual_norm.get(), residual_norm_collection.get(), krylov_bases.get(),
        hessenberg_iter.get(), iter, final_iter_nums.get(), stop_status.get());
    gko::kernels::dpcpp::gmres::step_1(
        dpcpp, d_x->get_size()[0], d_givens_sin.get(), d_givens_cos.get(),
        d_residual_norm.get(), d_residual_norm_collection.get(),
        d_krylov_bases.get(), d_hessenberg_iter.get(), iter,
        d_final_iter_nums.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_givens_sin, givens_sin, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_givens_cos, givens_cos, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm, residual_norm, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual_norm_collection, residual_norm_collection,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_hessenberg_iter, hessenberg_iter,
                        2 * r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_krylov_bases, krylov_bases, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_final_iter_nums, *final_iter_nums);
}


TEST_F(Gmres, DpcppGmresStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gmres::step_2(ref, residual_norm_collection.get(),
                                           krylov_bases.get(), hessenberg.get(),
                                           y.get(), before_preconditioner.get(),
                                           final_iter_nums.get());
    gko::kernels::dpcpp::gmres::step_2(dpcpp, d_residual_norm_collection.get(),
                                       d_krylov_bases.get(), d_hessenberg.get(),
                                       d_y.get(), d_before_preconditioner.get(),
                                       d_final_iter_nums.get());

    GKO_ASSERT_MTX_NEAR(d_y, y, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


}  // namespace
