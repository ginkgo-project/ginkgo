// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <limits>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/matrix/amp_helpers.hpp"
#include "core/test/utils.hpp"


namespace {


class Amp : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using AmpMtx = gko::matrix::AMP<value_type, index_type>;
    using Mtx = gko::matrix::Ell<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Amp() : rand_engine(42) {}

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
            num_rows, num_cols, std::uniform_int_distribution<>(1, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<Vec> gen_vec(int num_rows, int num_cols)
    {
        return gko::test::generate_random_dense_matrix<value_type>(
            num_rows, num_cols, std::normal_distribution<>(-1.0, 1.0),
            rand_engine, ref);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::OmpExecutor> omp;
    std::default_random_engine rand_engine;
};


TEST_F(Amp, GenerateEllRownormsStorageIsEquivalentToRef)
{
    using T = value_type;
    using IndexType = index_type;
    using real_T = gko::remove_complex<T>;
    constexpr int q = AmpMtx::num_precisions;
    const float tol = 1e-10;
    auto mtx = gen_mtx(532, 231);
    auto dmtx = gko::clone(omp, mtx);
    gko::amp::precision_array<int, T> ref_max_nnz;
    gko::amp::precision_array<int, T> omp_max_nnz;
    gko::array<real_T> ref_rownorms(ref, mtx->get_size()[0]);
    gko::array<real_T> omp_rownorms(omp, dmtx->get_size()[0]);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        ref, mtx.get(), tol, ref_max_nnz, ref_rownorms);
    gko::kernels::omp::amp::generate_ell_rownorms_storage(
        omp, dmtx.get(), tol, omp_max_nnz, omp_rownorms);

    for (int k = 0; k < q; k++) {
        EXPECT_EQ(omp_max_nnz[k], ref_max_nnz[k]);
    }
    GKO_ASSERT_ARRAY_NEAR(omp_rownorms, ref_rownorms,
                          std::numeric_limits<real_T>::epsilon());
}


TEST_F(Amp, SpmvIsEquivalentToRef)
{
    using T = value_type;
    using IndexType = index_type;
    const float tol = 1e-10;
    auto ell = gen_mtx(532, 231);
    auto amp_ref = AmpMtx::build().with_tolerance(tol).on(ref)->generate(
        gko::share(std::move(ell)));
    auto amp_omp = gko::clone(omp, amp_ref);
    auto b_ref = gen_vec(amp_ref->get_size()[1], 1);
    auto b_omp = gko::clone(omp, b_ref);
    auto c_ref = Vec::create(ref, gko::dim<2>{amp_ref->get_size()[0], 1});
    auto c_omp = Vec::create(omp, gko::dim<2>{amp_omp->get_size()[0], 1});

    gko::kernels::reference::amp::spmv(ref, amp_ref.get(), b_ref.get(),
                                       c_ref.get());
    gko::kernels::omp::amp::spmv(omp, amp_omp.get(), b_omp.get(), c_omp.get());

    GKO_ASSERT_MTX_NEAR(c_omp, c_ref, r<T>::value);
}


TEST_F(Amp, AdvancedSpmvIsEquivalentToRef)
{
    using T = value_type;
    using IndexType = index_type;
    const float tol = 1e-10;
    auto ell = gen_mtx(532, 231);
    auto amp_ref = AmpMtx::build().with_tolerance(tol).on(ref)->generate(
        gko::share(std::move(ell)));
    auto amp_omp = gko::clone(omp, amp_ref);
    auto b_ref = gen_vec(amp_ref->get_size()[1], 1);
    auto b_omp = gko::clone(omp, b_ref);
    auto c_ref = gen_vec(amp_ref->get_size()[0], 1);
    auto c_omp = gko::clone(omp, c_ref);
    auto alpha_ref = gko::initialize<Vec>({2.0}, ref);
    auto alpha_omp = gko::clone(omp, alpha_ref);
    auto beta_ref = gko::initialize<Vec>({-1.0}, ref);
    auto beta_omp = gko::clone(omp, beta_ref);

    gko::kernels::reference::amp::advanced_spmv(ref, alpha_ref.get(),
                                                amp_ref.get(), b_ref.get(),
                                                beta_ref.get(), c_ref.get());
    gko::kernels::omp::amp::advanced_spmv(omp, alpha_omp.get(), amp_omp.get(),
                                          b_omp.get(), beta_omp.get(),
                                          c_omp.get());

    GKO_ASSERT_MTX_NEAR(c_omp, c_ref, r<T>::value);
}


}  // namespace
