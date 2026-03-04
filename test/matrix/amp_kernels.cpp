// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/matrix/amp_helpers.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


class Amp : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Ell<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using AmpMtx = gko::matrix::AMP<value_type, index_type>;

    Amp() : rand_engine(42) {}

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

    std::default_random_engine rand_engine;
};


TEST_F(Amp, CanBeClonedToAnotherExecutor)
{
    using T = value_type;
    using IndexType = index_type;
    const float tol = 1e-10;
    auto ell = gen_mtx(532, 231);
    auto amp_ref = AmpMtx::build().with_tolerance(tol).on(ref)->generate(
        gko::share(std::move(ell)));

    auto amp_exec = gko::clone(exec, amp_ref);

    auto cloned_mtx = dynamic_cast<AmpMtx*>(amp_exec.get());
    ASSERT_NE(cloned_mtx, nullptr);
    EXPECT_NE(cloned_mtx, amp_ref.get());
    EXPECT_EQ(cloned_mtx->get_executor(), exec);
    EXPECT_EQ(cloned_mtx->get_size(), amp_ref->get_size());
    // Verify each precision bin's data matches
    using types_list = typename gko::amp::narrow_types<T>::type;
    gko::constexpr_for<0, AmpMtx::num_precisions, 1>([&](auto k) {
        using vt = typename std::tuple_element<k, types_list>::type;
        auto ref_bin = dynamic_cast<const gko::matrix::Ell<vt, IndexType>*>(
            amp_ref->get_bin_matrix(k));
        auto exec_bin = dynamic_cast<const gko::matrix::Ell<vt, IndexType>*>(
            cloned_mtx->get_bin_matrix(k));
        ASSERT_TRUE(ref_bin);
        ASSERT_TRUE(exec_bin);
        GKO_ASSERT_MTX_NEAR(ref_bin, exec_bin, 0);
    });
}


TEST_F(Amp, GenerateEllRownormsStorageIsEquivalentToRef)
{
    using T = value_type;
    using IndexType = index_type;
    using real_T = gko::remove_complex<T>;
    constexpr int q = gko::matrix::AMP<T, IndexType>::num_precisions;
    const float tol = 1e-10;
    auto mtx = gen_mtx(532, 231);
    auto dmtx = gko::clone(exec, mtx);
    gko::amp::precision_array<int, T> ref_max_nnz;
    gko::amp::precision_array<int, T> dev_max_nnz;
    gko::array<real_T> ref_rownorms(ref, mtx->get_size()[0]);
    gko::array<real_T> dev_rownorms(exec, dmtx->get_size()[0]);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        ref, mtx.get(), tol, ref_max_nnz, ref_rownorms);
    gko::kernels::GKO_DEVICE_NAMESPACE::amp::generate_ell_rownorms_storage(
        exec, dmtx.get(), tol, dev_max_nnz, dev_rownorms);

    for (int k = 0; k < q; k++) {
        EXPECT_EQ(dev_max_nnz[k], ref_max_nnz[k]);
    }
    GKO_ASSERT_ARRAY_NEAR(dev_rownorms, ref_rownorms,
                          std::numeric_limits<real_T>::epsilon());
}


TEST_F(Amp, GenerateEllScatterBinsIsEquivalentToRef)
{
    using T = value_type;
    using IndexType = index_type;
    constexpr int q = gko::matrix::AMP<T, IndexType>::num_precisions;
    const float tol = 1e-10;
    auto mtx = gen_mtx(532, 231);
    auto dmtx = gko::clone(exec, mtx);
    // Compute max_nnz per bin using reference kernel
    gko::amp::precision_array<int, T> max_nnz;
    gko::array<gko::remove_complex<T>> rownorms(ref, mtx->get_size()[0]);
    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        ref, mtx.get(), tol, max_nnz, rownorms);
    // Allocate bins on ref and exec with the same max_nnz
    auto ref_bins =
        gko::amp::allocate_bins<T, IndexType>(ref, mtx->get_size(), max_nnz);
    auto exec_bins =
        gko::amp::allocate_bins<T, IndexType>(exec, dmtx->get_size(), max_nnz);
    constexpr auto num_bins = std::tuple_size<decltype(ref_bins)>::value;
    static_assert(num_bins == q, "Wrong number of bins!");
    gko::amp::precision_array<gko::LinOp*, T> ref_amat;
    gko::amp::precision_array<gko::LinOp*, T> exec_amat;
    for (int k = 0; k < num_bins; k++) {
        ref_amat[k] = ref_bins[k].get();
        exec_amat[k] = exec_bins[k].get();
    }

    // Run kernel on ref and exec
    gko::kernels::reference::amp::generate_ell_scatter_bins(ref, mtx.get(), tol,
                                                            ref_amat);
    gko::kernels::GKO_DEVICE_NAMESPACE::amp::generate_ell_scatter_bins(
        exec, dmtx.get(), tol, exec_amat);

    // Compare each bin
    using types_list = typename gko::amp::narrow_types<T>::type;
    gko::constexpr_for<0, num_bins, 1>([&](auto k) {
        using vt = typename std::tuple_element<k, types_list>::type;
        auto ref_ell =
            dynamic_cast<gko::matrix::Ell<vt, IndexType>*>(ref_amat[k]);
        auto exec_ell =
            dynamic_cast<gko::matrix::Ell<vt, IndexType>*>(exec_amat[k]);
        ASSERT_TRUE(ref_ell);
        ASSERT_TRUE(exec_ell);
        GKO_ASSERT_MTX_NEAR(ref_ell, exec_ell, 0);
    });
}


TEST_F(Amp, SpmvIsEquivalentToRef)
{
    using T = value_type;
    using IndexType = index_type;
    const float tol = 1e-10;
    auto ell = gen_mtx(532, 231);
    auto amp_ref = AmpMtx::build().with_tolerance(tol).on(ref)->generate(
        gko::share(std::move(ell)));
    auto amp_d = gko::clone(exec, amp_ref);
    auto b_ref = gen_vec(amp_ref->get_size()[1], 1);
    auto b_d = gko::clone(exec, b_ref);
    auto c_ref = Vec::create(ref, gko::dim<2>{amp_ref->get_size()[0], 1});
    auto c_d = Vec::create(exec, gko::dim<2>{amp_d->get_size()[0], 1});

    gko::kernels::reference::amp::spmv(ref, amp_ref.get(), b_ref.get(),
                                       c_ref.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::amp::spmv(exec, amp_d.get(), b_d.get(),
                                                  c_d.get());

    GKO_ASSERT_MTX_NEAR(c_d, c_ref, r<T>::value);
}


TEST_F(Amp, FillInDenseIsEquivalentToRef)
{
    SKIP_IF_SINGLE_MODE;
    using T = value_type;
    using IndexType = index_type;
    const float tol = 1e-10;
    auto ell = gen_mtx(532, 231);
    // Build AMP matrix on ref, then clone to exec
    auto amp_ref = AmpMtx::build().with_tolerance(tol).on(ref)->generate(
        gko::share(std::move(ell)));
    auto amp_exec = gko::clone(exec, amp_ref);
    auto result_ref = Vec::create(ref, amp_ref->get_size());
    auto result_exec = Vec::create(exec, amp_exec->get_size());

    gko::kernels::reference::amp::fill_in_dense(ref, amp_ref.get(),
                                                result_ref.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::amp::fill_in_dense(exec, amp_exec.get(),
                                                           result_exec.get());

    GKO_ASSERT_MTX_NEAR(result_ref, result_exec, 0);
}


TEST_F(Amp, ExtractDiagonalIsEquivalentToRef)
{
    SKIP_IF_SINGLE_MODE;
    using T = value_type;
    using IndexType = index_type;
    using Diag = gko::matrix::Diagonal<T>;
    const float tol = 1e-10;
    auto ell = gen_mtx(532, 231);
    // Build AMP matrix on ref, then clone to exec
    auto amp_ref = AmpMtx::build().with_tolerance(tol).on(ref)->generate(
        gko::share(std::move(ell)));
    auto amp_exec = gko::clone(exec, amp_ref);
    const auto diag_size =
        std::min(amp_ref->get_size()[0], amp_ref->get_size()[1]);
    auto diag_ref = Diag::create(ref, diag_size);
    auto diag_exec = Diag::create(exec, diag_size);

    gko::kernels::reference::amp::extract_diagonal(ref, amp_ref.get(),
                                                   diag_ref.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::amp::extract_diagonal(
        exec, amp_exec.get(), diag_exec.get());

    GKO_ASSERT_MTX_NEAR(diag_ref, diag_exec, 0);
}
