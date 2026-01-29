// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/test/utils.hpp"
#include "ginkgo/core/base/amp_types.hpp"


namespace {


template <typename ValueType>
class AMPDouble : public ::testing::Test {
protected:
    using value_type = ValueType;
    using index_type = int;
    using Mtx = gko::matrix::AMP<value_type, index_type>;
    using Ell = gko::matrix::Ell<value_type, index_type>;
    using Dns = gko::matrix::Dense<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<gko::next_precision<value_type>>;

    AMPDouble() : exec(gko::ReferenceExecutor::create())
    {
        // clang-format off
        mtx1 = gko::initialize<Dns>({{1.1, 3.0e-9, 0.0, 4.5e-4},
                                     {0.0, 1.2e-11, 2.0, 0.0},
                                     {0.0, 0.0, 0.8, 0.0},
                                     {1.2e-11, 0.0, 1.6e-4, 0.0},
                                     {-2e-5, 0.0, -2.0, 0.0}}, exec);
        mtx2 = gko::initialize<Dns>(
            {{1.0, 3.0, 2.0},
             {0.0, 5.0, 0.0}}, exec);
        // clang-format on
        ell1 = gko::share(Ell::create(exec));
        mtx1->convert_to(ell1.get());
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Dns> mtx1;
    std::unique_ptr<Dns> mtx2;
    std::shared_ptr<Ell> ell1;
    const float tol = 1e-10;
};

using double_types = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(AMPDouble, double_types, TypenameNameGenerator);


TYPED_TEST(AMPDouble, GenerateComputesCorrectRowNorms)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<typename TestFixture::value_type>;
    static_assert(std::is_same<real_T, double>::value, "double only!");
    gko::amp::array_prec<int, T> max_nnz;
    gko::array<real_T> rownorms(this->exec, this->ell1->get_size()[0]);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        rexec, this->ell1.get(), this->tol, max_nnz, rownorms);

    auto rnv = rownorms.get_const_data();
    EXPECT_EQ(rnv[0], static_cast<real_T>(1.1) + static_cast<real_T>(3e-9) +
                          static_cast<real_T>(4.5e-4));
    EXPECT_EQ(rnv[1], static_cast<real_T>(2.0) + static_cast<real_T>(1.2e-11));
    EXPECT_EQ(rnv[2], static_cast<real_T>(0.8));
    EXPECT_EQ(rnv[3],
              static_cast<real_T>(1.2e-11) + static_cast<real_T>(1.6e-4));
    EXPECT_EQ(rnv[4], static_cast<real_T>(2.0) + static_cast<real_T>(2e-5));
}

TYPED_TEST(AMPDouble, GenerateComputesCorrectBinNNZs)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<typename TestFixture::value_type>;
    static_assert(std::is_same<real_T, double>::value, "double only!");
    static_assert(std::tuple_size<gko::amp::array_prec<int, T>>::value == 3,
                  "should be 3 available precisions");
    gko::amp::array_prec<int, T> max_nnz;
    gko::array<real_T> rownorms(this->exec, this->ell1->get_size()[0]);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        rexec, this->ell1.get(), this->tol, max_nnz, rownorms);

#if GKO_AMP_HALF_IS_FP16
    EXPECT_EQ(max_nnz[0], 1);
    EXPECT_EQ(max_nnz[1], 2);
    EXPECT_EQ(max_nnz[2], 0);
#elif GKO_AMP_IS_BFLOAT16
    EXPECT_EQ(max_nnz[0], 1);
    EXPECT_EQ(max_nnz[1], 1);
    EXPECT_EQ(max_nnz[2], 1);
#endif
}

TYPED_TEST(AMPDouble, GenerateEllScattersBinsCorrectly)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<typename TestFixture::value_type>;
    static_assert(std::is_same<real_T, double>::value, "double only!");
    static_assert(std::tuple_size<gko::amp::array_prec<int, T>>::value == 3,
                  "should be 3 available precisions");
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
#if GKO_AMP_HALF_IS_FP16
    auto max_nnzs = gko::amp::array_prec<int, T>{1, 2, 0};
#elif GKO_AMP_HALF_IS_BFLOAT16
    auto max_nnzs = gko::amp::array_prec<int, T>{1, 1, 1};
#else
    auto max_nnzs = gko::amp::array_prec<int, T>{1, 2};
#endif
    auto abins = gko::amp::allocate_bins<T, int>(
        this->exec, this->ell1->get_size(), max_nnzs);
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    gko::amp::array_prec<gko::LinOp*, T> amat;
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    gko::kernels::reference::amp::generate_ell_scatter_bins(
        rexec, this->ell1.get(), this->tol, amat);

    using types_list = typename gko::amp::narrow_types<T>::type;
    gko::constexpr_for<0, num_bins, 1>([&](auto k) {
        using value_type = typename std::tuple_element<k, types_list>::type;
        auto amat0 = dynamic_cast<gko::matrix::Ell<value_type, int>*>(amat[k]);
        ASSERT_TRUE(amat0);
        const auto nnzrow = amat0->get_num_stored_elements_per_row();
        auto vals = amat0->get_const_values();
        auto colids = amat0->get_const_col_idxs();
        if (k == 0) {
            EXPECT_EQ(nnzrow, 1);
            EXPECT_EQ(colids[0], 0);
            EXPECT_EQ(colids[1], 2);
            EXPECT_EQ(colids[2], 2);
            EXPECT_EQ(colids[3], 2);
            EXPECT_EQ(colids[4], 2);
            EXPECT_EQ(vals[0], static_cast<value_type>(1.1));
            EXPECT_EQ(vals[1], static_cast<value_type>(2.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.8));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.6e-4));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2.0));
        }
#if GKO_AMP_HALF_IS_FP16
        else if (k == 1) {
            EXPECT_EQ(nnzrow, 2);
            EXPECT_EQ(colids[0], 1);
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], 0);
            EXPECT_EQ(colids[4], 0);
            EXPECT_EQ(colids[5], 3);
            EXPECT_EQ(colids[6], gko::invalid_index<int>());
            EXPECT_EQ(colids[7], gko::invalid_index<int>());
            EXPECT_EQ(colids[8], gko::invalid_index<int>());
            EXPECT_EQ(colids[9], gko::invalid_index<int>());
            EXPECT_EQ(vals[0], static_cast<value_type>(3e-9));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.2e-11));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2e-5));
            EXPECT_EQ(vals[5], static_cast<value_type>(4.5e-4));
            EXPECT_EQ(vals[6], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[7], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[8], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[9], static_cast<value_type>(0.0));
        } else if (k == 2) {
            EXPECT_EQ(nnzrow, 0);
            EXPECT_FALSE(vals);
            EXPECT_FALSE(colids);
        }
#elif GKO_AMP_HALF_IS_BFLOAT16
        else if (k == 1) {
            EXPECT_EQ(nnzrow, 1);
            EXPECT_EQ(colids[0], 3);
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], 0);
            EXPECT_EQ(colids[4], 0);
            EXPECT_EQ(vals[0], static_cast<value_type>(4.5e-4));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.2e-11));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2e-5));
        } else if (k == 2) {
            EXPECT_EQ(colids[0], 1);
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], gko::invalid_index<int>());
            EXPECT_EQ(colids[4], gko::invalid_index<int>());
            EXPECT_EQ(vals[0], static_cast<value_type>(3e-9));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[4], static_cast<value_type>(0.0));
        }
#endif
    });
}

TYPED_TEST(AMPDouble, ApplyHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<T>;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(gko::share(this->ell1->clone()));
    // Create test vector (matrix is 5x4)
    auto x = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, this->exec);
    // Compute y_amp = AMP * x
    auto y_amp =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 1});

    amp_mtx->apply(x, y_amp);

    // Compute y_ref = original * x
    auto y_ref =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 1});
    this->ell1->apply(x, y_ref);
    // Check relative componentwise error
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < y_ref->get_size()[0]; i++) {
        auto ref_val = y_ref_vals[i];
        auto amp_val = y_amp_vals[i];
        // real_T abs_ref = 0;
        // for(int j = 0; j < this->mtx1->get_size()[1]; j++) {
        //     abs_ref += std::abs(this->mtx1->at(i,j)*x->at(j));
        // }
        const auto abs_ref = std::abs(ref_val);
        ASSERT_GT(abs_ref, real_T{1e-14});
        auto rel_error =
            std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
        EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
            << "Component " << i << ": amp=" << amp_val << ", ref=" << ref_val;
    }
}

TYPED_TEST(AMPDouble, AdvancedApplyHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<T>;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(gko::share(this->ell1->clone()));
    // Create alpha and beta scalars
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    // Create test vector (matrix is 5x4)
    auto x = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, this->exec);
    // Initialize y_amp with some values
    auto y_amp = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0, 5.0}, this->exec);
    // Initialize y_ref with the same values
    auto y_ref = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0, 5.0}, this->exec);

    amp_mtx->apply(alpha, x, beta, y_amp);

    // Compute y_ref = alpha * original * x + beta * y_ref
    this->ell1->apply(alpha, x, beta, y_ref);
    // Check relative componentwise error
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < y_ref->get_size()[0]; i++) {
        auto ref_val = y_ref_vals[i];
        auto amp_val = y_amp_vals[i];
        const auto abs_ref = std::abs(ref_val);
        ASSERT_GT(abs_ref, real_T{1e-14});
        auto rel_error =
            std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
        EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
            << "Component " << i << ": amp=" << amp_val << ", ref=" << ref_val;
    }
}

TYPED_TEST(AMPDouble, FillInDenseReconstructsOriginalMatrix)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dns = typename TestFixture::Dns;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(gko::share(this->ell1->clone()));
    // Create result dense matrix with same size
    auto result = Dns::create(this->exec, this->ell1->get_size());

    gko::kernels::reference::amp::fill_in_dense(rexec, amp_mtx.get(),
                                                result.get());

    GKO_ASSERT_MTX_NEAR(this->mtx1, result, this->tol);
}

TYPED_TEST(AMPDouble, ExtractDiagonalSumsOverBins)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<T>;
    using Mtx = typename TestFixture::Mtx;
    using Diag = gko::matrix::Diagonal<T>;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(gko::share(this->ell1->clone()));
    // The matrix is 5x4, so diagonal size is min(5,4) = 4
    auto diag = Diag::create(this->exec, 4);

    gko::kernels::reference::amp::extract_diagonal(rexec, amp_mtx.get(),
                                                   diag.get());

    // Diagonal entries from mtx1:
    // diag[0] = 1.1
    // diag[1] = 0 (dropped from 1.2e-11)
    // diag[2] = 0.8
    // diag[3] = 0.0
    auto diag_vals = diag->get_const_values();
    EXPECT_NEAR(std::abs(diag_vals[0] - static_cast<T>(1.1)), real_T{0},
                static_cast<real_T>(this->tol));
    EXPECT_NEAR(std::abs(diag_vals[1]), real_T{0}, 0.0);
    EXPECT_NEAR(std::abs(diag_vals[2] - static_cast<T>(0.8)), real_T{0},
                static_cast<real_T>(this->tol));
    EXPECT_NEAR(std::abs(diag_vals[3]), real_T{0}, 0.0);
}


template <typename ValueType>
class AMPFloat : public ::testing::Test {
protected:
    using value_type = ValueType;
    using real_T = gko::remove_complex<value_type>;
    static_assert(std::is_same<real_T, float>::value, "float only!");
    using index_type = int;
    using Mtx = gko::matrix::AMP<value_type, index_type>;
    using Ell = gko::matrix::Ell<value_type, index_type>;
    using Dns = gko::matrix::Dense<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<gko::next_precision<value_type>>;

    AMPFloat() : exec(gko::ReferenceExecutor::create())
    {
        // clang-format off
        mtx1 = gko::initialize<Dns>({{1.1, 3.0e-9, 0.0, 4.5e-4},
                                     {0.0, 1.2e-11, 2.0, 0.0},
                                     {0.0, 0.0, 0.8, 0.0},
                                     {1.2e-11, 0.0, 1.6e-4, 0.0},
                                     {-2e-5, 0.0, -2.0, 0.0}}, exec);
        mtx2 = gko::initialize<Dns>(
            {{1.0, 3.0, 2.0},
             {0.0, 5.0, 0.0}}, exec);
        // clang-format on
        ell1 = Ell::create(exec);
        mtx1->convert_to(ell1.get());
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Dns> mtx1;
    std::unique_ptr<Dns> mtx2;
    std::unique_ptr<Ell> ell1;
    const float tol = 1e-6;
};

using float_types = ::testing::Types<float, std::complex<float>>;
TYPED_TEST_SUITE(AMPFloat, float_types, TypenameNameGenerator);


TYPED_TEST(AMPFloat, GenerateComputesCorrectRowNorms)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    gko::amp::array_prec<int, T> max_nnz;
    gko::array<real_T> rownorms(this->exec, this->ell1->get_size()[0]);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        rexec, this->ell1.get(), this->tol, max_nnz, rownorms);

    auto rnv = rownorms.get_const_data();
    EXPECT_EQ(rnv[0], static_cast<real_T>(1.1) + static_cast<real_T>(3e-9) +
                          static_cast<real_T>(4.5e-4));
    EXPECT_EQ(rnv[1], static_cast<real_T>(2.0) + static_cast<real_T>(1.2e-11));
    EXPECT_EQ(rnv[2], static_cast<real_T>(0.8));
    EXPECT_EQ(rnv[3],
              static_cast<real_T>(1.2e-11) + static_cast<real_T>(1.6e-4));
    EXPECT_EQ(rnv[4], static_cast<real_T>(2.0) + static_cast<real_T>(2e-5));
}

TYPED_TEST(AMPFloat, GenerateComputesCorrectBinNNZs)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    static_assert(std::tuple_size<gko::amp::array_prec<int, T>>::value == 2,
                  "should be 2 available precisions");
    gko::amp::array_prec<int, T> max_nnz;
    gko::array<real_T> rownorms(this->exec, this->ell1->get_size()[0]);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        rexec, this->ell1.get(), this->tol, max_nnz, rownorms);

#if GINKGO_HAVE_AMP_HALF
    EXPECT_EQ(max_nnz[0], 2);
    EXPECT_EQ(max_nnz[1], 1);
#else
    EXPECT_EQ(max_nnz[0], 3);
#endif
}

TYPED_TEST(AMPFloat, GenerateEllScattersBinsCorrectly)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    static_assert(std::tuple_size<gko::amp::array_prec<int, T>>::value == 2,
                  "should be 2 available precisions");
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    const auto max_nnzs =
#if GINKGO_HAVE_AMP_HALF
        gko::amp::array_prec<int, T>{2, 1};
#else
        gko::amp::array_prec<int, T>{3};
#endif
    auto abins = gko::amp::allocate_bins<T, int>(
        this->exec, this->ell1->get_size(), max_nnzs);
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    gko::amp::array_prec<gko::LinOp*, T> amat;
#if GINKGO_HAVE_AMP_HALF
    static_assert(num_bins == 2, "Wrong num bins!");
    ASSERT_EQ(amat.size(), 2);
#else
    static_assert(num_bins == 1, "Wrong num bins!");
    ASSERT_EQ(amat.size(), 1);
#endif
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    gko::kernels::reference::amp::generate_ell_scatter_bins(
        rexec, this->ell1.get(), this->tol, amat);

    using types_list = typename gko::amp::narrow_types<T>::type;
    gko::constexpr_for<0, num_bins, 1>([&](auto k) {
        using value_type = typename std::tuple_element<k, types_list>::type;
        auto amat0 = dynamic_cast<gko::matrix::Ell<value_type, int>*>(amat[k]);
        ASSERT_TRUE(amat0);
        auto vals = amat0->get_const_values();
        auto colids = amat0->get_const_col_idxs();
#if GKO_AMP_HALF_IS_FP16
        if (k == 0) {
            EXPECT_EQ(amat0->get_num_stored_elements_per_row(), 2);
            EXPECT_EQ(colids[0], 0);
            EXPECT_EQ(colids[1], 2);
            EXPECT_EQ(colids[2], 2);
            EXPECT_EQ(colids[3], 2);
            EXPECT_EQ(colids[4], 0);
            EXPECT_EQ(colids[5], gko::invalid_index<int>());
            EXPECT_EQ(colids[6], gko::invalid_index<int>());
            EXPECT_EQ(colids[7], gko::invalid_index<int>());
            EXPECT_EQ(colids[8], gko::invalid_index<int>());
            EXPECT_EQ(colids[9], 2);
            EXPECT_EQ(vals[0], static_cast<value_type>(1.1));
            EXPECT_EQ(vals[1], static_cast<value_type>(2.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.8));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.6e-4));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2e-5));
            for (int j = 5; j < 9; j++) {
                EXPECT_EQ(vals[j], static_cast<value_type>(0));
            }
            EXPECT_EQ(vals[9], static_cast<value_type>(-2.0));
        } else if (k == 1) {
            EXPECT_EQ(amat0->get_num_stored_elements_per_row(), 1);
            EXPECT_EQ(colids[0], 3);
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], gko::invalid_index<int>());
            EXPECT_EQ(colids[4], gko::invalid_index<int>());
            EXPECT_EQ(vals[0], static_cast<value_type>(4.5e-4));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[4], static_cast<value_type>(0));
        }
#elif GKO_AMP_HALF_IS_BFLOAT16
        if (k == 0) {
            EXPECT_EQ(amat0->get_num_stored_elements_per_row(), 2);
            EXPECT_EQ(colids[0], 0);
            EXPECT_EQ(colids[1], 2);
            EXPECT_EQ(colids[2], 2);
            EXPECT_EQ(colids[3], 2);
            EXPECT_EQ(colids[4], 2);
            EXPECT_EQ(colids[5], 3);
            EXPECT_EQ(colids[6], gko::invalid_index<int>());
            EXPECT_EQ(colids[7], gko::invalid_index<int>());
            EXPECT_EQ(colids[8], gko::invalid_index<int>());
            EXPECT_EQ(colids[9], gko::invalid_index<int>());
            EXPECT_EQ(vals[0], static_cast<value_type>(1.1));
            EXPECT_EQ(vals[1], static_cast<value_type>(2.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.8));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.6e-4));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2.0));
            EXPECT_EQ(vals[5], static_cast<value_type>(4.5e-4));
            for (int j = 6; j < 10; j++) {
                EXPECT_EQ(vals[j], static_cast<value_type>(0));
            }
        } else if (k == 1) {
            EXPECT_EQ(amat0->get_num_stored_elements_per_row(), 1);
            EXPECT_EQ(colids[0], gko::invalid_index<int>());
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], gko::invalid_index<int>());
            EXPECT_EQ(colids[4], 0);
            EXPECT_EQ(vals[0], static_cast<value_type>(0));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2e-5));
        }
#else
#endif
    });
}

TYPED_TEST(AMPFloat, ApplyHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(gko::share(this->ell1->clone()));
    // Create test vector (matrix is 5x4)
    auto x = gko::initialize<Vec>({1.0, 2.0, 1.0, 2.0}, this->exec);
    // Compute y_amp = AMP * x
    auto y_amp =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 1});

    amp_mtx->apply(x, y_amp);

    // Compute y_ref = original * x
    auto y_ref =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 1});
    this->ell1->apply(x, y_ref);
    // Check relative componentwise error
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < y_ref->get_size()[0]; i++) {
        const auto ref_val = y_ref_vals[i];
        const auto amp_val = y_amp_vals[i];
        const auto abs_ref = std::abs(ref_val);
        // real_T abs_ref = 0;
        // for(int j = 0; j < this->mtx1->get_size()[1]; j++) {
        //     abs_ref += std::abs(this->mtx1->at(i,j)*x->at(j));
        // }
        ASSERT_GT(abs_ref, real_T{1e-6});
        auto rel_error =
            std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
        EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
            << "Component " << i << ": amp=" << amp_val << ", ref=" << ref_val;
    }
}

TYPED_TEST(AMPFloat, AdvancedApplyHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(gko::share(this->ell1->clone()));
    // Create alpha and beta scalars
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    // Create test vector (matrix is 5x4)
    auto x = gko::initialize<Vec>({1.0, 2.0, 1.0, 2.0}, this->exec);
    // Initialize y_amp with some values
    auto y_amp = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0, 5.0}, this->exec);
    // Initialize y_ref with the same values
    auto y_ref = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0, 5.0}, this->exec);

    amp_mtx->apply(alpha, x, beta, y_amp);

    // Compute y_ref = alpha * original * x + beta * y_ref
    this->ell1->apply(alpha, x, beta, y_ref);
    // Check relative componentwise error
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < y_ref->get_size()[0]; i++) {
        const auto ref_val = y_ref_vals[i];
        const auto amp_val = y_amp_vals[i];
        const auto abs_ref = std::abs(ref_val);
        ASSERT_GT(abs_ref, real_T{1e-6});
        auto rel_error =
            std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
        EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
            << "Component " << i << ": amp=" << amp_val << ", ref=" << ref_val;
    }
}

TYPED_TEST(AMPFloat, FillInDenseReconstructsOriginalMatrix)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dns = typename TestFixture::Dns;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(gko::share(this->ell1->clone()));
    // Create result dense matrix with same size
    auto result = Dns::create(this->exec, this->ell1->get_size());

    gko::kernels::reference::amp::fill_in_dense(rexec, amp_mtx.get(),
                                                result.get());

    GKO_ASSERT_MTX_NEAR(this->mtx1, result, this->tol);
}

TYPED_TEST(AMPFloat, ExtractDiagonalSumsOverBins)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Diag = gko::matrix::Diagonal<T>;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(gko::share(this->ell1->clone()));
    // The matrix is 5x4, so diagonal size is min(5,4) = 4
    auto diag = Diag::create(this->exec, 4);

    gko::kernels::reference::amp::extract_diagonal(rexec, amp_mtx.get(),
                                                   diag.get());

    // Diagonal entries from mtx1:
    // diag[0] = 1.1
    // diag[1] = 0 (dropped from 1.2e-11)
    // diag[2] = 0.8
    // diag[3] = 0.0
    auto diag_vals = diag->get_const_values();
    EXPECT_NEAR(std::abs(diag_vals[0] - static_cast<T>(1.1)), real_T{0},
                static_cast<real_T>(this->tol));
    EXPECT_NEAR(std::abs(diag_vals[1]), real_T{0}, 0.0);
    EXPECT_NEAR(std::abs(diag_vals[2] - static_cast<T>(0.8)), real_T{0},
                static_cast<real_T>(this->tol));
    EXPECT_NEAR(std::abs(diag_vals[3]), real_T{0}, 0.0);
}


#if 0
TYPED_TEST(AMP, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx1->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(AMP, MixedAppliesToDenseVector1)
{
    // Both vectors have the same value type which differs from the matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec = typename gko::matrix::Dense<next_T>;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx1->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(AMP, MixedAppliesToDenseVector2)
{
    // Input vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<next_T>;
    auto x = gko::initialize<Vec1>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec2::create(this->exec, gko::dim<2>{2, 1});

    this->mtx1->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(AMP, MixedAppliesToDenseVector3)
{
    // Output vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<gko::next_precision<T>>;
    auto x = gko::initialize<Vec2>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec1::create(this->exec, gko::dim<2>{2, 1});

    this->mtx1->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(AMP, AppliesToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    this->mtx1->apply(x, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, MixedAppliesToDenseMatrix1)
{
    // Both vectors have the same value type which differs from the matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec = gko::matrix::Dense<next_T>;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<next_T>{2.0, 3.0},
         I<next_T>{1.0, -1.5},
         I<next_T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    this->mtx1->apply(x, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, MixedAppliesToDenseMatrix2)
{
    // Input vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<next_T>;
    // clang-format off
    auto x = gko::initialize<Vec1>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec2::create(this->exec, gko::dim<2>{2});

    this->mtx1->apply(x, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, MixedAppliesToDenseMatrix3)
{
    // Output vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<next_T>;
    // clang-format off
    auto x = gko::initialize<Vec2>(
        {I<next_T>{2.0, 3.0},
         I<next_T>{1.0, -1.5},
         I<next_T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec1::create(this->exec, gko::dim<2>{2});

    this->mtx1->apply(x, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, AppliesLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx1->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(AMP, MixedAppliesLinearCombinationToDenseVector1)
{
    // Both vectors have the same value type which differs from the matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec = gko::matrix::Dense<next_T>;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx1->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(AMP, MixedAppliesLinearCombinationToDenseVector2)
{
    // Input vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<next_T>;
    auto alpha = gko::initialize<Vec1>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec2>({2.0}, this->exec);
    auto x = gko::initialize<Vec1>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec2>({1.0, 2.0}, this->exec);

    this->mtx1->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(AMP, MixedAppliesLinearCombinationToDenseVector3)
{
    // Output vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<next_T>;
    auto alpha = gko::initialize<Vec2>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec1>({2.0}, this->exec);
    auto x = gko::initialize<Vec2>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec1>({1.0, 2.0}, this->exec);

    this->mtx1->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(AMP, AppliesLinearCombinationToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx1->apply(alpha, x, beta, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{-11.0, -2.5},
                           { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, MixedAppliesLinearCombinationToDenseMatrix1)
{
    // Both vectors have the same value type which differs from the matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec = gko::matrix::Dense<next_T>;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<next_T>{2.0, 3.0},
         I<next_T>{1.0, -1.5},
         I<next_T>{4.0, 2.5}}, this->exec);
    auto y = gko::initialize<Vec>(
        {I<next_T>{1.0, 0.5},
         I<next_T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx1->apply(alpha, x, beta, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{-11.0, -2.5},
                           { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, MixedAppliesLinearCombinationToDenseMatrix2)
{
    // Input vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<next_T>;
    auto alpha = gko::initialize<Vec1>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec2>({2.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec1>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y = gko::initialize<Vec2>(
        {I<next_T>{1.0, 0.5},
         I<next_T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx1->apply(alpha, x, beta, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{-11.0, -2.5},
                           { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, MixedAppliesLinearCombinationToDenseMatrix3)
{
    // Output vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<next_T>;
    auto alpha = gko::initialize<Vec2>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec1>({2.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec2>(
        {I<next_T>{2.0, 3.0},
         I<next_T>{1.0, -1.5},
         I<next_T>{4.0, 2.5}}, this->exec);
    auto y = gko::initialize<Vec1>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx1->apply(alpha, x, beta, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{-11.0, -2.5},
                           { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, ApplyFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx1->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(AMP, ApplyFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx1->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(AMP, ApplyFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3}, 2);
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx1->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(AMP, ConvertsToDense)
{
    using Vec = typename TestFixture::Vec;
    auto dense_mtx = Vec::create(this->mtx1->get_executor());

    this->mtx1->convert_to(dense_mtx);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, MovesToDense)
{
    using Vec = typename TestFixture::Vec;
    auto dense_mtx = Vec::create(this->mtx1->get_executor());

    this->mtx1->move_to(dense_mtx);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, AppliesWithStrideToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx2->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(AMP, AppliesWithStrideToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    this->mtx2->apply(x, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{13.0, 3.5},
                           {5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(AMP, AppliesWithStrideLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx2->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Ell, AppliesWithStrideLinearCombinationToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx2->apply(alpha, x, beta, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{-11.0, -2.5},
                           {-1.0, 4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Ell, ApplyWithStrideFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx2->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Ell, ApplyWithStrideFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx2->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Ell, ApplyWithStrideFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3}, 2);
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx2->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Ell, ConvertsWithStrideToDense)
{
    using Vec = typename TestFixture::Vec;
    auto dense_mtx = Vec::create(this->mtx2->get_executor());
    // clang-format off
    auto dense_other = gko::initialize<Vec>(
        4, {{1.0, 3.0, 2.0},
            {0.0, 5.0, 0.0}}, this->exec);
    // clang-format on

    this->mtx2->convert_to(dense_mtx);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Ell, MovesWithStrideToDense)
{
    using Vec = typename TestFixture::Vec;
    auto dense_mtx = Vec::create(this->mtx2->get_executor());

    this->mtx2->move_to(dense_mtx);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Ell, ConvertsEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Ell = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty = Ell::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->convert_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Ell, MovesEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Ell = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty = Ell::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->move_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Ell, ExtractsDiagonal)
{
    using T = typename TestFixture::value_type;
    auto matrix = this->mtx2->clone();
    auto diag = matrix->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Ell, AppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Mtx = typename TestFixture::Mtx;
    using Vec = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx1->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{13.0, 14.0}, complex_type{19.0, 20.0}},
           {complex_type{10.0, 10.0}, complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Ell, AppliesToMixedComplex)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using Vec = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx1->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{13.0, 14.0}, mixed_complex_type{19.0, 20.0}},
           {mixed_complex_type{10.0, 10.0}, mixed_complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Ell, AdvancedAppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Mtx = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<value_type>;
    using DenseComplex = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    auto beta = gko::initialize<Dense>({2.0}, this->exec);
    // clang-format on

    this->mtx1->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{-11.0, -14.0}, complex_type{-15.0, -18.0}},
           {complex_type{-6.0, -6.0}, complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Ell, AdvancedAppliesToMixedComplex)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using MixedDense = gko::matrix::Dense<mixed_value_type>;
    using MixedDenseComplex = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({2.0}, this->exec);
    // clang-format on

    this->mtx1->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{-11.0, -14.0}, mixed_complex_type{-15.0, -18.0}},
           {mixed_complex_type{-6.0, -6.0}, mixed_complex_type{-9.0, -9.0}}}),
        0.0);
}
#endif


}  // namespace
