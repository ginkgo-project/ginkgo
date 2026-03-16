// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gauss_seidel_kernels.hpp"

#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class GaussSeidelKernel : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Ell<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    GaussSeidelKernel()
        : exec(gko::ReferenceExecutor::create()),
          // 4x4 matrix with 2 colors:
          //   color 0: rows 0, 1  (independent: no off-diagonal links within
          //   color) color 1: rows 2, 3  (depend only on color-0 rows)
          //
          // A = [ 2  0  1  0 ]
          //     [ 0  3  0  1 ]
          //     [ 1  0  4  0 ]
          //     [ 0  1  0  5 ]
          //
          // color_ptrs = {0, 2, 4}
          mtx(gko::initialize<Mtx>(
              // clang-format off
              {{2.0, 0.0, 1.0, 0.0},
               {0.0, 3.0, 0.0, 1.0},
               {1.0, 0.0, 4.0, 0.0},
               {0.0, 1.0, 0.0, 5.0}},
              // clang-format on
              exec)),
          color_ptrs{0, 2, 4}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx;
    std::vector<index_type> color_ptrs;
};

TYPED_TEST_SUITE(GaussSeidelKernel, gko::test::ValueIndexTypesBase,
                 PairTypenameNameGenerator);


// After one sweep starting from x=0:
//   Color 0: x[0] = b[0]/2,  x[1] = b[1]/3
//   Color 1: x[2] = (b[2] - A[2,0]*x[0]) / 4
//            x[3] = (b[3] - A[3,1]*x[1]) / 5
TYPED_TEST(GaussSeidelKernel, SingleIterationFromZero)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto b = gko::initialize<Vec>({2.0, 3.0, 3.0, 2.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 1);

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        this->exec, this->color_ptrs, this->mtx.get(), b.get(), x.get(), true,
        &stop);

    // x[0] = 2/2 = 1, x[1] = 3/3 = 1
    // x[2] = (3 - 1*1)/4 = 0.5, x[3] = (2 - 1*1)/5 = 0.2
    GKO_ASSERT_MTX_NEAR(x, l({1.0, 1.0, 0.5, 0.2}), r<value_type>::value);
}


TYPED_TEST(GaussSeidelKernel, UsesCurrentXAsInitialGuess)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto b = gko::initialize<Vec>({2.0, 3.0, 3.0, 2.0}, this->exec);
    // Start with non-zero x; color-0 rows see no updated neighbours yet, so
    // x[0] and x[1] change.  Color-1 rows use the newly computed x[0], x[1].
    auto x = gko::initialize<Vec>({0.5, 0.5, 0.5, -0.5}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 1);

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        this->exec, this->color_ptrs, this->mtx.get(), b.get(), x.get(), true,
        &stop);

    // Color-0 rows do not see each other; off-diag neighbours are color-1 rows
    // still at their initial values (0.5):
    //   x[0] = (2 - A[0,2]*x[2]) / 2 = (2 - 0.5) / 2 = 0.75
    //   x[1] = (3 - A[1,3]*x[3]) / 3 = (3 + 0.5) / 3 = 7/6
    // Color-1 rows use updated x[0], x[1]:
    //   x[2] = (3 - A[2,0]*0.75) / 4 = (3 - 0.75) / 4 = 2.25/4 = 0.5625
    //   x[3] = (2 - A[3,1]*(7/6)) / 5 = (2 - 7/6) / 5 = (5/6) / 5 = 1/6
    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{0.75}, value_type{7.0 / 6.0},
                           value_type{0.5625}, value_type{1.0 / 6.0}}),
                        r<value_type>::value);
}


TYPED_TEST(GaussSeidelKernel, MultipleRHS)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using T = value_type;

    // Two RHS columns side-by-side: [2,3,3,2] and [4,6,6,4]
    // Second column is exactly 2x the first, so result should also be 2x.
    auto b = gko::initialize<Vec>(
        {I<T>{2.0, 4.0}, I<T>{3.0, 6.0}, I<T>{3.0, 6.0}, I<T>{2.0, 4.0}},
        this->exec);
    auto x = gko::initialize<Vec>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}},
        this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 2);

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        this->exec, this->color_ptrs, this->mtx.get(), b.get(), x.get(), true,
        &stop);

    GKO_ASSERT_MTX_NEAR(x, l({{1.0, 2.0}, {1.0, 2.0}, {0.5, 1.0}, {0.2, 0.4}}),
                        r<value_type>::value);
}


TYPED_TEST(GaussSeidelKernel, FirstIterResetsStopStatus)
{
    using Vec = typename TestFixture::Vec;

    auto b = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 2);

    // Pre-set stop entries to "stopped"
    gko::stopping_status stopped{};
    stopped.stop(1);
    stop.get_data()[0] = stopped;
    stop.get_data()[1] = stopped;

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        this->exec, this->color_ptrs, this->mtx.get(), b.get(), x.get(), true,
        &stop);

    gko::stopping_status non_stopped{};
    non_stopped.reset();
    EXPECT_EQ(stop.get_data()[0], non_stopped);
    EXPECT_EQ(stop.get_data()[1], non_stopped);
}


TYPED_TEST(GaussSeidelKernel, SubsequentIterDoesNotResetStopStatus)
{
    using Vec = typename TestFixture::Vec;

    auto b = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 2);

    // Pre-set stop entries to "stopped"
    gko::stopping_status stopped{};
    stopped.stop(1);
    stop.get_data()[0] = stopped;
    stop.get_data()[1] = stopped;

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        this->exec, this->color_ptrs, this->mtx.get(), b.get(), x.get(), false,
        &stop);

    EXPECT_EQ(stop.get_data()[0], stopped);
    EXPECT_EQ(stop.get_data()[1], stopped);
}


TYPED_TEST(GaussSeidelKernel, DiagonalOnlyMatrixSolvesExactlyInOneStep)
{
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    // 3x3 diagonal matrix, single color
    auto diag = gko::initialize<Mtx>(
        {{4.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 5.0}}, this->exec);
    auto b = gko::initialize<Vec>({8.0, 6.0, 10.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 1);
    std::vector<index_type> single_color{0, 3};

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        this->exec, single_color, diag.get(), b.get(), x.get(), true, &stop);

    GKO_ASSERT_MTX_NEAR(x, l({2.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(GaussSeidelKernel, EmptyColorPtrsDoesNothing)
{
    using Vec = typename TestFixture::Vec;
    using index_type = typename TestFixture::index_type;

    auto b = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0}, this->exec);
    auto x = gko::initialize<Vec>({5.0, 6.0, 7.0, 8.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 1);
    std::vector<index_type> empty_ptrs{};

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        this->exec, empty_ptrs, this->mtx.get(), b.get(), x.get(), true, &stop);

    // x must be unchanged
    GKO_ASSERT_MTX_NEAR(x, l({5.0, 6.0, 7.0, 8.0}), 0.0);
}


// ============================================================
// AMP Gauss-Seidel kernel tests
// ============================================================

// The AMP kernel applies a Richardson-style correction:
//   x[row] += (b[row] - A_offdiag * x) / diag[row]
// rather than the direct solve used by the ELL kernel:
//   x[row]  = (b[row] - A_offdiag * x) / diag[row]
//
// When x starts at zero the two are equivalent; for non-zero
// initial guesses the expected values differ.

template <typename ValueIndexType>
class GaussSeidelKernelAMP : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using AMPMtx = gko::matrix::AMP<value_type, index_type>;
    using EllMtx = gko::matrix::Ell<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    GaussSeidelKernelAMP()
        : exec(gko::ReferenceExecutor::create()),
          // Same 4x4 matrix used by GaussSeidelKernel, 2 colors:
          //   color 0: rows 0, 1
          //   color 1: rows 2, 3
          //
          // A = [ 2  0  1  0 ]
          //     [ 0  3  0  1 ]
          //     [ 1  0  4  0 ]
          //     [ 0  1  0  5 ]
          //
          // color_ptrs = {0, 2, 4}
          color_ptrs{0, 2, 4}
    {
        auto ell = gko::initialize<EllMtx>(
            // clang-format off
            {{2.0, 0.0, 1.0, 0.0},
             {0.0, 3.0, 0.0, 1.0},
             {1.0, 0.0, 4.0, 0.0},
             {0.0, 1.0, 0.0, 5.0}},
            // clang-format on
            exec);
        // tolerance 1e-6: all O(1) integer values land in the
        // highest-precision bin, so no accuracy is lost.
        mtx = AMPMtx::build().with_tolerance(1e-6f).on(exec)->generate(
            gko::share(std::move(ell)));
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<AMPMtx> mtx;
    std::vector<index_type> color_ptrs;
};

TYPED_TEST_SUITE(GaussSeidelKernelAMP, gko::test::ValueIndexTypesBase,
                 PairTypenameNameGenerator);


// Starting from x = 0 the correction x += (b - A_off*x)/diag
// collapses to x = b/diag, matching the ELL result exactly.
//   Color 0: x[0] = 2/2 = 1,  x[1] = 3/3 = 1
//   Color 1: x[2] = (3-1*1)/4 = 0.5,  x[3] = (2-1*1)/5 = 0.2
TYPED_TEST(GaussSeidelKernelAMP, SingleIterationFromZero)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto b = gko::initialize<Vec>({2.0, 3.0, 3.0, 2.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 1);

    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        this->exec, this->color_ptrs, this->mtx.get(), b.get(), x.get(), true,
        &stop);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 1.0, 0.5, 0.2}), r<value_type>::value);
}


// Non-zero initial x
//
// With x_init = {0.5, 0.5, 0.5, -0.5}:
//   Color 0:
//     x[0] = (2 - 1*0.5)/2   = 3/4
//     x[1] = (3 - 1*(-0.5))/3 = 7/6
//   Color 1 (uses updated x[0], x[1]):
//     x[2] = (3 - 1*3/4)/4  = 9/16
//     x[3] = (2 - 1*(7/6))/5 = 1/6
TYPED_TEST(GaussSeidelKernelAMP, UpdatesCurrentX)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto b = gko::initialize<Vec>({2.0, 3.0, 3.0, 2.0}, this->exec);
    auto x = gko::initialize<Vec>({0.5, 0.5, 0.5, -0.5}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 1);

    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        this->exec, this->color_ptrs, this->mtx.get(), b.get(), x.get(), true,
        &stop);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{0.75}, value_type{7.0 / 6.0},
                           value_type{9.0 / 16.0}, value_type{1.0 / 6}}),
                        r<value_type>::value);
}


// Two RHS columns: second column is 2x the first; starting from x=0
// so AMP += agrees with ELL.
TYPED_TEST(GaussSeidelKernelAMP, MultipleRHS)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using T = value_type;

    auto b = gko::initialize<Vec>(
        {I<T>{2.0, 4.0}, I<T>{3.0, 6.0}, I<T>{3.0, 6.0}, I<T>{2.0, 4.0}},
        this->exec);
    auto x = gko::initialize<Vec>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}},
        this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 2);

    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        this->exec, this->color_ptrs, this->mtx.get(), b.get(), x.get(), true,
        &stop);

    GKO_ASSERT_MTX_NEAR(x, l({{1.0, 2.0}, {1.0, 2.0}, {0.5, 1.0}, {0.2, 0.4}}),
                        r<value_type>::value);
}


TYPED_TEST(GaussSeidelKernelAMP, FirstIterResetsStopStatus)
{
    using Vec = typename TestFixture::Vec;

    auto b = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 2);

    gko::stopping_status stopped{};
    stopped.stop(1);
    stop.get_data()[0] = stopped;
    stop.get_data()[1] = stopped;

    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        this->exec, this->color_ptrs, this->mtx.get(), b.get(), x.get(), true,
        &stop);

    gko::stopping_status non_stopped{};
    non_stopped.reset();
    EXPECT_EQ(stop.get_data()[0], non_stopped);
    EXPECT_EQ(stop.get_data()[1], non_stopped);
}


TYPED_TEST(GaussSeidelKernelAMP, SubsequentIterDoesNotResetStopStatus)
{
    using Vec = typename TestFixture::Vec;

    auto b = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 2);

    gko::stopping_status stopped{};
    stopped.stop(1);
    stop.get_data()[0] = stopped;
    stop.get_data()[1] = stopped;

    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        this->exec, this->color_ptrs, this->mtx.get(), b.get(), x.get(), false,
        &stop);

    EXPECT_EQ(stop.get_data()[0], stopped);
    EXPECT_EQ(stop.get_data()[1], stopped);
}


// Diagonal matrix, single color, x starts at zero:
// x += b/diag collapses to x = b/diag, giving the exact solution.
TYPED_TEST(GaussSeidelKernelAMP, DiagonalOnlyMatrixSolvesExactlyInOneStep)
{
    using AMPMtx = typename TestFixture::AMPMtx;
    using EllMtx = typename TestFixture::EllMtx;
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto diag_ell = gko::initialize<EllMtx>(
        {{4.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 5.0}}, this->exec);
    auto diag_amp = AMPMtx::build()
                        .with_tolerance(1e-6f)
                        .on(this->exec)
                        ->generate(gko::share(std::move(diag_ell)));
    auto b = gko::initialize<Vec>({8.0, 6.0, 10.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 1);
    std::vector<index_type> single_color{0, 3};

    gko::kernels::reference::gssdl::multicolor_fgs_amp(this->exec, single_color,
                                                       diag_amp.get(), b.get(),
                                                       x.get(), true, &stop);

    GKO_ASSERT_MTX_NEAR(x, l({2.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(GaussSeidelKernelAMP, EmptyColorPtrsDoesNothing)
{
    using Vec = typename TestFixture::Vec;
    using index_type = typename TestFixture::index_type;

    auto b = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0}, this->exec);
    auto x = gko::initialize<Vec>({5.0, 6.0, 7.0, 8.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 1);
    std::vector<index_type> empty_ptrs{};

    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        this->exec, empty_ptrs, this->mtx.get(), b.get(), x.get(), true, &stop);

    GKO_ASSERT_MTX_NEAR(x, l({5.0, 6.0, 7.0, 8.0}), 0.0);
}


// ============================================================
// AMP Gauss-Seidel accuracy tests (AMP result vs ELL result)
// ============================================================

// Test matrix (4x4, 2 colors):
//
//   color 0: rows 0, 1
//   color 1: rows 2, 3
//
//   A = [  4     0    0.8   3e-4 ]
//       [  0     5    3e-4  0.7  ]
//       [  0.8   3e-4  6    0    ]
//       [  3e-4  0.7   0    3    ]
//
// Row norms (L1) are ≈ {4.8, 5.7, 6.8, 3.7}.  With this norm:
//
//   For double (amp_tol = 1e-10):
//     lbs[0] (double) = norm * 1e-10 / float_eps ≈ 4e-3
//     → O(1) off-diagonals stay in the double bin
//     → 3e-4 falls below lbs[0] → goes into the float (or half) bin
//
//   For float (amp_tol = 1e-6):
//     lbs[0] (float) = norm * 1e-6 / half_eps  ≈ 4.9e-3 (fp16) / 6.2e-4 (bf16)
//     → 3e-4 falls below lbs[0] → goes into the half bin (when available)
//     → with no half support: 3e-4 > norm*tol stays in the only (float) bin
//
// The reference b is chosen so that a single ELL GS sweep from x=0 gives
// x_ell = {1, 1, 1, 1}.  x_amp must agree with x_ell to within the AMP
// relative tolerance.

template <typename ValueIndexType>
class GaussSeidelKernelAMPAccuracy : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using AMPMtx = gko::matrix::AMP<value_type, index_type>;
    using EllMtx = gko::matrix::Ell<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    GaussSeidelKernelAMPAccuracy()
        : exec(gko::ReferenceExecutor::create()),
          // tighter tolerance for double: matches AMPDouble fixture in
          // amp_kernels.cpp; relaxed for float: matches AMPFloat
          amp_tol(std::is_same<real_type, double>::value ? 1e-10f : 1e-6f),
          color_ptrs{0, 2, 4}
    {
        // clang-format off
        mtx_ell = gko::initialize<EllMtx>(
            {{4.0,  0.0,  0.8,  3e-4},
             {0.0,  5.0,  3e-4, 0.7 },
             {0.8,  3e-4, 6.0,  0.0 },
             {3e-4, 0.7,  0.0,  3.0 }},
            exec);
        // clang-format on
        // Clone the ELL matrix so that mtx_ell stays accessible for
        // the ELL kernel calls inside the tests.
        mtx_amp = AMPMtx::build().with_tolerance(amp_tol).on(exec)->generate(
            gko::share(mtx_ell->clone()));
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    const float amp_tol;
    std::unique_ptr<EllMtx> mtx_ell;
    std::unique_ptr<AMPMtx> mtx_amp;
    std::vector<index_type> color_ptrs;
};

TYPED_TEST_SUITE(GaussSeidelKernelAMPAccuracy, gko::test::ValueIndexTypesBase,
                 PairTypenameNameGenerator);


// b chosen so that one ELL sweep from x=0 gives x_ell = {1, 1, 1, 1}:
//   x[0] = 4.0/4 = 1
//   x[1] = 5.0/5 = 1
//   x[2] = (6.8003 - 0.8*1 - 3e-4*1) / 6 = 6.0/6 = 1
//   x[3] = (3.7003 - 3e-4*1 - 0.7*1 ) / 3 = 3.0/3 = 1
// The O(3e-4) off-diagonal terms are stored at lower precision in AMP,
// so x_amp[2] and x_amp[3] acquire a small error.  The test asserts the
// component-wise relative error is ≤ amp_tol.
TYPED_TEST(GaussSeidelKernelAMPAccuracy, SingleRHSMatchesELLWithinTol)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;

    // b[2] = 6 + 0.8 + 3e-4, b[3] = 3 + 0.7 + 3e-4
    auto b = gko::initialize<Vec>({4.0, 5.0, 6.8003, 3.7003}, this->exec);
    auto x_ell = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto x_amp = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 1);

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        this->exec, this->color_ptrs, this->mtx_ell.get(), b.get(), x_ell.get(),
        true, &stop);
    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        this->exec, this->color_ptrs, this->mtx_amp.get(), b.get(), x_amp.get(),
        true, &stop);

    const auto cmp_tol = static_cast<real_type>(this->amp_tol);
    const auto* ell_vals = x_ell->get_const_values();
    const auto* amp_vals = x_amp->get_const_values();
    const auto nrows = x_ell->get_size()[0];
    for (gko::size_type i = 0; i < nrows; ++i) {
        const auto abs_ref = std::abs(ell_vals[i]);
        ASSERT_GT(abs_ref, real_type{1e-14});
        const auto rel_err = std::abs(amp_vals[i] - ell_vals[i]) / abs_ref;
        EXPECT_LE(rel_err, cmp_tol) << "Row " << i << ": amp=" << amp_vals[i]
                                    << ", ell=" << ell_vals[i];
    }
}


// Two RHS columns: col 1 = 2 × col 0, so x_ell = {{1,2},{1,2},{1,2},{1,2}}.
//   b col 0: {4.0,  5.0,   6.8003,  3.7003}
//   b col 1: {8.0, 10.0,  13.6006,  7.4006}  = 2 × col 0
// The lower-precision AMP approximation of the O(3e-4) off-diagonals
// introduces small errors in both columns; the test checks each (row,col)
// component-wise relative error is ≤ amp_tol.
TYPED_TEST(GaussSeidelKernelAMPAccuracy, MultipleRHSMatchesELLWithinTol)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using T = value_type;

    auto b = gko::initialize<Vec>(
        // clang-format off
        {I<T>{4.0,  8.0 },
         I<T>{5.0, 10.0 },
         I<T>{6.8003, 13.6006},
         I<T>{3.7003,  7.4006}},
        // clang-format on
        this->exec);
    auto x_ell = gko::initialize<Vec>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}},
        this->exec);
    auto x_amp = gko::initialize<Vec>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}},
        this->exec);
    auto stop = gko::array<gko::stopping_status>(this->exec, 2);

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        this->exec, this->color_ptrs, this->mtx_ell.get(), b.get(), x_ell.get(),
        true, &stop);
    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        this->exec, this->color_ptrs, this->mtx_amp.get(), b.get(), x_amp.get(),
        true, &stop);

    const auto cmp_tol = static_cast<real_type>(this->amp_tol);
    const auto* ell_vals = x_ell->get_const_values();
    const auto* amp_vals = x_amp->get_const_values();
    const auto nrows = x_ell->get_size()[0];
    const auto ncols = x_ell->get_size()[1];
    const auto stride = x_ell->get_stride();
    for (gko::size_type row = 0; row < nrows; ++row) {
        for (gko::size_type col = 0; col < ncols; ++col) {
            const auto ell_val = ell_vals[row * stride + col];
            const auto amp_val = amp_vals[row * stride + col];
            const auto abs_ref = std::abs(ell_val);
            ASSERT_GT(abs_ref, real_type{1e-14});
            const auto rel_err = std::abs(amp_val - ell_val) / abs_ref;
            EXPECT_LE(rel_err, cmp_tol)
                << "Row " << row << " col " << col << ": amp=" << amp_val
                << ", ell=" << ell_val;
        }
    }
}


}  // namespace
