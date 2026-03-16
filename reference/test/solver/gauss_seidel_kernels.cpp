// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gauss_seidel_kernels.hpp"

#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
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

TYPED_TEST_SUITE(GaussSeidelKernel, gko::test::ValueIndexTypes,
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


}  // namespace
