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
        ell1 = Ell::create(exec);
        mtx1->convert_to(ell1.get());
        assert_equal_to_mtx1(ell1.get());
    }

    void assert_equal_to_mtx1(gko::ptr_param<const Ell> m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        const auto max_nnz_per_row = m->get_num_stored_elements_per_row();
        const auto stride = m->get_stride();

        ASSERT_EQ(m->get_size(), gko::dim<2>(5, 4));
        ASSERT_EQ(m->get_num_stored_elements_per_row(), 3);
        ASSERT_EQ(m->get_stride(), 5);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[5], 1);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[6], 2);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[7], gko::invalid_index<index_type>());
        EXPECT_EQ(c[3], 0);
        EXPECT_EQ(c[8], 2);
        EXPECT_EQ(c[4], 0);
        EXPECT_EQ(c[9], 2);
        EXPECT_EQ(c[10], 3);
        EXPECT_EQ(c[11], gko::invalid_index<index_type>());
        EXPECT_EQ(c[12], gko::invalid_index<index_type>());
        EXPECT_EQ(c[13], gko::invalid_index<index_type>());
        EXPECT_EQ(c[14], gko::invalid_index<index_type>());
        EXPECT_EQ(v[0], value_type{1.1});
        EXPECT_EQ(v[1], value_type{1.2e-11});
        EXPECT_EQ(v[2], value_type{0.8});
        EXPECT_EQ(v[3], value_type{1.2e-11});
        EXPECT_EQ(v[4], value_type{-2.0e-5});
        EXPECT_EQ(v[5], value_type{3.0e-9});
        EXPECT_EQ(v[6], value_type{2.0});
        EXPECT_EQ(v[7], value_type{0.0});
        EXPECT_EQ(v[8], value_type{1.6e-4});
        EXPECT_EQ(v[9], value_type{-2.0});
        EXPECT_EQ(v[10], value_type{4.5e-4});
        EXPECT_EQ(v[11], value_type{0.0});
        EXPECT_EQ(v[12], value_type{0.0});
        EXPECT_EQ(v[13], value_type{0.0});
        EXPECT_EQ(v[14], value_type{0.0});
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Dns> mtx1;
    std::unique_ptr<Dns> mtx2;
    std::unique_ptr<Ell> ell1;
};

using double_types = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(AMPDouble, double_types, TypenameNameGenerator);


TYPED_TEST(AMPDouble, GenerateComputesCorrectRowNorms)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<typename TestFixture::value_type>;
    static_assert(std::is_same<real_T, double>::value, "double only!");
    const float tol = 1e-10;
    gko::amp::array_prec<int, T> max_nnz;
    gko::array<real_T> rownorms(this->exec, this->ell1->get_size()[0]);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        rexec, this->ell1.get(), tol, max_nnz, rownorms);

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
    const float tol = 1e-10;
    gko::amp::array_prec<int, T> max_nnz;
    gko::array<real_T> rownorms(this->exec, this->ell1->get_size()[0]);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        rexec, this->ell1.get(), tol, max_nnz, rownorms);

    std::cout << "max nnz of bins are ";
    for (int i = 0; i < 3; i++) {
        std::cout << ", " << max_nnz[i];
    }
    std::cout << std::endl;
    EXPECT_EQ(max_nnz[0], 1);
    EXPECT_EQ(max_nnz[1], 1);
    EXPECT_EQ(max_nnz[2], 1);
}

TYPED_TEST(AMPDouble, GenerateEllScattersBinsCorrectly)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<typename TestFixture::value_type>;
    static_assert(std::is_same<real_T, double>::value, "double only!");
    static_assert(std::tuple_size<gko::amp::array_prec<int, T>>::value == 3,
                  "should be 3 available precisions");
    const float tol = 1e-10;
    // gko::kernels::amp::array_prec<int, T, int> max_nnz;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    auto max_nnzs = gko::amp::array_prec<int, T>{1, 1, 1};
    auto abins = gko::amp::allocate_bins<T, int>(
        this->exec, this->ell1->get_size(), max_nnzs);
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    gko::amp::array_prec<gko::LinOp*, T> amat;
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    gko::kernels::reference::amp::generate_ell_scatter_bins(
        rexec, this->ell1.get(), tol, amat);

    using types_list = typename gko::amp::narrow_types<T>::type;
    gko::constexpr_for<0, num_bins, 1>([&](auto k) {
        using value_type = typename std::tuple_element<k, types_list>::type;
        auto amat0 = dynamic_cast<gko::matrix::Ell<value_type, int>*>(amat[k]);
        EXPECT_TRUE(amat0);
        EXPECT_EQ(amat0->get_num_stored_elements_per_row(), 1);
    });
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
