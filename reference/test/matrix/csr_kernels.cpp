// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/csr.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


template <typename ValueIndexType>
class Csr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Sellp = gko::matrix::Sellp<value_type, index_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using Ell = gko::matrix::Ell<value_type, index_type>;
    using Hybrid = gko::matrix::Hybrid<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<gko::next_precision<value_type>>;
    using Perm = gko::matrix::Permutation<index_type>;
    using ScaledPerm = gko::matrix::ScaledPermutation<value_type, index_type>;

    Csr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4,
                          std::make_shared<typename Mtx::load_balance>(2))),
          mtx2(Mtx::create(exec, gko::dim<2>{2, 3}, 5,
                           std::make_shared<typename Mtx::classical>())),
          mtx3_sorted(Mtx::create(exec, gko::dim<2>(3, 3), 7,
                                  std::make_shared<typename Mtx::classical>())),
          mtx3_unsorted(
              Mtx::create(exec, gko::dim<2>(3, 3), 7,
                          std::make_shared<typename Mtx::classical>())),
          perm3(Perm::create(exec, gko::array<index_type>{exec, {1, 2, 0}})),
          perm3_rev(perm3->compute_inverse()),
          perm2(Perm::create(exec, gko::array<index_type>{exec, {1, 0}})),
          perm0(Perm::create(exec)),
          scale_perm3(ScaledPerm::create(
              exec, gko::array<value_type>{this->exec, {2.0, 3.0, 5.0}},
              gko::array<index_type>{exec, {1, 2, 0}})),
          scale_perm3_rev(ScaledPerm::create(
              exec, gko::array<value_type>{this->exec, {7.0, 11.0, 13.0}},
              gko::array<index_type>{exec, {1, 2, 0}})),
          scale_perm2(ScaledPerm::create(
              exec, gko::array<value_type>{this->exec, {17.0, 19.0}},
              gko::array<index_type>{exec, {1, 0}})),
          scale_perm0(ScaledPerm::create(exec))
    {
        this->create_mtx(mtx.get());
        this->create_mtx2(mtx2.get());
        this->create_mtx3(mtx3_sorted.get(), mtx3_unsorted.get());
    }

    void create_mtx(Mtx* m)
    {
        value_type* v = m->get_values();
        index_type* c = m->get_col_idxs();
        index_type* r = m->get_row_ptrs();
        auto* s = m->get_srow();
        /*
         * 1   3   2
         * 0   5   0
         */
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 5.0;
        s[0] = 0;
    }

    void create_mtx2(Mtx* m)
    {
        value_type* v = m->get_values();
        index_type* c = m->get_col_idxs();
        index_type* r = m->get_row_ptrs();
        // It keeps an explicit zero
        /*
         *  1    3   2
         * {0}   5   0
         */
        r[0] = 0;
        r[1] = 3;
        r[2] = 5;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 0;
        c[4] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 0.0;
        v[4] = 5.0;
    }

    void create_mtx3(Mtx* sorted, Mtx* unsorted)
    {
        auto vals_s = sorted->get_values();
        auto cols_s = sorted->get_col_idxs();
        auto rows_s = sorted->get_row_ptrs();
        auto vals_u = unsorted->get_values();
        auto cols_u = unsorted->get_col_idxs();
        auto rows_u = unsorted->get_row_ptrs();
        /* For both versions (sorted and unsorted), this matrix is stored:
         * 0  2  1
         * 3  1  8
         * 2  0  3
         * The unsorted matrix will have the (value, column) pair per row not
         * sorted, which we still consider a valid CSR format.
         */
        rows_s[0] = 0;
        rows_s[1] = 2;
        rows_s[2] = 5;
        rows_s[3] = 7;
        rows_u[0] = 0;
        rows_u[1] = 2;
        rows_u[2] = 5;
        rows_u[3] = 7;

        vals_s[0] = 2.;
        vals_s[1] = 1.;
        vals_s[2] = 3.;
        vals_s[3] = 1.;
        vals_s[4] = 8.;
        vals_s[5] = 2.;
        vals_s[6] = 3.;
        // Each row is stored rotated once to the left
        vals_u[0] = 1.;
        vals_u[1] = 2.;
        vals_u[2] = 1.;
        vals_u[3] = 8.;
        vals_u[4] = 3.;
        vals_u[5] = 3.;
        vals_u[6] = 2.;

        cols_s[0] = 1;
        cols_s[1] = 2;
        cols_s[2] = 0;
        cols_s[3] = 1;
        cols_s[4] = 2;
        cols_s[5] = 0;
        cols_s[6] = 2;
        // The same applies for the columns
        cols_u[0] = 2;
        cols_u[1] = 1;
        cols_u[2] = 1;
        cols_u[3] = 2;
        cols_u[4] = 0;
        cols_u[5] = 2;
        cols_u[6] = 0;
    }

    void assert_equal_to_mtx(const Coo* m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_idxs();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 0);
        EXPECT_EQ(r[2], 0);
        EXPECT_EQ(r[3], 1);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{3.0});
        EXPECT_EQ(v[2], value_type{2.0});
        EXPECT_EQ(v[3], value_type{5.0});
    }

    void assert_equal_to_mtx(const Sellp* m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto slice_sets = m->get_const_slice_sets();
        auto slice_lengths = m->get_const_slice_lengths();
        auto stride_factor = m->get_stride_factor();
        auto slice_size = m->get_slice_size();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(stride_factor, 1);
        ASSERT_EQ(slice_size, 64);
        EXPECT_EQ(slice_sets[0], 0);
        EXPECT_EQ(slice_lengths[0], 3);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[64], 1);
        EXPECT_EQ(c[65], this->invalid_index);
        EXPECT_EQ(c[128], 2);
        EXPECT_EQ(c[129], this->invalid_index);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{5.0});
        EXPECT_EQ(v[64], value_type{3.0});
        EXPECT_EQ(v[65], value_type{0.0});
        EXPECT_EQ(v[128], value_type{2.0});
        EXPECT_EQ(v[129], value_type{0.0});
    }

    void assert_equal_to_mtx(const SparsityCsr* m)
    {
        auto* c = m->get_const_col_idxs();
        auto* r = m->get_const_row_ptrs();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_nonzeros(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 3);
        EXPECT_EQ(r[2], 4);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
    }

    void assert_equal_to_mtx(const Ell* m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 6);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 1);
        EXPECT_EQ(c[3], invalid_index);
        EXPECT_EQ(c[4], 2);
        EXPECT_EQ(c[5], invalid_index);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{5.0});
        EXPECT_EQ(v[2], value_type{3.0});
        EXPECT_EQ(v[3], value_type{0.0});
        EXPECT_EQ(v[4], value_type{2.0});
        EXPECT_EQ(v[5], value_type{0.0});
    }

    void assert_equal_to_mtx(const Hybrid* m)
    {
        auto v = m->get_const_coo_values();
        auto c = m->get_const_coo_col_idxs();
        auto r = m->get_const_coo_row_idxs();
        auto n = m->get_ell_num_stored_elements_per_row();
        auto p = m->get_ell_stride();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_ell_num_stored_elements(), 0);
        ASSERT_EQ(m->get_coo_num_stored_elements(), 4);
        EXPECT_EQ(n, 0);
        EXPECT_EQ(p, 2);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 0);
        EXPECT_EQ(r[2], 0);
        EXPECT_EQ(r[3], 1);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{3.0});
        EXPECT_EQ(v[2], value_type{2.0});
        EXPECT_EQ(v[3], value_type{5.0});
    }

    void assert_equal_to_mtx2(const Hybrid* m)
    {
        auto v = m->get_const_coo_values();
        auto c = m->get_const_coo_col_idxs();
        auto r = m->get_const_coo_row_idxs();
        auto n = m->get_ell_num_stored_elements_per_row();
        auto p = m->get_ell_stride();
        auto ell_v = m->get_const_ell_values();
        auto ell_c = m->get_const_ell_col_idxs();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        // Test Coo values
        ASSERT_EQ(m->get_coo_num_stored_elements(), 1);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(c[0], 2);
        EXPECT_EQ(v[0], value_type{2.0});
        // Test Ell values
        ASSERT_EQ(m->get_ell_num_stored_elements(), 4);
        EXPECT_EQ(n, 2);
        EXPECT_EQ(p, 2);
        EXPECT_EQ(ell_v[0], value_type{1});
        EXPECT_EQ(ell_v[1], value_type{0});
        EXPECT_EQ(ell_v[2], value_type{3});
        EXPECT_EQ(ell_v[3], value_type{5});
        EXPECT_EQ(ell_c[0], 0);
        EXPECT_EQ(ell_c[1], 0);
        EXPECT_EQ(ell_c[2], 1);
        EXPECT_EQ(ell_c[3], 1);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3_sorted;
    std::unique_ptr<Mtx> mtx3_unsorted;
    std::unique_ptr<Perm> perm3;
    std::unique_ptr<Perm> perm3_rev;
    std::unique_ptr<Perm> perm2;
    std::unique_ptr<Perm> perm0;
    std::unique_ptr<ScaledPerm> scale_perm3;
    std::unique_ptr<ScaledPerm> scale_perm3_rev;
    std::unique_ptr<ScaledPerm> scale_perm2;
    std::unique_ptr<ScaledPerm> scale_perm0;
    index_type invalid_index = gko::invalid_index<index_type>();
};

TYPED_TEST_SUITE(Csr, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Csr, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx->apply(x, y);

    EXPECT_EQ(y->at(0), T{13.0});
    EXPECT_EQ(y->at(1), T{5.0});
}


TYPED_TEST(Csr, MixedAppliesToDenseVector1)
{
    // Both vectors have the same value type which differs from the matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec = typename gko::matrix::Dense<next_T>;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Csr, MixedAppliesToDenseVector2)
{
    // Input vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<next_T>;
    auto x = gko::initialize<Vec1>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec2::create(this->exec, gko::dim<2>{2, 1});

    this->mtx->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Csr, MixedAppliesToDenseVector3)
{
    // Output vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<gko::next_precision<T>>;
    auto x = gko::initialize<Vec2>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec1::create(this->exec, gko::dim<2>{2, 1});

    this->mtx->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Csr, AppliesToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0}, I<T>{1.0, -1.5}, I<T>{4.0, 2.5}}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    this->mtx->apply(x, y);

    EXPECT_EQ(y->at(0, 0), T{13.0});
    EXPECT_EQ(y->at(1, 0), T{5.0});
    EXPECT_EQ(y->at(0, 1), T{3.5});
    EXPECT_EQ(y->at(1, 1), T{-7.5});
}


TYPED_TEST(Csr, MixedAppliesToDenseMatrix1)
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

    this->mtx->apply(x, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Csr, MixedAppliesToDenseMatrix2)
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

    this->mtx->apply(x, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Csr, MixedAppliesToDenseMatrix3)
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

    this->mtx->apply(x, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Csr, AppliesLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx->apply(alpha, x, beta, y);

    EXPECT_EQ(y->at(0), T{-11.0});
    EXPECT_EQ(y->at(1), T{-1.0});
}


TYPED_TEST(Csr, MixedAppliesLinearCombinationToDenseVector1)
{
    // Both vectors have the same value type which differs from the matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec = gko::matrix::Dense<next_T>;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Csr, MixedAppliesLinearCombinationToDenseVector2)
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

    this->mtx->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Csr, MixedAppliesLinearCombinationToDenseVector3)
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

    this->mtx->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Csr, AppliesLinearCombinationToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0}, I<T>{1.0, -1.5}, I<T>{4.0, 2.5}}, this->exec);
    auto y =
        gko::initialize<Vec>({I<T>{1.0, 0.5}, I<T>{2.0, -1.5}}, this->exec);

    this->mtx->apply(alpha, x, beta, y);

    EXPECT_EQ(y->at(0, 0), T{-11.0});
    EXPECT_EQ(y->at(1, 0), T{-1.0});
    EXPECT_EQ(y->at(0, 1), T{-2.5});
    EXPECT_EQ(y->at(1, 1), T{4.5});
}


TYPED_TEST(Csr, MixedAppliesLinearCombinationToDenseMatrix1)
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

    this->mtx->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({{-11.0, -2.5}, {-1.0, 4.5}}), 0.0);
}


TYPED_TEST(Csr, MixedAppliesLinearCombinationToDenseMatrix2)
{
    // Input vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<next_T>;
    auto alpha = gko::initialize<Vec1>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec2>({2.0}, this->exec);
    auto x = gko::initialize<Vec1>(
        {I<T>{2.0, 3.0}, I<T>{1.0, -1.5}, I<T>{4.0, 2.5}}, this->exec);
    auto y = gko::initialize<Vec2>({I<next_T>{1.0, 0.5}, I<next_T>{2.0, -1.5}},
                                   this->exec);

    this->mtx->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({{-11.0, -2.5}, {-1.0, 4.5}}), 0.0);
}


TYPED_TEST(Csr, MixedAppliesLinearCombinationToDenseMatrix3)
{
    // Output vector has same value type as matrix
    using T = typename TestFixture::value_type;
    using next_T = gko::next_precision<T>;
    using Vec1 = typename TestFixture::Vec;
    using Vec2 = gko::matrix::Dense<next_T>;
    auto alpha = gko::initialize<Vec2>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec1>({2.0}, this->exec);
    auto x = gko::initialize<Vec2>(
        {I<next_T>{2.0, 3.0}, I<next_T>{1.0, -1.5}, I<next_T>{4.0, 2.5}},
        this->exec);
    auto y =
        gko::initialize<Vec1>({I<T>{1.0, 0.5}, I<T>{2.0, -1.5}}, this->exec);

    this->mtx->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({{-11.0, -2.5}, {-1.0, 4.5}}), 0.0);
}


TYPED_TEST(Csr, AppliesToCsrMatrix)
{
    using T = typename TestFixture::value_type;
    this->mtx->apply(this->mtx3_unsorted, this->mtx2);

    ASSERT_EQ(this->mtx2->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx2->get_num_stored_elements(), 6);
    ASSERT_TRUE(this->mtx2->is_sorted_by_column_index());
    auto r = this->mtx2->get_const_row_ptrs();
    auto c = this->mtx2->get_const_col_idxs();
    auto v = this->mtx2->get_const_values();
    // 13  5 31
    // 15  5 40
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 6);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 1);
    EXPECT_EQ(c[5], 2);
    EXPECT_EQ(v[0], T{13});
    EXPECT_EQ(v[1], T{5});
    EXPECT_EQ(v[2], T{31});
    EXPECT_EQ(v[3], T{15});
    EXPECT_EQ(v[4], T{5});
    EXPECT_EQ(v[5], T{40});
}


TYPED_TEST(Csr, AppliesLinearCombinationToCsrMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);

    this->mtx->apply(alpha, this->mtx3_unsorted, beta, this->mtx2);

    ASSERT_EQ(this->mtx2->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx2->get_num_stored_elements(), 6);
    ASSERT_TRUE(this->mtx2->is_sorted_by_column_index());
    auto r = this->mtx2->get_const_row_ptrs();
    auto c = this->mtx2->get_const_col_idxs();
    auto v = this->mtx2->get_const_values();
    // -11 1 -27
    // -15 5 -40
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 6);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 1);
    EXPECT_EQ(c[5], 2);
    EXPECT_EQ(v[0], T{-11});
    EXPECT_EQ(v[1], T{1});
    EXPECT_EQ(v[2], T{-27});
    EXPECT_EQ(v[3], T{-15});
    EXPECT_EQ(v[4], T{5});
    EXPECT_EQ(v[5], T{-40});
}


TYPED_TEST(Csr, AppliesLinearCombinationToIdentityMatrix)
{
    using T = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    auto alpha = gko::initialize<Vec>({-3.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto a = gko::initialize<Mtx>(
        {I<T>{2.0, 0.0, 3.0}, I<T>{0.0, 1.0, -1.5}, I<T>{0.0, -2.0, 0.0},
         I<T>{5.0, 0.0, 0.0}, I<T>{1.0, 0.0, 4.0}, I<T>{2.0, -2.0, 0.0},
         I<T>{0.0, 0.0, 0.0}},
        this->exec);
    auto b = gko::initialize<Mtx>(
        {I<T>{2.0, -2.0, 0.0}, I<T>{1.0, 0.0, 4.0}, I<T>{2.0, 0.0, 3.0},
         I<T>{0.0, 1.0, -1.5}, I<T>{1.0, 0.0, 0.0}, I<T>{0.0, 0.0, 0.0},
         I<T>{0.0, 0.0, 0.0}},
        this->exec);
    auto expect = gko::initialize<Mtx>(
        {I<T>{-2.0, -4.0, -9.0}, I<T>{2.0, -3.0, 12.5}, I<T>{4.0, 6.0, 6.0},
         I<T>{-15.0, 2.0, -3.0}, I<T>{-1.0, 0.0, -12.0}, I<T>{-6.0, 6.0, 0.0},
         I<T>{0.0, 0.0, 0.0}},
        this->exec);
    auto id = gko::matrix::Identity<T>::create(this->exec, a->get_size()[1]);

    a->apply(alpha, id, beta, b);

    GKO_ASSERT_MTX_NEAR(b, expect, r<T>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(b, expect);
    ASSERT_TRUE(b->is_sorted_by_column_index());
}


TYPED_TEST(Csr, ApplyFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Csr, ApplyFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Csr, ApplyFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Csr, ConvertsToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Csr = typename TestFixture::Mtx;
    using OtherCsr = gko::matrix::Csr<OtherType, IndexType>;
    auto tmp = OtherCsr::create(this->exec);
    auto res = Csr::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    // use mtx2 as mtx's strategy would involve creating a CudaExecutor
    this->mtx2->convert_to(tmp);
    tmp->convert_to(res);

    GKO_ASSERT_MTX_NEAR(this->mtx2, res, residual);
    auto first_strategy = this->mtx2->get_strategy();
    auto second_strategy = res->get_strategy();
    GKO_ASSERT_DYNAMIC_TYPE_EQ(first_strategy, second_strategy);
}


TYPED_TEST(Csr, MovesToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Csr = typename TestFixture::Mtx;
    using OtherCsr = gko::matrix::Csr<OtherType, IndexType>;
    auto tmp = OtherCsr::create(this->exec);
    auto res = Csr::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    // use mtx2 as mtx's strategy would involve creating a CudaExecutor
    this->mtx2->move_to(tmp);
    tmp->move_to(res);

    GKO_ASSERT_MTX_NEAR(this->mtx2, res, residual);
    auto first_strategy = this->mtx2->get_strategy();
    auto second_strategy = res->get_strategy();
    GKO_ASSERT_DYNAMIC_TYPE_EQ(first_strategy, second_strategy);
}


TYPED_TEST(Csr, ConvertsToDense)
{
    using Dense = typename TestFixture::Vec;
    auto dense_mtx = Dense::create(this->mtx->get_executor());
    auto dense_other = gko::initialize<Dense>(
        4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, this->exec);

    this->mtx->convert_to(dense_mtx);

    GKO_ASSERT_MTX_NEAR(dense_mtx, dense_other, 0.0);
}


TYPED_TEST(Csr, MovesToDense)
{
    using Dense = typename TestFixture::Vec;
    auto dense_mtx = Dense::create(this->mtx->get_executor());
    auto dense_other = gko::initialize<Dense>(
        4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, this->exec);

    this->mtx->move_to(dense_mtx);

    GKO_ASSERT_MTX_NEAR(dense_mtx, dense_other, 0.0);
}


TYPED_TEST(Csr, ConvertsToCoo)
{
    using Coo = typename TestFixture::Coo;
    auto coo_mtx = Coo::create(this->mtx->get_executor());

    this->mtx->convert_to(coo_mtx);

    this->assert_equal_to_mtx(coo_mtx.get());
}


TYPED_TEST(Csr, MovesToCoo)
{
    using Coo = typename TestFixture::Coo;
    auto coo_mtx = Coo::create(this->mtx->get_executor());

    this->mtx->move_to(coo_mtx);

    this->assert_equal_to_mtx(coo_mtx.get());
}


TYPED_TEST(Csr, ConvertsToSellp)
{
    using Sellp = typename TestFixture::Sellp;
    auto sellp_mtx = Sellp::create(this->mtx->get_executor());

    this->mtx->convert_to(sellp_mtx);

    this->assert_equal_to_mtx(sellp_mtx.get());
}


TYPED_TEST(Csr, MovesToSellp)
{
    using Sellp = typename TestFixture::Sellp;
    using Csr = typename TestFixture::Mtx;
    auto sellp_mtx = Sellp::create(this->mtx->get_executor());
    auto csr_ref = Csr::create(this->mtx->get_executor());

    csr_ref->copy_from(this->mtx);
    csr_ref->move_to(sellp_mtx);

    this->assert_equal_to_mtx(sellp_mtx.get());
}


TYPED_TEST(Csr, ConvertsToSparsityCsr)
{
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto sparsity_mtx = SparsityCsr::create(this->mtx->get_executor());

    this->mtx->convert_to(sparsity_mtx);

    this->assert_equal_to_mtx(sparsity_mtx.get());
}


TYPED_TEST(Csr, MovesToSparsityCsr)
{
    using SparsityCsr = typename TestFixture::SparsityCsr;
    using Csr = typename TestFixture::Mtx;
    auto sparsity_mtx = SparsityCsr::create(this->mtx->get_executor());
    auto csr_ref = Csr::create(this->mtx->get_executor());

    csr_ref->copy_from(this->mtx);
    csr_ref->move_to(sparsity_mtx);

    this->assert_equal_to_mtx(sparsity_mtx.get());
}


TYPED_TEST(Csr, ConvertsToHybridAutomatically)
{
    using Hybrid = typename TestFixture::Hybrid;
    auto hybrid_mtx = Hybrid::create(this->mtx->get_executor());

    this->mtx->convert_to(hybrid_mtx);

    this->assert_equal_to_mtx(hybrid_mtx.get());
}


TYPED_TEST(Csr, MovesToHybridAutomatically)
{
    using Hybrid = typename TestFixture::Hybrid;
    using Csr = typename TestFixture::Mtx;
    auto hybrid_mtx = Hybrid::create(this->mtx->get_executor());
    auto csr_ref = Csr::create(this->mtx->get_executor());

    csr_ref->copy_from(this->mtx);
    csr_ref->move_to(hybrid_mtx);

    this->assert_equal_to_mtx(hybrid_mtx.get());
}


TYPED_TEST(Csr, ConvertsToHybridByColumn2)
{
    using Hybrid = typename TestFixture::Hybrid;
    auto hybrid_mtx =
        Hybrid::create(this->mtx2->get_executor(),
                       std::make_shared<typename Hybrid::column_limit>(2));

    this->mtx2->convert_to(hybrid_mtx);

    this->assert_equal_to_mtx2(hybrid_mtx.get());
}


TYPED_TEST(Csr, MovesToHybridByColumn2)
{
    using Hybrid = typename TestFixture::Hybrid;
    using Csr = typename TestFixture::Mtx;
    auto hybrid_mtx =
        Hybrid::create(this->mtx2->get_executor(),
                       std::make_shared<typename Hybrid::column_limit>(2));
    auto csr_ref = Csr::create(this->mtx2->get_executor());

    csr_ref->copy_from(this->mtx2);
    csr_ref->move_to(hybrid_mtx);

    this->assert_equal_to_mtx2(hybrid_mtx.get());
}


TYPED_TEST(Csr, ConvertsEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Csr = typename TestFixture::Mtx;
    using OtherCsr = gko::matrix::Csr<OtherType, IndexType>;
    auto empty = OtherCsr::create(this->exec);
    empty->get_row_ptrs()[0] = 0;
    auto res = Csr::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, MovesEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Csr = typename TestFixture::Mtx;
    using OtherCsr = gko::matrix::Csr<OtherType, IndexType>;
    auto empty = OtherCsr::create(this->exec);
    empty->get_row_ptrs()[0] = 0;
    auto res = Csr::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, ConvertsEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using Csr = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty = Csr::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->convert_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, MovesEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using Csr = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty = Csr::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->move_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, ConvertsEmptyToCoo)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;
    auto empty = Csr::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, MovesEmptyToCoo)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;
    auto empty = Csr::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, ConvertsEmptyToEll)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Mtx;
    using Ell = gko::matrix::Ell<ValueType, IndexType>;
    auto empty = Csr::create(this->exec);
    auto res = Ell::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, MovesEmptyToEll)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Mtx;
    using Ell = gko::matrix::Ell<ValueType, IndexType>;
    auto empty = Csr::create(this->exec);
    auto res = Ell::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, ConvertsEmptyToSellp)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Mtx;
    using Sellp = gko::matrix::Sellp<ValueType, IndexType>;
    auto empty = Csr::create(this->exec);
    auto res = Sellp::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, MovesEmptyToSellp)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Mtx;
    using Sellp = gko::matrix::Sellp<ValueType, IndexType>;
    auto empty = Csr::create(this->exec);
    auto res = Sellp::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, ConvertsEmptyToSparsityCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Mtx;
    using SparsityCsr = gko::matrix::SparsityCsr<ValueType, IndexType>;
    auto empty = Csr::create(this->exec);
    empty->get_row_ptrs()[0] = 0;
    auto res = SparsityCsr::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
}


TYPED_TEST(Csr, MovesEmptyToSparsityCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Mtx;
    using SparsityCsr = gko::matrix::SparsityCsr<ValueType, IndexType>;
    auto empty = Csr::create(this->exec);
    empty->get_row_ptrs()[0] = 0;
    auto res = SparsityCsr::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
}


TYPED_TEST(Csr, ConvertsEmptyToHybrid)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Mtx;
    using Hybrid = gko::matrix::Hybrid<ValueType, IndexType>;
    auto empty = Csr::create(this->exec);
    auto res = Hybrid::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, MovesEmptyToHybrid)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Mtx;
    using Hybrid = gko::matrix::Hybrid<ValueType, IndexType>;
    auto empty = Csr::create(this->exec);
    auto res = Hybrid::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Csr, ConvertsToEll)
{
    using Ell = typename TestFixture::Ell;
    using Dense = typename TestFixture::Vec;
    auto ell_mtx = Ell::create(this->mtx->get_executor());
    auto dense_mtx = Dense::create(this->mtx->get_executor());
    auto ref_dense_mtx = Dense::create(this->mtx->get_executor());

    this->mtx->convert_to(ell_mtx);

    this->assert_equal_to_mtx(ell_mtx.get());
}


TYPED_TEST(Csr, MovesToEll)
{
    using Ell = typename TestFixture::Ell;
    using Dense = typename TestFixture::Vec;
    auto ell_mtx = Ell::create(this->mtx->get_executor());
    auto dense_mtx = Dense::create(this->mtx->get_executor());
    auto ref_dense_mtx = Dense::create(this->mtx->get_executor());

    this->mtx->move_to(ell_mtx);

    this->assert_equal_to_mtx(ell_mtx.get());
}


TYPED_TEST(Csr, SquareMtxIsTransposable)
{
    using Csr = typename TestFixture::Mtx;
    // clang-format off
    auto mtx2 = gko::initialize<Csr>(
                {{1.0, 3.0, 2.0},
                 {0.0, 5.0, 0.0},
                 {0.0, 1.5, 2.0}}, this->exec);
    // clang-format on

    auto trans_as_csr = gko::as<Csr>(mtx2->transpose());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(trans_as_csr,
                    l({{1.0, 0.0, 0.0},
                       {3.0, 5.0, 1.5},
                       {2.0, 0.0, 2.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Csr, NonSquareMtxIsTransposable)
{
    using Csr = typename TestFixture::Mtx;
    auto trans_as_csr = gko::as<Csr>(this->mtx->transpose());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(trans_as_csr,
                    l({{1.0, 0.0},
                       {3.0, 5.0},
                       {2.0, 0.0}}), 0.0);
    // clang-format on
}


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> csr_from_permutation(
    gko::matrix::Permutation<IndexType>* perm, bool invert)
{
    gko::matrix_data<double, IndexType> double_data;
    if (invert) {
        perm->compute_inverse()->write(double_data);
    } else {
        perm->write(double_data);
    }
    gko::matrix_data<ValueType, IndexType> data;
    data.size = double_data.size;
    for (auto entry : double_data.nonzeros) {
        data.nonzeros.emplace_back(entry.row, entry.column, 1.0);
    }
    auto mtx =
        gko::matrix::Csr<ValueType, IndexType>::create(perm->get_executor());
    mtx->read(data);
    return mtx;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> csr_from_permutation(
    gko::matrix::ScaledPermutation<ValueType, IndexType>* perm, bool invert)
{
    gko::matrix_data<ValueType, IndexType> data;
    if (invert) {
        perm->compute_inverse()->write(data);
    } else {
        perm->write(data);
    }
    auto mtx =
        gko::matrix::Csr<ValueType, IndexType>::create(perm->get_executor());
    mtx->read(data);
    return mtx;
}


template <typename ValueType, typename IndexType, typename Permutation>
std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> ref_permute(
    gko::matrix::Csr<ValueType, IndexType>* input, Permutation* permutation,
    gko::matrix::permute_mode mode)
{
    using gko::matrix::permute_mode;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto result = input->clone();
    auto permutation_csr = csr_from_permutation<ValueType>(
        permutation, (mode & permute_mode::inverse) == permute_mode::inverse);
    if ((mode & permute_mode::rows) == permute_mode::rows) {
        // compute P * A
        permutation_csr->apply(input, result);
    }
    if ((mode & permute_mode::columns) == permute_mode::columns) {
        // compute A * P^T = (P * A^T)^T
        auto tmp = result->transpose();
        auto tmp2 = gko::as<Csr>(tmp->clone());
        permutation_csr->apply(tmp, tmp2);
        result = gko::as<Csr>(tmp2->transpose());
    }
    return result;
}


template <typename ValueType, typename IndexType, typename Permutation>
std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> ref_permute(
    gko::matrix::Csr<ValueType, IndexType>* input, Permutation* row_permutation,
    Permutation* col_permutation, bool invert)
{
    using gko::matrix::permute_mode;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto result = input->clone();
    auto row_permutation_csr =
        csr_from_permutation<ValueType>(row_permutation, invert);
    auto col_permutation_csr =
        csr_from_permutation<ValueType>(col_permutation, invert);
    row_permutation_csr->apply(input, result);
    auto tmp = result->transpose();
    auto tmp2 = gko::as<Csr>(tmp->clone());
    col_permutation_csr->apply(tmp, tmp2);
    return gko::as<Csr>(tmp2->transpose());
}


TYPED_TEST(Csr, Permute)
{
    using gko::matrix::permute_mode;

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        auto permuted = this->mtx3_sorted->permute(this->perm3, mode);
        auto ref_permuted =
            ref_permute(this->mtx3_sorted.get(), this->perm3.get(), mode);

        GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
        ASSERT_TRUE(permuted->is_sorted_by_column_index());
    }
}


TYPED_TEST(Csr, PermuteRoundtrip)
{
    using gko::matrix::permute_mode;

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::symmetric,
          permute_mode::inverse_rows, permute_mode::inverse_columns,
          permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        auto permuted =
            this->mtx3_sorted->permute(this->perm3, mode)
                ->permute(this->perm3, mode ^ permute_mode::inverse);

        GKO_ASSERT_MTX_NEAR(this->mtx3_sorted, permuted, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, this->mtx3_sorted);
        ASSERT_TRUE(permuted->is_sorted_by_column_index());
    }
}


TYPED_TEST(Csr, PermuteInverted)
{
    using gko::matrix::permute_mode;

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::symmetric}) {
        SCOPED_TRACE(mode);

        auto permuted = this->mtx3_sorted->permute(this->perm3, mode);
        auto inv_inv_permuted = this->mtx3_sorted->permute(
            this->perm3->compute_inverse(), mode | permute_mode::inverse);

        GKO_ASSERT_MTX_NEAR(permuted, inv_inv_permuted, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, inv_inv_permuted);
        ASSERT_TRUE(permuted->is_sorted_by_column_index());
        ASSERT_TRUE(inv_inv_permuted->is_sorted_by_column_index());
    }
}


TYPED_TEST(Csr, PermuteRectangular)
{
    using gko::matrix::permute_mode;

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::inverse_rows,
          permute_mode::inverse_columns}) {
        auto perm = (mode & permute_mode::rows) == permute_mode::rows
                        ? this->perm2.get()
                        : this->perm3.get();

        auto permuted = this->mtx2->permute(perm, mode);
        auto ref_permuted = ref_permute(this->mtx2.get(), perm, mode);

        GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
        ASSERT_TRUE(permuted->is_sorted_by_column_index());
    }
}


TYPED_TEST(Csr, PermuteFailsWithIncorrectPermutationSize)
{
    using gko::matrix::permute_mode;

    for (auto mode :
         {/* no permute_mode::none */ permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        ASSERT_THROW(this->mtx3_sorted->permute(this->perm0, mode),
                     gko::DimensionMismatch);
    }
}


TYPED_TEST(Csr, NonsymmPermute)
{
    auto permuted = this->mtx3_sorted->permute(this->perm3, this->perm3_rev);
    auto ref_permuted = ref_permute(this->mtx3_sorted.get(), this->perm3.get(),
                                    this->perm3_rev.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmPermuteInverse)
{
    auto permuted =
        this->mtx3_sorted->permute(this->perm3, this->perm3_rev, true);
    auto ref_permuted = ref_permute(this->mtx3_sorted.get(), this->perm3.get(),
                                    this->perm3_rev.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmPermuteRectangular)
{
    auto permuted = this->mtx2->permute(this->perm2, this->perm3);
    auto ref_permuted = ref_permute(this->mtx2.get(), this->perm2.get(),
                                    this->perm3.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmPermuteInverseRectangular)
{
    auto permuted = this->mtx2->permute(this->perm2, this->perm3, true);
    auto ref_permuted = ref_permute(this->mtx2.get(), this->perm2.get(),
                                    this->perm3.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmPermuteRoundtrip)
{
    auto permuted = this->mtx3_sorted->permute(this->perm3, this->perm3_rev)
                        ->permute(this->perm3, this->perm3_rev, true);

    GKO_ASSERT_MTX_NEAR(this->mtx3_sorted, permuted, 0.0);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, this->mtx3_sorted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmPermuteInverted)
{
    auto permuted = this->mtx3_sorted->permute(this->perm3, this->perm3_rev);
    auto inv_inv_permuted =
        this->mtx3_sorted->permute(this->perm3->compute_inverse(),
                                   this->perm3_rev->compute_inverse(), true);

    GKO_ASSERT_MTX_NEAR(permuted, inv_inv_permuted, 0.0);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, inv_inv_permuted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
    ASSERT_TRUE(inv_inv_permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmPermuteFailsWithIncorrectPermutationSize)
{
    ASSERT_THROW(this->mtx3_sorted->permute(this->perm0, this->perm3_rev),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->mtx3_sorted->permute(this->perm3_rev, this->perm0),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->mtx3_sorted->permute(this->perm0, this->perm0),
                 gko::DimensionMismatch);
}


TYPED_TEST(Csr, ScaledPermute)
{
    using gko::matrix::permute_mode;
    using value_type = typename TestFixture::value_type;

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        auto permuted =
            this->mtx3_sorted->scale_permute(this->scale_perm3, mode);
        auto ref_permuted =
            ref_permute(this->mtx3_sorted.get(), this->scale_perm3.get(), mode);

        GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
        ASSERT_TRUE(permuted->is_sorted_by_column_index());
    }
}


TYPED_TEST(Csr, ScaledPermuteRoundtrip)
{
    using gko::matrix::permute_mode;
    using value_type = typename TestFixture::value_type;

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::symmetric,
          permute_mode::inverse_rows, permute_mode::inverse_columns,
          permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        auto permuted =
            this->mtx3_sorted->scale_permute(this->scale_perm3, mode)
                ->scale_permute(this->scale_perm3,
                                mode ^ permute_mode::inverse);

        GKO_ASSERT_MTX_NEAR(this->mtx3_sorted, permuted, r<value_type>::value);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, this->mtx3_sorted);
        ASSERT_TRUE(permuted->is_sorted_by_column_index());
    }
}


TYPED_TEST(Csr, ScaledPermuteRectangular)
{
    using gko::matrix::permute_mode;
    using value_type = typename TestFixture::value_type;

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::inverse_rows,
          permute_mode::inverse_columns}) {
        auto perm = (mode & permute_mode::rows) == permute_mode::rows
                        ? this->scale_perm2.get()
                        : this->scale_perm3.get();

        auto permuted = this->mtx2->scale_permute(perm, mode);
        auto ref_permuted = ref_permute(this->mtx2.get(), perm, mode);

        GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
        ASSERT_TRUE(permuted->is_sorted_by_column_index());
    }
}


TYPED_TEST(Csr, ScaledPermuteFailsWithIncorrectPermutationSize)
{
    using gko::matrix::permute_mode;

    for (auto mode :
         {/* no permute_mode::none */ permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        ASSERT_THROW(this->mtx3_sorted->scale_permute(this->scale_perm0, mode),
                     gko::DimensionMismatch);
    }
}


TYPED_TEST(Csr, NonsymmScaledPermute)
{
    using value_type = typename TestFixture::value_type;

    auto permuted = this->mtx3_sorted->scale_permute(this->scale_perm3,
                                                     this->scale_perm3_rev);
    auto ref_permuted =
        ref_permute(this->mtx3_sorted.get(), this->scale_perm3.get(),
                    this->scale_perm3_rev.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmScaledPermuteInverse)
{
    using value_type = typename TestFixture::value_type;

    auto permuted = this->mtx3_sorted->scale_permute(
        this->scale_perm3, this->scale_perm3_rev, true);
    auto ref_permuted =
        ref_permute(this->mtx3_sorted.get(), this->scale_perm3.get(),
                    this->scale_perm3_rev.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmScaledPermuteRectangular)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx2->scale_permute(this->scale_perm2, this->scale_perm3);
    auto ref_permuted = ref_permute(this->mtx2.get(), this->scale_perm2.get(),
                                    this->scale_perm3.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmScaledPermuteInverseRectangular)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx2->scale_permute(this->scale_perm2, this->scale_perm3, true);
    auto ref_permuted = ref_permute(this->mtx2.get(), this->scale_perm2.get(),
                                    this->scale_perm3.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, ref_permuted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmScaledPermuteRoundtrip)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx3_sorted
            ->scale_permute(this->scale_perm3, this->scale_perm3_rev)
            ->scale_permute(this->scale_perm3, this->scale_perm3_rev, true);

    GKO_ASSERT_MTX_NEAR(this->mtx3_sorted, permuted, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, this->mtx3_sorted);
    ASSERT_TRUE(permuted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, NonsymmScaledPermuteFailsWithIncorrectPermutationSize)
{
    ASSERT_THROW(this->mtx3_sorted->scale_permute(this->scale_perm0,
                                                  this->scale_perm3_rev),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->mtx3_sorted->scale_permute(this->scale_perm3_rev,
                                                  this->scale_perm0),
                 gko::DimensionMismatch);
    ASSERT_THROW(
        this->mtx3_sorted->scale_permute(this->scale_perm0, this->scale_perm0),
        gko::DimensionMismatch);
}


TYPED_TEST(Csr, SquareMatrixIsPermutable)
{
    using Csr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto p_mtx = gko::initialize<Csr>({{1.0, 3.0, 2.0},
                                       {0.0, 5.0, 0.0},
                                       {0.0, 1.5, 2.0}}, this->exec);
    // clang-format on
    gko::array<index_type> permute_idxs{this->exec, {1, 2, 0}};

    auto ref_permute_csr =
        gko::as<Csr>(gko::as<Csr>(p_mtx->row_permute(&permute_idxs))
                         ->column_permute(&permute_idxs));
    auto permute_csr = gko::as<Csr>(p_mtx->permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(ref_permute_csr, permute_csr, 0.0);
}


TYPED_TEST(Csr, SquareMatrixIsInversePermutable)
{
    using Csr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto p_mtx = gko::initialize<Csr>({{1.0, 3.0, 2.0},
                                       {0.0, 5.0, 0.0},
                                       {0.0, 1.5, 2.0}}, this->exec);
    // clang-format on
    gko::array<index_type> permute_idxs{this->exec, {1, 2, 0}};

    auto ref_permute_csr =
        gko::as<Csr>(gko::as<Csr>(p_mtx->inverse_row_permute(&permute_idxs))
                         ->inverse_column_permute(&permute_idxs));
    auto permute_csr = gko::as<Csr>(p_mtx->inverse_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(ref_permute_csr, permute_csr, 0.0);
}


TYPED_TEST(Csr, SquareMatrixIsRowPermutable)
{
    using Csr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto p_mtx = gko::initialize<Csr>({{1.0, 3.0, 2.0},
                                       {0.0, 5.0, 0.0},
                                       {0.0, 1.5, 2.0}}, this->exec);
    // clang-format on
    gko::array<index_type> permute_idxs{this->exec, {1, 2, 0}};

    auto row_permute_csr = gko::as<Csr>(p_mtx->row_permute(&permute_idxs));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_permute_csr,
                        l({{0.0, 5.0, 0.0},
                           {0.0, 1.5, 2.0},
                           {1.0, 3.0, 2.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Csr, NonSquareMatrixIsRowPermutable)
{
    using Csr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto p_mtx = gko::initialize<Csr>({{1.0, 3.0, 2.0},
                                       {0.0, 5.0, 0.0}}, this->exec);
    // clang-format on
    gko::array<index_type> permute_idxs{this->exec, {1, 0}};

    auto row_permute_csr = gko::as<Csr>(p_mtx->row_permute(&permute_idxs));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_permute_csr,
                        l({{0.0, 5.0, 0.0},
                           {1.0, 3.0, 2.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Csr, SquareMatrixIsColPermutable)
{
    using Csr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto p_mtx = gko::initialize<Csr>({{1.0, 3.0, 2.0},
                                       {0.0, 5.0, 0.0},
                                       {0.0, 1.5, 2.0}}, this->exec);
    // clang-format on
    gko::array<index_type> permute_idxs{this->exec, {1, 2, 0}};

    auto c_permute_csr = gko::as<Csr>(p_mtx->column_permute(&permute_idxs));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(c_permute_csr,
                        l({{3.0, 2.0, 1.0},
                           {5.0, 0.0, 0.0},
                           {1.5, 2.0, 0.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Csr, NonSquareMatrixIsColPermutable)
{
    using Csr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto p_mtx = gko::initialize<Csr>({{1.0, 0.0, 2.0},
                                       {0.0, 5.0, 0.0}}, this->exec);
    // clang-format on
    gko::array<index_type> permute_idxs{this->exec, {1, 2, 0}};

    auto c_permute_csr = gko::as<Csr>(p_mtx->column_permute(&permute_idxs));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(c_permute_csr,
                        l({{0.0, 2.0, 1.0},
                           {5.0, 0.0, 0.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Csr, SquareMatrixIsInverseRowPermutable)
{
    using Csr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto inverse_p_mtx = gko::initialize<Csr>({{1.0, 3.0, 2.0},
                                               {0.0, 5.0, 0.0},
                                               {0.0, 1.5, 2.0}}, this->exec);
    // clang-format on
    gko::array<index_type> inverse_permute_idxs{this->exec, {1, 2, 0}};

    auto inverse_row_permute_csr =
        gko::as<Csr>(inverse_p_mtx->inverse_row_permute(&inverse_permute_idxs));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_row_permute_csr,
                        l({{0.0, 1.5, 2.0},
                           {1.0, 3.0, 2.0},
                           {0.0, 5.0, 0.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Csr, NonSquareMatrixIsInverseRowPermutable)
{
    using Csr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto inverse_p_mtx = gko::initialize<Csr>({{1.0, 3.0, 2.0},
                                               {0.0, 5.0, 0.0}}, this->exec);
    // clang-format on
    gko::array<index_type> inverse_permute_idxs{this->exec, {1, 0}};

    auto inverse_row_permute_csr =
        gko::as<Csr>(inverse_p_mtx->inverse_row_permute(&inverse_permute_idxs));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_row_permute_csr,
                        l({{0.0, 5.0, 0.0},
                           {1.0, 3.0, 2.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Csr, SquareMatrixIsInverseColPermutable)
{
    using Csr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto inverse_p_mtx = gko::initialize<Csr>({{1.0, 3.0, 2.0},
                                               {0.0, 5.0, 0.0},
                                               {0.0, 1.5, 2.0}}, this->exec);
    // clang-format on
    gko::array<index_type> inverse_permute_idxs{this->exec, {1, 2, 0}};

    auto inverse_c_permute_csr = gko::as<Csr>(
        inverse_p_mtx->inverse_column_permute(&inverse_permute_idxs));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_c_permute_csr,
                        l({{2.0, 1.0, 3.0},
                           {0.0, 0.0, 5.0},
                           {2.0, 0.0, 1.5}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Csr, NonSquareMatrixIsInverseColPermutable)
{
    using Csr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto inverse_p_mtx = gko::initialize<Csr>({{1.0, 3.0, 2.0},
                                              {0.0, 5.0, 0.0}}, this->exec);
    // clang-format on
    gko::array<index_type> inverse_permute_idxs{this->exec, {1, 2, 0}};

    auto inverse_c_permute_csr = gko::as<Csr>(
        inverse_p_mtx->inverse_column_permute(&inverse_permute_idxs));

    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_c_permute_csr,
                        l({{2.0, 1.0, 3.0},
                           {0.0, 0.0, 5.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Csr, RecognizeSortedMatrix)
{
    ASSERT_TRUE(this->mtx->is_sorted_by_column_index());
    ASSERT_TRUE(this->mtx2->is_sorted_by_column_index());
    ASSERT_TRUE(this->mtx3_sorted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, RecognizeUnsortedMatrix)
{
    ASSERT_FALSE(this->mtx3_unsorted->is_sorted_by_column_index());
}


TYPED_TEST(Csr, SortSortedMatrix)
{
    auto matrix = this->mtx3_sorted->clone();

    matrix->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(matrix, this->mtx3_sorted, 0.0);
}


TYPED_TEST(Csr, SortUnsortedMatrix)
{
    auto matrix = this->mtx3_unsorted->clone();

    matrix->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(matrix, this->mtx3_sorted, 0.0);
}


TYPED_TEST(Csr, ExtractsDiagonal)
{
    using T = typename TestFixture::value_type;
    auto matrix = this->mtx3_unsorted->clone();
    auto diag = matrix->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 3);
    ASSERT_EQ(diag->get_size()[1], 3);
    ASSERT_EQ(diag->get_values()[0], T{0.});
    ASSERT_EQ(diag->get_values()[1], T{1.});
    ASSERT_EQ(diag->get_values()[2], T{3.});
}


TYPED_TEST(Csr, InplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec);

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
}


TYPED_TEST(Csr, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec);

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
    ASSERT_EQ(mtx->get_strategy()->get_name(),
              abs_mtx->get_strategy()->get_name());
}


TYPED_TEST(Csr, AppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Vec = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{13.0, 14.0}, complex_type{19.0, 20.0}},
           {complex_type{10.0, 10.0}, complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Csr, AppliesToMixedComplex)
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

    this->mtx->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{13.0, 14.0}, mixed_complex_type{19.0, 20.0}},
           {mixed_complex_type{10.0, 10.0}, mixed_complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Csr, AdvancedAppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
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

    this->mtx->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{-11.0, -14.0}, complex_type{-15.0, -18.0}},
           {complex_type{-6.0, -6.0}, complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Csr, AdvancedAppliesToMixedComplex)
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

    this->mtx->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{-11.0, -14.0}, mixed_complex_type{-15.0, -18.0}},
           {mixed_complex_type{-6.0, -6.0}, mixed_complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Csr, ScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Dense = gko::matrix::Dense<T>;
    auto alpha = gko::initialize<Dense>({I<T>{2.0}}, this->exec);
    auto to_scale = gko::clone(this->mtx2);

    to_scale->scale(alpha);

    GKO_ASSERT_MTX_EQ_SPARSITY(to_scale, this->mtx2);
    EXPECT_EQ(to_scale->get_values()[0], T{2.0});
    EXPECT_EQ(to_scale->get_values()[1], T{6.0});
    EXPECT_EQ(to_scale->get_values()[2], T{4.0});
    EXPECT_EQ(to_scale->get_values()[3], T{0.0});
    EXPECT_EQ(to_scale->get_values()[4], T{10.0});
}


TYPED_TEST(Csr, InvScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Dense = gko::matrix::Dense<T>;
    auto alpha = gko::initialize<Dense>({I<T>{2.0}}, this->exec);
    auto to_scale = gko::clone(this->mtx2);

    to_scale->inv_scale(alpha);

    GKO_ASSERT_MTX_EQ_SPARSITY(to_scale, this->mtx2);
    EXPECT_EQ(to_scale->get_values()[0], T{0.5});
    EXPECT_EQ(to_scale->get_values()[1], T{1.5});
    EXPECT_EQ(to_scale->get_values()[2], T{1.0});
    EXPECT_EQ(to_scale->get_values()[3], T{0.0});
    EXPECT_EQ(to_scale->get_values()[4], T{2.5});
}


TYPED_TEST(Csr, CanDetectMissingDiagonalEntry)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    using Csr = typename TestFixture::Mtx;
    auto b = gko::initialize<Csr>(
        {I<T>{2.0, 0.0, 1.1}, I<T>{1.0, 0.0, 2.5}, I<T>{0.0, -4.0, 1.0}},
        this->exec);
    bool has_diags{};

    gko::kernels::reference::csr::check_diagonal_entries_exist(
        this->exec, b.get(), has_diags);

    ASSERT_FALSE(has_diags);
}


TYPED_TEST(Csr, CanDetectWhenAllDiagonalEntriesArePresent)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    using Csr = typename TestFixture::Mtx;
    auto b = gko::initialize<Csr>({I<T>{2.0, 0.0, 1.1}, I<T>{1.0, -2.0, 2.5},
                                   I<T>{0.0, -4.0, 1.0}, I<T>{1.1, -3.0, 1.5}},
                                  this->exec);
    bool has_diags{};

    gko::kernels::reference::csr::check_diagonal_entries_exist(
        this->exec, b.get(), has_diags);

    ASSERT_TRUE(has_diags);
}


TYPED_TEST(Csr, ScaleCsrAddIdentityRectangular)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    using Csr = typename TestFixture::Mtx;
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto b = gko::initialize<Csr>(
        {I<T>{2.0, 0.0}, I<T>{1.0, 2.5}, I<T>{0.0, -4.0}}, this->exec);

    b->add_scaled_identity(alpha.get(), beta.get());

    GKO_ASSERT_MTX_NEAR(b, l({{0.0, 0.0}, {-1.0, -0.5}, {0.0, 4.0}}), 0.0);
}


TYPED_TEST(Csr, ScaleCsrAddIdentityThrowsOnZeroDiagonal)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    using Csr = typename TestFixture::Mtx;
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto b = gko::initialize<Csr>(
        {I<T>{2.0, 0.0}, I<T>{1.0, 0.0}, I<T>{0.0, -4.0}}, this->exec);

    ASSERT_THROW(b->add_scaled_identity(alpha.get(), beta.get()),
                 gko::UnsupportedMatrixProperty);
}


template <typename ValueIndexType>
class CsrComplex : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
};

TYPED_TEST_SUITE(CsrComplex, gko::test::ComplexValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(CsrComplex, MtxIsConjugateTransposable)
{
    using Csr = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;

    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx2 = gko::initialize<Csr>(
        {{T{1.0, 2.0}, T{3.0, 0.0}, T{2.0, 0.0}},
         {T{0.0, 0.0}, T{5.0, - 3.5}, T{0.0,0.0}},
         {T{0.0, 0.0}, T{0.0, 1.5}, T{2.0,0.0}}}, exec);
    // clang-format on

    auto trans_as_csr = gko::as<Csr>(mtx2->conj_transpose());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(trans_as_csr,
                        l({{T{1.0, - 2.0}, T{0.0, 0.0}, T{0.0, 0.0}},
                           {T{3.0, 0.0}, T{5.0, 3.5}, T{0.0, - 1.5}},
                           {T{2.0, 0.0}, T{0.0, 0.0}, T{2.0 + 0.0}}}), 0.0);
    // clang-format on
}


TYPED_TEST(CsrComplex, InplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


TYPED_TEST(CsrComplex, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
    ASSERT_EQ(mtx->get_strategy()->get_name(),
              abs_mtx->get_strategy()->get_name());
}


TYPED_TEST(Csr, CanGetSubmatrix)
{
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    /* this->mtx
     * 1   3   2
     * 0   5   0
     */
    auto sub_mat =
        this->mtx->create_submatrix(gko::span(0, 2), gko::span(0, 2));
    auto ref =
        gko::initialize<Mtx>({I<T>{1.0, 3.0}, I<T>{0.0, 5.0}}, this->exec);

    GKO_ASSERT_MTX_NEAR(sub_mat, ref, 0.0);
}


TYPED_TEST(Csr, CanGetSubmatrix2)
{
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mat = gko::initialize<Mtx>(
        {
            I<T>{1.0, 3.0, 4.5, 0.0, 2.0},   // 0
            I<T>{1.0, 0.0, 4.5, 7.5, 3.0},   // 1
            I<T>{0.0, 3.0, 4.5, 0.0, 2.0},   // 2
            I<T>{0.0, -1.0, 2.5, 0.0, 2.0},  // 3
            I<T>{1.0, 0.0, -1.0, 3.5, 1.0},  // 4
            I<T>{0.0, 1.0, 0.0, 0.0, 2.0},   // 5
            I<T>{0.0, 3.0, 0.0, 7.5, 1.0}    // 6
        },
        this->exec);
    ASSERT_EQ(mat->get_num_stored_elements(), 23);
    {
        SCOPED_TRACE("Left top corner: Square 2x2");
        auto sub_mat1 = mat->create_submatrix(gko::span(0, 2), gko::span(0, 2));
        auto ref1 =
            gko::initialize<Mtx>({I<T>{1.0, 3.0}, I<T>{1.0, 0.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1, ref1, 0.0);
    }
    {
        SCOPED_TRACE("Left boundary: Square 2x2");
        auto sub_mat2 = mat->create_submatrix(gko::span(2, 4), gko::span(0, 2));
        auto ref2 =
            gko::initialize<Mtx>({I<T>{0.0, 3.0}, I<T>{0.0, -1.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat2, ref2, 0.0);
    }
    {
        SCOPED_TRACE("Right boundary: Square 2x2");
        auto sub_mat3 = mat->create_submatrix(gko::span(0, 2), gko::span(3, 5));
        auto ref3 =
            gko::initialize<Mtx>({I<T>{0.0, 2.0}, I<T>{7.5, 3.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat3, ref3, 0.0);
    }
    {
        SCOPED_TRACE("Non-square 5x2");
        auto sub_mat4 = mat->create_submatrix(gko::span(1, 6), gko::span(2, 4));
        /*
           4.5, 7.5
           4.5, 0.0
           2.5, 0.0
          -1.0, 3.5
           0.0, 0.0
        */
        auto ref4 = gko::initialize<Mtx>(
            {I<T>{4.5, 7.5}, I<T>{4.5, 0.0}, I<T>{2.5, 0.0}, I<T>{-1.0, 3.5},
             I<T>{0.0, 0.0}},
            this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat4, ref4, 0.0);
    }
    {
        auto sub_mat5 = mat->create_submatrix(gko::span(0, 7), gko::span(0, 5));
        auto ref5 = gko::initialize<Mtx>(
            {
                I<T>{1.0, 3.0, 4.5, 0.0, 2.0},   // 0
                I<T>{1.0, 0.0, 4.5, 7.5, 3.0},   // 1
                I<T>{0.0, 3.0, 4.5, 0.0, 2.0},   // 2
                I<T>{0.0, -1.0, 2.5, 0.0, 2.0},  // 3
                I<T>{1.0, 0.0, -1.0, 3.5, 1.0},  // 4
                I<T>{0.0, 1.0, 0.0, 0.0, 2.0},   // 5
                I<T>{0.0, 3.0, 0.0, 7.5, 1.0}    // 6
            },
            this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat5, ref5, 0.0);
    }
    {
        auto sub_mat7 = mat->create_submatrix(gko::span(0, 1), gko::span(0, 1));
        auto ref7 = gko::initialize<Mtx>({I<T>{1.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat7, ref7, 0.0);
    }
}


TYPED_TEST(Csr, CanGetSubmatrixWithIndexSet)
{
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto mat = gko::initialize<Mtx>(
        {
            I<T>{1.0, 3.0, 4.5, 0.0, 2.0},   // 0
            I<T>{1.0, 0.0, 4.5, 7.5, 3.0},   // 1
            I<T>{0.0, 3.0, 4.5, 0.0, 2.0},   // 2
            I<T>{0.0, -1.0, 2.5, 0.0, 2.0},  // 3
            I<T>{1.0, 0.0, -1.0, 3.5, 1.0},  // 4
            I<T>{0.0, 1.0, 0.0, 0.0, 2.0},   // 5
            I<T>{0.0, 3.0, 0.0, 7.5, 1.0}    // 6
        },
        this->exec);

    ASSERT_EQ(mat->get_num_stored_elements(), 23);

    {
        SCOPED_TRACE("Both empty index sets");
        auto row_set = gko::index_set<index_type>(this->exec);
        auto col_set = gko::index_set<index_type>(this->exec);
        auto sub_mat1 = mat->create_submatrix(row_set, col_set);
        auto ref1 = Mtx::create(this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1, ref1, 0.0);
    }

    {
        SCOPED_TRACE("One empty index set");
        auto row_set = gko::index_set<index_type>(this->exec);
        auto col_set = gko::index_set<index_type>(this->exec, {0});
        auto sub_mat1 = mat->create_submatrix(row_set, col_set);
        auto ref1 = Mtx::create(this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1, ref1, 0.0);
    }

    {
        SCOPED_TRACE("Full index set");
        auto row_set =
            gko::index_set<index_type>(this->exec, {0, 1, 2, 3, 4, 5, 6});
        auto col_set = gko::index_set<index_type>(this->exec, {0, 1, 2, 3, 4});
        auto sub_mat1 = mat->create_submatrix(row_set, col_set);
        auto ref1 = gko::initialize<Mtx>(
            {
                I<T>{1.0, 3.0, 4.5, 0.0, 2.0},   // 0
                I<T>{1.0, 0.0, 4.5, 7.5, 3.0},   // 1
                I<T>{0.0, 3.0, 4.5, 0.0, 2.0},   // 2
                I<T>{0.0, -1.0, 2.5, 0.0, 2.0},  // 3
                I<T>{1.0, 0.0, -1.0, 3.5, 1.0},  // 4
                I<T>{0.0, 1.0, 0.0, 0.0, 2.0},   // 5
                I<T>{0.0, 3.0, 0.0, 7.5, 1.0}    // 6
            },
            this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1, ref1, 0.0);
    }

    {
        SCOPED_TRACE("Small square 2x2");
        auto row_set = gko::index_set<index_type>(this->exec, {0, 1});
        auto col_set = gko::index_set<index_type>(this->exec, {0, 1});
        auto sub_mat1 = mat->create_submatrix(row_set, col_set);
        auto ref1 =
            gko::initialize<Mtx>({I<T>{1.0, 3.0}, I<T>{1.0, 0.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1, ref1, 0.0);
    }

    {
        SCOPED_TRACE("Non-square 4x2");
        auto row_set = gko::index_set<index_type>(this->exec, {1, 2, 3, 4});
        auto col_set = gko::index_set<index_type>(this->exec, {1, 3});
        auto sub_mat1 = mat->create_submatrix(row_set, col_set);
        auto ref1 = gko::initialize<Mtx>(
            {I<T>{0.0, 7.5}, I<T>{3.0, 0.0}, I<T>{-1.0, 0.0}, I<T>{0.0, 3.5}},
            this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1, ref1, 0.0);
    }

    {
        SCOPED_TRACE("Square 3x3");
        auto row_set = gko::index_set<index_type>(this->exec, {1, 3, 4});
        auto col_set = gko::index_set<index_type>(this->exec, {1, 3, 0});
        auto sub_mat1 = mat->create_submatrix(row_set, col_set);
        auto ref1 = gko::initialize<Mtx>(
            {I<T>{1.0, 0.0, 7.5}, I<T>{0.0, -1.0, 0.0}, I<T>{1.0, 0.0, 3.5}},
            this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1, ref1, 0.0);
    }

    {
        SCOPED_TRACE("Square 4x4");
        auto row_set = gko::index_set<index_type>(this->exec, {1, 4, 5, 6});
        // This is unsorted to make sure that the output is correct (sorted)
        // even when the input is sorted.
        auto col_set = gko::index_set<index_type>(this->exec, {4, 3, 0, 1});
        auto sub_mat1 = mat->create_submatrix(row_set, col_set);
        auto ref1 = gko::initialize<Mtx>({I<T>{1.0, 0.0, 7.5, 3.0},   // 1
                                          I<T>{1.0, 0.0, 3.5, 1.0},   // 4
                                          I<T>{0.0, 1.0, 0.0, 2.0},   // 5
                                          I<T>{0.0, 3.0, 7.5, 1.0}},  // 6
                                         this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1, ref1, 0.0);
    }

    {
        SCOPED_TRACE("Non Square 2x4");
        auto row_set = gko::index_set<index_type>(this->exec, {5, 6});
        auto col_set = gko::index_set<index_type>(this->exec, {4, 3, 0, 1});
        auto sub_mat1 = mat->create_submatrix(row_set, col_set);
        auto ref1 = gko::initialize<Mtx>({I<T>{0.0, 1.0, 0.0, 2.0},   // 5
                                          I<T>{0.0, 3.0, 7.5, 1.0}},  // 6
                                         this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1, ref1, 0.0);
    }
}


template <typename ValueIndexType>
class CsrLookup : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;

    CsrLookup() : exec(gko::ReferenceExecutor::create())
    {
        mtx = Mtx::create(this->exec);
        typename Mtx::mat_data data{gko::dim<2>{6, 65}};
        for (int i = 0; i < 65; i++) {
            // row 0: empty row
            // row 1: full row
            data.nonzeros.emplace_back(1, i, 1.0);
            // row 2: pretty dense row
            if (i % 3 == 0) {
                data.nonzeros.emplace_back(2, i, 1.0);
            }
            // row 4-5: contiguous row
            if (i >= 10 && i < 30) {
                data.nonzeros.emplace_back(4, i, 1.0);
            }
            if (i >= 2 && i < 6) {
                data.nonzeros.emplace_back(5, i, 1.0);
            }
        }
        // row 3: very sparse
        data.nonzeros.emplace_back(3, 0, 1.0);
        data.nonzeros.emplace_back(3, 64, 1.0);
        data.sort_row_major();
        // 1000 as min-sentinel
        full_sizes = {0, 0, 1000, 1000, 0, 0};
        bitmap_sizes = {0, 6, 4, 6, 2, 2};
        hash_sizes = {1, 130, 44, 4, 40, 8};
        mtx->read(data);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx;
    std::vector<index_type> full_sizes;
    std::vector<index_type> bitmap_sizes;
    std::vector<index_type> hash_sizes;
    index_type invalid_index = gko::invalid_index<index_type>();
};

TYPED_TEST_SUITE(CsrLookup, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);

TYPED_TEST(CsrLookup, GeneratesLookupDataOffsets)
{
    using IndexType = typename TestFixture::index_type;
    using gko::matrix::csr::sparsity_type;
    const auto num_rows = this->mtx->get_size()[0];
    gko::array<IndexType> storage_offset_array(this->exec, num_rows + 1);
    const auto storage_offsets = storage_offset_array.get_data();
    const auto row_ptrs = this->mtx->get_const_row_ptrs();
    const auto col_idxs = this->mtx->get_const_col_idxs();

    for (auto allowed :
         {sparsity_type::full | sparsity_type::bitmap | sparsity_type::hash,
          sparsity_type::bitmap | sparsity_type::hash,
          sparsity_type::full | sparsity_type::hash, sparsity_type::hash}) {
        gko::kernels::reference::csr::build_lookup_offsets(
            this->exec, row_ptrs, col_idxs, num_rows, allowed, storage_offsets);
        bool allow_full =
            gko::matrix::csr::csr_lookup_allowed(allowed, sparsity_type::full);
        bool allow_bitmap = gko::matrix::csr::csr_lookup_allowed(
            allowed, sparsity_type::bitmap);

        for (gko::size_type row = 0; row < num_rows; row++) {
            const auto expected_size =
                std::min(allow_full ? this->full_sizes[row] : 1000,
                         std::min(allow_bitmap ? this->bitmap_sizes[row] : 1000,
                                  this->hash_sizes[row]));
            const auto size = storage_offsets[row + 1] - storage_offsets[row];

            ASSERT_EQ(size, expected_size);
        }
    }
}


TYPED_TEST(CsrLookup, GeneratesLookupData)
{
    using IndexType = typename TestFixture::index_type;
    using gko::matrix::csr::sparsity_type;
    const auto num_rows = this->mtx->get_size()[0];
    const auto num_cols = this->mtx->get_size()[1];
    gko::array<gko::int64> row_desc_array(this->exec, num_rows);
    gko::array<IndexType> storage_offset_array(this->exec, num_rows + 1);
    const auto row_descs = row_desc_array.get_data();
    const auto storage_offsets = storage_offset_array.get_data();
    const auto row_ptrs = this->mtx->get_const_row_ptrs();
    const auto col_idxs = this->mtx->get_const_col_idxs();
    for (auto allowed :
         {sparsity_type::full | sparsity_type::bitmap | sparsity_type::hash,
          sparsity_type::bitmap | sparsity_type::hash,
          sparsity_type::full | sparsity_type::hash, sparsity_type::hash}) {
        gko::kernels::reference::csr::build_lookup_offsets(
            this->exec, row_ptrs, col_idxs, num_rows, allowed, storage_offsets);
        gko::array<gko::int32> storage_array(this->exec,
                                             storage_offsets[num_rows]);
        const auto storage = storage_array.get_data();
        const auto bitmap_equivalent =
            csr_lookup_allowed(allowed, sparsity_type::bitmap)
                ? sparsity_type::bitmap
                : sparsity_type::hash;
        const auto full_equivalent =
            csr_lookup_allowed(allowed, sparsity_type::full)
                ? sparsity_type::full
                : bitmap_equivalent;

        gko::kernels::reference::csr::build_lookup(
            this->exec, row_ptrs, col_idxs, num_rows, allowed, storage_offsets,
            row_descs, storage);

        for (int row = 0; row < num_rows; row++) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
                row_ptrs, col_idxs,  storage_offsets,
                storage,  row_descs, static_cast<gko::size_type>(row)};
            for (auto nz = row_begin; nz < row_end; nz++) {
                const auto col = col_idxs[nz];
                ASSERT_EQ(lookup.lookup_unsafe(col) + row_begin, nz);
            }
            auto nz = row_begin;
            for (int col = 0; col < num_cols; col++) {
                auto found_nz = lookup[col];
                if (nz < row_end && col_idxs[nz] == col) {
                    ASSERT_EQ(found_nz, nz - row_begin);
                    nz++;
                } else {
                    ASSERT_EQ(found_nz, this->invalid_index);
                }
            }
        }
        ASSERT_EQ(row_descs[0] & 0xF, static_cast<int>(full_equivalent));
        ASSERT_EQ(row_descs[1] & 0xF, static_cast<int>(full_equivalent));
        ASSERT_EQ(row_descs[2] & 0xF, static_cast<int>(bitmap_equivalent));
        ASSERT_EQ(row_descs[3] & 0xF, static_cast<int>(sparsity_type::hash));
        ASSERT_EQ(row_descs[4] & 0xF, static_cast<int>(full_equivalent));
        ASSERT_EQ(row_descs[5] & 0xF, static_cast<int>(full_equivalent));
    }
}


}  // namespace
