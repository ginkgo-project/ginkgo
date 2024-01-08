// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/scaled_permutation.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class ScaledPermutation : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Vec = gko::matrix::Dense<value_type>;
    using Mtx = gko::matrix::ScaledPermutation<value_type, index_type>;

    ScaledPermutation() : exec(gko::ReferenceExecutor::create())
    {
        perm3 = Mtx::create(exec,
                            gko::array<value_type>{this->exec, {1.0, 2.0, 4.0}},
                            gko::array<index_type>{this->exec, {1, 2, 0}});
        perm2 =
            Mtx::create(exec, gko::array<value_type>{this->exec, {3.0, 5.0}},
                        gko::array<index_type>{this->exec, {1, 0}});
    }

    std::unique_ptr<Vec> ref_combine(const Mtx* first, const Mtx* second)
    {
        const auto exec = first->get_executor();
        gko::matrix_data<value_type, index_type> first_perm_data;
        gko::matrix_data<value_type, index_type> second_perm_data;
        first->write(first_perm_data);
        second->write(second_perm_data);
        const auto first_mtx = Vec::create(exec);
        const auto second_mtx = Vec::create(exec);
        first_mtx->read(first_perm_data);
        second_mtx->read(second_perm_data);
        auto combined_mtx = first_mtx->clone();
        second_mtx->apply(first_mtx, combined_mtx);
        return combined_mtx;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> perm3;
    std::unique_ptr<Mtx> perm2;
};

TYPED_TEST_SUITE(ScaledPermutation, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(ScaledPermutation, Invert)
{
    using T = typename TestFixture::value_type;
    auto inv = this->perm3->compute_inverse();

    EXPECT_EQ(inv->get_const_permutation()[0], 2);
    EXPECT_EQ(inv->get_const_permutation()[1], 0);
    EXPECT_EQ(inv->get_const_permutation()[2], 1);
    EXPECT_EQ(inv->get_const_scaling_factors()[0], T{0.5});
    EXPECT_EQ(inv->get_const_scaling_factors()[1], T{0.25});
    EXPECT_EQ(inv->get_const_scaling_factors()[2], T{1.0});
}


TYPED_TEST(ScaledPermutation, CreateFromPermutation)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto non_scaled = gko::matrix::Permutation<index_type>::create(
        this->exec, gko::array<index_type>{this->exec, {1, 2, 0}});

    auto scaled = Mtx::create(non_scaled);

    EXPECT_EQ(scaled->get_const_permutation()[0], 1);
    EXPECT_EQ(scaled->get_const_permutation()[1], 2);
    EXPECT_EQ(scaled->get_const_permutation()[2], 0);
    EXPECT_EQ(scaled->get_const_scaling_factors()[0], gko::one<value_type>());
    EXPECT_EQ(scaled->get_const_scaling_factors()[1], gko::one<value_type>());
    EXPECT_EQ(scaled->get_const_scaling_factors()[2], gko::one<value_type>());
}


TYPED_TEST(ScaledPermutation, Combine)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    const auto other_perm = Mtx::create(
        this->exec, gko::array<value_type>{this->exec, {3.0, 5.0, 7.0}},
        gko::array<index_type>{this->exec, {1, 0, 2}});
    const auto ref_combined =
        this->ref_combine(this->perm3.get(), other_perm.get());

    const auto combined = this->perm3->compose(other_perm);

    GKO_ASSERT_MTX_NEAR(combined, ref_combined, 0.0);
}


TYPED_TEST(ScaledPermutation, CombineLarger)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    const auto perm = Mtx::create(
        this->exec,
        gko::array<value_type>{
            this->exec,
            {1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0}},
        gko::array<index_type>{this->exec, {6, 2, 4, 0, 1, 5, 9, 8, 3, 7}});
    const auto perm2 = Mtx::create(
        this->exec,
        gko::array<value_type>{
            this->exec,
            {29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 61.0, 67.0}},
        gko::array<index_type>{this->exec, {9, 2, 1, 6, 3, 7, 8, 4, 0, 5}});
    const auto ref_combined = this->ref_combine(perm.get(), perm2.get());

    const auto combined = perm->compose(perm2);

    GKO_ASSERT_MTX_NEAR(combined, ref_combined, 0.0);
}


TYPED_TEST(ScaledPermutation, CombineWithInverse)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const gko::size_type size = 20;
    auto rng = std::default_random_engine{3754};
    auto dist = std::uniform_real_distribution<gko::remove_complex<value_type>>{
        1.0, 2.0};
    auto perm = gko::matrix::ScaledPermutation<value_type, index_type>::create(
        this->exec, size);
    std::iota(perm->get_permutation(), perm->get_permutation() + size, 0);
    std::shuffle(perm->get_permutation(), perm->get_permutation() + size, rng);
    for (gko::size_type i = 0; i < size; i++) {
        perm->get_scaling_factors()[i] = dist(rng);
    }

    auto combined = perm->compose(perm->compute_inverse());

    for (index_type i = 0; i < size; i++) {
        ASSERT_EQ(combined->get_const_permutation()[i], i);
        ASSERT_LT(gko::abs(combined->get_const_scaling_factors()[i] -
                           gko::one<value_type>()),
                  r<value_type>::value);
    }
}


TYPED_TEST(ScaledPermutation, CombineFailsWithMismatchingSize)
{
    ASSERT_THROW(this->perm3->compose(this->perm2), gko::DimensionMismatch);
}


TYPED_TEST(ScaledPermutation, Write)
{
    using T = typename TestFixture::value_type;

    GKO_ASSERT_MTX_NEAR(
        this->perm3, l<T>({{0.0, 2.0, 0.0}, {0.0, 0.0, 4.0}, {1.0, 0.0, 0.0}}),
        0.0);
}


TYPED_TEST(ScaledPermutation, AppliesToDense)
{
    using T = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({I<T>{2.0, 3.0}, I<T>{4.0, 2.5}}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    this->perm2->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({{20.0, 12.5}, {6.0, 9.0}}), 0.0);
}


TYPED_TEST(ScaledPermutation, AdvancedAppliesToDense)
{
    using T = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({I<T>{2.0, 3.0}, I<T>{4.0, 2.5}}, this->exec);
    auto y = x->clone();

    this->perm2->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({{38.0, 22.0}, {8.0, 15.5}}), 0.0);
}


}  // namespace
