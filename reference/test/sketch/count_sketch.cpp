// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/sketch/count_sketch.hpp>

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class CountSketchTest : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Sketch = gko::sketch::CountSketch<value_type, index_type>;

    CountSketchTest() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(CountSketchTest, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(CountSketchTest, HasCorrectDimensions)
{
    auto sketch = TestFixture::Sketch::create(this->exec, 5, 10, 42);

    EXPECT_EQ(sketch->get_size(), gko::dim<2>(5, 10));
    EXPECT_EQ(sketch->get_sketch_size(), 5);
    EXPECT_EQ(sketch->get_input_size(), 10);
}


TYPED_TEST(CountSketchTest, HashMapValuesAreInRange)
{
    using index_type = typename TestFixture::index_type;
    auto sketch = TestFixture::Sketch::create(this->exec, 5, 20, 42);

    auto hash_data = sketch->get_hash_map().get_const_data();
    for (gko::size_type i = 0; i < 20; ++i) {
        EXPECT_GE(hash_data[i], index_type{0});
        EXPECT_LT(hash_data[i], index_type{5});
    }
}


TYPED_TEST(CountSketchTest, SignsArePlusOrMinusOne)
{
    using value_type = typename TestFixture::value_type;
    auto sketch = TestFixture::Sketch::create(this->exec, 5, 20, 42);

    auto sign_data = sketch->get_signs().get_const_data();
    for (gko::size_type i = 0; i < 20; ++i) {
        auto val = sign_data[i];
        EXPECT_TRUE(val == gko::one<value_type>() ||
                    val == -gko::one<value_type>());
    }
}


TYPED_TEST(CountSketchTest, IsDeterministicWithSameSeed)
{
    auto sketch1 = TestFixture::Sketch::create(this->exec, 5, 10, 42);
    auto sketch2 = TestFixture::Sketch::create(this->exec, 5, 10, 42);

    auto h1 = sketch1->get_hash_map().get_const_data();
    auto h2 = sketch2->get_hash_map().get_const_data();
    auto s1 = sketch1->get_signs().get_const_data();
    auto s2 = sketch2->get_signs().get_const_data();
    for (gko::size_type i = 0; i < 10; ++i) {
        EXPECT_EQ(h1[i], h2[i]);
        EXPECT_EQ(s1[i], s2[i]);
    }
}


TYPED_TEST(CountSketchTest, ApplyMatchesExplicitConstruction)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Dense;
    gko::size_type k = 3, m = 5, n = 3;
    auto sketch = TestFixture::Sketch::create(this->exec, k, m, 42);
    auto b = gko::initialize<Dense>(
        n, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}, {13.0, 14.0, 15.0}},
        this->exec);
    auto x = Dense::create(this->exec, gko::dim<2>{k, n});

    sketch->apply(b, x);

    auto S = Dense::create(this->exec, gko::dim<2>{k, m});
    for (gko::size_type r = 0; r < k; ++r) {
        for (gko::size_type c = 0; c < m; ++c) {
            S->at(r, c) = gko::zero<value_type>();
        }
    }
    auto hash_data = sketch->get_hash_map().get_const_data();
    auto sign_data = sketch->get_signs().get_const_data();
    for (gko::size_type i = 0; i < m; ++i) {
        S->at(hash_data[i], i) = sign_data[i];
    }
    auto expected = Dense::create(this->exec, gko::dim<2>{k, n});
    S->apply(b, expected);

    GKO_ASSERT_MTX_NEAR(x, expected, r<value_type>::value);
}


TYPED_TEST(CountSketchTest, RapplyMatchesExplicitConstruction)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Dense;
    gko::size_type k = 3, m = 5, n = 3;
    auto sketch = TestFixture::Sketch::create(this->exec, k, m, 42);
    auto b = gko::initialize<Dense>(
        m, {{1.0, 2.0, 3.0, 4.0, 5.0}, {6.0, 7.0, 8.0, 9.0, 10.0},
            {11.0, 12.0, 13.0, 14.0, 15.0}},
        this->exec);
    auto x = Dense::create(this->exec, gko::dim<2>{n, k});

    sketch->rapply(b, x);

    auto ST = Dense::create(this->exec, gko::dim<2>{m, k});
    for (gko::size_type r = 0; r < m; ++r) {
        for (gko::size_type c = 0; c < k; ++c) {
            ST->at(r, c) = gko::zero<value_type>();
        }
    }
    auto hash_data = sketch->get_hash_map().get_const_data();
    auto sign_data = sketch->get_signs().get_const_data();
    for (gko::size_type i = 0; i < m; ++i) {
        ST->at(i, hash_data[i]) = sign_data[i];
    }
    auto expected = Dense::create(this->exec, gko::dim<2>{n, k});
    b->apply(ST, expected);

    GKO_ASSERT_MTX_NEAR(x, expected, r<value_type>::value);
}


TYPED_TEST(CountSketchTest, ApplyZerosOutputBeforeAccumulation)
{
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 5, 42);
    auto b = gko::initialize<Dense>(
        3, {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}},
        this->exec);
    auto x = gko::initialize<Dense>(
        3, {{999.0, 999.0, 999.0}, {999.0, 999.0, 999.0},
            {999.0, 999.0, 999.0}},
        this->exec);

    sketch->apply(b, x);

    for (gko::size_type i = 0; i < 3; ++i) {
        for (gko::size_type j = 0; j < 3; ++j) {
            EXPECT_NE(x->at(i, j), value_type{999.0});
        }
    }
}


}  // namespace
