// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/sketch/gaussian_sketch.hpp>

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename ValueType>
class GaussianSketchTest : public ::testing::Test {
protected:
    using value_type = ValueType;
    using real_type = gko::remove_complex<ValueType>;
    using Dense = gko::matrix::Dense<value_type>;
    using Sketch = gko::sketch::GaussianSketch<value_type>;

    GaussianSketchTest() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(GaussianSketchTest, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(GaussianSketchTest, HasCorrectDimensions)
{
    auto sketch = TestFixture::Sketch::create(this->exec, 5, 10, 42);

    EXPECT_EQ(sketch->get_size(), gko::dim<2>(5, 10));
    EXPECT_EQ(sketch->get_sketch_size(), 5);
    EXPECT_EQ(sketch->get_input_size(), 10);
}


TYPED_TEST(GaussianSketchTest, IsDeterministicWithSameSeed)
{
    auto sketch1 = TestFixture::Sketch::create(this->exec, 5, 10, 42);
    auto sketch2 = TestFixture::Sketch::create(this->exec, 5, 10, 42);

    GKO_ASSERT_MTX_NEAR(sketch1->get_sketch_matrix(),
                         sketch2->get_sketch_matrix(), 0.0);
}


TYPED_TEST(GaussianSketchTest, ProducesDifferentMatricesWithDifferentSeeds)
{
    auto sketch1 = TestFixture::Sketch::create(this->exec, 5, 10, 42);
    auto sketch2 = TestFixture::Sketch::create(this->exec, 5, 10, 99);

    auto mtx1 = sketch1->get_sketch_matrix();
    auto mtx2 = sketch2->get_sketch_matrix();
    bool found_diff = false;
    for (gko::size_type i = 0; i < 5; ++i) {
        for (gko::size_type j = 0; j < 10; ++j) {
            if (mtx1->at(i, j) != mtx2->at(i, j)) {
                found_diff = true;
                break;
            }
        }
        if (found_diff) break;
    }
    EXPECT_TRUE(found_diff);
}


TYPED_TEST(GaussianSketchTest, ApplyProducesCorrectDimensions)
{
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 8, 42);
    auto b = Dense::create(this->exec, gko::dim<2>{8, 2});
    auto x = Dense::create(this->exec, gko::dim<2>{3, 2});

    sketch->apply(b, x);

    EXPECT_EQ(x->get_size(), gko::dim<2>(3, 2));
}


TYPED_TEST(GaussianSketchTest, ApplyMatchesDenseGemm)
{
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 5, 42);
    auto b = gko::initialize<Dense>(
        3, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}, {13.0, 14.0, 15.0}},
        this->exec);
    auto x_sketch = Dense::create(this->exec, gko::dim<2>{3, 3});
    auto x_gemm = Dense::create(this->exec, gko::dim<2>{3, 3});

    sketch->apply(b, x_sketch);
    sketch->get_sketch_matrix()->apply(b, x_gemm);

    GKO_ASSERT_MTX_NEAR(x_sketch, x_gemm, r<value_type>::value);
}


TYPED_TEST(GaussianSketchTest, RapplyProducesCorrectDimensions)
{
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 8, 42);
    auto b = Dense::create(this->exec, gko::dim<2>{4, 8});
    auto x = Dense::create(this->exec, gko::dim<2>{4, 3});

    sketch->rapply(b, x);

    EXPECT_EQ(x->get_size(), gko::dim<2>(4, 3));
}


TYPED_TEST(GaussianSketchTest, RapplyMatchesDenseGemm)
{
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 5, 42);
    auto b = gko::initialize<Dense>(
        5, {{1.0, 2.0, 3.0, 4.0, 5.0}, {6.0, 7.0, 8.0, 9.0, 10.0},
            {11.0, 12.0, 13.0, 14.0, 15.0}},
        this->exec);
    auto x_sketch = Dense::create(this->exec, gko::dim<2>{3, 3});
    auto x_gemm = Dense::create(this->exec, gko::dim<2>{3, 3});

    sketch->rapply(b, x_sketch);
    auto st = gko::as<Dense>(sketch->get_sketch_matrix()->transpose());
    b->apply(st, x_gemm);

    GKO_ASSERT_MTX_NEAR(x_sketch, x_gemm, r<value_type>::value);
}


TYPED_TEST(GaussianSketchTest, AdvancedApplyWorks)
{
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 5, 42);
    auto b = gko::initialize<Dense>(
        3, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}, {13.0, 14.0, 15.0}},
        this->exec);
    auto x = gko::initialize<Dense>(
        3, {{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}}, this->exec);
    auto alpha = gko::initialize<Dense>({value_type{2.0}}, this->exec);
    auto beta = gko::initialize<Dense>({value_type{0.5}}, this->exec);

    auto expected = gko::clone(this->exec, x);
    auto tmp = Dense::create(this->exec, gko::dim<2>{3, 3});
    sketch->apply(b, tmp);
    expected->scale(beta);
    expected->add_scaled(alpha, tmp);

    sketch->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, expected, r<value_type>::value);
}


}  // namespace
