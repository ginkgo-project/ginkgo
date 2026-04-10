// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/sketch/gaussian_sketch.hpp>

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


class GaussianSketchKernels : public CommonTestFixture {
protected:
    using ValueType = value_type;
    using Dense = gko::matrix::Dense<ValueType>;
    using Sketch = gko::sketch::GaussianSketch<ValueType>;

    GaussianSketchKernels()
#ifdef GINKGO_FAST_TESTS
        : sketch_size(16),
          input_size(64),
#else
        : sketch_size(64),
          input_size(256),
#endif
          seed(42),
          rand_engine(seed)
    {}

    const gko::size_type sketch_size;
    const gko::size_type input_size;
    const gko::uint64 seed;
    std::default_random_engine rand_engine;
};


TEST_F(GaussianSketchKernels, GenerateIsDeterministic)
{
    auto sketch1 = Sketch::create(exec, sketch_size, input_size, seed);
    auto sketch2 = Sketch::create(exec, sketch_size, input_size, seed);

    GKO_ASSERT_MTX_NEAR(sketch1->get_sketch_matrix(),
                         sketch2->get_sketch_matrix(), 0.0);
}


TEST_F(GaussianSketchKernels, ApplyIsEquivalentToRef)
{
    // Create on reference and clone to device (different RNGs produce
    // different sequences, so we can't compare ref-created vs dev-created)
    auto ref_sketch = Sketch::create(ref, sketch_size, input_size, seed);
    auto dev_sketch = gko::clone(exec, ref_sketch);
    auto b = gko::test::generate_random_matrix<Dense>(
        input_size, 3,
        std::uniform_int_distribution<>(3, 3),
        std::normal_distribution<value_type>(0.0, 1.0), rand_engine, ref);
    auto db = gko::clone(exec, b);
    auto x_ref = Dense::create(ref, gko::dim<2>{sketch_size, 3});
    auto x_dev = Dense::create(exec, gko::dim<2>{sketch_size, 3});

    ref_sketch->apply(b, x_ref);
    dev_sketch->apply(db, x_dev);

    GKO_ASSERT_MTX_NEAR(x_ref, x_dev, r<value_type>::value);
}


TEST_F(GaussianSketchKernels, RapplyIsEquivalentToRef)
{
    auto ref_sketch = Sketch::create(ref, sketch_size, input_size, seed);
    auto dev_sketch = gko::clone(exec, ref_sketch);
    auto b = gko::test::generate_random_matrix<Dense>(
        3, input_size,
        std::uniform_int_distribution<>(static_cast<int>(input_size),
                                        static_cast<int>(input_size)),
        std::normal_distribution<value_type>(0.0, 1.0), rand_engine, ref);
    auto db = gko::clone(exec, b);
    auto x_ref = Dense::create(ref, gko::dim<2>{3, sketch_size});
    auto x_dev = Dense::create(exec, gko::dim<2>{3, sketch_size});

    ref_sketch->rapply(b, x_ref);
    dev_sketch->rapply(db, x_dev);

    GKO_ASSERT_MTX_NEAR(x_ref, x_dev, r<value_type>::value);
}
