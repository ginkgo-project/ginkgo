// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/sketch/count_sketch.hpp>

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


class CountSketchKernels : public CommonTestFixture {
protected:
    using ValueType = value_type;
    using IndexType = gko::int32;
    using Dense = gko::matrix::Dense<ValueType>;
    using Sketch = gko::sketch::CountSketch<ValueType, IndexType>;

    CountSketchKernels()
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


TEST_F(CountSketchKernels, GenerateIsDeterministic)
{
    auto sketch1 = Sketch::create(exec, sketch_size, input_size, seed);
    auto sketch2 = Sketch::create(exec, sketch_size, input_size, seed);

    auto h1 = sketch1->get_hash_map().copy_to_host();
    auto h2 = sketch2->get_hash_map().copy_to_host();
    auto s1 = sketch1->get_signs().copy_to_host();
    auto s2 = sketch2->get_signs().copy_to_host();
    for (gko::size_type i = 0; i < input_size; ++i) {
        EXPECT_EQ(h1[i], h2[i]);
        EXPECT_EQ(s1[i], s2[i]);
    }
}


TEST_F(CountSketchKernels, ApplyIsEquivalentToRef)
{
    auto ref_sketch = Sketch::create(ref, sketch_size, input_size, seed);
    auto dev_sketch = Sketch::create(exec, sketch_size, input_size, seed);
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


TEST_F(CountSketchKernels, RapplyIsEquivalentToRef)
{
    auto ref_sketch = Sketch::create(ref, sketch_size, input_size, seed);
    auto dev_sketch = Sketch::create(exec, sketch_size, input_size, seed);
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
