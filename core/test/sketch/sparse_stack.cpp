// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/sketch/sparse_stack.hpp>

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class SparseStackCore : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Dense = gko::matrix::Dense<value_type>;
    using Sketch = gko::sketch::SparseStack<value_type, index_type>;

    SparseStackCore() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(SparseStackCore, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(SparseStackCore, ReturnsCorrectSketchSize)
{
    auto sketch = TestFixture::Sketch::create(this->exec, 7, 20, 42);

    EXPECT_EQ(sketch->get_sketch_size(), 7);
}


TYPED_TEST(SparseStackCore, ReturnsCorrectInputSize)
{
    auto sketch = TestFixture::Sketch::create(this->exec, 7, 20, 42);

    EXPECT_EQ(sketch->get_input_size(), 20);
}


TYPED_TEST(SparseStackCore, ReturnsCorrectSeed)
{
    auto sketch = TestFixture::Sketch::create(this->exec, 7, 20, 123);

    EXPECT_EQ(sketch->get_seed(), 123);
}


TYPED_TEST(SparseStackCore, HashMapHasCorrectSize)
{
    auto sketch = TestFixture::Sketch::create(this->exec, 7, 20, 4, 42);

    EXPECT_EQ(sketch->get_hash_map().get_size(), 80); 
    EXPECT_EQ(sketch->get_signs().get_size(), 80);
}


TYPED_TEST(SparseStackCore, SignsHasCorrectSize)
{
    auto sketch = TestFixture::Sketch::create(this->exec, 7, 20, 42);

    EXPECT_EQ(sketch->get_signs().get_size(), 20);
}


TYPED_TEST(SparseStackCore, ApplyThrowsOnDimensionMismatch)
{
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 5, 42);
    // b has wrong number of rows (4 instead of 5)
    auto b = Dense::create(this->exec, gko::dim<2>{4, 3});
    auto x = Dense::create(this->exec, gko::dim<2>{3, 3});

    ASSERT_THROW(sketch->apply(b, x), gko::DimensionMismatch);
}


TYPED_TEST(SparseStackCore, ApplyThrowsOnOutputDimensionMismatch)
{
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 5, 42);
    auto b = Dense::create(this->exec, gko::dim<2>{5, 3});
    // x has wrong number of rows (4 instead of 3)
    auto x = Dense::create(this->exec, gko::dim<2>{4, 3});

    ASSERT_THROW(sketch->apply(b, x), gko::DimensionMismatch);
}


TYPED_TEST(SparseStackCore, ApplyThrowsOnColumnMismatch)
{
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 5, 42);
    auto b = Dense::create(this->exec, gko::dim<2>{5, 3});
    // x has wrong number of columns (2 instead of 3)
    auto x = Dense::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(sketch->apply(b, x), gko::DimensionMismatch);
}


TYPED_TEST(SparseStackCore, RapplyThrowsOnDimensionMismatch)
{
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 5, 42);
    // rapply: b must have cols == 5 (input_size), this has 4
    auto b = Dense::create(this->exec, gko::dim<2>{4, 4});
    auto x = Dense::create(this->exec, gko::dim<2>{4, 3});

    ASSERT_THROW(sketch->rapply(b, x), gko::DimensionMismatch);
}


TYPED_TEST(SparseStackCore, RapplyThrowsOnOutputColumnMismatch)
{
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 5, 42);
    auto b = Dense::create(this->exec, gko::dim<2>{4, 5});
    // x must have cols == 3 (sketch_size), this has 4
    auto x = Dense::create(this->exec, gko::dim<2>{4, 4});

    ASSERT_THROW(sketch->rapply(b, x), gko::DimensionMismatch);
}


TYPED_TEST(SparseStackCore, RapplyThrowsOnRowMismatch)
{
    using Dense = typename TestFixture::Dense;
    auto sketch = TestFixture::Sketch::create(this->exec, 3, 5, 42);
    auto b = Dense::create(this->exec, gko::dim<2>{4, 5});
    // x must have rows == b->rows (4), this has 3
    auto x = Dense::create(this->exec, gko::dim<2>{3, 3});

    ASSERT_THROW(sketch->rapply(b, x), gko::DimensionMismatch);
}


}  // namespace
