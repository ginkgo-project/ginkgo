// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/array.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Array : public ::testing::Test {
protected:
    Array() : exec(gko::ReferenceExecutor::create()), x(exec, 2)
    {
        x.get_data()[0] = 5;
        x.get_data()[1] = 2;
    }

    std::shared_ptr<const gko::Executor> exec;
    gko::array<T> x;
};

TYPED_TEST_SUITE(Array, gko::test::ValueAndIndexTypes, TypenameNameGenerator);


TYPED_TEST(Array, CanBeFilledWithValue)
{
    this->x.fill(TypeParam{42});

    ASSERT_EQ(this->x.get_size(), 2);
    ASSERT_EQ(this->x.get_data()[0], TypeParam{42});
    ASSERT_EQ(this->x.get_data()[1], TypeParam{42});
    ASSERT_EQ(this->x.get_const_data()[0], TypeParam{42});
    ASSERT_EQ(this->x.get_const_data()[1], TypeParam{42});
}


TYPED_TEST(Array, CanBeReduced)
{
    auto out = gko::array<TypeParam>(this->exec, I<TypeParam>{1});

    gko::reduce_add(this->x, out);

    ASSERT_EQ(out.get_data()[0], TypeParam{8});
}


TYPED_TEST(Array, CanBeReduced2)
{
    auto out = gko::reduce_add(this->x, TypeParam{2});

    ASSERT_EQ(out, TypeParam{9});
}


}  // namespace
