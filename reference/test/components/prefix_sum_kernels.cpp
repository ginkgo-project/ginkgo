// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/prefix_sum_kernels.hpp"


#include <algorithm>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class PrefixSum : public ::testing::Test {
protected:
    using index_type = T;
    PrefixSum()
        : exec(gko::ReferenceExecutor::create()),
          vals{3, 5, 6, 7, 1, 5, 9, 7, 2, 0, 5},
          expected{0, 3, 8, 14, 21, 22, 27, 36, 43, 45, 45}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::vector<index_type> vals;
    std::vector<index_type> expected;
};

using PrefixSumIndexTypes =
    ::testing::Types<gko::int32, gko::int64, gko::size_type>;

TYPED_TEST_SUITE(PrefixSum, PrefixSumIndexTypes, TypenameNameGenerator);


TYPED_TEST(PrefixSum, Works)
{
    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->exec, this->vals.data(), this->vals.size());

    ASSERT_EQ(this->vals, this->expected);
}


TYPED_TEST(PrefixSum, WorksCloseToOverflow)
{
    constexpr auto max = std::numeric_limits<TypeParam>::max() -
                         std::is_unsigned<TypeParam>::value;
    std::vector<TypeParam> vals{max - 1, 1, 0};
    std::vector<TypeParam> expected{0, max - 1, max};

    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->exec, vals.data(), vals.size());

    ASSERT_EQ(vals, expected);
}


TYPED_TEST(PrefixSum, DoesntOverflowFromLastElement)
{
    constexpr auto max = std::numeric_limits<TypeParam>::max() -
                         std::is_unsigned<TypeParam>::value;
    std::vector<TypeParam> vals{2, max - 1};
    std::vector<TypeParam> expected{0, 2};

    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->exec, vals.data(), vals.size());

    ASSERT_EQ(vals, expected);
}


TYPED_TEST(PrefixSum, ThrowsOnOverflow)
{
    constexpr auto max = std::numeric_limits<TypeParam>::max();
    std::vector<TypeParam> vals{0, 152, max / 2, 25, 147, max / 2, 0, 1};

    ASSERT_THROW(gko::kernels::reference::components::prefix_sum_nonnegative(
                     this->exec, vals.data(), vals.size()),
                 gko::OverflowError);
}


}  // namespace
