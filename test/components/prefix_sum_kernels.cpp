// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/prefix_sum_kernels.hpp"


#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


template <typename T>
class PrefixSum : public CommonTestFixture {
protected:
    using index_type = T;

    PrefixSum()
        : rand(293), total_size(42793), vals(ref, total_size), dvals(exec)
    {
        std::uniform_int_distribution<index_type> dist(0, 1000);
        for (gko::size_type i = 0; i < total_size; ++i) {
            vals.get_data()[i] = dist(rand);
        }
        dvals = vals;
    }

    std::default_random_engine rand;
    gko::size_type total_size;
    gko::array<index_type> vals;
    gko::array<index_type> dvals;
};

using PrefixSumIndexTypes =
    ::testing::Types<gko::int32, gko::int64, gko::size_type>;

TYPED_TEST_SUITE(PrefixSum, PrefixSumIndexTypes, TypenameNameGenerator);


TYPED_TEST(PrefixSum, EqualsReference)
{
    using gko::size_type;
    for (auto size :
         {size_type{0}, size_type{1}, size_type{131}, this->total_size}) {
        SCOPED_TRACE(size);
        gko::kernels::reference::components::prefix_sum_nonnegative(
            this->ref, this->vals.get_data(), size);
        gko::kernels::EXEC_NAMESPACE::components::prefix_sum_nonnegative(
            this->exec, this->dvals.get_data(), size);

        GKO_ASSERT_ARRAY_EQ(this->vals, this->dvals);
    }
}


TYPED_TEST(PrefixSum, WorksCloseToOverflow)
{
    // make sure the value we use as max isn't the sentinel used to mark
    // overflows for unsigned types
    // TODO remove with signed size_type
    const auto max = std::numeric_limits<TypeParam>::max() -
                     std::is_unsigned<TypeParam>::value;
    gko::array<TypeParam> data{this->exec, I<TypeParam>({max - 1, 1, 0})};

    gko::kernels::EXEC_NAMESPACE::components::prefix_sum_nonnegative(
        this->exec, data.get_data(), data.get_size());

    GKO_ASSERT_ARRAY_EQ(data, I<TypeParam>({0, max - 1, max}));
}


TYPED_TEST(PrefixSum, DoesntOverflowFromLastElement)
{
    const auto max = std::numeric_limits<TypeParam>::max();
    gko::array<TypeParam> data{this->exec, I<TypeParam>({2, max - 1})};

    gko::kernels::EXEC_NAMESPACE::components::prefix_sum_nonnegative(
        this->exec, data.get_data(), data.get_size());

    GKO_ASSERT_ARRAY_EQ(data, I<TypeParam>({0, 2}));
}


#ifndef GKO_COMPILING_DPCPP
// TODO implement overflow check for DPC++

TYPED_TEST(PrefixSum, ThrowsOnOverflow)
{
    const auto max = std::numeric_limits<TypeParam>::max();
    gko::array<TypeParam> data{this->exec,
                               {max / 3, max / 2, max / 4, max / 3, max / 4}};

    ASSERT_THROW(
        gko::kernels::EXEC_NAMESPACE::components::prefix_sum_nonnegative(
            this->exec, data.get_data(), data.get_size()),
        gko::OverflowError);
}

#endif
