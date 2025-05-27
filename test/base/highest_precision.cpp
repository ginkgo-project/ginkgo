// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <complex>

#include <gtest/gtest.h>

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/test/utils.hpp"


template <typename TwoTypes>
class HigestPrecision : public ::testing::Test {
public:
    using first_type =
        typename std::tuple_element<0, decltype(TwoTypes())>::type;
    using second_type =
        typename std::tuple_element<1, decltype(TwoTypes())>::type;
};

using TwoValueTypes = gko::test::merge_type_list_t<
    gko::test::cartesian_type_product_t<gko::test::RealValueTypes,
                                        gko::test::RealValueTypes>,
    gko::test::cartesian_type_product_t<gko::test::ComplexValueTypes,
                                        gko::test::ComplexValueTypes>>;

TYPED_TEST_SUITE(HigestPrecision, TwoValueTypes, PairTypenameNameGenerator);


template <typename T>
using device_type = gko::kernels::GKO_DEVICE_NAMESPACE::device_type<T>;

TYPED_TEST(HigestPrecision, DeviceShouldBeSameAsHost)
{
    using first_type = typename TestFixture::first_type;
    using second_type = typename TestFixture::second_type;

    // use std::is_same_v not StaticAssertTypeEq here. StaticAssertTypeEq shows
    // the final types are mismatched, so it is hard to know which pair is
    // failed.
    ASSERT_TRUE(
        (std::is_same_v<
            gko::highest_precision<device_type<first_type>,
                                   device_type<second_type>>,
            device_type<gko::highest_precision<first_type, second_type>>>));
}
