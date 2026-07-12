// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "accessor/sycl_helper.hpp"

#include <array>
#include <type_traits>

#include <gtest/gtest.h>

#include "accessor/block_col_major.hpp"
#include "accessor/range.hpp"
#include "accessor/reduced_row_major.hpp"
#include "accessor/row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"
#include "index_types.hpp"


namespace {


template <typename IndexType>
class SyclHelper : public ::testing::Test {
protected:
    using index_type = IndexType;

    // clang-format off
    double data[8]{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0
    };
    // clang-format on
    std::array<IndexType, 2> size{{4, 2}};
    std::array<IndexType, 1> stride{{2}};
};

TYPED_TEST_SUITE(SyclHelper, acc::test::AltIndexTypes);


TYPED_TEST(SyclHelper, MapsReducedRowMajorRange)
{
    using accessor = acc::reduced_row_major<2, double, double,
                                            typename TestFixture::index_type>;
    auto r = acc::range<accessor>(this->size, this->data, this->stride);

    auto device_r = acc::as_sycl_range(r);

    using device_accessor = typename decltype(device_r)::accessor;
    static_assert(std::is_same<device_accessor, accessor>::value,
                  "sycl_type maps double to itself, so the accessor type "
                  "(including IndexType) must be preserved");
    EXPECT_EQ(device_r.get_accessor().get_stored_data(), this->data);
    EXPECT_EQ(device_r(1, 1), 4.0);
}


TYPED_TEST(SyclHelper, MapsScaledReducedRowMajorRange)
{
    using accessor =
        acc::scaled_reduced_row_major<2, double, double, 0b10,
                                      typename TestFixture::index_type>;
    double scalar[4]{1.0, 1.0, 1.0, 1.0};
    auto r = acc::range<accessor>(this->size, this->data, this->stride, scalar);

    auto device_r = acc::as_sycl_range(r);

    using device_accessor = typename decltype(device_r)::accessor;
    static_assert(std::is_same<device_accessor, accessor>::value,
                  "sycl_type maps double to itself, so the accessor type "
                  "(including IndexType) must be preserved");
    EXPECT_EQ(device_r.get_accessor().get_stored_data(), this->data);
    EXPECT_EQ(device_r(1, 1), 4.0);
}


TYPED_TEST(SyclHelper, MapsRowMajorRange)
{
    using accessor =
        acc::row_major<double, 2, typename TestFixture::index_type>;
    auto r = acc::range<accessor>(this->size, this->data, this->stride);

    auto device_r = acc::as_sycl_range(r);

    using device_accessor = typename decltype(device_r)::accessor;
    static_assert(std::is_same<device_accessor, accessor>::value,
                  "mapping a row_major range must yield a row_major range, "
                  "not a different layout");
    EXPECT_EQ(device_r.get_accessor().data, this->data);
    // (2, 1) -> 2 * stride + 1 = 5 in row-major; a block_col_major
    // misinterpretation would read index 2 + 1 * stride = 4 instead
    EXPECT_EQ(device_r(2, 1), 6.0);
    EXPECT_EQ(device_r(1, 0), 3.0);
}


TYPED_TEST(SyclHelper, MapsBlockColMajorRange)
{
    using accessor =
        acc::block_col_major<double, 2, typename TestFixture::index_type>;
    auto r = acc::range<accessor>(this->size, this->data);

    auto device_r = acc::as_sycl_range(r);

    using device_accessor = typename decltype(device_r)::accessor;
    static_assert(std::is_same<device_accessor, accessor>::value,
                  "sycl_type maps double to itself, so the accessor type "
                  "(including IndexType) must be preserved");
    EXPECT_EQ(device_r.get_accessor().data, this->data);
    EXPECT_EQ(device_r(1, 1), 6.0);
}


}  // namespace
