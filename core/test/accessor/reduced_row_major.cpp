// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "accessor/reduced_row_major.hpp"

#include <array>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include "accessor/range.hpp"
#include "accessor/utils.hpp"
#include "core/test/accessor/utils.hpp"


namespace {


/**
 * This test makes sure reduced_row_major works independent of Ginkgo and with
 * dimensionalities 1 and 2.
 */
template <typename IndexSizeTypes>
class ReducedStorageXd : public ::testing::Test {
protected:
    using ar_type = double;
    using st_type = float;
    using index_type = typename IndexSizeTypes::index_type;
    using size_type = typename IndexSizeTypes::size_type;
    static constexpr ar_type delta{std::numeric_limits<st_type>::epsilon() *
                                   1e1};

    using accessor1d =
        gko::acc::reduced_row_major<1, ar_type, st_type, index_type, size_type>;
    using accessor2d =
        gko::acc::reduced_row_major<2, ar_type, st_type, index_type, size_type>;
    using const_accessor1d =
        gko::acc::reduced_row_major<1, ar_type, const st_type, index_type,
                                    size_type>;
    using const_accessor2d =
        gko::acc::reduced_row_major<2, ar_type, const st_type, index_type,
                                    size_type>;
    static_assert(std::is_same<const_accessor1d,
                               typename accessor1d::const_accessor>::value,
                  "Const accessors must be the same!");
    static_assert(std::is_same<const_accessor2d,
                               typename accessor2d::const_accessor>::value,
                  "Const accessors must be the same!");

    using reduced_storage1d = gko::acc::range<accessor1d>;
    using reduced_storage2d = gko::acc::range<accessor2d>;
    using const_reduced_storage2d = gko::acc::range<const_accessor2d>;
    using const_reduced_storage1d = gko::acc::range<const_accessor1d>;

    const std::array<size_type, 0> stride0{{}};
    const std::array<size_type, 1> stride1{{4}};
    const std::array<size_type, 1> size_1d{{8}};
    const std::array<size_type, 2> size_2d{{2, 4}};
    static constexpr size_type data_elements{8};
    st_type data[data_elements]{1.1f, 2.2f, 3.3f, 4.4f,
                                5.5f, 6.6f, 7.7f, -8.8f};
    reduced_storage1d r1{size_1d, data};
    reduced_storage2d r2{size_2d, data, stride1[0]};
    const_reduced_storage1d cr1{size_1d, data, stride0};
    const_reduced_storage2d cr2{size_2d, data, stride1};

    template <typename T>
    static ar_type c_st_ar(T val)
    {
        return static_cast<ar_type>(static_cast<st_type>(val));
    }

    void data_equal_except_for(int idx)
    {
        // clang-format off
        if (idx != 0) { EXPECT_EQ(data[0], c_st_ar(1.1)); }
        if (idx != 1) { EXPECT_EQ(data[1], c_st_ar(2.2)); }
        if (idx != 2) { EXPECT_EQ(data[2], c_st_ar(3.3)); }
        if (idx != 3) { EXPECT_EQ(data[3], c_st_ar(4.4)); }
        if (idx != 4) { EXPECT_EQ(data[4], c_st_ar(5.5)); }
        if (idx != 5) { EXPECT_EQ(data[5], c_st_ar(6.6)); }
        if (idx != 6) { EXPECT_EQ(data[6], c_st_ar(7.7)); }
        if (idx != 7) { EXPECT_EQ(data[7], c_st_ar(-8.8)); }
        // clang-format on
    }
};


TYPED_TEST_SUITE(ReducedStorageXd, gko::acc::test::IndexSizeTypes);


TYPED_TEST(ReducedStorageXd, CanRead)
{
    EXPECT_EQ(this->cr1(1), this->c_st_ar(2.2));
    EXPECT_EQ(this->cr2(0, 1), this->c_st_ar(2.2));
    EXPECT_EQ(this->r1(1), this->c_st_ar(2.2));
    EXPECT_EQ(this->r2(0, 1), this->c_st_ar(2.2));
}


TYPED_TEST(ReducedStorageXd, CanWrite1)
{
    this->r1(2) = 0.25;

    this->data_equal_except_for(2);
    EXPECT_EQ(this->r1(2), 0.25);  // expect exact since easy to store
}


TYPED_TEST(ReducedStorageXd, CanWrite2)
{
    this->r2(1, 1) = 0.75;

    this->data_equal_except_for(5);
    EXPECT_EQ(this->r2(1, 1), 0.75);  // expect exact since easy to store
}


TYPED_TEST(ReducedStorageXd, CanCreateConstSubrange)
{
    using span = gko::acc::index_span;
    auto sub = this->r2(span{0, 2}, span{1, 3});
    auto const_sub = sub->to_const();

    static_assert(std::is_same<decltype(const_sub.length(0)),
                               typename TypeParam::size_type>::value);
    EXPECT_EQ(const_sub.length(1), 2);
    EXPECT_EQ(const_sub(0, 0), this->c_st_ar(2.2));
    EXPECT_EQ(const_sub(1, 1), this->c_st_ar(7.7));
}


}  // namespace
