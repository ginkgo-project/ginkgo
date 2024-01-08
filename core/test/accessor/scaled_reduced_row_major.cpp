// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <cinttypes>
#include <limits>
#include <tuple>
#include <type_traits>


#include <gtest/gtest.h>


#include "accessor/index_span.hpp"
#include "accessor/range.hpp"
#include "accessor/scaled_reduced_row_major.hpp"


namespace {


template <typename ArithmeticStorageType>
class ScaledReducedStorage3d : public ::testing::Test {
protected:
    using ar_type =
        typename std::tuple_element<0, decltype(ArithmeticStorageType{})>::type;
    using st_type =
        typename std::tuple_element<1, decltype(ArithmeticStorageType{})>::type;
    // Type for `check_accessor_correctness` to forward the indices
    using t = std::tuple<int, int, int>;
    using i_span = gko::acc::index_span;

    static constexpr ar_type delta{std::numeric_limits<ar_type>::epsilon() *
                                   1e1};

    using accessor =
        gko::acc::scaled_reduced_row_major<3, ar_type, st_type, 0b0101>;
    using const_accessor =
        gko::acc::scaled_reduced_row_major<3, ar_type, const st_type, 0b0101>;

    using reduced_storage = gko::acc::range<accessor>;
    using const_reduced_storage = gko::acc::range<const_accessor>;

    const std::array<gko::acc::size_type, 3> size{{1u, 4u, 2u}};
    static constexpr gko::acc::size_type data_elements{8};
    static constexpr gko::acc::size_type scalar_elements{8};
    // clang-format off
    st_type data[8]{
        10, 11,
        -12, 13,
        14, -115,
        6, 77
    };
    ar_type scalar[scalar_elements]{
        1., 2., 3., 4., 5., 6., 7., 8.
    };
    // clang-format on
    const std::array<gko::acc::size_type, 2> storage_stride{{8, 2}};
    const std::array<gko::acc::size_type, 1> scalar_stride{{2}};
    reduced_storage r{size, data, storage_stride, scalar, scalar_stride};
    const_reduced_storage cr{size, data, scalar};

    template <typename Accessor>
    static void check_accessor_correctness(
        const Accessor& a,
        std::tuple<int, int, int> ignore = std::tuple<int, int, int>(99, 99,
                                                                     99))
    {
        // Test for equality is fine here since they should not be modified
        // clang-format off
        if (ignore != t(0, 0, 0)) { EXPECT_EQ(a(0, 0, 0), ar_type{10.});   }
        if (ignore != t(0, 0, 1)) { EXPECT_EQ(a(0, 0, 1), ar_type{22.});   }
        if (ignore != t(0, 1, 0)) { EXPECT_EQ(a(0, 1, 0), ar_type{-12.});  }
        if (ignore != t(0, 1, 1)) { EXPECT_EQ(a(0, 1, 1), ar_type{26.});   }
        if (ignore != t(0, 2, 0)) { EXPECT_EQ(a(0, 2, 0), ar_type{14.});   }
        if (ignore != t(0, 2, 1)) { EXPECT_EQ(a(0, 2, 1), ar_type{-230.}); }
        if (ignore != t(0, 3, 0)) { EXPECT_EQ(a(0, 3, 0), ar_type{6.});    }
        if (ignore != t(0, 3, 1)) { EXPECT_EQ(a(0, 3, 1), ar_type{154.});  }
        // clang-format on
    }
};

using ScaledReducedStorage3dTypes = ::testing::Types<
    std::tuple<double, std::int64_t>, std::tuple<double, std::int32_t>,
    std::tuple<double, std::int16_t>, std::tuple<float, std::int32_t>,
    std::tuple<float, std::int16_t>>;

TYPED_TEST_SUITE(ScaledReducedStorage3d, ScaledReducedStorage3dTypes);


TYPED_TEST(ScaledReducedStorage3d, CorrectLengths)
{
    EXPECT_EQ(this->r.length(0), this->size[0]);
    EXPECT_EQ(this->r.length(1), this->size[1]);
    EXPECT_EQ(this->r.length(2), this->size[2]);
    EXPECT_EQ(this->r.length(3), 1);
    EXPECT_EQ(this->r->get_size(), this->size);
}


TYPED_TEST(ScaledReducedStorage3d, CorrectStride)
{
    EXPECT_EQ(this->r->get_scalar_stride(), this->scalar_stride);
    EXPECT_EQ(this->r->get_storage_stride(), this->storage_stride);
}


TYPED_TEST(ScaledReducedStorage3d, CorrectStorage)
{
    EXPECT_EQ(this->r->get_stored_data(), this->data);
    EXPECT_EQ(this->r->get_const_storage(), this->data);
}


TYPED_TEST(ScaledReducedStorage3d, CorrectScale)
{
    EXPECT_EQ(this->r->get_scalar(), this->scalar);
    EXPECT_EQ(this->r->get_const_scalar(), this->scalar);
}


TYPED_TEST(ScaledReducedStorage3d, CanReadData)
{
    this->check_accessor_correctness(this->r);
    this->check_accessor_correctness(this->cr);
}


TYPED_TEST(ScaledReducedStorage3d, CanImplicitlyConvertToConst)
{
    using const_reduced_storage = typename TestFixture::const_reduced_storage;

    const_reduced_storage const_rs = this->r->to_const();
    const_reduced_storage const_rs2 = this->cr;

    this->check_accessor_correctness(const_rs);
    this->check_accessor_correctness(const_rs2);
}


TYPED_TEST(ScaledReducedStorage3d, ToConstWorks)
{
    using const_reduced_storage = typename TestFixture::const_reduced_storage;
    auto cr2 = this->r->to_const();

    static_assert(std::is_same<decltype(cr2), const_reduced_storage>::value,
                  "Types must be equal!");
    this->check_accessor_correctness(cr2);
}


TYPED_TEST(ScaledReducedStorage3d, CanCreateWithStride)
{
    using reduced_storage = typename TestFixture::reduced_storage;
    using ar_type = typename TestFixture::ar_type;
    std::array<gko::acc::size_type, 3> size{{2, 1, 2}};
    std::array<gko::acc::size_type, 2> stride_storage{{5, 2}};
    std::array<gko::acc::size_type, 1> stride_scalar{{4}};

    reduced_storage range{size, this->data, stride_storage, this->scalar,
                          stride_scalar};
    range(1, 0, 0) = ar_type{15};

    EXPECT_EQ(range(0, 0, 0), ar_type{10});
    EXPECT_EQ(range(0, 0, 1), ar_type{22});
    EXPECT_EQ(range(1, 0, 0), ar_type{15});
    EXPECT_EQ(range(1, 0, 1), ar_type{36});
}


TYPED_TEST(ScaledReducedStorage3d, Subrange)
{
    using i_span = typename TestFixture::i_span;
    auto subr = this->cr(0u, i_span{0u, 2u}, 1u);

    EXPECT_EQ(subr(0, 0, 0), 22.);
    EXPECT_EQ(subr(0, 1, 0), 26.);
}


TYPED_TEST(ScaledReducedStorage3d, CanWriteScale)
{
    using ar_type = typename TestFixture::ar_type;

    this->r->write_scalar_masked(10., 0, 0, 0);

    EXPECT_EQ(this->r(0, 0, 0), ar_type{100.});
    EXPECT_EQ(this->r(0, 0, 1), ar_type{22.});
    EXPECT_EQ(this->r(0, 1, 0), ar_type{-120.});
    EXPECT_EQ(this->r(0, 1, 1), ar_type{26.});
    EXPECT_EQ(this->r(0, 2, 0), ar_type{140.});
    EXPECT_EQ(this->r(0, 2, 1), ar_type{-230.});
    EXPECT_EQ(this->r(0, 3, 0), ar_type{60.});
    EXPECT_EQ(this->r(0, 3, 1), ar_type{154.});
}


TYPED_TEST(ScaledReducedStorage3d, CanWriteMaskedScale)
{
    using ar_type = typename TestFixture::ar_type;

    this->r->write_scalar_direct(10., 0, 0);

    EXPECT_EQ(this->r(0, 0, 0), ar_type{100.});
    EXPECT_EQ(this->r(0, 0, 1), ar_type{22.});
    EXPECT_EQ(this->r(0, 1, 0), ar_type{-120.});
    EXPECT_EQ(this->r(0, 1, 1), ar_type{26.});
    EXPECT_EQ(this->r(0, 2, 0), ar_type{140.});
    EXPECT_EQ(this->r(0, 2, 1), ar_type{-230.});
    EXPECT_EQ(this->r(0, 3, 0), ar_type{60.});
    EXPECT_EQ(this->r(0, 3, 1), ar_type{154.});
}


TYPED_TEST(ScaledReducedStorage3d, CanReadScale)
{
    EXPECT_EQ(this->r->read_scalar_masked(0, 0, 0), 1.);
    EXPECT_EQ(this->r->read_scalar_masked(0, 0, 1), 2.);
    EXPECT_EQ(this->r->read_scalar_direct(0, 0), 1.);
    EXPECT_EQ(this->r->read_scalar_direct(0, 1), 2.);
}


TYPED_TEST(ScaledReducedStorage3d, Addition)
{
    using ar_type = typename TestFixture::ar_type;
    using t = typename TestFixture::t;
    const ar_type expected = 10. + 3.;

    const auto result = this->cr(0, 0, 0) + 3.;
    this->r(0, 0, 0) += 3.;

    this->check_accessor_correctness(this->r, t(0, 0, 0));
    EXPECT_NEAR(this->r(0, 0, 0), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ScaledReducedStorage3d, Addition2)
{
    using ar_type = typename TestFixture::ar_type;
    using t = typename TestFixture::t;
    const ar_type expected = 10. + 22.;

    const auto result = this->cr(0, 0, 0) + this->r(0, 0, 1);
    this->r(0, 0, 0) += this->cr(0, 0, 1);

    this->check_accessor_correctness(this->r, t(0, 0, 0));
    EXPECT_NEAR(this->r(0, 0, 0), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ScaledReducedStorage3d, Subtraction)
{
    using ar_type = typename TestFixture::ar_type;
    using t = typename TestFixture::t;
    const ar_type expected = 22. - 2.;

    const auto result = this->cr(0, 0, 1) - 2.;
    this->r(0, 0, 1) -= 2.;

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(this->r(0, 0, 1), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ScaledReducedStorage3d, Subtraction2)
{
    using ar_type = typename TestFixture::ar_type;
    using t = typename TestFixture::t;
    const ar_type expected = -12. - 26.;

    const auto result = this->cr(0, 1, 0) - this->r(0, 1, 1);
    this->r(0, 1, 0) -= this->r(0, 1, 1);

    this->check_accessor_correctness(this->r, t(0, 1, 0));
    EXPECT_NEAR(this->r(0, 1, 0), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ScaledReducedStorage3d, Multiplication)
{
    using ar_type = typename TestFixture::ar_type;
    using t = typename TestFixture::t;
    const ar_type expected = 26. * 3.;

    const auto result = this->cr(0, 1, 1) * 3.;
    this->r(0, 1, 1) *= 3.;

    this->check_accessor_correctness(this->r, t(0, 1, 1));
    EXPECT_NEAR(this->r(0, 1, 1), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ScaledReducedStorage3d, Multiplication2)
{
    using ar_type = typename TestFixture::ar_type;
    using t = typename TestFixture::t;
    const ar_type expected = 14. * 10.;

    const auto result = this->r(0, 2, 0) * this->r(0, 0, 0);
    this->r(0, 2, 0) *= this->r(0, 0, 0);

    this->check_accessor_correctness(this->r, t(0, 2, 0));
    EXPECT_NEAR(this->r(0, 2, 0), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ScaledReducedStorage3d, Division)
{
    using ar_type = typename TestFixture::ar_type;
    using t = typename TestFixture::t;
    const ar_type expected = 10. / 2.;

    const auto result = this->cr(0, 0, 0) / 2.;
    this->r(0, 0, 0) /= 2.;

    this->check_accessor_correctness(this->r, t(0, 0, 0));
    EXPECT_NEAR(this->r(0, 0, 0), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ScaledReducedStorage3d, Division2)
{
    using ar_type = typename TestFixture::ar_type;
    using t = typename TestFixture::t;
    const ar_type expected = -12. / 6.;

    const auto result = this->r(0, 1, 0) / this->r(0, 3, 0);
    this->r(0, 1, 0) /= this->r(0, 3, 0);

    this->check_accessor_correctness(this->r, t(0, 1, 0));
    EXPECT_NEAR(this->r(0, 1, 0), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ScaledReducedStorage3d, UnaryMinus)
{
    using ar_type = typename TestFixture::ar_type;
    using t = typename TestFixture::t;
    const ar_type neg_expected = this->r(0, 1, 1);
    const ar_type expected = -neg_expected;

    auto result = -this->r(0, 1, 1);

    this->check_accessor_correctness(this->r);
    EXPECT_EQ(result, expected);
}


class ScaledReducedStorageXd : public ::testing::Test {
protected:
    using ar_type = double;
    using st_type = int;
    using size_type = gko::acc::size_type;
    static constexpr ar_type delta{0.1};

    using accessor1d =
        gko::acc::scaled_reduced_row_major<1, ar_type, st_type, 1>;
    using accessor2d =
        gko::acc::scaled_reduced_row_major<2, ar_type, st_type, 3>;
    using const_accessor1d =
        gko::acc::scaled_reduced_row_major<1, ar_type, const st_type, 1>;
    using const_accessor2d =
        gko::acc::scaled_reduced_row_major<2, ar_type, const st_type, 3>;
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
    const std::array<size_type, 1> stride_sc{{5}};
    const std::array<gko::acc::size_type, 1> size_1d{{8u}};
    const std::array<gko::acc::size_type, 2> size_2d{{2u, 2u}};

    static constexpr gko::acc::size_type data_elements{8};
    st_type data[data_elements]{10, 22, 32, 44, 54, 66, 76, -88};
    ar_type scalar[data_elements]{1e0,  5e-1, 1e-1, 5e-2,
                                  1e-2, 5e-3, 1e-3, 5e-4};

    reduced_storage1d r1{size_1d, data, scalar};
    reduced_storage2d r2{size_2d, data, stride1, scalar, stride_sc};
    const_reduced_storage1d cr1{size_1d, data, stride0, scalar};
    const_reduced_storage2d cr2{size_2d, data, stride1, scalar, stride_sc};

    void data_equal_except_for(int idx)
    {
        // clang-format off
        if (idx != 0) { EXPECT_EQ(data[0], 10); }
        if (idx != 1) { EXPECT_EQ(data[1], 22); }
        if (idx != 2) { EXPECT_EQ(data[2], 32); }
        if (idx != 3) { EXPECT_EQ(data[3], 44); }
        if (idx != 4) { EXPECT_EQ(data[4], 54); }
        if (idx != 5) { EXPECT_EQ(data[5], 66); }
        if (idx != 6) { EXPECT_EQ(data[6], 76); }
        if (idx != 7) { EXPECT_EQ(data[7], -88); }
        // clang-format on
    }
    void scalar_equal_except_for(int idx)
    {
        // clang-format off
        if (idx != 0) { EXPECT_EQ(scalar[0], ar_type{1e0}); }
        if (idx != 1) { EXPECT_EQ(scalar[1], ar_type{5e-1}); }
        if (idx != 2) { EXPECT_EQ(scalar[2], ar_type{1e-1}); }
        if (idx != 3) { EXPECT_EQ(scalar[3], ar_type{5e-2}); }
        if (idx != 4) { EXPECT_EQ(scalar[4], ar_type{1e-2}); }
        if (idx != 5) { EXPECT_EQ(scalar[5], ar_type{5e-3}); }
        if (idx != 6) { EXPECT_EQ(scalar[6], ar_type{1e-3}); }
        if (idx != 7) { EXPECT_EQ(scalar[7], ar_type{5e-4}); }
        // clang-format on
    }
};


TEST_F(ScaledReducedStorageXd, CanRead)
{
    EXPECT_NEAR(cr1(1), 11., delta);
    EXPECT_NEAR(cr2(0, 1), 11., delta);
    EXPECT_NEAR(cr2(1, 1), 66e-3, delta);
    EXPECT_NEAR(r1(1), 11., delta);
    EXPECT_NEAR(r2(0, 1), 11., delta);
    EXPECT_NEAR(r2(1, 1), 66e-3, delta);
}


TEST_F(ScaledReducedStorageXd, CanWrite1)
{
    r1(2) = 0.2;

    data_equal_except_for(2);
    scalar_equal_except_for(99);
    EXPECT_NEAR(r1(2), 0.2, delta);
}


TEST_F(ScaledReducedStorageXd, CanWrite2)
{
    r2(1, 1) = 0.5;

    data_equal_except_for(5);
    scalar_equal_except_for(99);
    EXPECT_NEAR(r2(1, 1), 0.5, delta);
}


}  // namespace
