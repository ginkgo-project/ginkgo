// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cinttypes>
#include <complex>
#include <limits>
#include <tuple>
#include <type_traits>


#include <gtest/gtest.h>


#include "accessor/math.hpp"
#include "accessor/reduced_row_major_reference.hpp"
#include "accessor/utils.hpp"


namespace {


template <typename ArithmeticStorageType>
class ReducedRowMajorReference : public ::testing::Test {
public:
    using ar_type =
        typename std::tuple_element<0, decltype(ArithmeticStorageType{})>::type;
    using st_type =
        typename std::tuple_element<1, decltype(ArithmeticStorageType{})>::type;
    using ref_type =
        gko::acc::reference_class::reduced_storage<ar_type, st_type>;
    using const_ref_type =
        gko::acc::reference_class::reduced_storage<ar_type, const st_type>;

protected:
    ReducedRowMajorReference() : storage{16} {}

    // writing only works on rvalue reference
    // and reading is only guaranteed to work on rvalue references
    auto get_ref() { return ref_type{&storage}; }

    auto get_const_ref() { return const_ref_type{&storage}; }

    auto get_conv_storage() { return static_cast<ar_type>(storage); }

    st_type storage;
};


using ReferenceTypes =
    ::testing::Types<std::tuple<std::int16_t, std::int8_t>,
                     std::tuple<std::int32_t, std::int16_t>,
                     std::tuple<std::int64_t, std::int16_t>,
                     std::tuple<double, std::int32_t>,
                     std::tuple<double, double>, std::tuple<double, float>,
                     std::tuple<float, float>,
                     std::tuple<std::complex<double>, std::complex<double>>,
                     std::tuple<std::complex<double>, std::complex<float>>,
                     std::tuple<std::complex<float>, std::complex<float>>>;

TYPED_TEST_SUITE(ReducedRowMajorReference, ReferenceTypes);


TYPED_TEST(ReducedRowMajorReference, Read)
{
    using ar_type = typename TestFixture::ar_type;
    using st_type = typename TestFixture::st_type;

    ar_type test = this->get_ref();
    ar_type c_test = this->get_const_ref();

    ASSERT_EQ(test, ar_type{16});
    ASSERT_EQ(test, this->get_conv_storage());
    ASSERT_EQ(this->get_ref(), static_cast<ar_type>(this->storage));
    ASSERT_EQ(c_test, this->get_conv_storage());
    ASSERT_EQ(static_cast<ar_type>(this->get_const_ref()),
              static_cast<ar_type>(this->storage));
}


TYPED_TEST(ReducedRowMajorReference, Write)
{
    using ar_type = typename TestFixture::ar_type;
    using st_type = typename TestFixture::st_type;
    const auto to_write = static_cast<ar_type>(19.1);

    this->get_ref() = to_write;

    ASSERT_EQ(this->storage, static_cast<st_type>(to_write));
    ASSERT_EQ(static_cast<ar_type>(this->get_ref()), this->get_conv_storage());
}


TYPED_TEST(ReducedRowMajorReference, Multiplication)
{
    using ar_type = typename TestFixture::ar_type;
    const ar_type mult{3};
    const ar_type expected_res =
        static_cast<ar_type>(this->get_conv_storage() * mult);
    const ar_type expected_self_res = static_cast<ar_type>(
        this->get_conv_storage() * this->get_conv_storage());

    auto res1 = mult * this->get_ref();
    auto res2 = this->get_ref() * mult;
    auto res3 = this->get_const_ref() * mult;
    auto res4 = mult * this->get_const_ref();
    // Not supported because of overload ambiguity:
    // auto self_res1 = this->get_ref() * this->get_const_ref();
    // auto self_res2 = this->get_const_ref() * this->get_ref();
    auto self_res1 = this->get_ref() * this->get_ref();
    auto self_res2 = this->get_const_ref() * this->get_const_ref();

    static_assert(std::is_same<decltype(res1), ar_type>::value,
                  "Types must match!");
    static_assert(std::is_same<decltype(self_res1), ar_type>::value,
                  "Types must match!");
    ASSERT_EQ(res1, expected_res);
    ASSERT_EQ(res2, expected_res);
    ASSERT_EQ(res3, expected_res);
    ASSERT_EQ(res4, expected_res);
    ASSERT_EQ(self_res1, expected_self_res);
    ASSERT_EQ(self_res2, expected_self_res);
}


TYPED_TEST(ReducedRowMajorReference, Division)
{
    using ar_type = typename TestFixture::ar_type;
    const ar_type div{4};
    const ar_type expected_res =
        static_cast<ar_type>(this->get_conv_storage() / div);
    const ar_type expected_self_res{1};

    auto res1 = ar_type{16} / this->get_ref();
    auto res2 = this->get_ref() / div;
    auto res3 = this->get_const_ref() / div;
    auto self_res1 = this->get_ref() / this->get_ref();
    auto self_res2 = this->get_const_ref() / this->get_const_ref();

    static_assert(std::is_same<decltype(res1), ar_type>::value,
                  "Types must match!");
    static_assert(std::is_same<decltype(self_res1), ar_type>::value,
                  "Types must match!");
    ASSERT_EQ(res1, ar_type{1});
    ASSERT_EQ(res2, expected_res);
    ASSERT_EQ(res3, expected_res);
    ASSERT_EQ(self_res1, expected_self_res);
    ASSERT_EQ(self_res2, expected_self_res);
}


TYPED_TEST(ReducedRowMajorReference, Plus)
{
    using ar_type = typename TestFixture::ar_type;
    const ar_type plus{3};
    const ar_type expected_res =
        static_cast<ar_type>(this->get_conv_storage() + plus);
    const ar_type expected_self_res = static_cast<ar_type>(
        this->get_conv_storage() + this->get_conv_storage());

    auto res1 = plus + this->get_ref();
    auto res2 = this->get_ref() + plus;
    auto res3 = this->get_const_ref() + plus;
    auto res4 = plus + this->get_const_ref();
    auto self_res1 = this->get_ref() + this->get_ref();
    auto self_res2 = this->get_const_ref() + this->get_const_ref();

    static_assert(std::is_same<decltype(res1), ar_type>::value,
                  "Types must match!");
    static_assert(std::is_same<decltype(self_res1), ar_type>::value,
                  "Types must match!");
    ASSERT_EQ(res1, expected_res);
    ASSERT_EQ(res2, expected_res);
    ASSERT_EQ(res3, expected_res);
    ASSERT_EQ(res4, expected_res);
    ASSERT_EQ(self_res1, expected_self_res);
    ASSERT_EQ(self_res2, expected_self_res);
}


TYPED_TEST(ReducedRowMajorReference, Minus)
{
    using ar_type = typename TestFixture::ar_type;
    const ar_type minus{3};
    const ar_type expected_res =
        static_cast<ar_type>(this->get_conv_storage() - minus);
    const ar_type expected_self_res{0};

    auto res1 = static_cast<ar_type>(this->get_conv_storage() + ar_type{1}) -
                this->get_ref();
    auto res2 = this->get_ref() - minus;
    auto res3 = this->get_const_ref() - minus;
    auto self_res1 = this->get_ref() - this->get_ref();
    auto self_res2 = this->get_const_ref() - this->get_const_ref();

    static_assert(std::is_same<decltype(res1), ar_type>::value,
                  "Types must match!");
    static_assert(std::is_same<decltype(self_res1), ar_type>::value,
                  "Types must match!");
    ASSERT_EQ(res1, ar_type{1});
    ASSERT_EQ(res2, expected_res);
    ASSERT_EQ(res3, expected_res);
    ASSERT_EQ(self_res1, expected_self_res);
    ASSERT_EQ(self_res2, expected_self_res);
}


TYPED_TEST(ReducedRowMajorReference, UnaryMinus)
{
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected_res =
        static_cast<ar_type>(-this->get_conv_storage());

    auto res1 = -this->get_ref();
    auto res2 = -this->get_const_ref();

    static_assert(std::is_same<decltype(res1), ar_type>::value,
                  "Types must match!");
    static_assert(std::is_same<decltype(res2), ar_type>::value,
                  "Types must match!");
    ASSERT_EQ(res1, expected_res);
    ASSERT_EQ(res2, expected_res);
}


TYPED_TEST(ReducedRowMajorReference, UnaryPlus)
{
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected_res =
        static_cast<ar_type>(+this->get_conv_storage());

    auto res1 = +this->get_ref();
    auto res2 = +this->get_const_ref();

    static_assert(std::is_same<decltype(res1), ar_type>::value,
                  "Types must match!");
    static_assert(std::is_same<decltype(res2), ar_type>::value,
                  "Types must match!");
    ASSERT_EQ(res1, expected_res);
    ASSERT_EQ(res2, expected_res);
}


TYPED_TEST(ReducedRowMajorReference, MultEquals)
{
    using ar_type = typename TestFixture::ar_type;
    using st_type = typename TestFixture::st_type;
    using ref_type = typename TestFixture::ref_type;
    const ar_type mult{2};
    st_type storage{3};
    const ar_type new_value{5};
    auto local_ref = [&storage]() { return ref_type{&storage}; };
    const ar_type expected_res1 =
        static_cast<ar_type>(mult * static_cast<ar_type>(storage));
    const ar_type expected_res2 =
        static_cast<ar_type>(expected_res1 * this->get_conv_storage());
    const ar_type expected_res3 = ar_type{new_value * new_value};

    local_ref() *= mult;
    ar_type res1 = local_ref();
    local_ref() *= this->get_const_ref();
    ar_type res2 = local_ref();
    this->get_ref() = new_value;
    this->get_ref() *= this->get_ref();
    ar_type res3 = this->get_ref();

    ASSERT_EQ(res1, expected_res1);
    ASSERT_EQ(res2, expected_res2);
    ASSERT_EQ(res3, expected_res3);
}


TYPED_TEST(ReducedRowMajorReference, DivEquals)
{
    using ar_type = typename TestFixture::ar_type;
    using st_type = typename TestFixture::st_type;
    using ref_type = typename TestFixture::ref_type;
    const ar_type div{2};
    st_type storage{64};
    const ar_type new_value{5};
    auto local_ref = [&storage]() { return ref_type{&storage}; };
    const ar_type expected_res1 =
        static_cast<ar_type>(static_cast<ar_type>(storage) / div);
    const ar_type expected_res2 =
        static_cast<ar_type>(expected_res1 / this->get_conv_storage());
    const ar_type expected_res3 = ar_type{1};

    local_ref() /= div;
    ar_type res1 = local_ref();
    local_ref() /= this->get_const_ref();
    ar_type res2 = local_ref();
    this->get_ref() = new_value;
    this->get_ref() /= this->get_ref();
    ar_type res3 = this->get_ref();

    ASSERT_EQ(res1, expected_res1);
    ASSERT_EQ(res2, expected_res2);
    ASSERT_EQ(res3, expected_res3);
}


TYPED_TEST(ReducedRowMajorReference, PlusEquals)
{
    using ar_type = typename TestFixture::ar_type;
    using st_type = typename TestFixture::st_type;
    using ref_type = typename TestFixture::ref_type;
    const ar_type plus{7};
    st_type storage{13};
    const ar_type new_value{5};
    auto local_ref = [&storage]() { return ref_type{&storage}; };
    const ar_type expected_res1 =
        static_cast<ar_type>(static_cast<ar_type>(storage) + plus);
    const ar_type expected_res2 =
        static_cast<ar_type>(expected_res1 + this->get_conv_storage());
    const ar_type expected_res3 = ar_type{new_value + new_value};

    local_ref() += plus;
    ar_type res1 = local_ref();
    local_ref() += this->get_const_ref();
    ar_type res2 = local_ref();
    this->get_ref() = new_value;
    this->get_ref() += this->get_ref();
    ar_type res3 = this->get_ref();

    ASSERT_EQ(res1, expected_res1);
    ASSERT_EQ(res2, expected_res2);
    ASSERT_EQ(res3, expected_res3);
}


TYPED_TEST(ReducedRowMajorReference, MinusEquals)
{
    using ar_type = typename TestFixture::ar_type;
    using st_type = typename TestFixture::st_type;
    using ref_type = typename TestFixture::ref_type;
    const ar_type minus{7};
    st_type storage{21};
    const ar_type new_value{5};
    auto local_ref = [&storage]() { return ref_type{&storage}; };
    const ar_type expected_res1 =
        static_cast<ar_type>(static_cast<ar_type>(storage) - minus);
    const ar_type expected_res2 =
        static_cast<ar_type>(expected_res1 - this->get_conv_storage());
    const ar_type expected_res3 = ar_type{0};

    local_ref() -= minus;
    ar_type res1 = local_ref();
    local_ref() -= this->get_const_ref();
    ar_type res2 = local_ref();
    this->get_ref() = new_value;
    this->get_ref() -= this->get_ref();
    ar_type res3 = this->get_ref();

    ASSERT_EQ(res1, expected_res1);
    ASSERT_EQ(res2, expected_res2);
    ASSERT_EQ(res3, expected_res3);
}


TYPED_TEST(ReducedRowMajorReference, Abs)
{
    using ar_type = typename TestFixture::ar_type;
    const auto expected_res = this->get_conv_storage();

    auto res1 = abs(this->get_ref());
    auto res2 = abs(this->get_const_ref());
    // Since unsigned types are also used in the test:
    if (std::is_signed<ar_type>::value ||
        gko::acc::is_complex<ar_type>::value) {
        this->get_ref() = -expected_res;
    }
    auto res3 = abs(this->get_ref());

    ASSERT_EQ(res1, expected_res);
    ASSERT_EQ(res2, expected_res);
    ASSERT_EQ(res3, expected_res);
}


TYPED_TEST(ReducedRowMajorReference, Real)
{
    using ar_type = typename TestFixture::ar_type;
    using gko::acc::real;  // required by some compilers, so ADL works properly
    const auto expected_res = this->get_conv_storage();

    auto res1 = real(this->get_ref());
    auto res2 = real(this->get_const_ref());

    ASSERT_EQ(res1, expected_res);
    ASSERT_EQ(res2, expected_res);
}


TYPED_TEST(ReducedRowMajorReference, Imag)
{
    using ar_type = typename TestFixture::ar_type;
    using gko::acc::imag;  // required by some compilers, so ADL works properly
    const ar_type expected_res{0};

    auto res1 = imag(this->get_ref());
    auto res2 = imag(this->get_const_ref());

    ASSERT_EQ(res1, expected_res);
    ASSERT_EQ(res2, expected_res);
}


TYPED_TEST(ReducedRowMajorReference, Conj)
{
    using ar_type = typename TestFixture::ar_type;
    using gko::acc::conj;  // required by some compilers, so ADL works properly
    const auto expected_res = this->get_conv_storage();

    auto res1 = conj(this->get_ref());
    auto res2 = conj(this->get_const_ref());

    ASSERT_EQ(res1, expected_res);
    ASSERT_EQ(res2, expected_res);
}


TYPED_TEST(ReducedRowMajorReference, SquaredNorm)
{
    using ar_type = typename TestFixture::ar_type;
    using gko::acc::squared_norm;  // required by some compilers, so ADL works
                                   // properly
    const auto expected_res =
        this->get_conv_storage() * this->get_conv_storage();

    auto res1 = squared_norm(this->get_ref());
    auto res2 = squared_norm(this->get_const_ref());
    // Since unsigned types are also used in the test:
    if (std::is_signed<ar_type>::value ||
        gko::acc::is_complex<ar_type>::value) {
        this->get_ref() = -this->get_ref();
    }
    auto res3 = squared_norm(this->get_ref());

    ASSERT_EQ(res1, expected_res);
    ASSERT_EQ(res2, expected_res);
    ASSERT_EQ(res3, expected_res);
}


template <typename ArithmeticStorageType>
class ComplexReducedRowMajorReference : public ::testing::Test {
public:
    using ar_type =
        typename std::tuple_element<0, decltype(ArithmeticStorageType{})>::type;
    using st_type =
        typename std::tuple_element<1, decltype(ArithmeticStorageType{})>::type;
    using rc_ar_type = gko::acc::remove_complex_t<ar_type>;
    using rc_st_type = gko::acc::remove_complex_t<st_type>;
    using ref_type =
        gko::acc::reference_class::reduced_storage<ar_type, st_type>;
    using const_ref_type =
        gko::acc::reference_class::reduced_storage<ar_type, const st_type>;

    const rc_ar_type delta{std::numeric_limits<rc_st_type>::epsilon()};

protected:
    ComplexReducedRowMajorReference() : storage{16.3, -12.19} {}

    // writing only works on rvalue reference
    // and reading is only guaranteed to work on rvalue references
    auto get_ref() { return ref_type{&storage}; }

    auto get_const_ref() { return const_ref_type{&storage}; }

    auto get_conv_storage() { return static_cast<ar_type>(storage); }

    st_type storage;
};


using ComplexReferenceTypes =
    ::testing::Types<std::tuple<std::complex<double>, std::complex<double>>,
                     std::tuple<std::complex<double>, std::complex<float>>,
                     std::tuple<std::complex<float>, std::complex<float>>>;

TYPED_TEST_SUITE(ComplexReducedRowMajorReference, ComplexReferenceTypes);


TYPED_TEST(ComplexReducedRowMajorReference, Abs)
{
    using ar_type = typename TestFixture::ar_type;
    using rc_ar_type = typename TestFixture::rc_ar_type;
    const rc_ar_type expected_res{std::abs(this->get_conv_storage())};

    auto res1 = abs(this->get_ref());
    auto res2 = abs(this->get_const_ref());
    this->get_ref() = -expected_res;
    auto res3 = abs(this->get_ref());

    // with sqrt in the computation, we need a slightly bigger delta
    ASSERT_NEAR(res1, expected_res, this->delta * 8);
    ASSERT_NEAR(res2, expected_res, this->delta * 8);
    ASSERT_NEAR(res3, expected_res, this->delta * 8);
}


TYPED_TEST(ComplexReducedRowMajorReference, Real)
{
    using ar_type = typename TestFixture::ar_type;
    using gko::acc::real;  // required by some compilers, so ADL works properly
    const auto expected_res = std::real(this->get_conv_storage());

    auto res1 = real(this->get_ref());
    auto res2 = real(this->get_const_ref());

    ASSERT_NEAR(res1, expected_res, this->delta);
    ASSERT_NEAR(res2, expected_res, this->delta);
}


TYPED_TEST(ComplexReducedRowMajorReference, Imag)
{
    using ar_type = typename TestFixture::ar_type;
    using gko::acc::imag;  // required by some compilers, so ADL works properly
    const auto expected_res = std::imag(this->get_conv_storage());

    auto res1 = imag(this->get_ref());
    auto res2 = imag(this->get_const_ref());

    ASSERT_NEAR(res1, expected_res, this->delta);
    ASSERT_NEAR(res2, expected_res, this->delta);
}


TYPED_TEST(ComplexReducedRowMajorReference, Conj)
{
    using ar_type = typename TestFixture::ar_type;
    using gko::acc::conj;  // required by some compilers, so ADL works properly
    const auto expected_res = std::conj(this->get_conv_storage());

    auto res1 = conj(this->get_ref());
    auto res2 = conj(this->get_const_ref());

    ASSERT_NEAR(std::real(res1), std::real(expected_res), this->delta);
    ASSERT_NEAR(std::imag(res1), std::imag(expected_res), this->delta);
    ASSERT_NEAR(std::real(res2), std::real(expected_res), this->delta);
    ASSERT_NEAR(std::imag(res2), std::imag(expected_res), this->delta);
}


TYPED_TEST(ComplexReducedRowMajorReference, SquaredNorm)
{
    using ar_type = typename TestFixture::ar_type;
    using rc_ar_type = typename TestFixture::rc_ar_type;
    using gko::acc::squared_norm;  // required by some compilers, so ADL works
                                   // properly
    const auto curr_val = this->get_conv_storage();
    const rc_ar_type expected_res{std::real(
        this->get_conv_storage() * std::conj(this->get_conv_storage()))};

    auto res1 = squared_norm(this->get_ref());
    auto res2 = squared_norm(this->get_ref());
    this->get_ref() = -this->get_ref();
    auto res3 = squared_norm(this->get_ref());

    ASSERT_NEAR(res1, expected_res, this->delta);
    ASSERT_NEAR(res2, expected_res, this->delta);
    ASSERT_NEAR(res3, expected_res, this->delta);
}


}  // namespace
