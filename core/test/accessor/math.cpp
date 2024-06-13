// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cinttypes>
#include <complex>


#include <gtest/gtest.h>


#include "accessor/math.hpp"


namespace {


template <typename ValueType>
class RealMath : public ::testing::Test {
public:
    using value_type = ValueType;
};

using RealTypes = ::testing::Types<std::int8_t, std::int16_t, std::int32_t,
                                   std::int64_t, float, double>;


TYPED_TEST_SUITE(RealMath, RealTypes);


TYPED_TEST(RealMath, Real)
{
    using value_type = typename TestFixture::value_type;
    value_type val{3};

    ASSERT_EQ(gko::acc::real(val), val);
}


TYPED_TEST(RealMath, Imag)
{
    using value_type = typename TestFixture::value_type;
    value_type val{3};

    ASSERT_EQ(gko::acc::imag(val), value_type{});
}


TYPED_TEST(RealMath, Conj)
{
    using value_type = typename TestFixture::value_type;
    value_type val{3};

    ASSERT_EQ(gko::acc::conj(val), val);
}


TYPED_TEST(RealMath, SquaredNorm)
{
    using value_type = typename TestFixture::value_type;
    value_type val{3};
    value_type expected{3 * 3};

    ASSERT_EQ(gko::acc::squared_norm(val), expected);
}


template <typename ValueType>
class ComplexMath : public ::testing::Test {
public:
    using value_type = ValueType;
    using real_type = typename ValueType::value_type;
};

using ComplexTypes =
    ::testing::Types<std::complex<float>, std::complex<double>>;


TYPED_TEST_SUITE(ComplexMath, ComplexTypes);


TYPED_TEST(ComplexMath, Real)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    real_type r{3};
    real_type i{-2};
    value_type val{r, i};

    ASSERT_EQ(gko::acc::real(val), r);
}


TYPED_TEST(ComplexMath, Imag)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    real_type r{3};
    real_type i{-2};
    value_type val{r, i};

    ASSERT_EQ(gko::acc::imag(val), i);
}


TYPED_TEST(ComplexMath, Conj)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    real_type r{3};
    real_type i{-2};
    value_type val{r, i};
    value_type expected{r, -i};

    ASSERT_EQ(gko::acc::conj(val), expected);
}


TYPED_TEST(ComplexMath, SquaredNorm)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    real_type r{3};
    real_type i{-2};
    value_type val{r, i};
    real_type expected{13};

    ASSERT_EQ(gko::acc::squared_norm(val), expected);
}


}  // namespace
