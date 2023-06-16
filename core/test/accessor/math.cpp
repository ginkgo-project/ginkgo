/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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

using RealTypes = ::testing::Types<std::int8_t, std::uint8_t, std::int16_t,
                                   std::uint16_t, std::int32_t, std::uint32_t,
                                   std::int64_t, std::uint64_t, float, double>;


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
