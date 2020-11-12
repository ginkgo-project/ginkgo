/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/base/range_accessors.hpp>


#include <gtest/gtest.h>


#include <tuple>
#include <type_traits>

#include <iostream>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/test/utils.hpp"


namespace {


class RowMajorAccessor : public ::testing::Test {
protected:
    using span = gko::span;

    using row_major_int_range = gko::range<gko::accessor::row_major<int, 2>>;

    // clang-format off
    int data[9]{
        1, 2, -1,
        3, 4, -2,
        5, 6, -3
    };
    // clang-format on
    row_major_int_range r{data, 3u, 2u, 3u};
};


TEST_F(RowMajorAccessor, CanAccessData)
{
    EXPECT_EQ(r(0, 0), 1);
    EXPECT_EQ(r(0, 1), 2);
    EXPECT_EQ(r(1, 0), 3);
    EXPECT_EQ(r(1, 1), 4);
    EXPECT_EQ(r(2, 0), 5);
    EXPECT_EQ(r(2, 1), 6);
}


TEST_F(RowMajorAccessor, CanWriteData)
{
    r(0, 0) = 4;

    EXPECT_EQ(r(0, 0), 4);
}


TEST_F(RowMajorAccessor, CanCreateSubrange)
{
    auto subr = r(span{1, 3}, span{0, 2});

    EXPECT_EQ(subr(0, 0), 3);
    EXPECT_EQ(subr(0, 1), 4);
    EXPECT_EQ(subr(1, 0), 5);
    EXPECT_EQ(subr(1, 1), 6);
}


TEST_F(RowMajorAccessor, CanCreateRowVector)
{
    auto subr = r(2, span{0, 2});

    EXPECT_EQ(subr(0, 0), 5);
    EXPECT_EQ(subr(0, 1), 6);
}


TEST_F(RowMajorAccessor, CanCreateColumnVector)
{
    auto subr = r(span{0, 3}, 0);

    EXPECT_EQ(subr(0, 0), 1);
    EXPECT_EQ(subr(1, 0), 3);
    EXPECT_EQ(subr(2, 0), 5);
}


TEST_F(RowMajorAccessor, CanAssignValues)
{
    r(1, 1) = r(0, 0);

    EXPECT_EQ(data[4], 1);
}


TEST_F(RowMajorAccessor, CanAssignSubranges)
{
    r(0, span{0, 2}) = r(1, span{0, 2});

    EXPECT_EQ(data[0], 3);
    EXPECT_EQ(data[1], 4);
    EXPECT_EQ(data[2], -1);
    EXPECT_EQ(data[3], 3);
    EXPECT_EQ(data[4], 4);
    EXPECT_EQ(data[5], -2);
    EXPECT_EQ(data[6], 5);
    EXPECT_EQ(data[7], 6);
    EXPECT_EQ(data[8], -3);
}


template <typename ArithmeticStorageType>
class ReducedStorage3d : public ::testing::Test {
protected:
    using ar_type =
        typename std::tuple_element<0, decltype(ArithmeticStorageType{})>::type;
    using st_type =
        typename std::tuple_element<1, decltype(ArithmeticStorageType{})>::type;
    static constexpr ar_type delta{::r<st_type>::value * 2};

    // Type for `check_accessor_correctness` to forward the indices
    using t = std::tuple<int, int, int>;

    using accessor = gko::accessor::reduced_row_major<3, ar_type, st_type>;
    using const_accessor =
        gko::accessor::reduced_row_major<3, ar_type, const st_type>;

    using reduced_storage = gko::range<accessor>;
    using const_reduced_storage = gko::range<const_accessor>;

    const gko::dim<3> size{4u, 3u, 2u};
    static constexpr gko::size_type data_elements{4 * 3 * 2};
    // clang-format off
    st_type data[data_elements] {
        // 0, y, z
        1.0, 2.01,
        -1.02, 3.03,
        4.04, -2.05,
        // 1, y, z
        5.06, 6.07,
        2.08, 3.09,
        -1.1, -9.11,
        // 2, y, z
        -2.12, 2.13,
        0.14, 15.15,
        -9.16, 8.17,
        // 3, y, z
        7.18, -6.19,
        5.2, -4.21,
        3.22, -2.23
    };
    // clang-format on
    reduced_storage r{size, data};
    const_reduced_storage cr{size, data};

    template <typename Accessor>
    static void check_accessor_correctness(
        const Accessor &a, std::tuple<int, int, int> ignore = t(99, 99, 99))
    {
        // Test for equality is fine here since they should not be modified
        // clang-format off
        if (ignore != t(0, 0, 0)) { EXPECT_EQ(a(0, 0, 0), st_type{1.0});     }
        if (ignore != t(0, 0, 1)) { EXPECT_EQ(a(0, 0, 1), st_type{2.01});    }
        if (ignore != t(0, 1, 0)) { EXPECT_EQ(a(0, 1, 0), st_type{-1.02});   }
        if (ignore != t(0, 1, 1)) { EXPECT_EQ(a(0, 1, 1), st_type{3.03});    }
        if (ignore != t(0, 2, 0)) { EXPECT_EQ(a(0, 2, 0), st_type{4.04});    }
        if (ignore != t(0, 2, 1)) { EXPECT_EQ(a(0, 2, 1), st_type{-2.05});   }
        if (ignore != t(1, 0, 0)) { EXPECT_EQ(a(1, 0, 0), st_type{5.06});    }
        if (ignore != t(1, 0, 1)) { EXPECT_EQ(a(1, 0, 1), st_type{6.07});    }
        if (ignore != t(1, 1, 0)) { EXPECT_EQ(a(1, 1, 0), st_type{2.08});    }
        if (ignore != t(1, 1, 1)) { EXPECT_EQ(a(1, 1, 1), st_type{3.09});    }
        if (ignore != t(1, 2, 0)) { EXPECT_EQ(a(1, 2, 0), st_type{-1.1});    }
        if (ignore != t(1, 2, 1)) { EXPECT_EQ(a(1, 2, 1), st_type{-9.11});   }
        if (ignore != t(2, 0, 0)) { EXPECT_EQ(a(2, 0, 0), st_type{-2.12});   }
        if (ignore != t(2, 0, 1)) { EXPECT_EQ(a(2, 0, 1), st_type{2.13});    }
        if (ignore != t(2, 1, 0)) { EXPECT_EQ(a(2, 1, 0), st_type{0.14});    }
        if (ignore != t(2, 1, 1)) { EXPECT_EQ(a(2, 1, 1), st_type{15.15});   }
        if (ignore != t(2, 2, 0)) { EXPECT_EQ(a(2, 2, 0), st_type{-9.16});   }
        if (ignore != t(2, 2, 1)) { EXPECT_EQ(a(2, 2, 1), st_type{8.17});    }
        if (ignore != t(3, 0, 0)) { EXPECT_EQ(a(3, 0, 0), st_type{7.18});    }
        if (ignore != t(3, 0, 1)) { EXPECT_EQ(a(3, 0, 1), st_type{-6.19});   }
        if (ignore != t(3, 1, 0)) { EXPECT_EQ(a(3, 1, 0), st_type{5.2});     }
        if (ignore != t(3, 1, 1)) { EXPECT_EQ(a(3, 1, 1), st_type{-4.21});   }
        if (ignore != t(3, 2, 0)) { EXPECT_EQ(a(3, 2, 0), st_type{3.22});    }
        if (ignore != t(3, 2, 1)) { EXPECT_EQ(a(3, 2, 1), st_type{-2.23});   }
        // clang-format on
    }
};

using ReducedStorage3dTypes =
    ::testing::Types<std::tuple<double, double>, std::tuple<double, float>,
                     std::tuple<float, float>
                     /*,
                     std::tuple<std::complex<double>, std::complex<double>>,
                     std::tuple<std::complex<double>, std::complex<float>>,
                     std::tuple<std::complex<float>, std::complex<float>>
                     */
                     >;

TYPED_TEST_SUITE(ReducedStorage3d, ReducedStorage3dTypes);


TYPED_TEST(ReducedStorage3d, CorrectLengths)
{
    EXPECT_EQ(this->r.length(0), this->size[0]);
    EXPECT_EQ(this->r.length(1), this->size[1]);
    EXPECT_EQ(this->r.length(2), this->size[2]);
    EXPECT_EQ(this->r.length(3), 1);
    EXPECT_EQ(this->r->get_size(), this->size);
}


TYPED_TEST(ReducedStorage3d, CorrectStride)
{
    EXPECT_EQ(this->r->get_stride()[0], this->size[1] * this->size[2]);
    EXPECT_EQ(this->r->get_stride().at(0), this->size[1] * this->size[2]);
    EXPECT_EQ(this->r->get_stride()[1], this->size[2]);
    EXPECT_EQ(this->r->get_stride().at(1), this->size[2]);
}


TYPED_TEST(ReducedStorage3d, CorrectStorage)
{
    EXPECT_EQ(this->r->get_stored_data(), this->data);
    EXPECT_EQ(this->r->get_const_storage(), this->data);
}


TYPED_TEST(ReducedStorage3d, CanReadData)
{
    this->check_accessor_correctness(this->r);
    this->check_accessor_correctness(this->cr);
}


TYPED_TEST(ReducedStorage3d, CopyFrom)
{
    using st_type = typename TestFixture::st_type;
    using reduced_storage = typename TestFixture::reduced_storage;
    st_type data2[TestFixture::data_elements];
    reduced_storage cpy(this->size, data2);

    // Do not use this in regular code since the implementation is slow
    cpy = this->r;

    this->check_accessor_correctness(cpy);
}


TYPED_TEST(ReducedStorage3d, CanImplicitlyConvertToConst)
{
    using const_reduced_storage = typename TestFixture::const_reduced_storage;

    const_reduced_storage const_rs = this->r->to_const();
    const_reduced_storage const_rs2 = this->cr;

    this->check_accessor_correctness(const_rs);
    this->check_accessor_correctness(const_rs2);
}


TYPED_TEST(ReducedStorage3d, ToConstWorks)
{
    using const_reduced_storage = typename TestFixture::const_reduced_storage;

    auto cr2 = this->r->to_const();

    static_assert(std::is_same<decltype(cr2), const_reduced_storage>::value,
                  "Types must be equal!");
    this->check_accessor_correctness(cr2);
}


TYPED_TEST(ReducedStorage3d, CanWriteData)
{
    using t = typename TestFixture::t;

    this->r(0, 1, 0) = 100.25;

    this->check_accessor_correctness(this->r, t(0, 1, 0));
    EXPECT_NEAR(this->r(0, 1, 0), 100.25, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Assignment)
{
    using t = typename TestFixture::t;

    this->r(0, 0, 1) = 10.2;

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(this->r(0, 0, 1), 10.2, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Assignment2)
{
    using t = typename TestFixture::t;

    this->r(0, 0, 1) = this->r(0, 1, 0);

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(this->r(0, 0, 1), -1.02, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Addition)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = 10.2 + 2.01;

    auto result = this->r(0, 0, 1) + ar_type{10.2};
    this->r(0, 0, 1) += 10.2;

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(this->r(0, 0, 1), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Addition2)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = 2.01 + -1.02;

    auto result = this->r(0, 0, 1) + this->r(0, 1, 0);
    this->r(0, 0, 1) += this->r(0, 1, 0);

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(this->r(0, 0, 1), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Subtraction)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = -2.23 - 1;

    auto result = this->r(3, 2, 1) - ar_type{1.};
    this->r(3, 2, 1) -= 1;

    this->check_accessor_correctness(this->r, t(3, 2, 1));
    EXPECT_NEAR(this->r(3, 2, 1), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Subtraction2)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = 3.22 - -2.23;

    auto result = this->cr(3, 2, 0) - this->r(3, 2, 1);
    this->r(3, 2, 0) -= this->r(3, 2, 1);

    this->check_accessor_correctness(this->r, t(3, 2, 0));
    EXPECT_NEAR(this->r(3, 2, 0), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Multiplication)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = 1 * 2;

    auto result = this->r(0, 0, 0) * ar_type{2.};
    this->r(0, 0, 0) *= 2;

    this->check_accessor_correctness(this->r, t(0, 0, 0));
    EXPECT_NEAR(this->r(0, 0, 0), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Multiplication2)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = 2.01 * 3.03;

    auto result = this->r(0, 0, 1) * this->cr(0, 1, 1);
    this->r(0, 0, 1) *= this->r(0, 1, 1);

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(this->r(0, 0, 1), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Division)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = 2.01 / 2.0;

    auto result = this->cr(0, 0, 1) / ar_type{2.};
    this->r(0, 0, 1) /= 2.;

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(this->r(0, 0, 1), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Division2)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = 5.06 / 4.04;

    auto result = this->r(1, 0, 0) / this->cr(0, 2, 0);
    this->r(1, 0, 0) /= this->r(0, 2, 0);

    this->check_accessor_correctness(this->r, t(1, 0, 0));
    EXPECT_NEAR(this->r(1, 0, 0), expected, TestFixture::delta);
    EXPECT_NEAR(result, expected, TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, UnaryMinus)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type neg_expected = this->r(2, 0, 0);
    const ar_type expected = -neg_expected;

    auto result = -this->r(2, 0, 0);

    this->check_accessor_correctness(this->r);
    EXPECT_EQ(result, expected);
}


TYPED_TEST(ReducedStorage3d, CanCreateSubrange)
{
    using st_type = typename TestFixture::st_type;

    auto subr = this->r(gko::span{1u, 3u}, gko::span{0u, 2u}, 0u);

    EXPECT_EQ(subr(0, 0, 0), st_type{5.06});
    EXPECT_EQ(subr(0, 1, 0), st_type{2.08});
    EXPECT_EQ(subr(1, 0, 0), st_type{-2.12});
    EXPECT_EQ(subr(1, 1, 0), st_type{0.14});
}


TYPED_TEST(ReducedStorage3d, CanCreateSubrange2)
{
    using st_type = typename TestFixture::st_type;

    auto subr =
        this->cr(gko::span{1u, 3u}, gko::span{0u, 2u}, gko::span{0u, 1u});

    EXPECT_EQ(subr(0, 0, 0), st_type{5.06});
    EXPECT_EQ(subr(0, 1, 0), st_type{2.08});
    EXPECT_EQ(subr(1, 0, 0), st_type{-2.12});
    EXPECT_EQ(subr(1, 1, 0), st_type{0.14});
}


class ReducedStorageXd : public ::testing::Test {
protected:
    using span = gko::span;
    using ar_type = double;
    using st_type = float;
    using size_type = gko::size_type;
    static constexpr ar_type delta{::r<st_type>::value};

    using accessor1d = gko::accessor::reduced_row_major<1, ar_type, st_type>;
    using accessor2d = gko::accessor::reduced_row_major<2, ar_type, st_type>;
    using const_accessor1d =
        gko::accessor::reduced_row_major<1, ar_type, const st_type>;
    using const_accessor2d =
        gko::accessor::reduced_row_major<2, ar_type, const st_type>;
    static_assert(std::is_same<const_accessor1d,
                               typename accessor1d::const_accessor>::value,
                  "Const accessors must be the same!");
    static_assert(std::is_same<const_accessor2d,
                               typename accessor2d::const_accessor>::value,
                  "Const accessors must be the same!");

    using reduced_storage1d = gko::range<accessor1d>;
    using reduced_storage2d = gko::range<accessor2d>;
    using const_reduced_storage2d = gko::range<const_accessor2d>;
    using const_reduced_storage1d = gko::range<const_accessor1d>;

    const std::array<size_type, 0> stride0{{}};
    const std::array<size_type, 1> stride1{{4}};
    const gko::dim<1> size_1d{8u};
    const gko::dim<2> size_2d{2u, 4u};
    static constexpr gko::size_type data_elements{8};
    st_type data[data_elements]{1.1f, 2.2f, 3.3f, 4.4f,
                                5.5f, 6.6f, 7.7f, -8.8f};
    reduced_storage1d r1{size_1d, data};
    reduced_storage2d r2{size_2d, data, stride1[0]};
    const_reduced_storage1d cr1{size_1d, data, stride0};
    const_reduced_storage2d cr2{size_2d, data, stride1};

    void data_equal_except_for(int idx)
    {
        // clang-format off
        if (idx != 0) { EXPECT_EQ(data[0], 1.1f); }
        if (idx != 1) { EXPECT_EQ(data[1], 2.2f); }
        if (idx != 2) { EXPECT_EQ(data[2], 3.3f); }
        if (idx != 3) { EXPECT_EQ(data[3], 4.4f); }
        if (idx != 4) { EXPECT_EQ(data[4], 5.5f); }
        if (idx != 5) { EXPECT_EQ(data[5], 6.6f); }
        if (idx != 6) { EXPECT_EQ(data[6], 7.7f); }
        if (idx != 7) { EXPECT_EQ(data[7], -8.8f); }
        // clang-format on
    }
};


TEST_F(ReducedStorageXd, CanRead)
{
    EXPECT_EQ(cr1(1), 2.2f);
    EXPECT_EQ(cr2(0, 1), 2.2f);
    EXPECT_EQ(r1(1), 2.2f);
    EXPECT_EQ(r2(0, 1), 2.2f);
}


TEST_F(ReducedStorageXd, CanWrite1)
{
    r1(2) = 0.25;

    data_equal_except_for(2);
    EXPECT_EQ(r1(2), 0.25);  // expect exact since easy to store
}


TEST_F(ReducedStorageXd, CanWrite2)
{
    r2(1, 1) = 0.75;

    data_equal_except_for(5);
    EXPECT_EQ(r2(1, 1), 0.75);  // expect exact since easy to store
}


template <typename ArithmeticStorageType>
class ScaledReducedStorage3d : public ::testing::Test {
protected:
    using ar_type =
        typename std::tuple_element<0, decltype(ArithmeticStorageType{})>::type;
    using st_type =
        typename std::tuple_element<1, decltype(ArithmeticStorageType{})>::type;
    // Type for `check_accessor_correctness` to forward the indices
    using t = std::tuple<int, int, int>;

    static constexpr ar_type delta{::r<ar_type>::value};

    using accessor =
        gko::accessor::scaled_reduced_row_major<3, ar_type, st_type, 0b0101>;
    using const_accessor =
        gko::accessor::scaled_reduced_row_major<3, ar_type, const st_type,
                                                0b0101>;

    using reduced_storage = gko::range<accessor>;
    using const_reduced_storage = gko::range<const_accessor>;

    const gko::dim<3> size{1u, 4u, 2u};
    static constexpr gko::size_type data_elements{8};
    static constexpr gko::size_type scalar_elements{8};
    // clang-format off
    st_type data[8]{
        10, 11,
        -12, 13,
        14, -115,
        6, 77
    };
    ar_type scalar[scalar_elements]{
        1., 2.,
    };
    // clang-format on
    reduced_storage r{size, data, scalar};
    const_reduced_storage cr{size, data, scalar};

    template <typename Accessor>
    static void check_accessor_correctness(
        const Accessor &a,
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
    std::tuple<double, gko::int64>, std::tuple<double, gko::int32>,
    std::tuple<double, gko::int16>, std::tuple<float, gko::int32>,
    std::tuple<float, gko::int16>>;

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
    EXPECT_EQ(this->r->get_stride()[0], this->size[1] * this->size[2]);
    EXPECT_EQ(this->r->get_stride().at(0), this->size[1] * this->size[2]);
    EXPECT_EQ(this->r->get_stride()[1], this->size[2]);
    EXPECT_EQ(this->r->get_stride().at(1), this->size[2]);
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


TYPED_TEST(ScaledReducedStorage3d, CopyFrom)
{
    using ar_type = typename TestFixture::ar_type;
    using st_type = typename TestFixture::st_type;
    using reduced_storage = typename TestFixture::reduced_storage;
    st_type data2[TestFixture::data_elements];
    ar_type scale2[TestFixture::scalar_elements];
    reduced_storage cpy(this->size, data2, scale2);

    // Do not use this in regular code since the implementation is slow
    cpy = this->r;

    this->check_accessor_correctness(cpy);
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


TYPED_TEST(ScaledReducedStorage3d, CanRead)
{
    this->check_accessor_correctness(this->cr);
    this->check_accessor_correctness(this->r);
}


TYPED_TEST(ScaledReducedStorage3d, Subrange)
{
    auto subr = this->cr(0u, gko::span{0u, 2u}, 1u);

    EXPECT_EQ(subr(0, 0, 0), 22.);
    EXPECT_EQ(subr(0, 1, 0), 26.);
}


TYPED_TEST(ScaledReducedStorage3d, CanWriteScale)
{
    this->r->write_scalar(10., 0, 0, 0);

    EXPECT_EQ(this->r(0, 0, 0), 100.);
    EXPECT_EQ(this->r(0, 0, 1), 22.);
    EXPECT_EQ(this->r(0, 1, 0), -120.);
    EXPECT_EQ(this->r(0, 1, 1), 26.);
    EXPECT_EQ(this->r(0, 2, 0), 140.);
    EXPECT_EQ(this->r(0, 2, 1), -230.);
    EXPECT_EQ(this->r(0, 3, 0), 60.);
    EXPECT_EQ(this->r(0, 3, 1), 154.);
}


TYPED_TEST(ScaledReducedStorage3d, CanReadScale)
{
    EXPECT_EQ(this->r->read_scalar(0, 0, 0), 1.);
    EXPECT_EQ(this->r->read_scalar(0, 0, 1), 2.);
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
    using span = gko::span;
    using ar_type = double;
    using st_type = int;
    using size_type = gko::size_type;
    static constexpr ar_type delta{0.1};

    using accessor1d =
        gko::accessor::scaled_reduced_row_major<1, ar_type, st_type, 1>;
    using accessor2d =
        gko::accessor::scaled_reduced_row_major<2, ar_type, st_type, 2>;
    using const_accessor1d =
        gko::accessor::scaled_reduced_row_major<1, ar_type, const st_type, 1>;
    using const_accessor2d =
        gko::accessor::scaled_reduced_row_major<2, ar_type, const st_type, 2>;
    static_assert(std::is_same<const_accessor1d,
                               typename accessor1d::const_accessor>::value,
                  "Const accessors must be the same!");
    static_assert(std::is_same<const_accessor2d,
                               typename accessor2d::const_accessor>::value,
                  "Const accessors must be the same!");

    using reduced_storage1d = gko::range<accessor1d>;
    using reduced_storage2d = gko::range<accessor2d>;
    using const_reduced_storage2d = gko::range<const_accessor2d>;
    using const_reduced_storage1d = gko::range<const_accessor1d>;

    const std::array<size_type, 0> stride0{{}};
    const std::array<size_type, 1> stride1{{4}};
    const size_type stride1nc[1] = {4};
    const gko::dim<1> size_1d{8u};
    const gko::dim<2> size_2d{2u, 4u};

    static constexpr gko::size_type data_elements{8};
    st_type data[data_elements]{11, 22, 33, 44, 55, 66, 77, -88};
    const double default_scalar{.1};
    ar_type scalar[data_elements]{
        default_scalar, default_scalar, default_scalar, default_scalar,
        default_scalar, default_scalar, default_scalar, default_scalar};

    reduced_storage1d r1{size_1d, data, scalar};
    reduced_storage2d r2{size_2d, data, stride1, scalar};
    const_reduced_storage1d cr1{size_1d, data, stride0, scalar};
    const_reduced_storage2d cr2{size_2d, data, stride1, scalar};

    void data_equal_except_for(int idx)
    {
        // clang-format off
        if (idx != 0) { EXPECT_EQ(data[0], 11); }
        if (idx != 1) { EXPECT_EQ(data[1], 22); }
        if (idx != 2) { EXPECT_EQ(data[2], 33); }
        if (idx != 3) { EXPECT_EQ(data[3], 44); }
        if (idx != 4) { EXPECT_EQ(data[4], 55); }
        if (idx != 5) { EXPECT_EQ(data[5], 66); }
        if (idx != 6) { EXPECT_EQ(data[6], 77); }
        if (idx != 7) { EXPECT_EQ(data[7], -88); }
        // clang-format on
    }
    void scalar_equal_except_for(int idx)
    {
        // clang-format off
        if (idx != 0) { EXPECT_EQ(scalar[0], default_scalar); }
        if (idx != 1) { EXPECT_EQ(scalar[1], default_scalar); }
        if (idx != 2) { EXPECT_EQ(scalar[2], default_scalar); }
        if (idx != 3) { EXPECT_EQ(scalar[3], default_scalar); }
        if (idx != 4) { EXPECT_EQ(scalar[4], default_scalar); }
        if (idx != 5) { EXPECT_EQ(scalar[5], default_scalar); }
        if (idx != 6) { EXPECT_EQ(scalar[6], default_scalar); }
        if (idx != 7) { EXPECT_EQ(scalar[7], default_scalar); }
        // clang-format on
    }
};


TEST_F(ScaledReducedStorageXd, CanRead)
{
    EXPECT_NEAR(cr1(1), 2.2, delta);
    EXPECT_NEAR(cr2(0, 1), 2.2, delta);
    EXPECT_NEAR(r1(1), 2.2, delta);
    EXPECT_NEAR(r2(0, 1), 2.2, delta);
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
    r2(1, 1) = 0.7;

    data_equal_except_for(5);
    scalar_equal_except_for(99);
    EXPECT_NEAR(r2(1, 1), 0.7, delta);
}


}  // namespace
