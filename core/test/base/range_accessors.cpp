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


class ReducedStorage3d : public ::testing::Test {
protected:
    using span = gko::span;
    using ar_type = double;
    using st_type = double;
    static constexpr ar_type delta{::r<st_type>::value};

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
    reduced_storage r{data, size};
    const_reduced_storage cr{data, size};

    template <typename Accessor>
    static void check_accessor_correctness(
        const Accessor &a, std::tuple<int, int, int> ignore = t(99, 99, 99))
    {
        // Test for equality is fine here since they should not be modified
        // clang-format off
        if (ignore != t(0, 0, 0)) { EXPECT_EQ(a(0, 0, 0), 1.0);     }
        if (ignore != t(0, 0, 1)) { EXPECT_EQ(a(0, 0, 1), 2.01);    }
        if (ignore != t(0, 1, 0)) { EXPECT_EQ(a(0, 1, 0), -1.02);   }
        if (ignore != t(0, 1, 1)) { EXPECT_EQ(a(0, 1, 1), 3.03);    }
        if (ignore != t(0, 2, 0)) { EXPECT_EQ(a(0, 2, 0), 4.04);    }
        if (ignore != t(0, 2, 1)) { EXPECT_EQ(a(0, 2, 1), -2.05);   }
        if (ignore != t(1, 0, 0)) { EXPECT_EQ(a(1, 0, 0), 5.06);    }
        if (ignore != t(1, 0, 1)) { EXPECT_EQ(a(1, 0, 1), 6.07);    }
        if (ignore != t(1, 1, 0)) { EXPECT_EQ(a(1, 1, 0), 2.08);    }
        if (ignore != t(1, 1, 1)) { EXPECT_EQ(a(1, 1, 1), 3.09);    }
        if (ignore != t(1, 2, 0)) { EXPECT_EQ(a(1, 2, 0), -1.1);    }
        if (ignore != t(1, 2, 1)) { EXPECT_EQ(a(1, 2, 1), -9.11);   }
        if (ignore != t(2, 0, 0)) { EXPECT_EQ(a(2, 0, 0), -2.12);   }
        if (ignore != t(2, 0, 1)) { EXPECT_EQ(a(2, 0, 1), 2.13);    }
        if (ignore != t(2, 1, 0)) { EXPECT_EQ(a(2, 1, 0), 0.14);    }
        if (ignore != t(2, 1, 1)) { EXPECT_EQ(a(2, 1, 1), 15.15);   }
        if (ignore != t(2, 2, 0)) { EXPECT_EQ(a(2, 2, 0), -9.16);   }
        if (ignore != t(2, 2, 1)) { EXPECT_EQ(a(2, 2, 1), 8.17);    }
        if (ignore != t(3, 0, 0)) { EXPECT_EQ(a(3, 0, 0), 7.18);    }
        if (ignore != t(3, 0, 1)) { EXPECT_EQ(a(3, 0, 1), -6.19);   }
        if (ignore != t(3, 1, 0)) { EXPECT_EQ(a(3, 1, 0), 5.2);     }
        if (ignore != t(3, 1, 1)) { EXPECT_EQ(a(3, 1, 1), -4.21);   }
        if (ignore != t(3, 2, 0)) { EXPECT_EQ(a(3, 2, 0), 3.22);    }
        if (ignore != t(3, 2, 1)) { EXPECT_EQ(a(3, 2, 1), -2.23);   }
        // clang-format on
    }
};


TEST_F(ReducedStorage3d, CorrectLengths)
{
    EXPECT_EQ(r.length(0), size[0]);
    EXPECT_EQ(r.length(1), size[1]);
    EXPECT_EQ(r.length(2), size[2]);
    EXPECT_EQ(r.length(3), 1);
    EXPECT_EQ(r->get_size(), size);
}


TEST_F(ReducedStorage3d, CorrectStride)
{
    EXPECT_EQ(r->get_stride()[0], size[1] * size[2]);
    EXPECT_EQ(r->get_stride().at(0), size[1] * size[2]);
    EXPECT_EQ(r->get_stride()[1], size[2]);
    EXPECT_EQ(r->get_stride().at(1), size[2]);
}


TEST_F(ReducedStorage3d, CorrectStorage)
{
    EXPECT_EQ(r->get_stored_data(), data);
    EXPECT_EQ(r->get_const_storage(), data);
}


TEST_F(ReducedStorage3d, CanReadData)
{
    check_accessor_correctness(r);
    check_accessor_correctness(cr);
}


TEST_F(ReducedStorage3d, CopyFrom)
{
    st_type data2[data_elements];
    reduced_storage cpy(data2, size);

    // Do not use this in regular code since the implementation is slow
    cpy = r;

    check_accessor_correctness(cpy);
}


TEST_F(ReducedStorage3d, CanImplicitlyConvertToConst)
{
    const_reduced_storage const_rs = r->to_const();
    const_reduced_storage const_rs2 = cr;

    check_accessor_correctness(const_rs);
    check_accessor_correctness(const_rs2);
}


TEST_F(ReducedStorage3d, ToConstWorks)
{
    auto cr2 = r->to_const();

    static_assert(std::is_same<decltype(cr2), const_reduced_storage>::value,
                  "Types must be equal!");
    check_accessor_correctness(cr2);
}


TEST_F(ReducedStorage3d, CanWriteData)
{
    r(0, 1, 0) = 100.2;

    check_accessor_correctness(r, t(0, 1, 0));
    EXPECT_EQ(r(0, 1, 0), 100.2);
}


TEST_F(ReducedStorage3d, Assignment)
{
    r(0, 0, 1) = 10.2;

    check_accessor_correctness(r, t(0, 0, 1));
    EXPECT_NEAR(r(0, 0, 1), 10.2, delta);
}


TEST_F(ReducedStorage3d, Assignment2)
{
    r(0, 0, 1) = r(0, 1, 0);

    check_accessor_correctness(r, t(0, 0, 1));
    EXPECT_NEAR(r(0, 0, 1), -1.02, delta);
}


TEST_F(ReducedStorage3d, Addition)
{
    const ar_type expected = 10.2 + 2.01;

    auto result = r(0, 0, 1) + 10.2;
    r(0, 0, 1) += 10.2;

    check_accessor_correctness(r, t(0, 0, 1));
    EXPECT_NEAR(r(0, 0, 1), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ReducedStorage3d, Addition2)
{
    const ar_type expected = 2.01 + -1.02;

    auto result = r(0, 0, 1) + r(0, 1, 0);
    r(0, 0, 1) += r(0, 1, 0);

    check_accessor_correctness(r, t(0, 0, 1));
    EXPECT_NEAR(r(0, 0, 1), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ReducedStorage3d, Subtraction)
{
    const ar_type expected = -2.23 - 1;

    auto result = r(3, 2, 1) - 1.;
    r(3, 2, 1) -= 1;

    check_accessor_correctness(r, t(3, 2, 1));
    EXPECT_NEAR(r(3, 2, 1), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ReducedStorage3d, Subtraction2)
{
    const ar_type expected = 3.22 - -2.23;

    auto result = cr(3, 2, 0) - r(3, 2, 1);
    r(3, 2, 0) -= r(3, 2, 1);

    check_accessor_correctness(r, t(3, 2, 0));
    EXPECT_NEAR(r(3, 2, 0), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ReducedStorage3d, Multiplication)
{
    const ar_type expected = 1 * 2;

    auto result = r(0, 0, 0) * 2.;
    r(0, 0, 0) *= 2;

    check_accessor_correctness(r, t(0, 0, 0));
    EXPECT_NEAR(r(0, 0, 0), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ReducedStorage3d, Multiplication2)
{
    const ar_type expected = 2.01 * 3.03;

    auto result = r(0, 0, 1) * cr(0, 1, 1);
    r(0, 0, 1) *= r(0, 1, 1);

    check_accessor_correctness(r, t(0, 0, 1));
    EXPECT_NEAR(r(0, 0, 1), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ReducedStorage3d, Division)
{
    const ar_type expected = 2.01 / 2.0;

    auto result = cr(0, 0, 1) / 2.;
    r(0, 0, 1) /= 2.;

    check_accessor_correctness(r, t(0, 0, 1));
    EXPECT_NEAR(r(0, 0, 1), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ReducedStorage3d, Division2)
{
    const ar_type expected = 5.06 / 4.04;

    auto result = r(1, 0, 0) / cr(0, 2, 0);
    r(1, 0, 0) /= r(0, 2, 0);

    check_accessor_correctness(r, t(1, 0, 0));
    EXPECT_NEAR(r(1, 0, 0), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ReducedStorage3d, UnaryMinus)
{
    const ar_type neg_expected = r(2, 0, 0);
    const ar_type expected = -neg_expected;

    auto result = -r(2, 0, 0);

    check_accessor_correctness(r);
    EXPECT_EQ(result, expected);
}


TEST_F(ReducedStorage3d, CanCreateSubrange)
{
    auto subr = r(span{1, 3}, span{0, 2}, 0);

    EXPECT_EQ(subr(0, 0, 0), 5.06);
    EXPECT_EQ(subr(0, 1, 0), 2.08);
    EXPECT_EQ(subr(1, 0, 0), -2.12);
    EXPECT_EQ(subr(1, 1, 0), 0.14);
}


TEST_F(ReducedStorage3d, CanCreateSubrange2)
{
    auto subr = cr(span{1, 3}, span{0, 2}, span{0, 1});

    EXPECT_EQ(subr(0, 0, 0), 5.06);
    EXPECT_EQ(subr(0, 1, 0), 2.08);
    EXPECT_EQ(subr(1, 0, 0), -2.12);
    EXPECT_EQ(subr(1, 1, 0), 0.14);
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

    using reduced_storage1d = gko::range<accessor1d>;
    using reduced_storage2d = gko::range<accessor2d>;
    using const_reduced_storage2d = gko::range<const_accessor2d>;
    using const_reduced_storage1d = gko::range<const_accessor1d>;

    const std::array<const size_type, 0> stride0{};
    const std::array<const size_type, 1> stride1{4};
    const gko::dim<1> size_1d{8u};
    const gko::dim<2> size_2d{2u, 4u};
    static constexpr gko::size_type data_elements{8};
    st_type data[data_elements]{1.1f, 2.2f, 3.3f, 4.4f,
                                5.5f, 6.6f, 7.7f, -8.8f};
    reduced_storage1d r1{data, size_1d /*, stride0*/};
    reduced_storage2d r2{data, size_2d, stride1[0]};
    const_reduced_storage1d cr1{data, size_1d, stride0};
    const_reduced_storage2d cr2{data, size_2d, stride1};

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


class ScaledReducedStorage3d : public ::testing::Test {
protected:
    using span = gko::span;
    using ar_type = double;
    using st_type = gko::int32;
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
    reduced_storage r{data, scalar, size};
    const_reduced_storage cr{data, scalar, size};

    template <typename Accessor>
    static void check_accessor_correctness(
        const Accessor &a,
        std::tuple<int, int, int> ignore = std::tuple<int, int, int>(99, 99,
                                                                     99))
    {
        // Test for equality is fine here since they should not be modified
        // clang-format off
        if (ignore != t(0, 0, 0)) { EXPECT_EQ(a(0, 0, 0), 10.);   }
        if (ignore != t(0, 0, 1)) { EXPECT_EQ(a(0, 0, 1), 22.);   }
        if (ignore != t(0, 1, 0)) { EXPECT_EQ(a(0, 1, 0), -12.);  }
        if (ignore != t(0, 1, 1)) { EXPECT_EQ(a(0, 1, 1), 26.);   }
        if (ignore != t(0, 2, 0)) { EXPECT_EQ(a(0, 2, 0), 14.);   }
        if (ignore != t(0, 2, 1)) { EXPECT_EQ(a(0, 2, 1), -230.); }
        if (ignore != t(0, 3, 0)) { EXPECT_EQ(a(0, 3, 0), 6.);    }
        if (ignore != t(0, 3, 1)) { EXPECT_EQ(a(0, 3, 1), 154.);  }
        // clang-format on
    }
};


TEST_F(ScaledReducedStorage3d, CorrectLengths)
{
    EXPECT_EQ(r.length(0), size[0]);
    EXPECT_EQ(r.length(1), size[1]);
    EXPECT_EQ(r.length(2), size[2]);
    EXPECT_EQ(r.length(3), 1);
    EXPECT_EQ(r->get_size(), size);
}


TEST_F(ScaledReducedStorage3d, CorrectStride)
{
    EXPECT_EQ(r->get_stride()[0], size[1] * size[2]);
    EXPECT_EQ(r->get_stride().at(0), size[1] * size[2]);
    EXPECT_EQ(r->get_stride()[1], size[2]);
    EXPECT_EQ(r->get_stride().at(1), size[2]);
}


TEST_F(ScaledReducedStorage3d, CorrectStorage)
{
    EXPECT_EQ(r->get_stored_data(), data);
    EXPECT_EQ(r->get_const_storage(), data);
}


TEST_F(ScaledReducedStorage3d, CorrectScale)
{
    EXPECT_EQ(r->get_scalar(), scalar);
    EXPECT_EQ(r->get_const_scalar(), scalar);
}


TEST_F(ScaledReducedStorage3d, CanReadData)
{
    check_accessor_correctness(r);
    check_accessor_correctness(cr);
}


TEST_F(ScaledReducedStorage3d, CopyFrom)
{
    st_type data2[data_elements];
    ar_type scale2[scalar_elements];
    reduced_storage cpy(data2, scale2, size);

    // Do not use this in regular code since the implementation is slow
    cpy = r;

    check_accessor_correctness(cpy);
}


TEST_F(ScaledReducedStorage3d, CanImplicitlyConvertToConst)
{
    const_reduced_storage const_rs = r->to_const();
    const_reduced_storage const_rs2 = cr;

    check_accessor_correctness(const_rs);
    check_accessor_correctness(const_rs2);
}


TEST_F(ScaledReducedStorage3d, ToConstWorks)
{
    auto cr2 = r->to_const();

    static_assert(std::is_same<decltype(cr2), const_reduced_storage>::value,
                  "Types must be equal!");
    check_accessor_correctness(cr2);
}


TEST_F(ScaledReducedStorage3d, CanRead)
{
    check_accessor_correctness(cr);
    check_accessor_correctness(r);
}


TEST_F(ScaledReducedStorage3d, Subrange)
{
    auto subr = cr(0, gko::span{0, 2}, 1);

    EXPECT_EQ(subr(0, 0, 0), 22.);
    EXPECT_EQ(subr(0, 1, 0), 26.);
}


TEST_F(ScaledReducedStorage3d, CanWriteScale)
{
    r->write_scalar(10., 0, 0, 0);

    EXPECT_EQ(r(0, 0, 0), 100.);
    EXPECT_EQ(r(0, 0, 1), 22.);
    EXPECT_EQ(r(0, 1, 0), -120.);
    EXPECT_EQ(r(0, 1, 1), 26.);
    EXPECT_EQ(r(0, 2, 0), 140.);
    EXPECT_EQ(r(0, 2, 1), -230.);
    EXPECT_EQ(r(0, 3, 0), 60.);
    EXPECT_EQ(r(0, 3, 1), 154.);
}


TEST_F(ScaledReducedStorage3d, CanReadScale)
{
    EXPECT_EQ(r->read_scalar(0, 0, 0), 1.);
    EXPECT_EQ(r->read_scalar(0, 0, 1), 2.);
}


TEST_F(ScaledReducedStorage3d, Addition)
{
    const ar_type expected = 10. + 3.;

    const auto result = cr(0, 0, 0) + 3.;
    r(0, 0, 0) += 3.;

    check_accessor_correctness(r, t(0, 0, 0));
    EXPECT_NEAR(r(0, 0, 0), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ScaledReducedStorage3d, Addition2)
{
    const ar_type expected = 10. + 22.;

    const auto result = cr(0, 0, 0) + r(0, 0, 1);
    r(0, 0, 0) += cr(0, 0, 1);

    check_accessor_correctness(r, t(0, 0, 0));
    EXPECT_NEAR(r(0, 0, 0), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ScaledReducedStorage3d, Subtraction)
{
    const ar_type expected = 22. - 2.;

    const auto result = cr(0, 0, 1) - 2.;
    r(0, 0, 1) -= 2.;

    check_accessor_correctness(r, t(0, 0, 1));
    EXPECT_NEAR(r(0, 0, 1), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ScaledReducedStorage3d, Subtraction2)
{
    const ar_type expected = -12. - 26.;

    const auto result = cr(0, 1, 0) - r(0, 1, 1);
    r(0, 1, 0) -= r(0, 1, 1);

    check_accessor_correctness(r, t(0, 1, 0));
    EXPECT_NEAR(r(0, 1, 0), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ScaledReducedStorage3d, Multiplication)
{
    const ar_type expected = 26. * 3.;

    const auto result = cr(0, 1, 1) * 3.;
    r(0, 1, 1) *= 3.;

    check_accessor_correctness(r, t(0, 1, 1));
    EXPECT_NEAR(r(0, 1, 1), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ScaledReducedStorage3d, Multiplication2)
{
    const ar_type expected = 14. * 10.;

    const auto result = r(0, 2, 0) * r(0, 0, 0);
    r(0, 2, 0) *= r(0, 0, 0);

    check_accessor_correctness(r, t(0, 2, 0));
    EXPECT_NEAR(r(0, 2, 0), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ScaledReducedStorage3d, Division)
{
    const ar_type expected = 10. / 2.;

    const auto result = cr(0, 0, 0) / 2.;
    r(0, 0, 0) /= 2.;

    check_accessor_correctness(r, t(0, 0, 0));
    EXPECT_NEAR(r(0, 0, 0), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ScaledReducedStorage3d, Division2)
{
    const ar_type expected = -12. / 6.;

    const auto result = r(0, 1, 0) / r(0, 3, 0);
    r(0, 1, 0) /= r(0, 3, 0);

    check_accessor_correctness(r, t(0, 1, 0));
    EXPECT_NEAR(r(0, 1, 0), expected, delta);
    EXPECT_NEAR(result, expected, delta);
}


TEST_F(ScaledReducedStorage3d, UnaryMinus)
{
    const ar_type neg_expected = r(0, 1, 1);
    const ar_type expected = -neg_expected;

    auto result = -r(0, 1, 1);

    check_accessor_correctness(r);
    EXPECT_EQ(result, expected);
}


}  // namespace
