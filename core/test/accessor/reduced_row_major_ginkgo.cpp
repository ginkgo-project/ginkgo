// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>


#include <gtest/gtest.h>


#include "accessor/index_span.hpp"
#include "accessor/range.hpp"
#include "accessor/reduced_row_major.hpp"
#include "accessor/utils.hpp"
#include "core/base/extended_float.hpp"  // necessary for gko::half
#include "core/test/utils.hpp"


namespace {


/**
 * This test makes sure reduced_row_major works properly with various types.
 * Note that this tests has a dependency on Ginkgo because of gko::half.
 */
template <typename ArithmeticStorageType>
class ReducedStorage3d : public ::testing::Test {
protected:
    using ar_type =
        typename std::tuple_element<0, decltype(ArithmeticStorageType{})>::type;
    using st_type =
        typename std::tuple_element<1, decltype(ArithmeticStorageType{})>::type;
    using rcar_type = gko::acc::remove_complex_t<ar_type>;
    static constexpr rcar_type delta{
        std::is_same<ar_type, st_type>::value
            ? 0
            : std::numeric_limits<
                  gko::acc::remove_complex_t<st_type>>::epsilon() *
                  1e1};

    // Type for `check_accessor_correctness` to forward the indices
    using t = std::tuple<int, int, int>;
    using i_span = gko::acc::index_span;

    using accessor = gko::acc::reduced_row_major<3, ar_type, st_type>;
    using const_accessor =
        gko::acc::reduced_row_major<3, ar_type, const st_type>;

    using reduced_storage = gko::acc::range<accessor>;
    using const_reduced_storage = gko::acc::range<const_accessor>;

    const std::array<gko::acc::size_type, 3> size{{4u, 3u, 2u}};
    static constexpr gko::acc::size_type data_elements{4 * 3 * 2};
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

    // Casts val first to `st_type`, then to `ar_type` in order to be allowed
    // to test for equality
    template <typename T>
    static ar_type c_st_ar(T val)
    {
        return static_cast<ar_type>(static_cast<st_type>(val));
    }

    template <typename Accessor>
    void check_accessor_correctness(const Accessor& a,
                                    std::tuple<int, int, int> ignore = t(99, 99,
                                                                         99))
    {
        // Test for equality is fine here since they should not be modified
        // clang-format off
        if (ignore != t(0, 0, 0)) { EXPECT_EQ(a(0, 0, 0), c_st_ar(1.0));     }
        if (ignore != t(0, 0, 1)) { EXPECT_EQ(a(0, 0, 1), c_st_ar(2.01));    }
        if (ignore != t(0, 1, 0)) { EXPECT_EQ(a(0, 1, 0), c_st_ar(-1.02));   }
        if (ignore != t(0, 1, 1)) { EXPECT_EQ(a(0, 1, 1), c_st_ar(3.03));    }
        if (ignore != t(0, 2, 0)) { EXPECT_EQ(a(0, 2, 0), c_st_ar(4.04));    }
        if (ignore != t(0, 2, 1)) { EXPECT_EQ(a(0, 2, 1), c_st_ar(-2.05));   }
        if (ignore != t(1, 0, 0)) { EXPECT_EQ(a(1, 0, 0), c_st_ar(5.06));    }
        if (ignore != t(1, 0, 1)) { EXPECT_EQ(a(1, 0, 1), c_st_ar(6.07));    }
        if (ignore != t(1, 1, 0)) { EXPECT_EQ(a(1, 1, 0), c_st_ar(2.08));    }
        if (ignore != t(1, 1, 1)) { EXPECT_EQ(a(1, 1, 1), c_st_ar(3.09));    }
        if (ignore != t(1, 2, 0)) { EXPECT_EQ(a(1, 2, 0), c_st_ar(-1.1));    }
        if (ignore != t(1, 2, 1)) { EXPECT_EQ(a(1, 2, 1), c_st_ar(-9.11));   }
        if (ignore != t(2, 0, 0)) { EXPECT_EQ(a(2, 0, 0), c_st_ar(-2.12));   }
        if (ignore != t(2, 0, 1)) { EXPECT_EQ(a(2, 0, 1), c_st_ar(2.13));    }
        if (ignore != t(2, 1, 0)) { EXPECT_EQ(a(2, 1, 0), c_st_ar(0.14));    }
        if (ignore != t(2, 1, 1)) { EXPECT_EQ(a(2, 1, 1), c_st_ar(15.15));   }
        if (ignore != t(2, 2, 0)) { EXPECT_EQ(a(2, 2, 0), c_st_ar(-9.16));   }
        if (ignore != t(2, 2, 1)) { EXPECT_EQ(a(2, 2, 1), c_st_ar(8.17));    }
        if (ignore != t(3, 0, 0)) { EXPECT_EQ(a(3, 0, 0), c_st_ar(7.18));    }
        if (ignore != t(3, 0, 1)) { EXPECT_EQ(a(3, 0, 1), c_st_ar(-6.19));   }
        if (ignore != t(3, 1, 0)) { EXPECT_EQ(a(3, 1, 0), c_st_ar(5.2));     }
        if (ignore != t(3, 1, 1)) { EXPECT_EQ(a(3, 1, 1), c_st_ar(-4.21));   }
        if (ignore != t(3, 2, 0)) { EXPECT_EQ(a(3, 2, 0), c_st_ar(3.22));    }
        if (ignore != t(3, 2, 1)) { EXPECT_EQ(a(3, 2, 1), c_st_ar(-2.23));   }
        // clang-format on
    }
};

using ReducedStorage3dTypes =
    ::testing::Types<std::tuple<double, double>, std::tuple<double, float>,
                     std::tuple<float, float>, std::tuple<double, gko::half>,
                     std::tuple<float, gko::half>,
                     std::tuple<std::complex<double>, std::complex<double>>,
                     std::tuple<std::complex<double>, std::complex<float>>,
                     std::tuple<std::complex<float>, std::complex<float>>>;

TYPED_TEST_SUITE(ReducedStorage3d, ReducedStorage3dTypes,
                 PairTypenameNameGenerator);


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


TYPED_TEST(ReducedStorage3d, CanCreateWithStride)
{
    using reduced_storage = typename TestFixture::reduced_storage;
    using ar_type = typename TestFixture::ar_type;
    auto size = std::array<gko::acc::size_type, 3>{{2, 2, 2}};
    auto stride = std::array<gko::acc::size_type, 2>{{12, 2}};

    auto range = reduced_storage{size, this->data, stride};
    range(1, 1, 0) = ar_type{2.};

    EXPECT_EQ(range(0, 0, 0), this->c_st_ar(1.0));
    EXPECT_EQ(range(0, 0, 1), this->c_st_ar(2.01));
    EXPECT_EQ(range(0, 1, 0), this->c_st_ar(-1.02));
    EXPECT_EQ(range(0, 1, 1), this->c_st_ar(3.03));
    EXPECT_EQ(range(1, 0, 0), this->c_st_ar(-2.12));
    EXPECT_EQ(range(1, 0, 1), this->c_st_ar(2.13));
    EXPECT_EQ(range(1, 1, 0), this->c_st_ar(2.));
    EXPECT_EQ(range(1, 1, 1), this->c_st_ar(15.15));
}


TYPED_TEST(ReducedStorage3d, CanWriteData)
{
    using t = typename TestFixture::t;

    this->r(0, 1, 0) = 100.25;

    this->check_accessor_correctness(this->r, t(0, 1, 0));
    EXPECT_EQ(this->r(0, 1, 0), this->c_st_ar(100.25));
}


TYPED_TEST(ReducedStorage3d, Assignment)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = 1.2;

    this->r(0, 0, 1) = expected;

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_EQ(this->r(0, 0, 1), this->c_st_ar(expected));
}


TYPED_TEST(ReducedStorage3d, Assignment2)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = -1.02;

    this->r(0, 0, 1) = this->r(0, 1, 0);

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_EQ(this->r(0, 0, 1), this->c_st_ar(expected));
}


TYPED_TEST(ReducedStorage3d, Addition)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    using std::abs;
    const ar_type expected = 1.2 + 2.01;

    ar_type result = this->r(0, 0, 1) + ar_type{1.2};
    this->r(0, 0, 1) += 1.2;

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(abs(this->r(0, 0, 1)), abs(expected), TestFixture::delta);
    EXPECT_NEAR(abs(result), abs(expected), TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Addition2)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    using std::abs;
    const ar_type expected = 2.01 + -1.02;

    auto result = this->r(0, 0, 1) + this->r(0, 1, 0);
    this->r(0, 0, 1) += this->r(0, 1, 0);

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(abs(this->r(0, 0, 1)), abs(this->c_st_ar(expected)),
                TestFixture::delta);
    EXPECT_NEAR(abs(result), abs(this->c_st_ar(expected)), TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Subtraction)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    using std::abs;
    const ar_type expected = -2.23 - 1;

    auto result = this->r(3, 2, 1) - ar_type{1.};
    this->r(3, 2, 1) -= 1;

    this->check_accessor_correctness(this->r, t(3, 2, 1));
    EXPECT_NEAR(abs(this->r(3, 2, 1)), abs(this->c_st_ar(expected)),
                TestFixture::delta);
    EXPECT_NEAR(abs(result), abs(this->c_st_ar(expected)), TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Subtraction2)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    using std::abs;
    const ar_type expected = 3.22 - -2.23;

    auto result = this->cr(3, 2, 0) - this->r(3, 2, 1);
    this->r(3, 2, 0) -= this->r(3, 2, 1);

    this->check_accessor_correctness(this->r, t(3, 2, 0));
    EXPECT_NEAR(abs(this->r(3, 2, 0)), abs(this->c_st_ar(expected)),
                TestFixture::delta);
    EXPECT_NEAR(abs(result), abs(this->c_st_ar(expected)), TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Multiplication)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    const ar_type expected = 1 * 2;

    auto result = this->r(0, 0, 0) * ar_type{2.};
    this->r(0, 0, 0) *= 2;

    this->check_accessor_correctness(this->r, t(0, 0, 0));
    EXPECT_EQ(this->r(0, 0, 0), this->c_st_ar(expected));
    EXPECT_EQ(result, this->c_st_ar(expected));
}


TYPED_TEST(ReducedStorage3d, Multiplication2)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    using std::abs;
    const ar_type expected = 2.01 * 3.03;

    auto result = this->r(0, 0, 1) * this->cr(0, 1, 1);
    this->r(0, 0, 1) *= this->r(0, 1, 1);

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(abs(this->r(0, 0, 1)), abs(expected), TestFixture::delta);
    EXPECT_NEAR(abs(result), abs(expected), TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Division)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    using std::abs;
    const ar_type expected = 2.01 / 2.0;

    auto result = this->cr(0, 0, 1) / ar_type{2.};
    this->r(0, 0, 1) /= 2.;

    this->check_accessor_correctness(this->r, t(0, 0, 1));
    EXPECT_NEAR(abs(this->r(0, 0, 1)), abs(expected), TestFixture::delta);
    EXPECT_NEAR(abs(result), abs(expected), TestFixture::delta);
}


TYPED_TEST(ReducedStorage3d, Division2)
{
    using t = typename TestFixture::t;
    using ar_type = typename TestFixture::ar_type;
    using std::abs;
    const ar_type expected = 5.06 / 4.04;

    auto result = this->r(1, 0, 0) / this->cr(0, 2, 0);
    this->r(1, 0, 0) /= this->r(0, 2, 0);

    this->check_accessor_correctness(this->r, t(1, 0, 0));
    EXPECT_NEAR(abs(this->r(1, 0, 0)), abs(expected), TestFixture::delta);
    EXPECT_NEAR(abs(result), abs(expected), TestFixture::delta);
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
    using i_span = typename TestFixture::i_span;
    auto subr = this->r(i_span{1u, 3u}, i_span{0u, 2u}, 0u);

    EXPECT_EQ(subr(0, 0, 0), this->c_st_ar(5.06));
    EXPECT_EQ(subr(0, 1, 0), this->c_st_ar(2.08));
    EXPECT_EQ(subr(1, 0, 0), this->c_st_ar(-2.12));
    EXPECT_EQ(subr(1, 1, 0), this->c_st_ar(0.14));
}


TYPED_TEST(ReducedStorage3d, CanCreateSubrange2)
{
    using i_span = typename TestFixture::i_span;
    auto subr = this->cr(i_span{1u, 3u}, i_span{0u, 2u}, i_span{0u, 1u});

    EXPECT_EQ(subr(0, 0, 0), this->c_st_ar(5.06));
    EXPECT_EQ(subr(0, 1, 0), this->c_st_ar(2.08));
    EXPECT_EQ(subr(1, 0, 0), this->c_st_ar(-2.12));
    EXPECT_EQ(subr(1, 1, 0), this->c_st_ar(0.14));
}


}  // namespace
