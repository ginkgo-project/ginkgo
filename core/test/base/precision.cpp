// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/precision.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/base/math.hpp>

namespace {
namespace gko_test {


auto precisions = {
    gko::precision::none, gko::precision::any,
    gko::precision::fp32, gko::precision::complex_fp32,
    gko::precision::fp64, gko::precision::complex_fp64,
#if GINKGO_ENABLE_BFLOAT16
    gko::precision::bf16, gko::precision::complex_bf16,
#endif
#if GINKGO_ENABLE_HALF
    gko::precision::fp16, gko::precision::complex_fp16,
#endif
};


TEST(Precision, EnumOpsEqual)
{
    auto any_p = gko::precision::any;

    for (auto p1 : precisions) {
        auto same_p = p1;

        EXPECT_EQ(p1, same_p);
        EXPECT_EQ(same_p, p1);
        EXPECT_EQ(p1, any_p);
        EXPECT_EQ(any_p, p1);
    }
}


TEST(Precision, EnumOpsNotEqual)
{
    for (auto p1 : precisions) {
        for (auto p2 : precisions) {
            if (p1 == p2) {
                continue;
            }

            EXPECT_NE(p1, p2);
            EXPECT_NE(p2, p1);
        }
    }
}


TEST(Precision, TypeToPrecision)
{
    EXPECT_EQ(gko::precision_v<float>, gko::precision::fp32);
    EXPECT_EQ(gko::precision_v<std::complex<float>>,
              gko::precision::complex_fp32);
    EXPECT_EQ(gko::precision_v<double>, gko::precision::fp64);
    EXPECT_EQ(gko::precision_v<std::complex<double>>,
              gko::precision::complex_fp64);
#if GINKGO_ENABLE_BFLOAT16
    EXPECT_EQ(gko::precision_v<gko::bfloat16>, gko::precision::bf16);
    EXPECT_EQ(gko::precision_v<std::complex<gko::bfloat16>>,
              gko::precision::complex_bf16);
#endif
#if GINKGO_ENABLE_HALF
    EXPECT_EQ(gko::precision_v<gko::half>, gko::precision::fp16);
    EXPECT_EQ(gko::precision_v<std::complex<gko::half>>,
              gko::precision::complex_fp16);
#endif
}


template <typename T>
void test_is_complex()
{
    EXPECT_EQ(gko::is_complex<T>(), gko::is_complex(gko::precision_v<T>));
}

TEST(Precision, IsComplex)
{
    test_is_complex<float>();
    test_is_complex<double>();
    test_is_complex<std::complex<float>>();
    test_is_complex<std::complex<double>>();
#if GINKGO_ENABLE_HALF
    test_is_complex<gko::half>();
    test_is_complex<std::complex<gko::half>>();
#endif
#if GINKGO_ENABLE_BFLOAT16
    test_is_complex<gko::bfloat16>();
    test_is_complex<std::complex<gko::bfloat16>>();
#endif
}


template <typename T>
void test_is_real()
{
    EXPECT_EQ(!gko::is_complex<T>(), gko::is_real(gko::precision_v<T>));
}

TEST(Precision, IsReal)
{
    test_is_real<float>();
    test_is_real<double>();
    test_is_real<std::complex<float>>();
    test_is_real<std::complex<double>>();
#if GINKGO_ENABLE_HALF
    test_is_real<gko::half>();
    test_is_real<std::complex<gko::half>>();
#endif
#if GINKGO_ENABLE_BFLOAT16
    test_is_real<gko::bfloat16>();
    test_is_real<std::complex<gko::bfloat16>>();
#endif
}

TEST(Precision, AsReal)
{
    EXPECT_EQ(gko::as_real(gko::precision::fp32), gko::precision::fp32);
    EXPECT_EQ(gko::as_real(gko::precision::fp64), gko::precision::fp64);
    EXPECT_EQ(gko::as_real(gko::precision::complex_fp32), gko::precision::fp32);
    EXPECT_EQ(gko::as_real(gko::precision::complex_fp64), gko::precision::fp64);
#if GINKGO_ENABLE_BFLOAT16
    EXPECT_EQ(gko::as_real(gko::precision::bf16), gko::precision::bf16);
    EXPECT_EQ(gko::as_real(gko::precision::complex_bf16), gko::precision::bf16);
#endif
#if GINKGO_ENABLE_HALF
    EXPECT_EQ(gko::as_real(gko::precision::fp16), gko::precision::fp16);
    EXPECT_EQ(gko::as_real(gko::precision::complex_fp16), gko::precision::fp16);
#endif
    EXPECT_EQ(gko::as_real(gko::precision::any), gko::precision::any);
    EXPECT_THROW(gko::as_real(gko::precision::none), gko::InvalidStateError);
}

TEST(Precision, AsComplex)
{
    EXPECT_EQ(gko::as_complex(gko::precision::fp32),
              gko::precision::complex_fp32);
    EXPECT_EQ(gko::as_complex(gko::precision::fp64),
              gko::precision::complex_fp64);
    EXPECT_EQ(gko::as_complex(gko::precision::complex_fp32),
              gko::precision::complex_fp32);
    EXPECT_EQ(gko::as_complex(gko::precision::complex_fp64),
              gko::precision::complex_fp64);
#if GINKGO_ENABLE_BFLOAT16
    EXPECT_EQ(gko::as_complex(gko::precision::bf16),
              gko::precision::complex_bf16);
    EXPECT_EQ(gko::as_complex(gko::precision::complex_bf16),
              gko::precision::complex_bf16);
#endif
#if GINKGO_ENABLE_HALF
    EXPECT_EQ(gko::as_complex(gko::precision::fp16),
              gko::precision::complex_fp16);
    EXPECT_EQ(gko::as_complex(gko::precision::complex_fp16),
              gko::precision::complex_fp16);
#endif
    EXPECT_EQ(gko::as_real(gko::precision::any), gko::precision::any);
    EXPECT_THROW(gko::as_complex(gko::precision::none), gko::InvalidStateError);
}

TEST(Precision, PrecisionToVariant)
{
    EXPECT_TRUE(std::holds_alternative<float>(
        gko::precision_to_variant(gko::precision::fp32)));
    EXPECT_TRUE(std::holds_alternative<double>(
        gko::precision_to_variant(gko::precision::fp64)));
    EXPECT_TRUE(std::holds_alternative<std::complex<float>>(
        gko::precision_to_variant(gko::precision::complex_fp32)));
    EXPECT_TRUE(std::holds_alternative<std::complex<double>>(
        gko::precision_to_variant(gko::precision::complex_fp64)));
#if GINKGO_ENABLE_HALF
    EXPECT_TRUE(std::holds_alternative<gko::half>(
        gko::precision_to_variant(gko::precision::fp16)));
    EXPECT_TRUE(std::holds_alternative<std::complex<gko::half>>(
        gko::precision_to_variant(gko::precision::complex_fp16)));
#endif
#if GINKGO_ENABLE_BFLOAT16
    EXPECT_TRUE(std::holds_alternative<gko::bfloat16>(
        gko::precision_to_variant(gko::precision::bf16)));
    EXPECT_TRUE(std::holds_alternative<std::complex<gko::bfloat16>>(
        gko::precision_to_variant(gko::precision::complex_bf16)));
#endif
    EXPECT_THROW(gko::precision_to_variant(gko::precision::none),
                 gko::InvalidStateError);
    EXPECT_THROW(gko::precision_to_variant(gko::precision::any),
                 gko::InvalidStateError);
}


}  // namespace gko_test
}  // namespace
