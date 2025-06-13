// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_
#define GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/bfloat16.hpp>
#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/types.hpp>


#ifdef GINKGO_MIXED_PRECISION


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1_BASE(_macro, ...) \
    template _macro(float, float, float, __VA_ARGS__);                     \
    template _macro(float, float, double, __VA_ARGS__);                    \
    template _macro(float, double, float, __VA_ARGS__);                    \
    template _macro(float, double, double, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1(_macro, ...)          \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1_BASE(_macro,              \
                                                          __VA_ARGS__);        \
    GKO_ADAPT_HF(template _macro(float, float16, float16, __VA_ARGS__));       \
    GKO_ADAPT_HF(template _macro(float, float16, float, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(float, float16, double, __VA_ARGS__));        \
    GKO_ADAPT_HF(template _macro(float, float, float16, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(float, double, float16, __VA_ARGS__));        \
    GKO_ADAPT_BF(template _macro(float, bfloat16, bfloat16, __VA_ARGS__));     \
    GKO_ADAPT_BF(template _macro(float, bfloat16, float, __VA_ARGS__));        \
    GKO_ADAPT_BF(template _macro(float, bfloat16, double, __VA_ARGS__));       \
    GKO_ADAPT_BF(template _macro(float, float, bfloat16, __VA_ARGS__));        \
    GKO_ADAPT_BF(template _macro(float, double, bfloat16, __VA_ARGS__));       \
    GKO_ADAPT_HF(                                                              \
        GKO_ADAPT_BF(template _macro(float, bfloat16, float16, __VA_ARGS__))); \
    GKO_ADAPT_HF(                                                              \
        GKO_ADAPT_BF(template _macro(float, float16, bfloat16, __VA_ARGS__)))


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2_BASE(_macro, ...) \
    template _macro(double, float, float, __VA_ARGS__);                    \
    template _macro(double, float, double, __VA_ARGS__);                   \
    template _macro(double, double, float, __VA_ARGS__);                   \
    template _macro(double, double, double, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2(_macro, ...)       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2_BASE(_macro,           \
                                                          __VA_ARGS__);     \
    GKO_ADAPT_HF(template _macro(double, float16, float16, __VA_ARGS__));   \
    GKO_ADAPT_HF(template _macro(double, float16, float, __VA_ARGS__));     \
    GKO_ADAPT_HF(template _macro(double, float16, double, __VA_ARGS__));    \
    GKO_ADAPT_HF(template _macro(double, float, float16, __VA_ARGS__));     \
    GKO_ADAPT_HF(template _macro(double, double, float16, __VA_ARGS__));    \
    GKO_ADAPT_BF(template _macro(double, bfloat16, bfloat16, __VA_ARGS__)); \
    GKO_ADAPT_BF(template _macro(double, bfloat16, float, __VA_ARGS__));    \
    GKO_ADAPT_BF(template _macro(double, bfloat16, double, __VA_ARGS__));   \
    GKO_ADAPT_BF(template _macro(double, float, bfloat16, __VA_ARGS__));    \
    GKO_ADAPT_BF(template _macro(double, double, bfloat16, __VA_ARGS__));   \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                              \
        template _macro(double, bfloat16, float16, __VA_ARGS__)));          \
    GKO_ADAPT_HF(                                                           \
        GKO_ADAPT_BF(template _macro(double, float16, bfloat16, __VA_ARGS__)))


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3_BASE(_macro, ...) \
    template _macro(std::complex<float>, std::complex<float>,              \
                    std::complex<float>, __VA_ARGS__);                     \
    template _macro(std::complex<float>, std::complex<float>,              \
                    std::complex<double>, __VA_ARGS__);                    \
    template _macro(std::complex<float>, std::complex<double>,             \
                    std::complex<float>, __VA_ARGS__);                     \
    template _macro(std::complex<float>, std::complex<double>,             \
                    std::complex<double>, __VA_ARGS__)
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3(_macro, ...)         \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3_BASE(_macro,             \
                                                          __VA_ARGS__);       \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<float16>,  \
                                 std::complex<float16>, __VA_ARGS__));        \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<float16>,  \
                                 std::complex<float>, __VA_ARGS__));          \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<float16>,  \
                                 std::complex<double>, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<float>,    \
                                 std::complex<float16>, __VA_ARGS__));        \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<double>,   \
                                 std::complex<float16>, __VA_ARGS__));        \
    GKO_ADAPT_BF(template _macro(std::complex<float>, std::complex<bfloat16>, \
                                 std::complex<bfloat16>, __VA_ARGS__));       \
    GKO_ADAPT_BF(template _macro(std::complex<float>, std::complex<bfloat16>, \
                                 std::complex<float>, __VA_ARGS__));          \
    GKO_ADAPT_BF(template _macro(std::complex<float>, std::complex<bfloat16>, \
                                 std::complex<double>, __VA_ARGS__));         \
    GKO_ADAPT_BF(template _macro(std::complex<float>, std::complex<float>,    \
                                 std::complex<bfloat16>, __VA_ARGS__));       \
    GKO_ADAPT_BF(template _macro(std::complex<float>, std::complex<double>,   \
                                 std::complex<bfloat16>, __VA_ARGS__));       \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                \
        template _macro(std::complex<float>, std::complex<bfloat16>,          \
                        std::complex<float16>, __VA_ARGS__)));                \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                \
        template _macro(std::complex<float>, std::complex<float16>,           \
                        std::complex<bfloat16>, __VA_ARGS__)))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4_BASE(_macro, ...) \
    template _macro(std::complex<double>, std::complex<float>,             \
                    std::complex<float>, __VA_ARGS__);                     \
    template _macro(std::complex<double>, std::complex<float>,             \
                    std::complex<double>, __VA_ARGS__);                    \
    template _macro(std::complex<double>, std::complex<double>,            \
                    std::complex<float>, __VA_ARGS__);                     \
    template _macro(std::complex<double>, std::complex<double>,            \
                    std::complex<double>, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4(_macro, ...)          \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4_BASE(_macro,              \
                                                          __VA_ARGS__);        \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<float16>,  \
                                 std::complex<float16>, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<float16>,  \
                                 std::complex<float>, __VA_ARGS__));           \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<float16>,  \
                                 std::complex<double>, __VA_ARGS__));          \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<float>,    \
                                 std::complex<float16>, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<double>,   \
                                 std::complex<float16>, __VA_ARGS__));         \
    GKO_ADAPT_BF(template _macro(std::complex<double>, std::complex<bfloat16>, \
                                 std::complex<bfloat16>, __VA_ARGS__));        \
    GKO_ADAPT_BF(template _macro(std::complex<double>, std::complex<bfloat16>, \
                                 std::complex<float>, __VA_ARGS__));           \
    GKO_ADAPT_BF(template _macro(std::complex<double>, std::complex<bfloat16>, \
                                 std::complex<double>, __VA_ARGS__));          \
    GKO_ADAPT_BF(template _macro(std::complex<double>, std::complex<float>,    \
                                 std::complex<bfloat16>, __VA_ARGS__));        \
    GKO_ADAPT_BF(template _macro(std::complex<double>, std::complex<double>,   \
                                 std::complex<bfloat16>, __VA_ARGS__));        \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(std::complex<double>, std::complex<bfloat16>,          \
                        std::complex<float16>, __VA_ARGS__)));                 \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(std::complex<double>, std::complex<float16>,           \
                        std::complex<bfloat16>, __VA_ARGS__)))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT5(_macro, ...)          \
    GKO_ADAPT_HF(template _macro(float16, float16, float16, __VA_ARGS__));     \
    GKO_ADAPT_HF(template _macro(float16, float16, float, __VA_ARGS__));       \
    GKO_ADAPT_HF(template _macro(float16, float16, double, __VA_ARGS__));      \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(float16, float16, bfloat16, __VA_ARGS__)));            \
    GKO_ADAPT_HF(template _macro(float16, float, float16, __VA_ARGS__));       \
    GKO_ADAPT_HF(template _macro(float16, float, float, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(float16, float, double, __VA_ARGS__));        \
    GKO_ADAPT_HF(                                                              \
        GKO_ADAPT_BF(template _macro(float16, float, bfloat16, __VA_ARGS__))); \
    GKO_ADAPT_HF(template _macro(float16, double, float16, __VA_ARGS__));      \
    GKO_ADAPT_HF(template _macro(float16, double, float, __VA_ARGS__));        \
    GKO_ADAPT_HF(template _macro(float16, double, double, __VA_ARGS__));       \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(float16, double, bfloat16, __VA_ARGS__)));             \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(float16, bfloat16, float16, __VA_ARGS__)));            \
    GKO_ADAPT_HF(                                                              \
        GKO_ADAPT_BF(template _macro(float16, bfloat16, float, __VA_ARGS__))); \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(float16, bfloat16, double, __VA_ARGS__)));             \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(float16, bfloat16, bfloat16, __VA_ARGS__)))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT6(_macro, ...)          \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<float16>, \
                                 std::complex<float16>, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<float16>, \
                                 std::complex<float>, __VA_ARGS__));           \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<float16>, \
                                 std::complex<double>, __VA_ARGS__));          \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(std::complex<float16>, std::complex<float16>,          \
                        std::complex<bfloat16>, __VA_ARGS__)));                \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<float>,   \
                                 std::complex<float16>, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<float>,   \
                                 std::complex<float>, __VA_ARGS__));           \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<float>,   \
                                 std::complex<double>, __VA_ARGS__));          \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(std::complex<float16>, std::complex<float>,            \
                        std::complex<bfloat16>, __VA_ARGS__)));                \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<double>,  \
                                 std::complex<float16>, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<double>,  \
                                 std::complex<float>, __VA_ARGS__));           \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<double>,  \
                                 std::complex<double>, __VA_ARGS__));          \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(std::complex<float16>, std::complex<double>,           \
                        std::complex<bfloat16>, __VA_ARGS__)));                \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(std::complex<float16>, std::complex<bfloat16>,         \
                        std::complex<float16>, __VA_ARGS__)));                 \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(std::complex<float16>, std::complex<bfloat16>,         \
                        std::complex<float>, __VA_ARGS__)));                   \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(std::complex<float16>, std::complex<bfloat16>,         \
                        std::complex<double>, __VA_ARGS__)));                  \
    GKO_ADAPT_HF(GKO_ADAPT_BF(                                                 \
        template _macro(std::complex<float16>, std::complex<bfloat16>,         \
                        std::complex<bfloat16>, __VA_ARGS__)))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT7(_macro, ...)          \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(bfloat16, float16, float16, __VA_ARGS__)));            \
    GKO_ADAPT_BF(                                                              \
        GKO_ADAPT_HF(template _macro(bfloat16, float16, float, __VA_ARGS__))); \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(bfloat16, float16, double, __VA_ARGS__)));             \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(bfloat16, float16, bfloat16, __VA_ARGS__)));           \
    GKO_ADAPT_BF(                                                              \
        GKO_ADAPT_HF(template _macro(bfloat16, float, float16, __VA_ARGS__))); \
    GKO_ADAPT_BF(template _macro(bfloat16, float, float, __VA_ARGS__));        \
    GKO_ADAPT_BF(template _macro(bfloat16, float, double, __VA_ARGS__));       \
    GKO_ADAPT_BF(template _macro(bfloat16, float, bfloat16, __VA_ARGS__));     \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(bfloat16, double, float16, __VA_ARGS__)));             \
    GKO_ADAPT_BF(template _macro(bfloat16, double, float, __VA_ARGS__));       \
    GKO_ADAPT_BF(template _macro(bfloat16, double, double, __VA_ARGS__));      \
    GKO_ADAPT_BF(template _macro(bfloat16, double, bfloat16, __VA_ARGS__));    \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(bfloat16, bfloat16, float16, __VA_ARGS__)));           \
    GKO_ADAPT_BF(template _macro(bfloat16, bfloat16, float, __VA_ARGS__));     \
    GKO_ADAPT_BF(template _macro(bfloat16, bfloat16, double, __VA_ARGS__));    \
    GKO_ADAPT_BF(template _macro(bfloat16, bfloat16, bfloat16, __VA_ARGS__))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT8(_macro, ...)          \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(std::complex<bfloat16>, std::complex<float16>,         \
                        std::complex<float16>, __VA_ARGS__)));                 \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(std::complex<bfloat16>, std::complex<float16>,         \
                        std::complex<float>, __VA_ARGS__)));                   \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(std::complex<bfloat16>, std::complex<float16>,         \
                        std::complex<double>, __VA_ARGS__)));                  \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(std::complex<bfloat16>, std::complex<float16>,         \
                        std::complex<bfloat16>, __VA_ARGS__)));                \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(std::complex<bfloat16>, std::complex<float>,           \
                        std::complex<float16>, __VA_ARGS__)));                 \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>, std::complex<float>,  \
                                 std::complex<float>, __VA_ARGS__));           \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>, std::complex<float>,  \
                                 std::complex<double>, __VA_ARGS__));          \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>, std::complex<float>,  \
                                 std::complex<bfloat16>, __VA_ARGS__));        \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(std::complex<bfloat16>, std::complex<double>,          \
                        std::complex<float16>, __VA_ARGS__)));                 \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>, std::complex<double>, \
                                 std::complex<float>, __VA_ARGS__));           \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>, std::complex<double>, \
                                 std::complex<double>, __VA_ARGS__));          \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>, std::complex<double>, \
                                 std::complex<bfloat16>, __VA_ARGS__));        \
    GKO_ADAPT_BF(GKO_ADAPT_HF(                                                 \
        template _macro(std::complex<bfloat16>, std::complex<bfloat16>,        \
                        std::complex<float16>, __VA_ARGS__)));                 \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>,                       \
                                 std::complex<bfloat16>, std::complex<float>,  \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>,                       \
                                 std::complex<bfloat16>, std::complex<double>, \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>,                       \
                                 std::complex<bfloat16>,                       \
                                 std::complex<bfloat16>, __VA_ARGS__))

#else

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1_BASE(_macro, ...) \
    template _macro(float, float, float, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1(_macro, ...) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1_BASE(_macro, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2_BASE(_macro, ...) \
    template _macro(double, double, double, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2(_macro, ...) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2_BASE(_macro, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3_BASE(_macro, ...) \
    template _macro(std::complex<float>, std::complex<float>,              \
                    std::complex<float>, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3(_macro, ...) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3_BASE(_macro, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4_BASE(_macro, ...) \
    template _macro(std::complex<double>, std::complex<double>,            \
                    std::complex<double>, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4(_macro, ...) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4_BASE(_macro, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT5(_macro, ...) \
    GKO_ADAPT_HF(template _macro(float16, float16, float16, __VA_ARGS__))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT6(_macro, ...)          \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<float16>, \
                                 std::complex<float16>, __VA_ARGS__))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT7(_macro, ...) \
    GKO_ADAPT_BF(template _macro(bfloat16, bfloat16, bfloat16, __VA_ARGS__))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT8(_macro, ...) \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>,              \
                                 std::complex<bfloat16>,              \
                                 std::complex<bfloat16>, __VA_ARGS__))


#endif


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_BASE(_macro, ...)     \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1_BASE(_macro,       \
                                                          __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2_BASE(_macro,       \
                                                          __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3_BASE(_macro,       \
                                                          __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4_BASE(_macro, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE(_macro, ...)             \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1(_macro, __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2(_macro, __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3(_macro, __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4(_macro, __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT5(_macro, __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT6(_macro, __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT7(_macro, __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT8(_macro, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_BASE(_macro, int32);       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_BASE(_macro, int64)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE(_macro, int32);       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE(_macro, int64)

#ifdef GINKGO_MIXED_PRECISION
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2_BASE(_macro, ...)        \
    template _macro(float, float, __VA_ARGS__);                              \
    template _macro(float, double, __VA_ARGS__);                             \
    template _macro(double, float, __VA_ARGS__);                             \
    template _macro(double, double, __VA_ARGS__);                            \
    template _macro(std::complex<float>, std::complex<float>, __VA_ARGS__);  \
    template _macro(std::complex<float>, std::complex<double>, __VA_ARGS__); \
    template _macro(std::complex<double>, std::complex<float>, __VA_ARGS__); \
    template _macro(std::complex<double>, std::complex<double>, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, ...)               \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2_BASE(_macro, __VA_ARGS__);     \
    GKO_ADAPT_HF(template _macro(float16, float16, __VA_ARGS__));              \
    GKO_ADAPT_HF(template _macro(float16, float, __VA_ARGS__));                \
    GKO_ADAPT_HF(template _macro(float16, double, __VA_ARGS__));               \
    GKO_ADAPT_HF(                                                              \
        GKO_ADAPT_BF(template _macro(float16, bfloat16, __VA_ARGS__)));        \
    GKO_ADAPT_HF(template _macro(float, float16, __VA_ARGS__));                \
    GKO_ADAPT_HF(template _macro(double, float16, __VA_ARGS__));               \
    GKO_ADAPT_BF(                                                              \
        GKO_ADAPT_HF(template _macro(bfloat16, float16, __VA_ARGS__)));        \
    GKO_ADAPT_BF(template _macro(bfloat16, float, __VA_ARGS__));               \
    GKO_ADAPT_BF(template _macro(bfloat16, double, __VA_ARGS__));              \
    GKO_ADAPT_BF(template _macro(bfloat16, bfloat16, __VA_ARGS__));            \
    GKO_ADAPT_BF(template _macro(float, bfloat16, __VA_ARGS__));               \
    GKO_ADAPT_BF(template _macro(double, bfloat16, __VA_ARGS__));              \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<float16>, \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<float>,   \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<double>,  \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_HF(GKO_ADAPT_BF(template _macro(                                 \
        std::complex<float16>, std::complex<bfloat16>, __VA_ARGS__)));         \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<float16>,   \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<float16>,  \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_BF(GKO_ADAPT_HF(template _macro(                                 \
        std::complex<bfloat16>, std::complex<float16>, __VA_ARGS__)));         \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>, std::complex<float>,  \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>, std::complex<double>, \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>,                       \
                                 std::complex<bfloat16>, __VA_ARGS__));        \
    GKO_ADAPT_BF(template _macro(std::complex<float>, std::complex<bfloat16>,  \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_BF(template _macro(std::complex<double>, std::complex<bfloat16>, \
                                 __VA_ARGS__))
#else
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2_BASE(_macro, ...)       \
    template _macro(float, float, __VA_ARGS__);                             \
    template _macro(double, double, __VA_ARGS__);                           \
    template _macro(std::complex<float>, std::complex<float>, __VA_ARGS__); \
    template _macro(std::complex<double>, std::complex<double>, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, ...)               \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2_BASE(_macro, __VA_ARGS__);     \
    GKO_ADAPT_HF(template _macro(float16, float16, __VA_ARGS__));              \
    GKO_ADAPT_BF(template _macro(bfloat16, bfloat16, __VA_ARGS__));            \
    GKO_ADAPT_HF(template _macro(std::complex<float16>, std::complex<float16>, \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_BF(template _macro(std::complex<bfloat16>,                       \
                                 std::complex<bfloat16>, __VA_ARGS__))
#endif


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2_BASE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2_BASE(_macro, int32);       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2_BASE(_macro, int64)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(_macro) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, int32);       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, int64)

#endif  // GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_
