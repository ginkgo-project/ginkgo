// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_
#define GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/types.hpp>


#ifdef GINKGO_MIXED_PRECISION


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1_BASE(_macro, ...) \
    template _macro(float, float, float, __VA_ARGS__);                     \
    template _macro(float, float, double, __VA_ARGS__);                    \
    template _macro(float, double, float, __VA_ARGS__);                    \
    template _macro(float, double, double, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1(_macro, ...)   \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1_BASE(_macro,       \
                                                          __VA_ARGS__); \
    GKO_ADAPT_HF(template _macro(float, half, half, __VA_ARGS__));      \
    GKO_ADAPT_HF(template _macro(float, half, float, __VA_ARGS__));     \
    GKO_ADAPT_HF(template _macro(float, half, double, __VA_ARGS__));    \
    GKO_ADAPT_HF(template _macro(float, float, half, __VA_ARGS__));     \
    GKO_ADAPT_HF(template _macro(float, double, half, __VA_ARGS__))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2_BASE(_macro, ...) \
    template _macro(double, float, float, __VA_ARGS__);                    \
    template _macro(double, float, double, __VA_ARGS__);                   \
    template _macro(double, double, float, __VA_ARGS__);                   \
    template _macro(double, double, double, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2(_macro, ...)   \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2_BASE(_macro,       \
                                                          __VA_ARGS__); \
    GKO_ADAPT_HF(template _macro(double, half, half, __VA_ARGS__));     \
    GKO_ADAPT_HF(template _macro(double, half, float, __VA_ARGS__));    \
    GKO_ADAPT_HF(template _macro(double, half, double, __VA_ARGS__));   \
    GKO_ADAPT_HF(template _macro(double, float, half, __VA_ARGS__));    \
    GKO_ADAPT_HF(template _macro(double, double, half, __VA_ARGS__))


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3_BASE(_macro, ...) \
    template _macro(std::complex<float>, std::complex<float>,              \
                    std::complex<float>, __VA_ARGS__);                     \
    template _macro(std::complex<float>, std::complex<float>,              \
                    std::complex<double>, __VA_ARGS__);                    \
    template _macro(std::complex<float>, std::complex<double>,             \
                    std::complex<float>, __VA_ARGS__);                     \
    template _macro(std::complex<float>, std::complex<double>,             \
                    std::complex<double>, __VA_ARGS__)
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3(_macro, ...)       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3_BASE(_macro,           \
                                                          __VA_ARGS__);     \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<half>,   \
                                 std::complex<half>, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<half>,   \
                                 std::complex<float>, __VA_ARGS__));        \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<half>,   \
                                 std::complex<double>, __VA_ARGS__));       \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<float>,  \
                                 std::complex<half>, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<double>, \
                                 std::complex<half>, __VA_ARGS__))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4_BASE(_macro, ...) \
    template _macro(std::complex<double>, std::complex<float>,             \
                    std::complex<float>, __VA_ARGS__);                     \
    template _macro(std::complex<double>, std::complex<float>,             \
                    std::complex<double>, __VA_ARGS__);                    \
    template _macro(std::complex<double>, std::complex<double>,            \
                    std::complex<float>, __VA_ARGS__);                     \
    template _macro(std::complex<double>, std::complex<double>,            \
                    std::complex<double>, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4(_macro, ...)        \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4_BASE(_macro,            \
                                                          __VA_ARGS__);      \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<half>,   \
                                 std::complex<half>, __VA_ARGS__));          \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<half>,   \
                                 std::complex<float>, __VA_ARGS__));         \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<half>,   \
                                 std::complex<double>, __VA_ARGS__));        \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<float>,  \
                                 std::complex<half>, __VA_ARGS__));          \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<double>, \
                                 std::complex<half>, __VA_ARGS__))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT5(_macro, ...) \
    GKO_ADAPT_HF(template _macro(half, half, half, __VA_ARGS__));     \
    GKO_ADAPT_HF(template _macro(half, half, float, __VA_ARGS__));    \
    GKO_ADAPT_HF(template _macro(half, half, double, __VA_ARGS__));   \
    GKO_ADAPT_HF(template _macro(half, float, half, __VA_ARGS__));    \
    GKO_ADAPT_HF(template _macro(half, float, float, __VA_ARGS__));   \
    GKO_ADAPT_HF(template _macro(half, float, double, __VA_ARGS__));  \
    GKO_ADAPT_HF(template _macro(half, double, half, __VA_ARGS__));   \
    GKO_ADAPT_HF(template _macro(half, double, float, __VA_ARGS__));  \
    GKO_ADAPT_HF(template _macro(half, double, double, __VA_ARGS__))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT6(_macro, ...)      \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<half>,   \
                                 std::complex<half>, __VA_ARGS__));        \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<half>,   \
                                 std::complex<float>, __VA_ARGS__));       \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<half>,   \
                                 std::complex<double>, __VA_ARGS__));      \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<float>,  \
                                 std::complex<half>, __VA_ARGS__));        \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<float>,  \
                                 std::complex<float>, __VA_ARGS__));       \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<float>,  \
                                 std::complex<double>, __VA_ARGS__));      \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<double>, \
                                 std::complex<half>, __VA_ARGS__));        \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<double>, \
                                 std::complex<float>, __VA_ARGS__));       \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<double>, \
                                 std::complex<double>, __VA_ARGS__))

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
    GKO_ADAPT_HF(template _macro(half, half, half, __VA_ARGS__))

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT6(_macro, ...)    \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<half>, \
                                 std::complex<half>, __VA_ARGS__))


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
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT6(_macro, __VA_ARGS__)

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
    GKO_ADAPT_HF(template _macro(half, half, __VA_ARGS__));                    \
    GKO_ADAPT_HF(template _macro(half, float, __VA_ARGS__));                   \
    GKO_ADAPT_HF(template _macro(half, double, __VA_ARGS__));                  \
    GKO_ADAPT_HF(template _macro(float, half, __VA_ARGS__));                   \
    GKO_ADAPT_HF(template _macro(double, half, __VA_ARGS__));                  \
    GKO_ADAPT_HF(                                                              \
        template _macro(std::complex<half>, std::complex<half>, __VA_ARGS__)); \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<float>,      \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_HF(template _macro(std::complex<half>, std::complex<double>,     \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_HF(template _macro(std::complex<float>, std::complex<half>,      \
                                 __VA_ARGS__));                                \
    GKO_ADAPT_HF(template _macro(std::complex<double>, std::complex<half>,     \
                                 __VA_ARGS__))
#else
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2_BASE(_macro, ...)       \
    template _macro(float, float, __VA_ARGS__);                             \
    template _macro(double, double, __VA_ARGS__);                           \
    template _macro(std::complex<float>, std::complex<float>, __VA_ARGS__); \
    template _macro(std::complex<double>, std::complex<double>, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, ...)           \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2_BASE(_macro, __VA_ARGS__); \
    GKO_ADAPT_HF(template _macro(half, half, __VA_ARGS__));                \
    GKO_ADAPT_HF(                                                          \
        template _macro(std::complex<half>, std::complex<half>, __VA_ARGS__))
#endif


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2_BASE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2_BASE(_macro, int32);       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2_BASE(_macro, int64)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(_macro) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, int32);       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, int64)

#endif  // GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_
