// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_
#define GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/types.hpp>


#ifdef GINKGO_MIXED_PRECISION

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1(_macro, ...) \
    template _macro(float, float, float, __VA_ARGS__);                \
    template _macro(float, float, double, __VA_ARGS__);               \
    template _macro(float, double, float, __VA_ARGS__);               \
    template _macro(float, double, double, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2(_macro, ...) \
    template _macro(double, float, float, __VA_ARGS__);               \
    template _macro(double, float, double, __VA_ARGS__);              \
    template _macro(double, double, float, __VA_ARGS__);              \
    template _macro(double, double, double, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3(_macro, ...) \
    template _macro(std::complex<float>, std::complex<float>,         \
                    std::complex<float>, __VA_ARGS__);                \
    template _macro(std::complex<float>, std::complex<float>,         \
                    std::complex<double>, __VA_ARGS__);               \
    template _macro(std::complex<float>, std::complex<double>,        \
                    std::complex<float>, __VA_ARGS__);                \
    template _macro(std::complex<float>, std::complex<double>,        \
                    std::complex<double>, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4(_macro, ...) \
    template _macro(std::complex<double>, std::complex<float>,        \
                    std::complex<float>, __VA_ARGS__);                \
    template _macro(std::complex<double>, std::complex<float>,        \
                    std::complex<double>, __VA_ARGS__);               \
    template _macro(std::complex<double>, std::complex<double>,       \
                    std::complex<float>, __VA_ARGS__);                \
    template _macro(std::complex<double>, std::complex<double>,       \
                    std::complex<double>, __VA_ARGS__)

#else

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1(_macro, ...) \
    template _macro(float, float, float, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2(_macro, ...) \
    template _macro(double, double, double, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3(_macro, ...) \
    template _macro(std::complex<float>, std::complex<float>,         \
                    std::complex<float>, __VA_ARGS__)

#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4(_macro, ...) \
    template _macro(std::complex<double>, std::complex<double>,       \
                    std::complex<double>, __VA_ARGS__)

#endif


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE(_macro, ...)             \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT1(_macro, __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT2(_macro, __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT3(_macro, __VA_ARGS__); \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_SPLIT4(_macro, __VA_ARGS__)


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE(_macro, int32);       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE(_macro, int64)


#ifdef GINKGO_MIXED_PRECISION
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, ...)             \
    template _macro(float, float, __VA_ARGS__);                              \
    template _macro(float, double, __VA_ARGS__);                             \
    template _macro(double, float, __VA_ARGS__);                             \
    template _macro(double, double, __VA_ARGS__);                            \
    template _macro(std::complex<float>, std::complex<float>, __VA_ARGS__);  \
    template _macro(std::complex<float>, std::complex<double>, __VA_ARGS__); \
    template _macro(std::complex<double>, std::complex<float>, __VA_ARGS__); \
    template _macro(std::complex<double>, std::complex<double>, __VA_ARGS__)
#else
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, ...)            \
    template _macro(float, float, __VA_ARGS__);                             \
    template _macro(double, double, __VA_ARGS__);                           \
    template _macro(std::complex<float>, std::complex<float>, __VA_ARGS__); \
    template _macro(std::complex<double>, std::complex<double>, __VA_ARGS__)
#endif


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(_macro) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, int32);       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, int64)


#endif  // GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_
