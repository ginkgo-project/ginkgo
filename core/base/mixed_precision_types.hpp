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

#ifndef GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_
#define GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/types.hpp>


#ifdef GINKGO_MIXED_PRECISION
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE(_macro, ...)        \
    template _macro(half, half, half, __VA_ARGS__);                   \
    template _macro(half, half, float, __VA_ARGS__);                  \
    template _macro(half, half, double, __VA_ARGS__);                 \
    template _macro(half, float, half, __VA_ARGS__);                  \
    template _macro(half, float, float, __VA_ARGS__);                 \
    template _macro(half, float, double, __VA_ARGS__);                \
    template _macro(half, double, half, __VA_ARGS__);                 \
    template _macro(half, double, float, __VA_ARGS__);                \
    template _macro(half, double, double, __VA_ARGS__);               \
    template _macro(float, half, half, __VA_ARGS__);                  \
    template _macro(float, half, float, __VA_ARGS__);                 \
    template _macro(float, half, double, __VA_ARGS__);                \
    template _macro(float, float, half, __VA_ARGS__);                 \
    template _macro(float, float, float, __VA_ARGS__);                \
    template _macro(float, float, double, __VA_ARGS__);               \
    template _macro(float, double, half, __VA_ARGS__);                \
    template _macro(float, double, float, __VA_ARGS__);               \
    template _macro(float, double, double, __VA_ARGS__);              \
    template _macro(double, half, half, __VA_ARGS__);                 \
    template _macro(double, half, float, __VA_ARGS__);                \
    template _macro(double, half, double, __VA_ARGS__);               \
    template _macro(double, float, half, __VA_ARGS__);                \
    template _macro(double, float, float, __VA_ARGS__);               \
    template _macro(double, float, double, __VA_ARGS__);              \
    template _macro(double, double, half, __VA_ARGS__);               \
    template _macro(double, double, float, __VA_ARGS__);              \
    template _macro(double, double, double, __VA_ARGS__);             \
    GKO_ADAPT_CPHF(_macro(std::complex<half>, std::complex<half>,     \
                          std::complex<half>, __VA_ARGS__));          \
    GKO_ADAPT_CPHF(_macro(std::complex<half>, std::complex<half>,     \
                          std::complex<float>, __VA_ARGS__));         \
    GKO_ADAPT_CPHF(_macro(std::complex<half>, std::complex<half>,     \
                          std::complex<double>, __VA_ARGS__));        \
    GKO_ADAPT_CPHF(_macro(std::complex<half>, std::complex<float>,    \
                          std::complex<half>, __VA_ARGS__));          \
    GKO_ADAPT_CPHF(_macro(std::complex<half>, std::complex<float>,    \
                          std::complex<float>, __VA_ARGS__));         \
    GKO_ADAPT_CPHF(_macro(std::complex<half>, std::complex<float>,    \
                          std::complex<double>, __VA_ARGS__));        \
    GKO_ADAPT_CPHF(_macro(std::complex<half>, std::complex<double>,   \
                          std::complex<half>, __VA_ARGS__));          \
    GKO_ADAPT_CPHF(_macro(std::complex<half>, std::complex<double>,   \
                          std::complex<float>, __VA_ARGS__));         \
    GKO_ADAPT_CPHF(_macro(std::complex<half>, std::complex<double>,   \
                          std::complex<double>, __VA_ARGS__));        \
    GKO_ADAPT_CPHF(_macro(std::complex<float>, std::complex<half>,    \
                          std::complex<half>, __VA_ARGS__));          \
    GKO_ADAPT_CPHF(_macro(std::complex<float>, std::complex<half>,    \
                          std::complex<float>, __VA_ARGS__));         \
    GKO_ADAPT_CPHF(_macro(std::complex<float>, std::complex<half>,    \
                          std::complex<double>, __VA_ARGS__));        \
    GKO_ADAPT_CPHF(_macro(std::complex<float>, std::complex<float>,   \
                          std::complex<half>, __VA_ARGS__));          \
    template _macro(std::complex<float>, std::complex<float>,         \
                    std::complex<float>, __VA_ARGS__);                \
    template _macro(std::complex<float>, std::complex<float>,         \
                    std::complex<double>, __VA_ARGS__);               \
    GKO_ADAPT_CPHF(_macro(std::complex<float>, std::complex<double>,  \
                          std::complex<half>, __VA_ARGS__));          \
    template _macro(std::complex<float>, std::complex<double>,        \
                    std::complex<float>, __VA_ARGS__);                \
    template _macro(std::complex<float>, std::complex<double>,        \
                    std::complex<double>, __VA_ARGS__);               \
    GKO_ADAPT_CPHF(_macro(std::complex<double>, std::complex<half>,   \
                          std::complex<half>, __VA_ARGS__));          \
    GKO_ADAPT_CPHF(_macro(std::complex<double>, std::complex<half>,   \
                          std::complex<float>, __VA_ARGS__));         \
    GKO_ADAPT_CPHF(_macro(std::complex<double>, std::complex<half>,   \
                          std::complex<double>, __VA_ARGS__));        \
    GKO_ADAPT_CPHF(_macro(std::complex<double>, std::complex<float>,  \
                          std::complex<half>, __VA_ARGS__));          \
    template _macro(std::complex<double>, std::complex<float>,        \
                    std::complex<float>, __VA_ARGS__);                \
    template _macro(std::complex<double>, std::complex<float>,        \
                    std::complex<double>, __VA_ARGS__);               \
    GKO_ADAPT_CPHF(_macro(std::complex<double>, std::complex<double>, \
                          std::complex<half>, __VA_ARGS__));          \
    template _macro(std::complex<double>, std::complex<double>,       \
                    std::complex<float>, __VA_ARGS__);                \
    template _macro(std::complex<double>, std::complex<double>,       \
                    std::complex<double>, __VA_ARGS__)
#else
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE(_macro, ...)    \
    template _macro(half, half, half, __VA_ARGS__);               \
    template _macro(float, float, float, __VA_ARGS__);            \
    template _macro(double, double, double, __VA_ARGS__);         \
    GKO_ADAPT_CPHF(_macro(std::complex<half>, std::complex<half>, \
                          std::complex<half>, __VA_ARGS__));      \
    template _macro(std::complex<float>, std::complex<float>,     \
                    std::complex<float>, __VA_ARGS__);            \
    template _macro(std::complex<double>, std::complex<double>,   \
                    std::complex<double>, __VA_ARGS__)
#endif


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE(_macro, int32);       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE(_macro, int64)


#ifdef GINKGO_MIXED_PRECISION
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, ...)             \
    template _macro(half, half, __VA_ARGS__);                                \
    template _macro(half, float, __VA_ARGS__);                               \
    template _macro(half, double, __VA_ARGS__);                              \
    template _macro(float, half, __VA_ARGS__);                               \
    template _macro(float, float, __VA_ARGS__);                              \
    template _macro(float, double, __VA_ARGS__);                             \
    template _macro(double, half, __VA_ARGS__);                              \
    template _macro(double, float, __VA_ARGS__);                             \
    template _macro(double, double, __VA_ARGS__);                            \
    GKO_ADAPT_CPHF(                                                          \
        _macro(std::complex<half>, std::complex<half>, __VA_ARGS__));        \
    GKO_ADAPT_CPHF(                                                          \
        _macro(std::complex<half>, std::complex<float>, __VA_ARGS__));       \
    GKO_ADAPT_CPHF(                                                          \
        _macro(std::complex<half>, std::complex<double>, __VA_ARGS__));      \
    GKO_ADAPT_CPHF(                                                          \
        _macro(std::complex<float>, std::complex<half>, __VA_ARGS__));       \
    template _macro(std::complex<float>, std::complex<float>, __VA_ARGS__);  \
    template _macro(std::complex<float>, std::complex<double>, __VA_ARGS__); \
    GKO_ADAPT_CPHF(                                                          \
        _macro(std::complex<double>, std::complex<half>, __VA_ARGS__));      \
    template _macro(std::complex<double>, std::complex<float>, __VA_ARGS__); \
    template _macro(std::complex<double>, std::complex<double>, __VA_ARGS__)
#else
#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, ...)            \
    template _macro(half, half, __VA_ARGS__);                               \
    template _macro(float, float, __VA_ARGS__);                             \
    template _macro(double, double, __VA_ARGS__);                           \
    GKO_ADAPT_CPHF(                                                         \
        _macro(std::complex<half>, std::complex<half>, __VA_ARGS__));       \
    template _macro(std::complex<float>, std::complex<float>, __VA_ARGS__); \
    template _macro(std::complex<double>, std::complex<double>, __VA_ARGS__)
#endif


#define GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(_macro) \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, int32);       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(_macro, int64)


#endif  // GKO_CORE_BASE_MIXED_PRECISION_TYPES_HPP_
