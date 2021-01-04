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

#ifndef GKO_HIP_BASE_HIPRAND_BINDINGS_HIP_HPP_
#define GKO_HIP_BASE_HIPRAND_BINDINGS_HIP_HPP_


#include <hiprand.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The HIPRAND namespace.
 *
 * @ingroup hiprand
 */
namespace hiprand {


template <typename ValueType>
struct is_supported : std::false_type {};

template <>
struct is_supported<float> : std::true_type {};

template <>
struct is_supported<double> : std::true_type {};

template <>
struct is_supported<std::complex<float>> : std::true_type {};

template <>
struct is_supported<std::complex<double>> : std::true_type {};


inline hiprandGenerator_t rand_generator(int64 seed,
                                         hiprandRngType generator_type)
{
    hiprandGenerator_t gen;
    hiprandCreateGenerator(&gen, generator_type);
    hiprandSetPseudoRandomGeneratorSeed(gen, seed);
    return gen;
}


#define GKO_BIND_HIPRAND_RANDOM_VECTOR(ValueType, HiprandName)               \
    inline void rand_vector(                                                 \
        hiprandGenerator_t &gen, int n, remove_complex<ValueType> mean,      \
        remove_complex<ValueType> stddev, ValueType *values)                 \
    {                                                                        \
        n = is_complex<ValueType>() ? 2 * n : n;                             \
        GKO_ASSERT_NO_HIPRAND_ERRORS(HiprandName(                            \
            gen, reinterpret_cast<remove_complex<ValueType> *>(values), n,   \
            mean, stddev));                                                  \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPRAND_RANDOM_VECTOR(float, hiprandGenerateNormal);
GKO_BIND_HIPRAND_RANDOM_VECTOR(double, hiprandGenerateNormalDouble);
GKO_BIND_HIPRAND_RANDOM_VECTOR(std::complex<float>, hiprandGenerateNormal);
GKO_BIND_HIPRAND_RANDOM_VECTOR(std::complex<double>,
                               hiprandGenerateNormalDouble);


#undef GKO_BIND_HIPRAND_RANDOM_VECTOR


}  // namespace hiprand
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_HIPRAND_BINDINGS_HIP_HPP_
