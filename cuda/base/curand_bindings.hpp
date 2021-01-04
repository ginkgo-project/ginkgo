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

#ifndef GKO_CUDA_BASE_CURAND_BINDINGS_HPP_
#define GKO_CUDA_BASE_CURAND_BINDINGS_HPP_


#include <curand.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CURAND namespace.
 *
 * @ingroup curand
 */
namespace curand {


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


inline curandGenerator_t rand_generator(int64 seed,
                                        curandRngType generator_type)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, generator_type);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    return gen;
}


#define GKO_BIND_CURAND_RANDOM_VECTOR(ValueType, CurandName)                 \
    inline void rand_vector(                                                 \
        curandGenerator_t &gen, int n, remove_complex<ValueType> mean,       \
        remove_complex<ValueType> stddev, ValueType *values)                 \
    {                                                                        \
        n = is_complex<ValueType>() ? 2 * n : n;                             \
        GKO_ASSERT_NO_CURAND_ERRORS(CurandName(                              \
            gen, reinterpret_cast<remove_complex<ValueType> *>(values), n,   \
            mean, stddev));                                                  \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CURAND_RANDOM_VECTOR(float, curandGenerateNormal);
GKO_BIND_CURAND_RANDOM_VECTOR(double, curandGenerateNormalDouble);
GKO_BIND_CURAND_RANDOM_VECTOR(std::complex<float>, curandGenerateNormal);
GKO_BIND_CURAND_RANDOM_VECTOR(std::complex<double>, curandGenerateNormalDouble);


#undef GKO_BIND_CURAND_RANDOM_VECTOR


}  // namespace curand
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CURAND_BINDINGS_HPP_
