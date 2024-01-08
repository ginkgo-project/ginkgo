// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
                                        curandRngType generator_type,
                                        cudaStream_t stream)
{
    curandGenerator_t gen;
    GKO_ASSERT_NO_CURAND_ERRORS(curandCreateGenerator(&gen, generator_type));
    GKO_ASSERT_NO_CURAND_ERRORS(curandSetStream(gen, stream));
    GKO_ASSERT_NO_CURAND_ERRORS(curandSetPseudoRandomGeneratorSeed(gen, seed));
    return gen;
}


inline void destroy(curandGenerator_t gen)
{
    GKO_ASSERT_NO_CURAND_ERRORS(curandDestroyGenerator(gen));
}


#define GKO_BIND_CURAND_RANDOM_VECTOR(ValueType, CurandName)                 \
    inline void rand_vector(                                                 \
        curandGenerator_t& gen, int n, remove_complex<ValueType> mean,       \
        remove_complex<ValueType> stddev, ValueType* values)                 \
    {                                                                        \
        n = is_complex<ValueType>() ? 2 * n : n;                             \
        GKO_ASSERT_NO_CURAND_ERRORS(CurandName(                              \
            gen, reinterpret_cast<remove_complex<ValueType>*>(values), n,    \
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
