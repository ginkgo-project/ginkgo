// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_HIPRAND_BINDINGS_HIP_HPP_
#define GKO_HIP_BASE_HIPRAND_BINDINGS_HIP_HPP_


#if HIP_VERSION >= 50200000
#include <hiprand/hiprand.h>
#else
#include <hiprand.h>
#endif


#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The HIPRAND namespace.
 *
 * @ingroup hiprand
 */
namespace hiprand {
namespace detail {


template <typename... Args>
inline int64 not_implemented(Args...)
{
    return static_cast<int64>(HIPRAND_STATUS_TYPE_ERROR);
}


}  // namespace detail


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
                                         hiprandRngType generator_type,
                                         hipStream_t stream)
{
    hiprandGenerator_t gen;
    GKO_ASSERT_NO_HIPRAND_ERRORS(hiprandCreateGenerator(&gen, generator_type));
    GKO_ASSERT_NO_HIPRAND_ERRORS(
        hiprandSetPseudoRandomGeneratorSeed(gen, seed));
    return gen;
}

inline void destroy(hiprandGenerator_t gen)
{
    GKO_ASSERT_NO_HIPRAND_ERRORS(hiprandDestroyGenerator(gen));
}


#define GKO_BIND_HIPRAND_RANDOM_VECTOR(ValueType, HiprandName)               \
    inline void rand_vector(                                                 \
        hiprandGenerator_t& gen, int n, remove_complex<ValueType> mean,      \
        remove_complex<ValueType> stddev, ValueType* values)                 \
    {                                                                        \
        n = is_complex<ValueType>() ? 2 * n : n;                             \
        GKO_ASSERT_NO_HIPRAND_ERRORS(HiprandName(                            \
            gen, reinterpret_cast<remove_complex<ValueType>*>(values), n,    \
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
template <typename ValueType>
GKO_BIND_HIPRAND_RANDOM_VECTOR(ValueType, detail::not_implemented);


#undef GKO_BIND_HIPRAND_RANDOM_VECTOR


}  // namespace hiprand


namespace randlib {


using namespace hiprand;


#define RANDLIB_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT


}  // namespace randlib
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_HIPRAND_BINDINGS_HIP_HPP_
