// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_TYPES_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_TYPES_HPP_

#include "common/cuda_hip/base/bf16_alias.hpp"
#include "common/cuda_hip/base/math.hpp"
#if defined(GKO_COMPILING_CUDA)
#include "cuda/base/types.hpp"
#elif defined(GKO_COMPILING_HIP)
#include "hip/base/types.hip.hpp"
#else
#error "Executor definition missing"
#endif

// We overwrite the thrust complex operation for 16 bit, or its default will use
// abs which is not supported.

#define THRUST_HALF_FRIEND_OPERATOR(_op, _opeq)                     \
    GKO_ATTRIBUTES GKO_INLINE GKO_THRUST_QUALIFIER::complex<__half> \
    operator _op(const GKO_THRUST_QUALIFIER::complex<__half> lhs,   \
                 const GKO_THRUST_QUALIFIER::complex<__half> rhs)   \
    {                                                               \
        return GKO_THRUST_QUALIFIER::complex<float>{                \
            lhs} _op GKO_THRUST_QUALIFIER::complex<float>(rhs);     \
    }

THRUST_HALF_FRIEND_OPERATOR(+, +=)
THRUST_HALF_FRIEND_OPERATOR(-, -=)
THRUST_HALF_FRIEND_OPERATOR(*, *=)
THRUST_HALF_FRIEND_OPERATOR(/, /=)

#undef THRUST_HALF_FRIEND_OPERATOR


#define THRUST_BF16_FRIEND_OPERATOR(_op, _opeq)                               \
    GKO_ATTRIBUTES GKO_INLINE GKO_THRUST_QUALIFIER::complex<gko::vendor_bf16> \
    operator _op(const GKO_THRUST_QUALIFIER::complex<gko::vendor_bf16> lhs,   \
                 const GKO_THRUST_QUALIFIER::complex<gko::vendor_bf16> rhs)   \
    {                                                                         \
        return GKO_THRUST_QUALIFIER::complex<float>{                          \
            lhs} _op GKO_THRUST_QUALIFIER::complex<float>(rhs);               \
    }

THRUST_BF16_FRIEND_OPERATOR(+, +=)
THRUST_BF16_FRIEND_OPERATOR(-, -=)
THRUST_BF16_FRIEND_OPERATOR(*, *=)
THRUST_BF16_FRIEND_OPERATOR(/, /=)

#undef THRUST_BF16_FRIEND_OPERATOR


#endif  // GKO_COMMON_CUDA_HIP_BASE_TYPES_HPP_
