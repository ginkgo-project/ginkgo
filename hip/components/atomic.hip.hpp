// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_COMPONENTS_ATOMIC_HIP_HPP_
#define GKO_HIP_COMPONENTS_ATOMIC_HIP_HPP_


#include <type_traits>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


#include "common/cuda_hip/components/atomic.hpp.inc"


/**
 * @internal
 *
 * @note It is not 'real' complex<float> atomic add operation
 */
__forceinline__ __device__ thrust::complex<float> atomic_add(
    thrust::complex<float>* __restrict__ address, thrust::complex<float> val)
{
    hipComplex* addr = reinterpret_cast<hipComplex*>(address);
    // Separate to real part and imag part
    auto real = atomic_add(static_cast<float*>(&(addr->x)), val.real());
    auto imag = atomic_add(static_cast<float*>(&(addr->y)), val.imag());
    return {real, imag};
}


/**
 * @internal
 *
 * @note It is not 'real' complex<double> atomic add operation
 */
__forceinline__ __device__ thrust::complex<double> atomic_add(
    thrust::complex<double>* __restrict__ address, thrust::complex<double> val)
{
    hipDoubleComplex* addr = reinterpret_cast<hipDoubleComplex*>(address);
    // Separate to real part and imag part
    auto real = atomic_add(static_cast<double*>(&(addr->x)), val.real());
    auto imag = atomic_add(static_cast<double*>(&(addr->y)), val.imag());
    return {real, imag};
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_ATOMIC_HIP_HPP_
