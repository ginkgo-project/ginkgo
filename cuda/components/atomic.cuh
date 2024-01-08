// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_ATOMIC_CUH_
#define GKO_CUDA_COMPONENTS_ATOMIC_CUH_


#include <type_traits>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/cuda_hip/components/atomic.hpp.inc"


/**
 * @internal
 *
 * @note It is not 'real' complex<float> atomic add operation
 */
__forceinline__ __device__ thrust::complex<float> atomic_add(
    thrust::complex<float>* __restrict__ address, thrust::complex<float> val)
{
    cuComplex* addr = reinterpret_cast<cuComplex*>(address);
    // Separate to real part and imag part
    auto real = atomic_add(&(addr->x), val.real());
    auto imag = atomic_add(&(addr->y), val.imag());
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
    cuDoubleComplex* addr = reinterpret_cast<cuDoubleComplex*>(address);
    // Separate to real part and imag part
    auto real = atomic_add(&(addr->x), val.real());
    auto imag = atomic_add(&(addr->y), val.imag());
    return {real, imag};
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_ATOMIC_CUH_
