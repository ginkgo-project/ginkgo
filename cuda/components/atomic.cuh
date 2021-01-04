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

#ifndef GKO_CUDA_COMPONENTS_ATOMIC_CUH_
#define GKO_CUDA_COMPONENTS_ATOMIC_CUH_


#include <type_traits>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/components/atomic.hpp.inc"


/**
 * @internal
 *
 * @note It is not 'real' complex<float> atomic add operation
 */
__forceinline__ __device__ thrust::complex<float> atomic_add(
    thrust::complex<float> *__restrict__ address, thrust::complex<float> val)
{
    cuComplex *addr = reinterpret_cast<cuComplex *>(address);
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
    thrust::complex<double> *__restrict__ address, thrust::complex<double> val)
{
    cuDoubleComplex *addr = reinterpret_cast<cuDoubleComplex *>(address);
    // Separate to real part and imag part
    auto real = atomic_add(&(addr->x), val.real());
    auto imag = atomic_add(&(addr->y), val.imag());
    return {real, imag};
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_ATOMIC_CUH_
