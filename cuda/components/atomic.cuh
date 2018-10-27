/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CUDA_COMPONENTS_ATOMIC_CUH_
#define GKO_CUDA_COMPONENTS_ATOMIC_CUH_


namespace gko {
namespace kernels {
namespace cuda {


template <typename T>
__forceinline__ __device__ void atomic_add(T *, T)
{
    GKO_ASSERT(false);
    // TODO: add proper implementation of generic atomic add
}


#define GKO_BIND_ATOMIC_ADD(ValueType)                                       \
    __forceinline__ __device__ void atomic_add(ValueType *__restrict__ addr, \
                                               ValueType val)                \
    {                                                                        \
        atomicAdd(addr, val);                                                \
    }

GKO_BIND_ATOMIC_ADD(int);
GKO_BIND_ATOMIC_ADD(unsigned int);
GKO_BIND_ATOMIC_ADD(unsigned long long int);
GKO_BIND_ATOMIC_ADD(float);


#if (defined(CUDA_VERSION) && (CUDA_VERSION < 8000)) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600))


// the function is copied from CUDA 9.2 documentation
__forceinline__ __device__ void atomic_add(double *__restrict__ addr,
                                           double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)addr;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN
        // != NaN)
    } while (assumed != old);
}


#else


GKO_BIND_ATOMIC_ADD(double);


#endif


#undef GKO_BIND_ATOMIC_ADD


/**
 * @internal
 *
 * @note It is not 'real' complex<float> atomic add opeartion
 */
__forceinline__ __device__ void atomic_add(
    thrust::complex<float> *__restrict__ address, thrust::complex<float> val)
{
    cuComplex *cuaddr = reinterpret_cast<cuComplex *>(address);
    // Separate to real part and imag part
    atomic_add(&(cuaddr->x), val.real());
    atomic_add(&(cuaddr->y), val.imag());
}

/**
 * @internal
 *
 * @note It is not 'real' complex<double> atomic add opeartion
 */
__forceinline__ __device__ void atomic_add(
    thrust::complex<double> *__restrict__ address, thrust::complex<double> val)
{
    cuDoubleComplex *cuaddr = reinterpret_cast<cuDoubleComplex *>(address);
    // Separate to real part and imag part
    atomic_add(&(cuaddr->x), val.real());
    atomic_add(&(cuaddr->y), val.imag());
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_ATOMIC_CUH_
