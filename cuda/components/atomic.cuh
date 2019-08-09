/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


namespace gko {
namespace kernels {
namespace cuda {


namespace detail {


template <typename ValueType, typename = void>
struct atomic_helper {
    __forceinline__ __device__ static void atomic_add(ValueType *, ValueType)
    {
        static_assert(sizeof(ValueType) == 0,
                      "This default function is not implemented, only the "
                      "specializations are.");
        // TODO: add proper implementation of generic atomic add
    }
};


template <typename ResultType, typename ValueType>
__forceinline__ __device__ ResultType reinterpret(ValueType val)
{
    static_assert(sizeof(ValueType) == sizeof(ResultType),
                  "The type to reinterpret to must be of the same size as the "
                  "original type.");
    return reinterpret_cast<ResultType &>(val);
}


#define GKO_BIND_ATOMIC_HELPER_STRUCTURE(CONVERTER_TYPE)                     \
    template <typename ValueType>                                            \
    struct atomic_helper<ValueType,                                          \
                         gko::xstd::enable_if_t<(sizeof(ValueType) ==        \
                                                 sizeof(CONVERTER_TYPE))>> { \
        __forceinline__ __device__ static void atomic_add(                   \
            ValueType *__restrict__ addr, ValueType val)                     \
        {                                                                    \
            CONVERTER_TYPE *address_as_ull =                                 \
                reinterpret_cast<CONVERTER_TYPE *>(addr);                    \
            CONVERTER_TYPE old = *address_as_ull;                            \
            CONVERTER_TYPE assumed;                                          \
            do {                                                             \
                assumed = old;                                               \
                old = atomicCAS(address_as_ull, assumed,                     \
                                reinterpret<CONVERTER_TYPE>(                 \
                                    val + reinterpret<ValueType>(assumed))); \
            } while (assumed != old);                                        \
        }                                                                    \
    };

// Support 64-bit ATOMIC_ADD
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned long long int);
// Support 32-bit ATOMIC_ADD
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned int);


#if !(defined(CUDA_VERSION) && (CUDA_VERSION < 10100))
// CUDA 10.1 starts supporting 16-bit unsigned short int atomicCAS
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned short int);
#endif

#undef GKO_BIND_ATOMIC_HELPER__STRUCTURE


}  // namespace detail


template <typename T>
__forceinline__ __device__ void atomic_add(T *__restrict__ addr, T val)
{
    detail::atomic_helper<T>::atomic_add(addr, val);
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


#if !((defined(CUDA_VERSION) && (CUDA_VERSION < 8000)) || \
      (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)))
// CUDA 8.0 starts suppoting 64-bit double atomicAdd on devices of compute
// capability 6.x and higher
GKO_BIND_ATOMIC_ADD(double);
#endif

#if !((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) || \
      (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
// CUDA 10.0 starts supporting 16-bit __half floating-point atomicAdd on devices
// of compute capability 7.x and higher.
GKO_BIND_ATOMIC_ADD(__half);
#endif

#if !((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) || \
      (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)))
// CUDA 10.0 starts supporting 32-bit __half2 floating-point atomicAdd on
// devices of compute capability 6.x and higher. note: The atomicity of the
// __half2 add operation is guaranteed separately for each of the two __half
// elements; the entire __half2 is not guaranteed to be atomic as a single
// 32-bit access.
GKO_BIND_ATOMIC_ADD(__half2);
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
