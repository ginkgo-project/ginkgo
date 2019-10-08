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

#ifndef GKO_HIP_COMPONENTS_ATOMIC_CUH_
#define GKO_HIP_COMPONENTS_ATOMIC_CUH_


namespace gko {
namespace kernels {
namespace hip {


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

#undef GKO_BIND_ATOMIC_HELPER_STRUCTURE


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

#undef GKO_BIND_ATOMIC_ADD


/**
 * @internal
 *
 * @note It is not 'real' complex<float> atomic add opeartion
 */
__forceinline__ __device__ void atomic_add(
    thrust::complex<float> *__restrict__ address, thrust::complex<float> val)
{
    hipComplex *cuaddr = reinterpret_cast<hipComplex *>(address);
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
    hipDoubleComplex *cuaddr = reinterpret_cast<hipDoubleComplex *>(address);
    // Separate to real part and imag part
    atomic_add(&(cuaddr->x), val.real());
    atomic_add(&(cuaddr->y), val.imag());
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_ATOMIC_CUH_
