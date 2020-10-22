/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_DPCPP_COMPONENTS_ATOMIC_CUH_
#define GKO_DPCPP_COMPONENTS_ATOMIC_CUH_


#include <CL/sycl.hpp>
#include <type_traits>
#include "dpcpp/base/dpct.hpp"

namespace gko {
namespace kernels {
namespace dpcpp {
namespace {


template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
T atomic_compare_exchange_strong(
    cl::sycl::multi_ptr<T, cl::sycl::access::address_space::global_space> addr,
    T expected, T desired,
    cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
    cl::sycl::memory_order fail = cl::sycl::memory_order::relaxed)
{
    cl::sycl::atomic<T, addressSpace> obj(addr);
    obj.compare_exchange_strong(expected, desired, success, fail);
    return expected;
}

template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
T atomic_compare_exchange_strong(
    T *addr, T expected, T desired,
    cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
    cl::sycl::memory_order fail = cl::sycl::memory_order::relaxed)
{
    return atomic_compare_exchange_strong(
        cl::sycl::multi_ptr<T, addressSpace>(addr), expected, desired, success,
        fail);
}


template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
inline T atomic_fetch_add(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed)
{
    cl::sycl::atomic<T, addressSpace> obj(
        (cl::sycl::multi_ptr<T, addressSpace>(addr)));
    return cl::sycl::atomic_fetch_add(obj, operand, memoryOrder);
}

template <typename T>
struct fake_complex {
    T x;
    T y;
};

}  // namespace


namespace detail {


template <typename ValueType, typename = void>
struct atomic_helper {
    __dpct_inline__ static ValueType atomic_add(ValueType *, ValueType)
    {
        static_assert(sizeof(ValueType) == 0,
                      "This default function is not implemented, only the "
                      "specializations are.");
        // TODO: add proper implementation of generic atomic add
    }
};


template <typename ResultType, typename ValueType>
__dpct_inline__ ResultType reinterpret(ValueType val)
{
    static_assert(sizeof(ValueType) == sizeof(ResultType),
                  "The type to reinterpret to must be of the same size as the "
                  "original type.");
    return reinterpret_cast<ResultType &>(val);
}


#define GKO_BIND_ATOMIC_HELPER_STRUCTURE(CONVERTER_TYPE)                   \
    template <typename ValueType>                                          \
    struct atomic_helper<                                                  \
        ValueType,                                                         \
        std::enable_if_t<(sizeof(ValueType) == sizeof(CONVERTER_TYPE))>> { \
        __dpct_inline__ static ValueType atomic_add(                       \
            ValueType *__restrict__ addr, ValueType val)                   \
        {                                                                  \
            CONVERTER_TYPE *address_as_converter =                         \
                reinterpret_cast<CONVERTER_TYPE *>(addr);                  \
            CONVERTER_TYPE old = *address_as_converter;                    \
            CONVERTER_TYPE assumed;                                        \
            do {                                                           \
                assumed = old;                                             \
                old = atomic_compare_exchange_strong(                      \
                    address_as_converter, assumed,                         \
                    reinterpret<CONVERTER_TYPE>(                           \
                        val + reinterpret<ValueType>(assumed)));           \
            } while (assumed != old);                                      \
            return reinterpret<ValueType>(old);                            \
        }                                                                  \
    };

// Support 64-bit ATOMIC_ADD
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned long long int);
// Support 32-bit ATOMIC_ADD
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned int);


#undef GKO_BIND_ATOMIC_HELPER_STRUCTURE


}  // namespace detail


template <typename T>
__dpct_inline__ T atomic_add(T *__restrict__ addr, T val)
{
    return detail::atomic_helper<T>::atomic_add(addr, val);
}


#define GKO_BIND_ATOMIC_ADD(ValueType)                                 \
    __dpct_inline__ ValueType atomic_add(ValueType *__restrict__ addr, \
                                         ValueType val)                \
    {                                                                  \
        return atomic_fetch_add(addr, val);                            \
    }

GKO_BIND_ATOMIC_ADD(int);
GKO_BIND_ATOMIC_ADD(unsigned int);
GKO_BIND_ATOMIC_ADD(unsigned long long int);


#undef GKO_BIND_ATOMIC_ADD


/**
 * @internal
 *
 * @note It is not 'real' complex<float> atomic add operation
 */
__dpct_inline__ std::complex<float> atomic_add(
    std::complex<float> *__restrict__ address, std::complex<float> val)
{
    fake_complex<float> *addr =
        reinterpret_cast<fake_complex<float> *>(address);
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
__dpct_inline__ std::complex<double> atomic_add(
    std::complex<double> *__restrict__ address, std::complex<double> val)
{
    fake_complex<double> *addr =
        reinterpret_cast<fake_complex<double> *>(address);
    // Separate to real part and imag part
    auto real = atomic_add(&(addr->x), val.real());
    auto imag = atomic_add(&(addr->y), val.imag());
    return {real, imag};
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_ATOMIC_CUH_
