// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_ATOMIC_DP_HPP_
#define GKO_DPCPP_COMPONENTS_ATOMIC_DP_HPP_


#include <type_traits>


#include <CL/sycl.hpp>


#include "dpcpp/base/dpct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace atomic {


constexpr auto local_space = sycl::access::address_space::local_space;
constexpr auto global_space = sycl::access::address_space::global_space;


}  // namespace atomic

namespace {


template <sycl::access::address_space addressSpace = atomic::global_space,
          typename T>
T atomic_compare_exchange_strong(
    sycl::multi_ptr<T, addressSpace> addr, T expected, T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed)
{
    sycl::atomic<T, addressSpace> obj(addr);
    obj.compare_exchange_strong(expected, desired, success, fail);
    return expected;
}

template <sycl::access::address_space addressSpace = atomic::global_space,
          typename T>
T atomic_compare_exchange_strong(
    T* addr, T expected, T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed)
{
    return atomic_compare_exchange_strong(
        sycl::multi_ptr<T, addressSpace>(addr), expected, desired, success,
        fail);
}


template <sycl::access::address_space addressSpace = atomic::global_space,
          typename T>
inline T atomic_fetch_add(
    T* addr, T operand,
    sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
{
    sycl::atomic<T, addressSpace> obj((sycl::multi_ptr<T, addressSpace>(addr)));
    return sycl::atomic_fetch_add(obj, operand, memoryOrder);
}


template <sycl::access::address_space addressSpace = atomic::global_space,
          typename T>
inline T atomic_fetch_max(
    T* addr, T operand,
    sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
{
    sycl::atomic<T, addressSpace> obj((sycl::multi_ptr<T, addressSpace>(addr)));
    return sycl::atomic_fetch_max(obj, operand, memoryOrder);
}


}  // namespace


namespace detail {


template <sycl::access::address_space addressSpace, typename ValueType,
          typename = void>
struct atomic_helper {
    __dpct_inline__ static ValueType atomic_add(ValueType*, ValueType)
    {
        static_assert(sizeof(ValueType) == 0,
                      "This default function is not implemented, only the "
                      "specializations are.");
        // TODO: add proper implementation of generic atomic add
    }
};


template <sycl::access::address_space addressSpace, typename ValueType,
          typename = void>
struct atomic_max_helper {
    __dpct_inline__ static ValueType atomic_max(ValueType*, ValueType)
    {
        static_assert(sizeof(ValueType) == 0,
                      "This default function is not implemented, only the "
                      "specializations are.");
        // TODO: add proper implementation of generic atomic max
    }
};


template <typename ResultType, typename ValueType>
__dpct_inline__ ResultType reinterpret(ValueType val)
{
    static_assert(sizeof(ValueType) == sizeof(ResultType),
                  "The type to reinterpret to must be of the same size as the "
                  "original type.");
    return reinterpret_cast<ResultType&>(val);
}


#define GKO_BIND_ATOMIC_HELPER_STRUCTURE(CONVERTER_TYPE)                    \
    template <sycl::access::address_space addressSpace, typename ValueType> \
    struct atomic_helper<                                                   \
        addressSpace, ValueType,                                            \
        std::enable_if_t<(sizeof(ValueType) == sizeof(CONVERTER_TYPE))>> {  \
        __dpct_inline__ static ValueType atomic_add(                        \
            ValueType* __restrict__ addr, ValueType val)                    \
        {                                                                   \
            CONVERTER_TYPE* address_as_converter =                          \
                reinterpret_cast<CONVERTER_TYPE*>(addr);                    \
            CONVERTER_TYPE old = *address_as_converter;                     \
            CONVERTER_TYPE assumed;                                         \
            do {                                                            \
                assumed = old;                                              \
                old = atomic_compare_exchange_strong<addressSpace>(         \
                    address_as_converter, assumed,                          \
                    reinterpret<CONVERTER_TYPE>(                            \
                        val + reinterpret<ValueType>(assumed)));            \
            } while (assumed != old);                                       \
            return reinterpret<ValueType>(old);                             \
        }                                                                   \
    };

// Support 64-bit ATOMIC_ADD
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned long long int);
// Support 32-bit ATOMIC_ADD
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned int);


#undef GKO_BIND_ATOMIC_HELPER_STRUCTURE

#define GKO_BIND_ATOMIC_HELPER_VALUETYPE(ValueType)                         \
    template <sycl::access::address_space addressSpace>                     \
    struct atomic_helper<addressSpace, ValueType, std::enable_if_t<true>> { \
        __dpct_inline__ static ValueType atomic_add(                        \
            ValueType* __restrict__ addr, ValueType val)                    \
        {                                                                   \
            return atomic_fetch_add<addressSpace>(addr, val);               \
        }                                                                   \
    };

GKO_BIND_ATOMIC_HELPER_VALUETYPE(int);
GKO_BIND_ATOMIC_HELPER_VALUETYPE(unsigned int);
GKO_BIND_ATOMIC_HELPER_VALUETYPE(unsigned long long int);

#undef GKO_BIND_ATOMIC_HELPER_VALUETYPE


template <sycl::access::address_space addressSpace, typename ValueType>
struct atomic_helper<
    addressSpace, ValueType,
    std::enable_if_t<is_complex<ValueType>() && sizeof(ValueType) >= 16>> {
    __dpct_inline__ static ValueType atomic_add(ValueType* __restrict__ addr,
                                                ValueType val)
    {
        using real_type = remove_complex<ValueType>;
        real_type* real_addr = reinterpret_cast<real_type*>(addr);
        // Separate to real part and imag part
        auto real = atomic_helper<addressSpace, real_type>::atomic_add(
            &real_addr[0], val.real());
        auto imag = atomic_helper<addressSpace, real_type>::atomic_add(
            &real_addr[1], val.imag());
        return {real, imag};
    }
};


#define GKO_BIND_ATOMIC_MAX_STRUCTURE(CONVERTER_TYPE)                       \
    template <sycl::access::address_space addressSpace, typename ValueType> \
    struct atomic_max_helper<                                               \
        addressSpace, ValueType,                                            \
        std::enable_if_t<(sizeof(ValueType) == sizeof(CONVERTER_TYPE))>> {  \
        __dpct_inline__ static ValueType atomic_max(                        \
            ValueType* __restrict__ addr, ValueType val)                    \
        {                                                                   \
            CONVERTER_TYPE* address_as_converter =                          \
                reinterpret_cast<CONVERTER_TYPE*>(addr);                    \
            CONVERTER_TYPE old = *address_as_converter;                     \
            CONVERTER_TYPE assumed;                                         \
            do {                                                            \
                assumed = old;                                              \
                if (reinterpret<ValueType>(assumed) < val) {                \
                    old = atomic_compare_exchange_strong<addressSpace>(     \
                        address_as_converter, assumed,                      \
                        reinterpret<CONVERTER_TYPE>(val));                  \
                }                                                           \
            } while (assumed != old);                                       \
            return reinterpret<ValueType>(old);                             \
        }                                                                   \
    };

// Support 64-bit ATOMIC_ADD
GKO_BIND_ATOMIC_MAX_STRUCTURE(unsigned long long int);
// Support 32-bit ATOMIC_ADD
GKO_BIND_ATOMIC_MAX_STRUCTURE(unsigned int);


#undef GKO_BIND_ATOMIC_MAX_STRUCTURE

#define GKO_BIND_ATOMIC_MAX_VALUETYPE(ValueType)              \
    template <sycl::access::address_space addressSpace>       \
    struct atomic_max_helper<addressSpace, ValueType,         \
                             std::enable_if_t<true>> {        \
        __dpct_inline__ static ValueType atomic_max(          \
            ValueType* __restrict__ addr, ValueType val)      \
        {                                                     \
            return atomic_fetch_max<addressSpace>(addr, val); \
        }                                                     \
    };

GKO_BIND_ATOMIC_MAX_VALUETYPE(int);
GKO_BIND_ATOMIC_MAX_VALUETYPE(unsigned int);
GKO_BIND_ATOMIC_MAX_VALUETYPE(unsigned long long int);

#undef GKO_BIND_ATOMIC_MAX_VALUETYPE


}  // namespace detail


template <sycl::access::address_space addressSpace = atomic::global_space,
          typename T>
__dpct_inline__ T atomic_add(T* __restrict__ addr, T val)
{
    return detail::atomic_helper<addressSpace, T>::atomic_add(addr, val);
}


template <sycl::access::address_space addressSpace = atomic::global_space,
          typename T>
__dpct_inline__ T atomic_max(T* __restrict__ addr, T val)
{
    return detail::atomic_max_helper<addressSpace, T>::atomic_max(addr, val);
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_ATOMIC_DP_HPP_
