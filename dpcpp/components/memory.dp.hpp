// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_MEMORY_DP_HPP_
#define GKO_DPCPP_COMPONENTS_MEMORY_DP_HPP_


#include <complex>
#include <type_traits>

#include <ginkgo/core/base/math.hpp>

#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/types.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


/**
 * Loads a value from memory using an atomic operation.
 *
 * @tparam memorder  The GCC memory ordering type
 * (https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html) to use
 * for this atomic operation.
 * @tparam scope  The visibility of this operation, i.e. which threads may have
 * written to this memory location before. HIP_SCOPE_GPU means that we want to
 * observe writes from all threads on this device, HIP_SCOPE_THREADBLOCK means
 * we want to observe only writes from within the same threadblock.
 */
template <sycl::memory_order memorder, sycl::memory_scope scope,
          sycl::access::address_space space =
              sycl::access::address_space::generic_space,
          typename ValueType>
__dpct_inline__ ValueType load_generic(const ValueType* ptr)
{
    sycl::atomic_ref<ValueType, memorder, scope, space> obj(
        *const_cast<ValueType*>(ptr));
    return obj.load();
}


/**
 * Stores a value to memory using an atomic operation.
 *
 * @tparam memorder  The GCC memory ordering type
 * (https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html) to use
 * for this atomic operation.
 * @tparam scope  The visibility of this operation, i.e. which threads may
 * observe the write to this memory location. HIP_SCOPE_GPU means that we want
 * to all threads on this device to observe it, HIP_SCOPE_THREADBLOCK means we
 * want only threads within the same threadblock to observe it.
 */
template <sycl::memory_order memorder, sycl::memory_scope scope,
          sycl::access::address_space space =
              sycl::access::address_space::generic_space,
          typename ValueType>
__dpct_inline__ ValueType store_generic(ValueType* ptr, ValueType value)
{
    sycl::atomic_ref<ValueType, memorder, scope, space> obj(*ptr);
    obj.store(value);
}


// template <typename ValueType, typename AddType>
// __dpct_inline__ ValueType atomic_add_relaxed(ValueType* ptr,
//                                                         AddType value)
// {
//     return __atomic_fetch_add(ptr, value, __ATOMIC_RELAXED);
// }


// template <typename ValueType>
// __dpct_inline__ ValueType atomic_min_relaxed(ValueType* ptr,
//                                                         ValueType value)
// {
//     return __atomic_fetch_min(ptr, value, __ATOMIC_RELAXED);
// }


// template <typename ValueType>
// __dpct_inline__ ValueType atomic_max_relaxed(ValueType* ptr,
//                                                         ValueType value)
// {
//     return __atomic_fetch_max(ptr, value, __ATOMIC_RELAXED);
// }


// template <typename ValueType>
// __dpct_inline__ ValueType atomic_cas_relaxed(ValueType* ptr,
//                                                         ValueType old_val,
//                                                         ValueType new_val)
// {
//     __atomic_compare_exchange_n(ptr, &old_val, new_val, false,
//     __ATOMIC_RELAXED,
//                                 __ATOMIC_RELAXED);
//     return old_val;
// }


// template <typename ValueType>
// __dpct_inline__ ValueType atomic_cas_relaxed_local(ValueType* ptr,
//                                                               ValueType
//                                                               old_val,
//                                                               ValueType
//                                                               new_val)
// {
//     // no special optimization available for threadblock-local atomic CAS
//     return atomic_cas_relaxed(ptr, old_val, new_val);
// }


template <typename ValueType>
__dpct_inline__ ValueType load_relaxed(const ValueType* ptr)
{
    return load_generic<sycl::memory_order::relaxed,
                        sycl::memory_scope::device>(ptr);
}


template <typename ValueType>
__dpct_inline__ ValueType load_relaxed_shared(const ValueType* ptr)
{
    return load_generic<sycl::memory_order::relaxed, sycl::memory_scope::device,
                        sycl::access::address_space::generic_space>(ptr);
}


template <typename ValueType>
__dpct_inline__ ValueType load_relaxed_local(const ValueType* ptr)
{
    return load_generic<sycl::memory_order::relaxed, sycl::memory_scope::device,
                        sycl::access::address_space::generic_space>(ptr);
}


template <typename ValueType>
__dpct_inline__ ValueType load_acquire(const ValueType* ptr)
{
    load_generic<sycl::memory_order::acq_rel, sycl::memory_scope::device>(ptr);
}


template <typename ValueType>
__dpct_inline__ ValueType load_acquire_shared(const ValueType* ptr)
{
    return load_generic<sycl::memory_order::acq_rel, sycl::memory_scope::device,
                        sycl::access::address_space::generic_space>(ptr);
}


template <typename ValueType>
__dpct_inline__ ValueType load_acquire_local(const ValueType* ptr)
{
    return load_generic<sycl::memory_order::acq_rel, sycl::memory_scope::device,
                        sycl::access::address_space::generic_space>(ptr);
}


template <typename ValueType>
__dpct_inline__ void store_relaxed(ValueType* ptr, ValueType value)
{
    store_generic<sycl::memory_order::relaxed, sycl::memory_scope::device>(
        ptr, value);
}


template <typename ValueType>
__dpct_inline__ void store_relaxed_shared(ValueType* ptr, ValueType value)
{
    store_generic<sycl::memory_order::relaxed, sycl::memory_scope::device,
                  sycl::access::address_space::generic_space>(ptr, value);
}


template <typename ValueType>
__dpct_inline__ void store_relaxed_local(ValueType* ptr, ValueType value)
{
    store_generic<sycl::memory_order::relaxed, sycl::memory_scope::device,
                  sycl::access::address_space::generic_space>(ptr, value);
}


template <typename ValueType>
__dpct_inline__ void store_release(ValueType* ptr, ValueType value)
{
    store_generic<sycl::memory_order::acq_rel, sycl::memory_scope::device>(
        ptr, value);
}


template <typename ValueType>
__dpct_inline__ void store_release_shared(ValueType* ptr, ValueType value)
{
    store_generic<sycl::memory_order::acq_rel, sycl::memory_scope::device,
                  sycl::access::address_space::generic_space>(ptr, value);
}


template <typename ValueType>
__dpct_inline__ void store_release_local(ValueType* ptr, ValueType value)
{
    store_generic<sycl::memory_order::acq_rel, sycl::memory_scope::device,
                  sycl::access::address_space::generic_space>(ptr, value);
}


template <typename ValueType>
__dpct_inline__ std::complex<ValueType> load_relaxed(
    const std::complex<ValueType>* ptr)
{
    auto real_ptr = reinterpret_cast<const ValueType*>(ptr);
    auto real = load_relaxed(real_ptr);
    auto imag = load_relaxed(real_ptr + 1);
    return {real, imag};
}


template <typename ValueType>
__dpct_inline__ std::complex<ValueType> load_relaxed_shared(
    const std::complex<ValueType>* ptr)
{
    auto real_ptr = reinterpret_cast<const ValueType*>(ptr);
    auto real = load_relaxed_shared(real_ptr);
    auto imag = load_relaxed_shared(real_ptr + 1);
    return {real, imag};
}


template <typename ValueType>
__dpct_inline__ std::complex<ValueType> load_relaxed_local(
    const std::complex<ValueType>* ptr)
{
    auto real_ptr = reinterpret_cast<const ValueType*>(ptr);
    auto real = load_relaxed_local(real_ptr);
    auto imag = load_relaxed_local(real_ptr + 1);
    return {real, imag};
}


template <typename ValueType>
__dpct_inline__ void store_relaxed(std::complex<ValueType>* ptr,
                                   std::complex<ValueType> value)
{
    auto real_ptr = reinterpret_cast<ValueType*>(ptr);
    store_relaxed(real_ptr, value.real());
    store_relaxed(real_ptr + 1, value.imag());
}


template <typename ValueType>
__dpct_inline__ void store_relaxed_shared(std::complex<ValueType>* ptr,
                                          std::complex<ValueType> value)
{
    auto real_ptr = reinterpret_cast<ValueType*>(ptr);
    store_relaxed_shared(real_ptr, value.real());
    store_relaxed_shared(real_ptr + 1, value.imag());
}


template <typename ValueType>
__dpct_inline__ void store_relaxed_local(std::complex<ValueType>* ptr,
                                         std::complex<ValueType> value)
{
    auto real_ptr = reinterpret_cast<ValueType*>(ptr);
    store_relaxed_local(real_ptr, value.real());
    store_relaxed_local(real_ptr + 1, value.imag());
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko

#endif  // GKO_DPCPP_COMPONENTS_MEMORY_DP_HPP_
