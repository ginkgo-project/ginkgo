// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_UTILS_HPP_
#define GKO_ACCESSOR_UTILS_HPP_

#include <cassert>
#include <cinttypes>
#include <complex>


#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/complex.h>
#endif


#if defined(__CUDACC__) || defined(__HIPCC__)
#define GKO_ACC_ATTRIBUTES __host__ __device__
#define GKO_ACC_INLINE __forceinline__
#define GKO_ACC_RESTRICT __restrict__
#else
#define GKO_ACC_ATTRIBUTES
#define GKO_ACC_INLINE inline
#define GKO_ACC_RESTRICT
#endif  // defined(__CUDACC__) || defined(__HIPCC__)


#if (defined(__CUDA_ARCH__) && defined(__APPLE__)) || \
    defined(__HIP_DEVICE_COMPILE__)

#ifdef NDEBUG
#define GKO_ACC_ASSERT(condition) ((void)0)
#else  // NDEBUG
// Poor man's assertions on GPUs for MACs. They won't terminate the program
// but will at least print something on the screen
#define GKO_ACC_ASSERT(condition)                                           \
    ((condition)                                                            \
         ? ((void)0)                                                        \
         : ((void)printf("%s: %d: %s: Assertion `" #condition "' failed\n", \
                         __FILE__, __LINE__, __func__)))
#endif  // NDEBUG

#else  // (defined(__CUDA_ARCH__) && defined(__APPLE__)) ||
       // defined(__HIP_DEVICE_COMPILE__)

// Handle assertions normally on other systems
#define GKO_ACC_ASSERT(condition) assert(condition)

#endif  // (defined(__CUDA_ARCH__) && defined(__APPLE__)) ||
        // defined(__HIP_DEVICE_COMPILE__)


namespace gko {
namespace acc {
namespace xstd {


template <typename...>
using void_t = void;


}  // namespace xstd


using size_type = std::int64_t;


namespace detail {


template <typename T>
struct remove_complex_impl {
    using type = T;
};


template <typename T>
struct remove_complex_impl<std::complex<T>> {
    using type = T;
};


#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct remove_complex_impl<thrust::complex<T>> {
    using type = T;
};
#endif


template <typename T>
struct is_complex_impl {
    static constexpr bool value{false};
};


template <typename T>
struct is_complex_impl<std::complex<T>> {
    static constexpr bool value{true};
};


#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct is_complex_impl<thrust::complex<T>> {
    static constexpr bool value{true};
};
#endif


}  // namespace detail


template <typename T>
using remove_complex_t = typename detail::remove_complex_impl<T>::type;


template <typename T>
using is_complex = typename detail::is_complex_impl<T>;


/**
 * Evaluates if all template arguments Args fulfill std::is_integral. If that is
 * the case, this class inherits from `std::true_type`, otherwise, it inherits
 * from `std::false_type`.
 * If no values are passed in, `std::true_type` is inherited from.
 *
 * @tparam Args...  Arguments to test for std::is_integral
 */
template <typename... Args>
struct are_all_integral : public std::true_type {};

template <typename First, typename... Args>
struct are_all_integral<First, Args...>
    : public std::conditional_t<std::is_integral<std::decay_t<First>>::value,
                                are_all_integral<Args...>, std::false_type> {};


namespace detail {


/**
 * @internal
 * Tests if a member function `Ref::to_arithmetic_type` exists
 */
template <typename Ref, typename Dummy = xstd::void_t<>>
struct has_to_arithmetic_type : std::false_type {
    static_assert(std::is_same<Dummy, void>::value,
                  "Do not modify the Dummy value!");
    using type = Ref;
};

template <typename Ref>
struct has_to_arithmetic_type<
    Ref, xstd::void_t<decltype(std::declval<Ref>().to_arithmetic_type())>>
    : std::true_type {
    using type = decltype(std::declval<Ref>().to_arithmetic_type());
};


/**
 * @internal
 * Tests if the type `Ref::arithmetic_type` exists
 */
template <typename Ref, typename Dummy = xstd::void_t<>>
struct has_arithmetic_type : std::false_type {
    static_assert(std::is_same<Dummy, void>::value,
                  "Do not modify the Dummy value!");
};

template <typename Ref>
struct has_arithmetic_type<Ref, xstd::void_t<typename Ref::arithmetic_type>>
    : std::true_type {};


/**
 * @internal
 * converts `ref` to an arithmetic type. It performs the following three steps:
 * 1. If a function `to_arithmetic_type()` is available, it will return the
 *    result of that function
 * 2. Otherwise, if the type `Ref::arithmetic_type` exists, it will return the
 *    implicit cast from Ref -> `Ref::arithmetic_type`
 * 3. Otherwise, it will return `ref` itself.
 */
template <typename Ref>
constexpr GKO_ACC_ATTRIBUTES
    std::enable_if_t<has_to_arithmetic_type<Ref>::value,
                     typename has_to_arithmetic_type<Ref>::type>
    to_arithmetic_type(const Ref& ref)
{
    return ref.to_arithmetic_type();
}

template <typename Ref>
constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<
    !has_to_arithmetic_type<Ref>::value && has_arithmetic_type<Ref>::value,
    typename Ref::arithmetic_type>
to_arithmetic_type(const Ref& ref)
{
    return ref;
}

template <typename Ref>
constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<
    !has_to_arithmetic_type<Ref>::value && !has_arithmetic_type<Ref>::value,
    Ref>
to_arithmetic_type(const Ref& ref)
{
    return ref;
}


/**
 * @internal
 * Struct used for testing if an implicit cast is present. The constructor only
 * takes an OutType, so any argument of a type that is not implicitly
 * convertible to OutType is incompatible.
 */
template <typename OutType>
struct test_for_implicit_cast {
    constexpr GKO_ACC_ATTRIBUTES test_for_implicit_cast(const OutType&) {}
};


/**
 * @internal
 * Checks if an implicit cast is defined from InType to OutType.
 */
template <typename OutType, typename InType, typename Dummy = void>
struct has_implicit_cast {
    static_assert(std::is_same<Dummy, void>::value,
                  "Don't touch the Dummy type!");
    static constexpr bool value = false;
};

template <typename OutType, typename InType>
struct has_implicit_cast<OutType, InType,
                         xstd::void_t<decltype(test_for_implicit_cast<OutType>(
                             std::declval<InType>()))>> {
    static constexpr bool value = true;
};


/**
 * Converts in from InType to OutType with an implicit cast if it is defined,
 * otherwise, a `static_cast` is used.
 */
template <typename OutType, typename InType>
constexpr GKO_ACC_ATTRIBUTES
    std::enable_if_t<has_implicit_cast<OutType, InType>::value, OutType>
    implicit_explicit_conversion(const InType& in)
{
    return in;
}

template <typename OutType, typename InType>
constexpr GKO_ACC_ATTRIBUTES
    std::enable_if_t<!has_implicit_cast<OutType, InType>::value, OutType>
    implicit_explicit_conversion(const InType& in)
{
    return static_cast<OutType>(in);
}


}  // namespace detail
}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_UTILS_HPP_
