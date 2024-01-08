// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_MATH_HPP_
#define GKO_PUBLIC_CORE_BASE_MATH_HPP_


#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


// HIP should not see std::abs or std::sqrt, we want the custom implementation.
// Hence, provide the using declaration only for some cases
namespace kernels {
namespace reference {


using std::abs;


using std::sqrt;


}  // namespace reference
}  // namespace kernels


namespace kernels {
namespace omp {


using std::abs;


using std::sqrt;


}  // namespace omp
}  // namespace kernels


namespace kernels {
namespace cuda {


using std::abs;


using std::sqrt;


}  // namespace cuda
}  // namespace kernels


namespace kernels {
namespace dpcpp {


using std::abs;


using std::sqrt;


}  // namespace dpcpp
}  // namespace kernels


namespace test {


using std::abs;


using std::sqrt;


}  // namespace test


// type manipulations


/**
 * @internal
 * @brief The detail namespace.
 * @ingroup detail
 */
namespace detail {


/**
 * Keep the same data type if it is not complex.
 */
template <typename T>
struct remove_complex_impl {
    using type = T;
};

/**
 * Use the underlying real type if it is complex type.
 */
template <typename T>
struct remove_complex_impl<std::complex<T>> {
    using type = T;
};


/**
 * Use the complex type if it is not complex.
 *
 * @tparam T  the type being made complex
 */
template <typename T>
struct to_complex_impl {
    using type = std::complex<T>;
};

/**
 * Use the same type if it is complex type.
 *
 * @tparam T  the type being made complex
 */
template <typename T>
struct to_complex_impl<std::complex<T>> {
    using type = std::complex<T>;
};


template <typename T>
struct is_complex_impl : public std::integral_constant<bool, false> {};

template <typename T>
struct is_complex_impl<std::complex<T>>
    : public std::integral_constant<bool, true> {};


template <typename T>
struct is_complex_or_scalar_impl : std::is_scalar<T> {};

template <typename T>
struct is_complex_or_scalar_impl<std::complex<T>> : std::is_scalar<T> {};


/**
 * template_converter is converting the template parameters of a class by
 * converter<type>.
 *
 * @tparam  converter<type> which convert one type to another type
 * @tparam  T  type
 */
template <template <typename> class converter, typename T>
struct template_converter {};

/**
 * template_converter is converting the template parameters of a class by
 * converter<type>. Converting class<T1, T2, ...> to class<converter<T1>,
 * converter<T2>, converter<...>>.
 *
 * @tparam  converter<type> which convert one type to another type
 * @tparam  template <...> T  class template base
 * @tparam  ...Rest  the template parameter of T
 */
template <template <typename> class converter, template <typename...> class T,
          typename... Rest>
struct template_converter<converter, T<Rest...>> {
    using type = T<typename converter<Rest>::type...>;
};


template <typename T, typename = void>
struct remove_complex_s {};

/**
 * Obtains a real counterpart of a std::complex type, and leaves the type
 * unchanged if it is not a complex type for complex/scalar type.
 *
 * @tparam T  complex or scalar type
 */
template <typename T>
struct remove_complex_s<T,
                        std::enable_if_t<is_complex_or_scalar_impl<T>::value>> {
    using type = typename detail::remove_complex_impl<T>::type;
};

/**
 * Obtains a real counterpart of a class with template parameters, which
 * converts complex parameters to real parameters.
 *
 * @tparam T  class with template parameters
 */
template <typename T>
struct remove_complex_s<
    T, std::enable_if_t<!is_complex_or_scalar_impl<T>::value>> {
    using type =
        typename detail::template_converter<detail::remove_complex_impl,
                                            T>::type;
};


template <typename T, typename = void>
struct to_complex_s {};

/**
 * Obtains a complex counterpart of a real type, and leaves the type
 * unchanged if it is a complex type for complex/scalar type.
 *
 * @tparam T  complex or scalar type
 */
template <typename T>
struct to_complex_s<T, std::enable_if_t<is_complex_or_scalar_impl<T>::value>> {
    using type = typename detail::to_complex_impl<T>::type;
};

/**
 * Obtains a complex counterpart of a class with template parameters, which
 * converts real parameters to complex parameters.
 *
 * @tparam T  class with template parameters
 */
template <typename T>
struct to_complex_s<T, std::enable_if_t<!is_complex_or_scalar_impl<T>::value>> {
    using type =
        typename detail::template_converter<detail::to_complex_impl, T>::type;
};


}  // namespace detail


/**
 * Access the underlying real type of a complex number.
 *
 * @tparam T  the type being checked.
 */
template <typename T>
struct cpx_real_type {
    /** The type. When the type is not complex, return the type itself.*/
    using type = T;
};

/**
 * Specialization for complex types.
 *
 * @copydoc cpx_real_type
 */
template <typename T>
struct cpx_real_type<std::complex<T>> {
    /** The type. When the type is complex, return the underlying value_type.*/
    using type = typename std::complex<T>::value_type;
};


/**
 * Allows to check if T is a complex value during compile time by accessing the
 * `value` attribute of this struct.
 * If `value` is `true`, T is a complex type, if it is `false`, T is not a
 * complex type.
 *
 * @tparam T  type to check
 */
template <typename T>
using is_complex_s = detail::is_complex_impl<T>;

/**
 * Checks if T is a complex type.
 *
 * @tparam T  type to check
 *
 * @return `true` if T is a complex type, `false` otherwise
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr bool is_complex()
{
    return detail::is_complex_impl<T>::value;
}


/**
 * Allows to check if T is a complex or scalar value during compile time by
 * accessing the `value` attribute of this struct. If `value` is `true`, T is a
 * complex/scalar type, if it is `false`, T is not a complex/scalar type.
 *
 * @tparam T  type to check
 */
template <typename T>
using is_complex_or_scalar_s = detail::is_complex_or_scalar_impl<T>;

/**
 * Checks if T is a complex/scalar type.
 *
 * @tparam T  type to check
 *
 * @return `true` if T is a complex/scalar type, `false` otherwise
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr bool is_complex_or_scalar()
{
    return detail::is_complex_or_scalar_impl<T>::value;
}


/**
 * Obtain the type which removed the complex of complex/scalar type or the
 * template parameter of class by accessing the `type` attribute of this struct.
 *
 * @tparam T  type to remove complex
 *
 * @note remove_complex<class> can not be used in friend class declaration.
 */
template <typename T>
using remove_complex = typename detail::remove_complex_s<T>::type;


/**
 * Obtain the type which adds the complex of complex/scalar type or the
 * template parameter of class by accessing the `type` attribute of this struct.
 *
 * @tparam T  type to complex_type
 *
 * @note to_complex<class> can not be used in friend class declaration.
 *       the followings are the error message from different combination.
 *       friend to_complex<Csr>;
 *         error: can not recognize it is class correctly.
 *       friend class to_complex<Csr>;
 *         error: using alias template specialization
 *       friend class to_complex_s<Csr<ValueType,IndexType>>::type;
 *         error: can not recognize it is class correctly.
 */
template <typename T>
using to_complex = typename detail::to_complex_s<T>::type;


/**
 * to_real is alias of remove_complex
 *
 * @tparam T  type to real
 */
template <typename T>
using to_real = remove_complex<T>;


namespace detail {


// singly linked list of all our supported precisions
template <typename T>
struct next_precision_impl {};

template <>
struct next_precision_impl<float> {
    using type = double;
};

template <>
struct next_precision_impl<double> {
    using type = float;
};

template <typename T>
struct next_precision_impl<std::complex<T>> {
    using type = std::complex<typename next_precision_impl<T>::type>;
};


template <typename T>
struct reduce_precision_impl {
    using type = T;
};

template <typename T>
struct reduce_precision_impl<std::complex<T>> {
    using type = std::complex<typename reduce_precision_impl<T>::type>;
};

template <>
struct reduce_precision_impl<double> {
    using type = float;
};

template <>
struct reduce_precision_impl<float> {
    using type = half;
};


template <typename T>
struct increase_precision_impl {
    using type = T;
};

template <typename T>
struct increase_precision_impl<std::complex<T>> {
    using type = std::complex<typename increase_precision_impl<T>::type>;
};

template <>
struct increase_precision_impl<float> {
    using type = double;
};

template <>
struct increase_precision_impl<half> {
    using type = float;
};


template <typename T>
struct infinity_impl {
    // CUDA doesn't allow us to call std::numeric_limits functions
    // so we need to store the value instead.
    static constexpr auto value = std::numeric_limits<T>::infinity();
};


/**
 * Computes the highest-precision type from a list of types
 */
template <typename T1, typename T2>
struct highest_precision_impl {
    using type = decltype(T1{} + T2{});
};

template <typename T1, typename T2>
struct highest_precision_impl<std::complex<T1>, std::complex<T2>> {
    using type = std::complex<typename highest_precision_impl<T1, T2>::type>;
};

template <typename Head, typename... Tail>
struct highest_precision_variadic {
    using type = typename highest_precision_impl<
        Head, typename highest_precision_variadic<Tail...>::type>::type;
};

template <typename Head>
struct highest_precision_variadic<Head> {
    using type = Head;
};


}  // namespace detail


/**
 * Obtains the next type in the singly-linked precision list.
 */
template <typename T>
using next_precision = typename detail::next_precision_impl<T>::type;


/**
 * Obtains the previous type in the singly-linked precision list.
 *
 * @note Currently our lists contains only two elements, so this is the same as
 *       next_precision.
 */
template <typename T>
using previous_precision = next_precision<T>;


/**
 * Obtains the next type in the hierarchy with lower precision than T.
 */
template <typename T>
using reduce_precision = typename detail::reduce_precision_impl<T>::type;


/**
 * Obtains the next type in the hierarchy with higher precision than T.
 */
template <typename T>
using increase_precision = typename detail::increase_precision_impl<T>::type;


/**
 * Obtains the smallest arithmetic type that is able to store elements of all
 * template parameter types exactly. All template type parameters need to be
 * either real or complex types, mixing them is not possible.
 *
 * Formally, it computes a right-fold over the type list, with the highest
 * precision of a pair of real arithmetic types T1, T2 computed as
 * `decltype(T1{} + T2{})`, or
 * `std::complex<highest_precision<remove_complex<T1>, remove_complex<T2>>>` for
 * complex types.
 */
template <typename... Ts>
using highest_precision =
    typename detail::highest_precision_variadic<Ts...>::type;


/**
 * Reduces the precision of the input parameter.
 *
 * @tparam T  the original precision
 *
 * @param val  the value to round down
 *
 * @return the rounded down value
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr reduce_precision<T> round_down(T val)
{
    return static_cast<reduce_precision<T>>(val);
}


/**
 * Increases the precision of the input parameter.
 *
 * @tparam T  the original precision
 *
 * @param val  the value to round up
 *
 * @return the rounded up value
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr increase_precision<T> round_up(T val)
{
    return static_cast<increase_precision<T>>(val);
}


template <typename FloatType, size_type NumComponents, size_type ComponentId>
class truncated;


namespace detail {


template <typename T>
struct truncate_type_impl {
    using type = truncated<T, 2, 0>;
};

template <typename T, size_type Components>
struct truncate_type_impl<truncated<T, Components, 0>> {
    using type = truncated<T, 2 * Components, 0>;
};

template <typename T>
struct truncate_type_impl<std::complex<T>> {
    using type = std::complex<typename truncate_type_impl<T>::type>;
};


template <typename T>
struct type_size_impl {
    static constexpr auto value = sizeof(T) * byte_size;
};

template <typename T>
struct type_size_impl<std::complex<T>> {
    static constexpr auto value = sizeof(T) * byte_size;
};


}  // namespace detail


/**
 * Truncates the type by half (by dropping bits), but ensures that it is at
 * least `Limit` bits wide.
 */
template <typename T, size_type Limit = sizeof(uint16) * byte_size>
using truncate_type =
    std::conditional_t<detail::type_size_impl<T>::value >= 2 * Limit,
                       typename detail::truncate_type_impl<T>::type, T>;


/**
 * Used to convert objects of type `S` to objects of type `R` using static_cast.
 *
 * @tparam S  source type
 * @tparam R  result type
 */
template <typename S, typename R>
struct default_converter {
    /**
     * Converts the object to result type.
     *
     * @param val  the object to convert
     * @return the converted object
     */
    GKO_ATTRIBUTES R operator()(S val) { return static_cast<R>(val); }
};


// mathematical functions


/**
 * Performs integer division with rounding up.
 *
 * @param num  numerator
 * @param den  denominator
 *
 * @return returns the ceiled quotient.
 */
GKO_INLINE GKO_ATTRIBUTES constexpr int64 ceildiv(int64 num, int64 den)
{
    return (num + den - 1) / den;
}


#if defined(__HIPCC__) && GINKGO_HIP_PLATFORM_HCC


/**
 * Returns the additive identity for T.
 *
 * @return additive identity for T
 */
template <typename T>
GKO_INLINE __host__ constexpr T zero()
{
    return T{};
}


/**
 * Returns the additive identity for T.
 *
 * @return additive identity for T
 *
 * @note This version takes an unused reference argument to avoid
 *       complicated calls like `zero<decltype(x)>()`. Instead, it allows
 *       `zero(x)`.
 */
template <typename T>
GKO_INLINE __host__ constexpr T zero(const T&)
{
    return zero<T>();
}


/**
 * Returns the multiplicative identity for T.
 *
 * @return the multiplicative identity for T
 */
template <typename T>
GKO_INLINE __host__ constexpr T one()
{
    return T(1);
}


/**
 * Returns the multiplicative identity for T.
 *
 * @return the multiplicative identity for T
 *
 * @note This version takes an unused reference argument to avoid
 *       complicated calls like `one<decltype(x)>()`. Instead, it allows
 *       `one(x)`.
 */
template <typename T>
GKO_INLINE __host__ constexpr T one(const T&)
{
    return one<T>();
}


/**
 * Returns the additive identity for T.
 *
 * @return additive identity for T
 */
template <typename T>
GKO_INLINE __device__ constexpr std::enable_if_t<
    !std::is_same<T, std::complex<remove_complex<T>>>::value, T>
zero()
{
    return T{};
}


/**
 * Returns the additive identity for T.
 *
 * @return additive identity for T
 *
 * @note This version takes an unused reference argument to avoid
 *       complicated calls like `zero<decltype(x)>()`. Instead, it allows
 *       `zero(x)`.
 */
template <typename T>
GKO_INLINE __device__ constexpr T zero(const T&)
{
    return zero<T>();
}


/**
 * Returns the multiplicative identity for T.
 *
 * @return the multiplicative identity for T
 */
template <typename T>
GKO_INLINE __device__ constexpr std::enable_if_t<
    !std::is_same<T, std::complex<remove_complex<T>>>::value, T>
one()
{
    return T(1);
}


/**
 * Returns the multiplicative identity for T.
 *
 * @return the multiplicative identity for T
 *
 * @note This version takes an unused reference argument to avoid
 *       complicated calls like `one<decltype(x)>()`. Instead, it allows
 *       `one(x)`.
 */
template <typename T>
GKO_INLINE __device__ constexpr T one(const T&)
{
    return one<T>();
}


#else


/**
 * Returns the additive identity for T.
 *
 * @return additive identity for T
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T zero()
{
    return T{};
}


/**
 * Returns the additive identity for T.
 *
 * @return additive identity for T
 *
 * @note This version takes an unused reference argument to avoid
 *       complicated calls like `zero<decltype(x)>()`. Instead, it allows
 *       `zero(x)`.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T zero(const T&)
{
    return zero<T>();
}


/**
 * Returns the multiplicative identity for T.
 *
 * @return the multiplicative identity for T
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T one()
{
    return T(1);
}


/**
 * Returns the multiplicative identity for T.
 *
 * @return the multiplicative identity for T
 *
 * @note This version takes an unused reference argument to avoid
 *       complicated calls like `one<decltype(x)>()`. Instead, it allows
 *       `one(x)`.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T one(const T&)
{
    return one<T>();
}


#endif  // defined(__HIPCC__) && GINKGO_HIP_PLATFORM_HCC


#undef GKO_BIND_ZERO_ONE


/**
 * Returns true if and only if the given value is zero.
 *
 * @tparam T  the type of the value
 *
 * @param value  the given value
 * @return true iff the given value is zero, i.e. `value == zero<T>()`
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr bool is_zero(T value)
{
    return value == zero<T>();
}


/**
 * Returns true if and only if the given value is not zero.
 *
 * @tparam T  the type of the value
 *
 * @param value  the given value
 * @return true iff the given value is not zero, i.e. `value != zero<T>()`
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr bool is_nonzero(T value)
{
    return value != zero<T>();
}


/**
 * Returns the larger of the arguments.
 *
 * @tparam T  type of the arguments
 *
 * @param x  first argument
 * @param y  second argument
 *
 * @return x >= y ? x : y
 *
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T max(const T& x, const T& y)
{
    return x >= y ? x : y;
}


/**
 * Returns the smaller of the arguments.
 *
 * @tparam T  type of the arguments
 *
 * @param x  first argument
 * @param y  second argument
 *
 * @return x <= y ? x : y
 *
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T min(const T& x, const T& y)
{
    return x <= y ? x : y;
}


namespace detail {


/**
 * @internal
 * Tests if a member function `Ref::to_arithmetic_type` exists
 * This is used for the accessor references to convert to the arithmetic type in
 * the math-functions
 *
 * @note
 * This basically mirrors the accessor functionality
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
 * converts `ref` to arithmetic type. This is used for accessor references to
 * convert the type before performing math operations.
 * It performs the following three steps in order:
 * 1. If a function `to_arithmetic_type()` is available, it will return the
 *    result of that function
 * 2. Otherwise, if the type `Ref::arithmetic_type` exists, it will return the
 *    implicit cast from Ref -> `Ref::arithmetic_type`
 * 3. Otherwise, it will return `ref` itself.
 */
template <typename Ref>
constexpr GKO_ATTRIBUTES
    std::enable_if_t<has_to_arithmetic_type<Ref>::value,
                     typename has_to_arithmetic_type<Ref>::type>
    to_arithmetic_type(const Ref& ref)
{
    return ref.to_arithmetic_type();
}

template <typename Ref>
constexpr GKO_ATTRIBUTES std::enable_if_t<!has_to_arithmetic_type<Ref>::value &&
                                              has_arithmetic_type<Ref>::value,
                                          typename Ref::arithmetic_type>
to_arithmetic_type(const Ref& ref)
{
    return ref;
}

template <typename Ref>
constexpr GKO_ATTRIBUTES std::enable_if_t<!has_to_arithmetic_type<Ref>::value &&
                                              !has_arithmetic_type<Ref>::value,
                                          Ref>
to_arithmetic_type(const Ref& ref)
{
    return ref;
}


// Note: All functions have postfix `impl` so they are not considered for
// overload resolution (in case a class / function also is in the namespace
// `detail`)
template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<!is_complex_s<T>::value, T>
real_impl(const T& x)
{
    return x;
}

template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<is_complex_s<T>::value,
                                                     remove_complex<T>>
real_impl(const T& x)
{
    return x.real();
}


template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<!is_complex_s<T>::value, T>
imag_impl(const T&)
{
    return T{};
}

template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<is_complex_s<T>::value,
                                                     remove_complex<T>>
imag_impl(const T& x)
{
    return x.imag();
}


template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<!is_complex_s<T>::value, T>
conj_impl(const T& x)
{
    return x;
}

template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<is_complex_s<T>::value, T>
conj_impl(const T& x)
{
    return T{real_impl(x), -imag_impl(x)};
}


}  // namespace detail


/**
 * Returns the real part of the object.
 *
 * @tparam T  type of the object
 *
 * @param x  the object
 *
 * @return real part of the object (by default, the object itself)
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr auto real(const T& x)
{
    return detail::real_impl(detail::to_arithmetic_type(x));
}


/**
 * Returns the imaginary part of the object.
 *
 * @tparam T  type of the object
 *
 * @param x  the object
 *
 * @return imaginary part of the object (by default, zero<T>())
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr auto imag(const T& x)
{
    return detail::imag_impl(detail::to_arithmetic_type(x));
}


/**
 * Returns the conjugate of an object.
 *
 * @param x  the number to conjugate
 *
 * @return  conjugate of the object (by default, the object itself)
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr auto conj(const T& x)
{
    return detail::conj_impl(detail::to_arithmetic_type(x));
}


/**
 * Returns the squared norm of the object.
 *
 * @tparam T type of the object.
 *
 * @return  The squared norm of the object.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr auto squared_norm(const T& x)
    -> decltype(real(conj(x) * x))
{
    return real(conj(x) * x);
}


/**
 * Returns the absolute value of the object.
 *
 * @tparam T  the type of the object
 *
 * @param x  the object
 *
 * @return x >= zero<T>() ? x : -x;
 */
template <typename T>
GKO_INLINE
    GKO_ATTRIBUTES constexpr xstd::enable_if_t<!is_complex_s<T>::value, T>
    abs(const T& x)
{
    return x >= zero<T>() ? x : -x;
}


template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr xstd::enable_if_t<is_complex_s<T>::value,
                                                      remove_complex<T>>
abs(const T& x)
{
    return sqrt(squared_norm(x));
}


/**
 * Returns the value of pi.
 *
 * @tparam T  the value type to return
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T pi()
{
    return static_cast<T>(3.1415926535897932384626433);
}


/**
 * Returns the value of exp(2 * pi * i * k / n), i.e. an nth root of unity.
 *
 * @param n  the denominator of the argument
 * @param k  the numerator of the argument. Defaults to 1.
 *
 * @tparam T  the corresponding real value type.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr std::complex<remove_complex<T>> unit_root(
    int64 n, int64 k = 1)
{
    return std::polar(one<remove_complex<T>>(),
                      remove_complex<T>{2} * pi<remove_complex<T>>() * k / n);
}


/**
 * Returns the position of the most significant bit of the number.
 *
 * This is the same as the rounded down base-2 logarithm of the number.
 *
 * @tparam T  a numeric type supporting bit shift and comparison
 *
 * @param n  a number
 * @param hint  a lower bound for the position o the significant bit
 *
 * @return maximum of `hint` and the significant bit position of `n`
 */
template <typename T>
constexpr uint32 get_significant_bit(const T& n, uint32 hint = 0u) noexcept
{
    return (T{1} << (hint + 1)) > n ? hint : get_significant_bit(n, hint + 1u);
}


/**
 * Returns the smallest power of `base` not smaller than `limit`.
 *
 * @tparam T  a numeric type supporting multiplication and comparison
 *
 * @param base  the base of the power to be returned
 * @param limit  the lower limit on the size of the power returned
 * @param hint  a lower bound on the result, has to be a power of base
 *
 * @return the smallest power of `base` not smaller than `limit`
 */
template <typename T>
constexpr T get_superior_power(const T& base, const T& limit,
                               const T& hint = T{1}) noexcept
{
    return hint >= limit ? hint : get_superior_power(base, limit, hint * base);
}


/**
 * Checks if a floating point number is finite, meaning it is
 * neither +/- infinity nor NaN.
 *
 * @tparam T  type of the value to check
 *
 * @param value  value to check
 *
 * @return `true` if the value is finite, meaning it are neither
 *         +/- infinity nor NaN.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES std::enable_if_t<!is_complex_s<T>::value, bool>
is_finite(const T& value)
{
    constexpr T infinity{detail::infinity_impl<T>::value};
    return abs(value) < infinity;
}


/**
 * Checks if all components of a complex value are finite, meaning they are
 * neither +/- infinity nor NaN.
 *
 * @tparam T  complex type of the value to check
 *
 * @param value  complex value to check
 *
 * @return `true` if both components of the given value are finite, meaning
 *         they are neither +/- infinity nor NaN.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES std::enable_if_t<is_complex_s<T>::value, bool>
is_finite(const T& value)
{
    return is_finite(value.real()) && is_finite(value.imag());
}


/**
 * Computes the quotient of the given parameters, guarding against division by
 * zero.
 *
 * @tparam T  value type of the parameters
 *
 * @param a  the dividend
 * @param b  the divisor
 *
 * @return the value of `a / b` if b is non-zero, zero otherwise.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES T safe_divide(T a, T b)
{
    return b == zero<T>() ? zero<T>() : a / b;
}


/**
 * Checks if a floating point number is NaN.
 *
 * @tparam T  type of the value to check
 *
 * @param value  value to check
 *
 * @return `true` if the value is NaN.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES std::enable_if_t<!is_complex_s<T>::value, bool>
is_nan(const T& value)
{
    return std::isnan(value);
}


/**
 * Checks if any component of a complex value is NaN.
 *
 * @tparam T  complex type of the value to check
 *
 * @param value  complex value to check
 *
 * @return `true` if any component of the given value is NaN.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES std::enable_if_t<is_complex_s<T>::value, bool> is_nan(
    const T& value)
{
    return std::isnan(value.real()) || std::isnan(value.imag());
}


/**
 * Returns a quiet NaN of the given type.
 *
 * @tparam T  the type of the object
 *
 * @return NaN.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr std::enable_if_t<!is_complex_s<T>::value, T>
nan()
{
    return std::numeric_limits<T>::quiet_NaN();
}


/**
 * Returns a complex with both components quiet NaN.
 *
 * @tparam T  the type of the object
 *
 * @return complex{NaN, NaN}.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr std::enable_if_t<is_complex_s<T>::value, T>
nan()
{
    return T{nan<remove_complex<T>>(), nan<remove_complex<T>>()};
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_MATH_HPP_
