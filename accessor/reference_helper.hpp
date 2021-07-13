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

#ifndef GKO_ACCESSOR_REFERENCE_HELPER_HPP_
#define GKO_ACCESSOR_REFERENCE_HELPER_HPP_


#include <type_traits>
#include <utility>


#include "utils.hpp"


// CUDA TOOLKIT < 11 does not support constexpr in combination with
// thrust::complex, which is why constexpr is only present in later versions
#if defined(__CUDA_ARCH__) && defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ < 11)

#define GKO_ACC_ENABLE_REFERENCE_CONSTEXPR

#else

#define GKO_ACC_ENABLE_REFERENCE_CONSTEXPR constexpr

#endif  // __CUDA_ARCH__ && __CUDACC_VER_MAJOR__ && __CUDACC_VER_MAJOR__ < 11


namespace gko {
namespace acc {
/**
 * This namespace is not part of the public interface and can change without
 * notice.
 */
namespace detail {


// tests if the cast operator to `ValueType` is present
template <typename Ref, typename ValueType, typename = xstd::void_t<>>
struct has_cast_operator : std::false_type {};

template <typename Ref, typename ValueType>
struct has_cast_operator<
    Ref, ValueType,
    xstd::void_t<decltype(std::declval<Ref>().Ref::operator ValueType())>>
    : std::true_type {};


/**
 * @internal
 * converts `ref` to ValueType while preferring the cast operator overload
 * from class `Ref` before falling back to a simple
 * `static_cast<ValueType>`.
 *
 * This function is only needed for CUDA TOOLKIT < 11 because
 * thrust::complex has a constructor call: `template<T> complex(const T
 * &other) : real(other), imag()`, which is always preferred over the
 * overloaded `operator value_type()`.
 */
template <typename ValueType, typename Ref>
constexpr GKO_ACC_ATTRIBUTES
    std::enable_if_t<has_cast_operator<Ref, ValueType>::value, ValueType>
    to_value_type(const Ref &ref)
{
    return ref.Ref::operator ValueType();
}

template <typename ValueType, typename Ref>
constexpr GKO_ACC_ATTRIBUTES
    std::enable_if_t<!has_cast_operator<Ref, ValueType>::value, ValueType>
    to_value_type(const Ref &ref)
{
    return static_cast<ValueType>(ref);
}


/**
 * This is a mixin which defines the binary operators for *, /, +, - for the
 * Reference class, the unary operator -, and the assignment operators
 * *=, /=, +=, -=
 * All assignment operators expect an rvalue reference (Reference &&) for
 * the Reference class in order to prevent copying the Reference object.
 *
 * @tparam Reference  The reference class this mixin provides operator overloads
 *                    for. The reference class needs to overload the cast
 *                    operator to ValueType
 *
 * @tparam ArithmeticType  arithmetic type the Reference class is supposed
 *         to represent.
 *
 * @warning  This struct should only be used by reference classes.
 */
template <typename Reference, typename ArithmeticType>
struct enable_reference_operators {
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;

#define GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(_op)              \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE         \
        GKO_ACC_ATTRIBUTES arithmetic_type                           \
        operator _op(const Reference &ref1, const Reference &ref2)   \
    {                                                                \
        return to_value_type<arithmetic_type>(ref1)                  \
            _op to_value_type<arithmetic_type>(ref2);                \
    }                                                                \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE         \
        GKO_ACC_ATTRIBUTES arithmetic_type                           \
        operator _op(const Reference &ref, const arithmetic_type &a) \
    {                                                                \
        return to_value_type<arithmetic_type>(ref) _op a;            \
    }                                                                \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE         \
        GKO_ACC_ATTRIBUTES arithmetic_type                           \
        operator _op(const arithmetic_type &a, const Reference &ref) \
    {                                                                \
        return a _op to_value_type<arithmetic_type>(ref);            \
    }

    GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(*)
    GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(/)
    GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(+)
    GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(-)
#undef GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD

#define GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(_oper, _op)         \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE               \
        GKO_ACC_ATTRIBUTES arithmetic_type                                 \
        _oper(Reference &&ref1, const Reference &ref2)                     \
    {                                                                      \
        return std::move(ref1) = to_value_type<arithmetic_type>(ref1)      \
                   _op to_value_type<arithmetic_type>(ref2);               \
    }                                                                      \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE               \
        GKO_ACC_ATTRIBUTES arithmetic_type                                 \
        _oper(Reference &&ref, const arithmetic_type &a)                   \
    {                                                                      \
        return std::move(ref) = to_value_type<arithmetic_type>(ref) _op a; \
    }

    GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator*=, *)
    GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator/=, /)
    GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator+=, +)
    GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator-=, -)
#undef GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD

#define GKO_ACC_REFERENCE_COMPARISON_OPERATOR_OVERLOAD(_op)          \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE         \
        GKO_ACC_ATTRIBUTES bool                                      \
        operator _op(const Reference &ref1, const Reference &ref2)   \
    {                                                                \
        return to_value_type<arithmetic_type>(ref1)                  \
            _op to_value_type<arithmetic_type>(ref2);                \
    }                                                                \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE         \
        GKO_ACC_ATTRIBUTES bool                                      \
        operator _op(const Reference &ref, const arithmetic_type &a) \
    {                                                                \
        return to_value_type<arithmetic_type>(ref) _op a;            \
    }                                                                \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE         \
        GKO_ACC_ATTRIBUTES bool                                      \
        operator _op(const arithmetic_type &a, const Reference &ref) \
    {                                                                \
        return a _op to_value_type<arithmetic_type>(ref);            \
    }

    GKO_ACC_REFERENCE_COMPARISON_OPERATOR_OVERLOAD(==)
#undef GKO_ACC_REFERENCE_COMPARISON_OPERATOR_OVERLOAD

    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE GKO_ACC_ATTRIBUTES
        arithmetic_type
        operator-(const Reference &ref)
    {
        return -to_value_type<arithmetic_type>(ref);
    }

    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE GKO_ACC_ATTRIBUTES
        arithmetic_type
        operator+(const Reference &ref)
    {
        return +to_value_type<arithmetic_type>(ref);
    }
};

// There is no more need for this macro in this file
#undef GKO_ACC_ENABLE_REFERENCE_CONSTEXPR


}  // namespace detail
}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_REFERENCE_HELPER_HPP_
