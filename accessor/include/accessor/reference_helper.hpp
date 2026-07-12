// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ACCESSOR_REFERENCE_HELPER_HPP_
#define ACCESSOR_REFERENCE_HELPER_HPP_


#include <type_traits>
#include <utility>

#include "accessor/utils.hpp"


// NVC++ disallows a constexpr function with a nonliteral return type like half.
// A consumer that may instantiate the accessor with such a type under NVC++
// defines ACC_NONLITERAL_ARITHMETIC_TYPE to drop constexpr from the reference
// operators.
#if defined(__NVCOMPILER) && defined(ACC_NONLITERAL_ARITHMETIC_TYPE)

#define MACC_ENABLE_REFERENCE_CONSTEXPR

#else

#define MACC_ENABLE_REFERENCE_CONSTEXPR constexpr

#endif


namespace acc {
/**
 * This namespace is not part of the public interface and can change without
 * notice.
 */
namespace detail {


/**
 * This is a mixin which defines the binary operators for *, /, +, - for the
 * Reference class, the unary operator -, and the assignment operators
 * *=, /=, +=, -=
 * All assignment operators expect an rvalue reference (Reference &&) for
 * the Reference class in order to prevent copying the Reference object.
 *
 * @tparam Reference  The reference class this mixin provides operator overloads
 *                    for. The reference class must overload the cast
 *                    operator to ArithmeticType
 *
 * @tparam ArithmeticType  arithmetic type the Reference class is supposed
 *         to represent.
 *
 * @warning  This struct should only be used by reference classes.
 */
template <typename Reference, typename ArithmeticType>
struct enable_reference_operators {
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;

    /**
     * @internal
     * This function calls the cast operator to arithmetic_type of *this.
     * To achieve that, it needs to cast *this to a Reference object because
     * the cast operation must be defined there (this is a requirement for this
     * Mixin).
     * This function is also used to detect if a proxy object is used or not.
     */
    constexpr MACC_ATTRIBUTES MACC_INLINE arithmetic_type
    to_arithmetic_type() const
    {
        return *static_cast<const Reference*>(this);
    }

#define MACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(_op)                    \
    friend MACC_ENABLE_REFERENCE_CONSTEXPR MACC_INLINE MACC_ATTRIBUTES  \
        arithmetic_type                                                 \
        operator _op(const Reference& ref1, const Reference& ref2)      \
    {                                                                   \
        return ref1.to_arithmetic_type() _op ref2.to_arithmetic_type(); \
    }                                                                   \
    friend MACC_ENABLE_REFERENCE_CONSTEXPR MACC_INLINE MACC_ATTRIBUTES  \
        arithmetic_type                                                 \
        operator _op(const Reference& ref, const arithmetic_type& a)    \
    {                                                                   \
        return ref.to_arithmetic_type() _op a;                          \
    }                                                                   \
    friend MACC_ENABLE_REFERENCE_CONSTEXPR MACC_INLINE MACC_ATTRIBUTES  \
        arithmetic_type                                                 \
        operator _op(const arithmetic_type& a, const Reference& ref)    \
    {                                                                   \
        return a _op ref.to_arithmetic_type();                          \
    }

    MACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(*)
    MACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(/)
    MACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(+)
    MACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(-)
#undef MACC_REFERENCE_BINARY_OPERATOR_OVERLOAD

#define MACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(_oper, _op)             \
    friend MACC_ENABLE_REFERENCE_CONSTEXPR MACC_INLINE MACC_ATTRIBUTES      \
        arithmetic_type                                                     \
        _oper(Reference&& ref1, const Reference& ref2)                      \
    {                                                                       \
        return std::move(ref1) =                                            \
                   ref1.to_arithmetic_type() _op ref2.to_arithmetic_type(); \
    }                                                                       \
    friend MACC_ENABLE_REFERENCE_CONSTEXPR MACC_INLINE MACC_ATTRIBUTES      \
        arithmetic_type                                                     \
        _oper(Reference&& ref, const arithmetic_type& a)                    \
    {                                                                       \
        return std::move(ref) = ref.to_arithmetic_type() _op a;             \
    }

    MACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator*=, *)
    MACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator/=, /)
    MACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator+=, +)
    MACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator-=, -)
#undef MACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD

#define MACC_REFERENCE_COMPARISON_OPERATOR_OVERLOAD(_op)                    \
    friend MACC_ENABLE_REFERENCE_CONSTEXPR MACC_INLINE MACC_ATTRIBUTES bool \
    operator _op(const Reference& ref1, const Reference& ref2)              \
    {                                                                       \
        return ref1.to_arithmetic_type() _op ref2.to_arithmetic_type();     \
    }                                                                       \
    friend MACC_ENABLE_REFERENCE_CONSTEXPR MACC_INLINE MACC_ATTRIBUTES bool \
    operator _op(const Reference& ref, const arithmetic_type& a)            \
    {                                                                       \
        return ref.to_arithmetic_type() _op a;                              \
    }                                                                       \
    friend MACC_ENABLE_REFERENCE_CONSTEXPR MACC_INLINE MACC_ATTRIBUTES bool \
    operator _op(const arithmetic_type& a, const Reference& ref)            \
    {                                                                       \
        return a _op ref.to_arithmetic_type();                              \
    }

    MACC_REFERENCE_COMPARISON_OPERATOR_OVERLOAD(==)
#undef MACC_REFERENCE_COMPARISON_OPERATOR_OVERLOAD

    friend MACC_ENABLE_REFERENCE_CONSTEXPR MACC_INLINE MACC_ATTRIBUTES
        arithmetic_type
        operator-(const Reference& ref)
    {
        return -ref.to_arithmetic_type();
    }

    friend MACC_ENABLE_REFERENCE_CONSTEXPR MACC_INLINE MACC_ATTRIBUTES
        arithmetic_type
        operator+(const Reference& ref)
    {
        return +ref.to_arithmetic_type();
    }
};

// There is no more need for this macro in this file
#undef MACC_ENABLE_REFERENCE_CONSTEXPR


}  // namespace detail
}  // namespace acc


#endif  // ACCESSOR_REFERENCE_HELPER_HPP_
