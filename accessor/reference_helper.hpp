// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE arithmetic_type
    to_arithmetic_type() const
    {
        return *static_cast<const Reference*>(this);
    }

#define GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(_op)                 \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE            \
        GKO_ACC_ATTRIBUTES arithmetic_type                              \
        operator _op(const Reference& ref1, const Reference& ref2)      \
    {                                                                   \
        return ref1.to_arithmetic_type() _op ref2.to_arithmetic_type(); \
    }                                                                   \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE            \
        GKO_ACC_ATTRIBUTES arithmetic_type                              \
        operator _op(const Reference& ref, const arithmetic_type& a)    \
    {                                                                   \
        return ref.to_arithmetic_type() _op a;                          \
    }                                                                   \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE            \
        GKO_ACC_ATTRIBUTES arithmetic_type                              \
        operator _op(const arithmetic_type& a, const Reference& ref)    \
    {                                                                   \
        return a _op ref.to_arithmetic_type();                          \
    }

    GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(*)
    GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(/)
    GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(+)
    GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD(-)
#undef GKO_ACC_REFERENCE_BINARY_OPERATOR_OVERLOAD

#define GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(_oper, _op)          \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE                \
        GKO_ACC_ATTRIBUTES arithmetic_type                                  \
        _oper(Reference&& ref1, const Reference& ref2)                      \
    {                                                                       \
        return std::move(ref1) =                                            \
                   ref1.to_arithmetic_type() _op ref2.to_arithmetic_type(); \
    }                                                                       \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE                \
        GKO_ACC_ATTRIBUTES arithmetic_type                                  \
        _oper(Reference&& ref, const arithmetic_type& a)                    \
    {                                                                       \
        return std::move(ref) = ref.to_arithmetic_type() _op a;             \
    }

    GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator*=, *)
    GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator/=, /)
    GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator+=, +)
    GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator-=, -)
#undef GKO_ACC_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD

#define GKO_ACC_REFERENCE_COMPARISON_OPERATOR_OVERLOAD(_op)             \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE            \
        GKO_ACC_ATTRIBUTES bool                                         \
        operator _op(const Reference& ref1, const Reference& ref2)      \
    {                                                                   \
        return ref1.to_arithmetic_type() _op ref2.to_arithmetic_type(); \
    }                                                                   \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE            \
        GKO_ACC_ATTRIBUTES bool                                         \
        operator _op(const Reference& ref, const arithmetic_type& a)    \
    {                                                                   \
        return ref.to_arithmetic_type() _op a;                          \
    }                                                                   \
    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE            \
        GKO_ACC_ATTRIBUTES bool                                         \
        operator _op(const arithmetic_type& a, const Reference& ref)    \
    {                                                                   \
        return a _op ref.to_arithmetic_type();                          \
    }

    GKO_ACC_REFERENCE_COMPARISON_OPERATOR_OVERLOAD(==)
#undef GKO_ACC_REFERENCE_COMPARISON_OPERATOR_OVERLOAD

    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE GKO_ACC_ATTRIBUTES
        arithmetic_type
        operator-(const Reference& ref)
    {
        return -ref.to_arithmetic_type();
    }

    friend GKO_ACC_ENABLE_REFERENCE_CONSTEXPR GKO_ACC_INLINE GKO_ACC_ATTRIBUTES
        arithmetic_type
        operator+(const Reference& ref)
    {
        return +ref.to_arithmetic_type();
    }
};

// There is no more need for this macro in this file
#undef GKO_ACC_ENABLE_REFERENCE_CONSTEXPR


}  // namespace detail
}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_REFERENCE_HELPER_HPP_
