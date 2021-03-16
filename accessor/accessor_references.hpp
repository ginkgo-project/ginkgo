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

#ifndef GKO_ACCESSOR_ACCESSOR_REFERENCES_HPP_
#define GKO_ACCESSOR_ACCESSOR_REFERENCES_HPP_


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
};

// There is no more need for this macro in this file
#undef GKO_ACC_ENABLE_REFERENCE_CONSTEXPR


}  // namespace detail


/**
 * This namespace contains reference classes used inside accessors.
 *
 * @warning These classes should only be used by accessors.
 */
namespace reference_class {


/**
 * Reference class for a different storage than arithmetic type. The
 * conversion between both formats is done with a simple static_cast.
 *
 * Copying this reference is disabled, but move construction is possible to
 * allow for an additional layer (like gko::acc::range).
 * The assignment operator only works for an rvalue reference (&&) to
 * prevent accidentally copying the reference and working on a reference.
 *
 * @tparam ArithmeticType  Type used for arithmetic operations, therefore,
 *                         the type which is used for input and output of this
 *                         class.
 *
 * @tparam StorageType  Type actually used as a storage, which is converted
 *                      to ArithmeticType before usage
 */
template <typename ArithmeticType, typename StorageType>
class reduced_storage
    : public detail::enable_reference_operators<
          reduced_storage<ArithmeticType, StorageType>, ArithmeticType> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;

    // Allow move construction, so perfect forwarding is possible (required
    // for `range` support)
    reduced_storage(reduced_storage &&) = default;

    reduced_storage() = delete;

    ~reduced_storage() = default;

    // Forbid copy construction
    reduced_storage(const reduced_storage &) = delete;

    constexpr explicit GKO_ACC_ATTRIBUTES reduced_storage(
        storage_type *const GKO_ACC_RESTRICT ptr)
        : ptr_{ptr}
    {}

    constexpr GKO_ACC_ATTRIBUTES operator arithmetic_type() const
    {
        const storage_type *const GKO_ACC_RESTRICT r_ptr = ptr_;
        return static_cast<arithmetic_type>(*r_ptr);
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(arithmetic_type val) &&noexcept
    {
        storage_type *const GKO_ACC_RESTRICT r_ptr = ptr_;
        *r_ptr = static_cast<storage_type>(val);
        return val;
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(const reduced_storage &ref) &&
    {
        std::move(*this) = static_cast<arithmetic_type>(ref);
        return static_cast<arithmetic_type>(*this);
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(reduced_storage &&ref) &&noexcept
    {
        std::move(*this) = static_cast<arithmetic_type>(ref);
        return static_cast<arithmetic_type>(*this);
    }

private:
    storage_type *const GKO_ACC_RESTRICT ptr_;
};

// Specialization for const storage_type to prevent `operator=`
template <typename ArithmeticType, typename StorageType>
class reduced_storage<ArithmeticType, const StorageType>
    : public detail::enable_reference_operators<
          reduced_storage<ArithmeticType, const StorageType>, ArithmeticType> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = const StorageType;

    // Allow move construction, so perfect forwarding is possible
    reduced_storage(reduced_storage &&) = default;

    reduced_storage() = delete;

    ~reduced_storage() = default;

    // Forbid copy construction and move assignment
    reduced_storage(const reduced_storage &) = delete;

    reduced_storage &operator=(reduced_storage &&) = delete;

    constexpr explicit GKO_ACC_ATTRIBUTES reduced_storage(
        storage_type *const GKO_ACC_RESTRICT ptr)
        : ptr_{ptr}
    {}

    constexpr GKO_ACC_ATTRIBUTES operator arithmetic_type() const
    {
        const storage_type *const GKO_ACC_RESTRICT r_ptr = ptr_;
        return static_cast<arithmetic_type>(*r_ptr);
    }

private:
    storage_type *const GKO_ACC_RESTRICT ptr_;
};


template <typename ArithmeticType, typename StorageType>
constexpr remove_complex_t<ArithmeticType> abs(
    const reduced_storage<ArithmeticType, StorageType> &ref)
{
    using std::abs;
    return abs(static_cast<ArithmeticType>(ref));
}


/**
 * Reference class for a different storage than arithmetic type with the
 * addition of a scaling factor. The conversion between both formats is done
 * with a static_cast to the ArithmeticType, followed by a multiplication
 * of the scalar (when reading; for writing, the new value is divided by the
 * scalar before casting to the StorageType).
 *
 * Copying this reference is disabled, but move construction is possible to
 * allow for an additional layer (like gko::acc::range).
 * The assignment operator only works for an rvalue reference (&&) to
 * prevent accidentally copying and working on the reference.
 *
 * @tparam ArithmeticType  Type used for arithmetic operations, therefore,
 *                         the type which is used for input and output of this
 *                         class.
 *
 * @tparam StorageType  Type actually used as a storage, which is converted
 *                      to ArithmeticType before usage
 */
template <typename ArithmeticType, typename StorageType>
class scaled_reduced_storage
    : public detail::enable_reference_operators<
          scaled_reduced_storage<ArithmeticType, StorageType>, ArithmeticType> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;

    // Allow move construction, so perfect forwarding is possible
    scaled_reduced_storage(scaled_reduced_storage &&) = default;

    scaled_reduced_storage() = delete;

    ~scaled_reduced_storage() = default;

    // Forbid copy construction
    scaled_reduced_storage(const scaled_reduced_storage &) = delete;

    constexpr explicit GKO_ACC_ATTRIBUTES scaled_reduced_storage(
        storage_type *const GKO_ACC_RESTRICT ptr, arithmetic_type scalar)
        : ptr_{ptr}, scalar_{scalar}
    {}

    constexpr GKO_ACC_ATTRIBUTES operator arithmetic_type() const
    {
        const storage_type *const GKO_ACC_RESTRICT r_ptr = ptr_;
        return static_cast<arithmetic_type>(*r_ptr) * scalar_;
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(arithmetic_type val) &&noexcept
    {
        storage_type *const GKO_ACC_RESTRICT r_ptr = ptr_;
        *r_ptr = static_cast<storage_type>(val / scalar_);
        return val;
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(const scaled_reduced_storage &ref) &&
    {
        std::move(*this) = static_cast<arithmetic_type>(ref);
        return static_cast<arithmetic_type>(*this);
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(scaled_reduced_storage &&ref) &&noexcept
    {
        std::move(*this) = static_cast<arithmetic_type>(ref);
        return static_cast<arithmetic_type>(*this);
    }

private:
    storage_type *const GKO_ACC_RESTRICT ptr_;
    const arithmetic_type scalar_;
};

// Specialization for constant storage_type (no `operator=`)
template <typename ArithmeticType, typename StorageType>
class scaled_reduced_storage<ArithmeticType, const StorageType>
    : public detail::enable_reference_operators<
          scaled_reduced_storage<ArithmeticType, const StorageType>,
          ArithmeticType> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = const StorageType;

    // Allow move construction, so perfect forwarding is possible
    scaled_reduced_storage(scaled_reduced_storage &&) = default;

    scaled_reduced_storage() = delete;

    ~scaled_reduced_storage() = default;

    // Forbid copy construction and move assignment
    scaled_reduced_storage(const scaled_reduced_storage &) = delete;

    scaled_reduced_storage &operator=(scaled_reduced_storage &&) = delete;

    constexpr explicit GKO_ACC_ATTRIBUTES scaled_reduced_storage(
        storage_type *const GKO_ACC_RESTRICT ptr, arithmetic_type scalar)
        : ptr_{ptr}, scalar_{scalar}
    {}

    constexpr GKO_ACC_ATTRIBUTES operator arithmetic_type() const
    {
        const storage_type *const GKO_ACC_RESTRICT r_ptr = ptr_;
        return static_cast<arithmetic_type>(*r_ptr) * scalar_;
    }

private:
    storage_type *const GKO_ACC_RESTRICT ptr_;
    const arithmetic_type scalar_;
};


template <typename ArithmeticType, typename StorageType>
constexpr remove_complex_t<ArithmeticType> abs(
    const scaled_reduced_storage<ArithmeticType, StorageType> &ref)
{
    using std::abs;
    return abs(static_cast<ArithmeticType>(ref));
}


}  // namespace reference_class
}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_ACCESSOR_REFERENCES_HPP_
