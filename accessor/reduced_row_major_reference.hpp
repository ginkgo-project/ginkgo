// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_REDUCED_ROW_MAJOR_REFERENCE_HPP_
#define GKO_ACCESSOR_REDUCED_ROW_MAJOR_REFERENCE_HPP_


#include <cmath>
#include <type_traits>


#include "math.hpp"
#include "reference_helper.hpp"
#include "utils.hpp"


namespace gko {
namespace acc {
/**
 * This namespace contains reference classes used inside accessors.
 *
 * @warning These classes should only be used by accessors.
 */
namespace reference_class {


/**
 * Reference class for a different storage than arithmetic type. The
 * conversion between both formats is done with an implicit cast.
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
    reduced_storage(reduced_storage&&) noexcept = default;

    reduced_storage() = delete;

    ~reduced_storage() = default;

    // Forbid copy construction
    reduced_storage(const reduced_storage&) = delete;

    constexpr explicit GKO_ACC_ATTRIBUTES reduced_storage(
        storage_type* const GKO_ACC_RESTRICT ptr)
        : ptr_{ptr}
    {}

    constexpr GKO_ACC_ATTRIBUTES operator arithmetic_type() const
    {
        const storage_type* const GKO_ACC_RESTRICT r_ptr = ptr_;
        return detail::implicit_explicit_conversion<arithmetic_type>(*r_ptr);
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(arithmetic_type val) &&
    {
        storage_type* const GKO_ACC_RESTRICT r_ptr = ptr_;
        *r_ptr = val;
        return val;
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(const reduced_storage& ref) &&
    {
        std::move(*this) = ref.implicit_conversion();
        return *this;
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(reduced_storage&& ref) && noexcept
    {
        std::move(*this) = ref.implicit_conversion();
        return *this;
    }

private:
    storage_type* const GKO_ACC_RESTRICT ptr_;

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type implicit_conversion() const
    {
        return *this;
    }
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
    reduced_storage(reduced_storage&&) noexcept = default;

    reduced_storage() = delete;

    ~reduced_storage() = default;

    // Forbid copy construction and move assignment
    reduced_storage(const reduced_storage&) = delete;

    reduced_storage& operator=(reduced_storage&&) = delete;

    constexpr explicit GKO_ACC_ATTRIBUTES reduced_storage(
        const storage_type* const GKO_ACC_RESTRICT ptr)
        : ptr_{ptr}
    {}

    constexpr GKO_ACC_ATTRIBUTES operator arithmetic_type() const
    {
        const storage_type* const GKO_ACC_RESTRICT r_ptr = ptr_;
        return detail::implicit_explicit_conversion<arithmetic_type>(*r_ptr);
    }

private:
    const storage_type* const GKO_ACC_RESTRICT ptr_;
};


template <typename ArithmeticType, typename StorageType>
constexpr GKO_ACC_ATTRIBUTES remove_complex_t<ArithmeticType> abs(
    const reduced_storage<ArithmeticType, StorageType>& ref)
{
    using std::abs;
    auto implicit_cast = [](ArithmeticType val) { return val; };
    return abs(implicit_cast(ref));
}


}  // namespace reference_class
}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_REDUCED_ROW_MAJOR_REFERENCE_HPP_
