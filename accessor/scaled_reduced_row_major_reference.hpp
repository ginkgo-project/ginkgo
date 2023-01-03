/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_ACCESSOR_SCALED_REDUCED_ROW_MAJOR_REFERENCE_HPP_
#define GKO_ACCESSOR_SCALED_REDUCED_ROW_MAJOR_REFERENCE_HPP_


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
 * Reference class for a different storage than arithmetic type with the
 * addition of a scaling factor. The conversion between both formats is done
 * with an implicit cast to the ArithmeticType, followed by a multiplication
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
    scaled_reduced_storage(scaled_reduced_storage&&) noexcept = default;

    scaled_reduced_storage() = delete;

    ~scaled_reduced_storage() = default;

    // Forbid copy construction
    scaled_reduced_storage(const scaled_reduced_storage&) = delete;

    constexpr explicit GKO_ACC_ATTRIBUTES scaled_reduced_storage(
        storage_type* const GKO_ACC_RESTRICT ptr, arithmetic_type scalar)
        : ptr_{ptr}, scalar_{scalar}
    {}

    constexpr GKO_ACC_ATTRIBUTES operator arithmetic_type() const
    {
        const storage_type* const GKO_ACC_RESTRICT r_ptr = ptr_;
        return detail::implicit_explicit_conversion<arithmetic_type>(*r_ptr) *
               scalar_;
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(arithmetic_type val) &&
    {
        storage_type* const GKO_ACC_RESTRICT r_ptr = ptr_;
        *r_ptr = val / scalar_;
        return val;
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(const scaled_reduced_storage& ref) &&
    {
        std::move(*this) = ref.implicit_conversion();
        return *this;
    }

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type
    operator=(scaled_reduced_storage&& ref) && noexcept
    {
        std::move(*this) = ref.implicit_conversion();
        return *this;
    }

private:
    storage_type* const GKO_ACC_RESTRICT ptr_;
    const arithmetic_type scalar_;

    constexpr GKO_ACC_ATTRIBUTES arithmetic_type implicit_conversion() const
    {
        return *this;
    }
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
    scaled_reduced_storage(scaled_reduced_storage&&) noexcept = default;

    scaled_reduced_storage() = delete;

    ~scaled_reduced_storage() = default;

    // Forbid copy construction and move assignment
    scaled_reduced_storage(const scaled_reduced_storage&) = delete;

    scaled_reduced_storage& operator=(scaled_reduced_storage&&) = delete;

    constexpr explicit GKO_ACC_ATTRIBUTES scaled_reduced_storage(
        const storage_type* const GKO_ACC_RESTRICT ptr, arithmetic_type scalar)
        : ptr_{ptr}, scalar_{scalar}
    {}

    constexpr GKO_ACC_ATTRIBUTES operator arithmetic_type() const
    {
        const storage_type* const GKO_ACC_RESTRICT r_ptr = ptr_;
        return detail::implicit_explicit_conversion<arithmetic_type>(*r_ptr) *
               scalar_;
    }

private:
    const storage_type* const GKO_ACC_RESTRICT ptr_;
    const arithmetic_type scalar_;
};


template <typename ArithmeticType, typename StorageType>
constexpr remove_complex_t<ArithmeticType> abs(
    const scaled_reduced_storage<ArithmeticType, StorageType>& ref)
{
    using std::abs;
    auto implicit_cast = [](ArithmeticType val) { return val; };
    return abs(implicit_cast(ref));
}


}  // namespace reference_class
}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_SCALED_REDUCED_ROW_MAJOR_REFERENCE_HPP_
