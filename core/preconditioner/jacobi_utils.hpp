/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_PRECONDITIONER_JACOBI_UTILS_HPP_
#define GKO_CORE_PRECONDITIONER_JACOBI_UTILS_HPP_


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include "core/base/extended_float.hpp"


#define GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(_type, _prec, ...) \
    if (_prec == ::gko::precision_reduction(0, 1)) {                   \
        using resolved_precision = ::gko::reduce_precision<_type>;     \
        __VA_ARGS__;                                                   \
    } else if (_prec == ::gko::precision_reduction(0, 2)) {            \
        using resolved_precision =                                     \
            ::gko::reduce_precision<::gko::reduce_precision<_type>>;   \
        __VA_ARGS__;                                                   \
    } else if (_prec == ::gko::precision_reduction(1, 0)) {            \
        using resolved_precision = ::gko::truncate_type<_type>;        \
        __VA_ARGS__;                                                   \
    } else if (_prec == ::gko::precision_reduction(1, 1)) {            \
        using resolved_precision =                                     \
            ::gko::truncate_type<::gko::reduce_precision<_type>>;      \
        __VA_ARGS__;                                                   \
    } else if (_prec == ::gko::precision_reduction(2, 0)) {            \
        using resolved_precision =                                     \
            ::gko::truncate_type<::gko::truncate_type<_type>>;         \
        __VA_ARGS__;                                                   \
    } else {                                                           \
        using resolved_precision = _type;                              \
        __VA_ARGS__;                                                   \
    }


namespace gko {
namespace preconditioner {
namespace detail {


/**
 * @internal
 *
 * A descriptor encoding multiple available precision_reduction values.
 */
struct precision_reduction_descriptor {
    enum : uint32 {
        p0n0 = 0x00,  // precision_reduction(0, 0)
        p0n2 = 0x01,  // precision_reduction(0, 2)
        p1n1 = 0x02,  // precision_reduction(1, 1)
        p2n0 = 0x04,  // precision_reduction(2, 0)
        p0n1 = 0x08,  // precision_reduction(0, 1)
        p1n0 = 0x10,  // precision_reduction(1, 0)
    };

    static constexpr GKO_ATTRIBUTES uint32
    singleton(const precision_reduction &pr)
    {
        return pr == precision_reduction(0, 0)
                   ? p0n0
                   : pr == precision_reduction(0, 1)
                         ? p0n1
                         : pr == precision_reduction(0, 2)
                               ? p0n2
                               : pr == precision_reduction(1, 0)
                                     ? p1n0
                                     : pr == precision_reduction(1, 1)
                                           ? p1n1
                                           : pr == precision_reduction(2, 0)
                                                 ? p2n0
                                                 : p0n0;
    }
};


/**
 * @internal
 *
 * Returns an encoded list of precision reductions of ValueType accurate enough
 * with respect to accuracy and cond.
 *
 * The precision reduction is present in the list if and only if the following
 * is true:
 *
 * -   the roundoff error of the reduction is at most accuracy / cond
 * -   verificator1 returned true or the returned reduction contains only
 *     range preserving reductions
 * -   verificator2 returned true or the returned reduction contains at most
 *     one range non-preserving reduction
 *
 * The function optimizes the number of calls to verificators, and at most 1
 * call to each one will be made.
 *
 * @note The returned list is encoded as a bit vector, so bitwise operations can
 *       be used to manipulate it (`~x` returns all reductions not in `x`,
 *       `x & y` all reduction that are both in `x` and `y`, `x | y` all
 *       reductions that are at least in one of `x` and `y`, `x ^ y` all
 *       reduction that are in exactly one of `x` and `y`, etc.)
 * @note The "best" reduction in the set can be obtained by using
 *       get_optimal_storage_reduction()
 */
template <typename ValueType, typename AccuracyType, typename CondType,
          typename Predicate1, typename Predicate2>
GKO_ATTRIBUTES GKO_INLINE uint32 get_supported_storage_reductions(
    AccuracyType accuracy, CondType cond, Predicate1 verificator1,
    Predicate2 verificator2)
{
    using gko::detail::float_traits;
    using type = remove_complex<ValueType>;
    using prd = precision_reduction_descriptor;
    auto accurate = [&cond, &accuracy](double eps) {
        return cond * eps < accuracy;
    };
    uint8 is_verified1 = 2;
    auto supported = static_cast<uint32>(prd::p0n0);
    // the following code uses short-circuiting to avoid calling possibly
    // expensive verificatiors multiple times
    if (accurate(float_traits<truncate_type<truncate_type<type>>>::eps)) {
        supported |= prd::p2n0;
    }
    if (accurate(float_traits<truncate_type<reduce_precision<type>>>::eps) &&
        (is_verified1 = verificator1())) {
        supported |= prd::p1n1;
    }
    if (accurate(float_traits<reduce_precision<reduce_precision<type>>>::eps) &&
        is_verified1 != 0 && verificator2()) {
        supported |= prd::p0n2;
    }
    if (accurate(float_traits<truncate_type<type>>::eps)) {
        supported |= prd::p1n0;
    }
    if (accurate(float_traits<reduce_precision<type>>::eps) &&
        (is_verified1 == 1 ||
         (is_verified1 == 2 && (is_verified1 = verificator1())))) {
        supported |= prd::p0n1;
    }
    return supported;
}


/**
 * @internal
 *
 * Returns the largest precision reduction in a list of available reductions.
 *
 * If there are multiple precision reductions of the same size in the set, the
 * one with the most range non-preserving transformations (i.e. the one with the
 * largest amount of significand bits) will be used.
 *
 * @param supported  encoded list of available precisions reduction
 *
 * @return the largest precision reduction among the reductions encoded in
 *         `supported`
 */
GKO_ATTRIBUTES GKO_INLINE precision_reduction
get_optimal_storage_reduction(uint32 supported)
{
    using prd = precision_reduction_descriptor;
    if (supported & prd::p0n2) {
        return precision_reduction(0, 2);
    } else if (supported & prd::p1n1) {
        return precision_reduction(1, 1);
    } else if (supported & prd::p2n0) {
        return precision_reduction(2, 0);
    } else if (supported & prd::p0n1) {
        return precision_reduction(0, 1);
    } else if (supported & prd::p1n0) {
        return precision_reduction(1, 0);
    } else {
        return precision_reduction(0, 0);
    }
}


}  // namespace detail
}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_JACOBI_UTILS_HPP_
