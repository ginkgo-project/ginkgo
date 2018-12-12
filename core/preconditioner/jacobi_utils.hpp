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


#include "core/base/extended_float.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"


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
 * Returns the largest precision reduction of ValueType accurate enough with
 * respect to accuracy and cond.
 *
 * The precision returned is the smallest precision with the following
 * properties:
 *
 * -   the roundoff error of the reduction is at most accuracy / cond
 * -   if verificator1 returned false, the returned reduction contains only
 *     range preserving reductions
 * -   if verificator2 returned false, the returned reduction contains at most
 *     one range non-preserving reduction
 *
 * The function optimizes the number of calls to verificators, and at most 1
 * call to each will be made.
 */
template <typename ValueType, typename AccuracyType, typename CondType,
          typename Predicate1, typename Predicate2>
GKO_ATTRIBUTES precision_reduction
get_optimal_storage_reduction(AccuracyType accuracy, CondType cond,
                              Predicate1 verificator1, Predicate2 verificator2)
{
    using gko::detail::float_traits;
    using type = remove_complex<ValueType>;
    auto accurate = [&cond, &accuracy](double eps) {
        return cond * eps < accuracy;
    };
    uint8 is_verified1 = 2;
    // the following code uses short-circuiting to avoid calling possibly
    // expensive verificatiors multiple times
    if (accurate(float_traits<truncate_type<truncate_type<type>>>::eps)) {
        return precision_reduction(2, 0);
    } else if (accurate(
                   float_traits<truncate_type<reduce_precision<type>>>::eps) &&
               (is_verified1 = verificator1())) {
        return precision_reduction(1, 1);
    } else if (accurate(float_traits<
                        reduce_precision<reduce_precision<type>>>::eps) &&
               is_verified1 != 0 && verificator2()) {
        return precision_reduction(0, 2);
    } else if (accurate(float_traits<truncate_type<type>>::eps)) {
        return precision_reduction(1, 0);
    } else if (accurate(float_traits<reduce_precision<type>>::eps) &&
               (is_verified1 == 1 ||
                (is_verified1 == 2 && (is_verified1 = verificator1())))) {
        return precision_reduction(0, 1);
    } else {
        return precision_reduction(0, 0);
    }
}

}  // namespace detail
}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_JACOBI_UTILS_HPP_
