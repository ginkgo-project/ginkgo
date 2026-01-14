// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_AMP_HELPERS_H
#define GKO_CORE_MATRIX_AMP_HELPERS_H

#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/amp_utils.hpp"
#include "ginkgo/core/base/amp_types.hpp"


namespace gko {
namespace amp {


/**
 * A fixed-size array holding an item for each supported precision starting at
 * the precision of the template parameter ValueType as the highest precision.
 *
 * @tparam T  Type of object to hold for each supported precision.
 * @tparam ValueType  A scalar type of the highest precision needed.
 */
template <typename T, typename ValueType>
using array_prec = std::array<T, matrix::AMP<ValueType, int>::num_precisions>;


/**
 * Allocate an Ell matrix for each precision bin supported, starting at the
 * precision of the parameter ValueType.
 *
 * @tparam ValueType  Scalar type of the highest precision bin.
 * @tparam IndexType  Index type for the concrete matrix.
 *
 * @param dims  (Common) dimensions of all bins.
 * @param mnpr  Max nonzeros per row for each bin.
 */
template <typename ValueType, typename IndexType>
inline array_prec<std::unique_ptr<LinOp>, ValueType> allocate_bins(
    std::shared_ptr<const Executor> exec, const dim<2>& dims,
    const array_prec<int, ValueType> mnpr)
{
    using last_precision =
        std::tuple_element<num_amp_precisions - 1, supported_precisions>::type;
    constexpr int highest_idx = precision_index<last_precision>::index;
    constexpr int starting_idx =
        precision_index<gko::remove_complex<ValueType>>::index;
    array_prec<std::unique_ptr<LinOp>, ValueType> bins;
    gko::constexpr_for<starting_idx, highest_idx + 1, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::supported_types<ValueType>::type>::type;
        bins[k - starting_idx] =
            std::move(matrix::Ell<value_type, IndexType>::create(
                exec, dims, mnpr[k - starting_idx]));
    });
    return bins;
}


}  // namespace amp
}  // namespace gko


#endif  // GKO_CORE_MATRIX_AMP_HELPERS_H
