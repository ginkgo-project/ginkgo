// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_AMP_HELPERS_H
#define GKO_CORE_MATRIX_AMP_HELPERS_H

#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/utils.hpp"


namespace gko {
namespace amp {


/**
 * Allocate an Ell matrix for each precision bin supported, starting at the
 * precision of the parameter ValueType.
 *
 * @tparam ValueType  Scalar type of the highest precision bin.
 * @tparam IndexType  Index type for the concrete matrix.
 *
 * @param dims  (Common) dimensions of all bins.
 * @param mnpr  Max nonzeros per row for each bin.
 * @return  Fixed-size array of LinOps, one for each allocated bin.
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

/**
 * Allocate an Ell matrix for each precision bin supported, starting at the
 * precision of the parameter ValueType.
 *
 * @tparam ValueType  Scalar type of the highest precision bin.
 * @tparam IndexType  Index type for the concrete matrix.
 *
 * @param dims  (Common) dimensions of all bins.
 * @param mnpr  Max nonzeros per row for each bin.
 * @return  Tuple of unique_ptrs to Ell matrices, one for each allocated bin.
 */
template <typename ValueType, typename IndexType>
inline auto allocate_bins_tuple(std::shared_ptr<const Executor> exec,
                                const dim<2>& dims,
                                const array_prec<int, ValueType> mnpr)
{
    using last_precision =
        std::tuple_element<num_amp_precisions - 1, supported_precisions>::type;
    constexpr int highest_idx = precision_index<last_precision>::index;
    constexpr int starting_idx = precision_index<ValueType>::index;
    // array_prec<std::unique_ptr<LinOp>, ValueType> bins;
    using EllTuple = gko::transformed_instantiation_tuple_t<
        std::unique_ptr, gko::generator_partial<gko::matrix::Ell, IndexType>,
        typename gko::amp::narrow_types<ValueType>::type>;
    EllTuple bins;
    gko::constexpr_for<starting_idx, highest_idx + 1, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::supported_types<ValueType>::type>::type;
        std::get<k - starting_idx>(bins) =
            std::move(matrix::Ell<value_type, IndexType>::create(
                exec, dims, mnpr[k - starting_idx]));
    });
    return bins;
}


}  // namespace amp
}  // namespace gko


#endif  // GKO_CORE_MATRIX_AMP_HELPERS_H
