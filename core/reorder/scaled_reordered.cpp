// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/reorder/scaled_reordered.hpp"

#include <utility>

#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace experimental {
namespace reorder {


template <typename ValueType, typename IndexType>
void ScaledReordered<ValueType, IndexType>::apply_impl(
    const AbstractMultiVector* b, AbstractMultiVector* x) const
{
    auto exec = this->get_executor();
    auto converted_b = b->as_precision(precision_v<ValueType>);
    auto converted_x = x->as_precision(precision_v<ValueType>);
    this->set_cache_to(converted_b.get(), converted_x.get());

    // Preprocess the input vectors before applying the inner operator.
    if (row_scaling_) {
        row_scaling_->apply(cache_.inner_b, cache_.intermediate);
        std::swap(cache_.inner_b, cache_.intermediate);
    }
    // Col scaling for x is only necessary if the inner operator uses an
    // initial guess. Otherwise x is overwritten anyway.
    if (col_scaling_ && inner_operator_->apply_uses_initial_guess()) {
        col_scaling_->inverse_apply(cache_.inner_x, cache_.intermediate);
        std::swap(cache_.inner_x, cache_.intermediate);
    }
    if (permutation_array_.get_size() > 0) {
        cache_.inner_b->row_permute(&permutation_array_, cache_.intermediate);
        std::swap(cache_.inner_b, cache_.intermediate);
        if (inner_operator_->apply_uses_initial_guess()) {
            cache_.inner_x->row_permute(&permutation_array_,
                                        cache_.intermediate);
            std::swap(cache_.inner_x, cache_.intermediate);
        }
    }

    inner_operator_->apply(cache_.inner_b, cache_.inner_x);

    // Permute and scale the solution vector back.
    if (permutation_array_.get_size() > 0) {
        cache_.inner_x->inverse_row_permute(&permutation_array_,
                                            cache_.intermediate);
        std::swap(cache_.inner_x, cache_.intermediate);
    }
    if (col_scaling_) {
        col_scaling_->apply(cache_.inner_x, cache_.intermediate);
        std::swap(cache_.inner_x, cache_.intermediate);
    }
    // @todo: this needs two copies in the mixed precision case:
    //        inner->converted and converted->x
    converted_x->copy_from(cache_.inner_x);
}


template <typename ValueType, typename IndexType>
void ScaledReordered<ValueType, IndexType>::apply_impl(
    const AbstractMultiVector* alpha, const AbstractMultiVector* b,
    const AbstractMultiVector* beta, AbstractMultiVector* x) const
{
    auto x_clone = x->clone();
    this->apply_impl(b, x_clone.get());
    x->scale(beta);
    x->add_scaled(alpha, x_clone);
}


#define GKO_DECLARE_SCALED_REORDERED(ValueType, IndexType) \
    class ScaledReordered<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCALED_REORDERED);


}  // namespace reorder
}  // namespace experimental
}  // namespace gko
