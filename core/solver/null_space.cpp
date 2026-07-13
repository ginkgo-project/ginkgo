// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/null_space.hpp"

#include <ginkgo/core/base/precision_dispatch.hpp>


namespace gko {


template <typename ValueType>
void NullSpace<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            dense_x->copy_from(dense_b);
            this->project(dense_x);
        },
        b, x);
}


template <typename ValueType>
void NullSpace<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                      const LinOp* beta, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto projected = dense_b->clone();
            this->project(projected.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, projected);
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_NULL_SPACE(ValueType) class NullSpace<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_NULL_SPACE);


}  // namespace gko
