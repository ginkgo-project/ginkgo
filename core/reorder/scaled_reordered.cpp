// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/reorder/scaled_reordered.hpp"

#include <utility>

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace experimental {
namespace reorder {


template <typename ValueType, typename IndexType>
ScaledReordered<ValueType, IndexType>::ScaledReordered(
    std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec), dim<2>{}, type_to_precision<ValueType>),
      permutation_array_{exec}
{}


template <typename ValueType, typename IndexType>
ScaledReordered<ValueType, IndexType>::ScaledReordered(
    const Factory* factory, std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), system_matrix->get_size(),
            type_to_precision<ValueType>),
      parameters_{factory->get_parameters()},
      permutation_array_{factory->get_executor()}
{
    // For now only support square matrices.
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    auto exec = this->get_executor();

    system_matrix_ = gko::clone(exec, system_matrix);

    // Scale the system matrix if scaling coefficients are provided
    if (parameters_.row_scaling) {
        GKO_ASSERT_EQUAL_DIMENSIONS(parameters_.row_scaling, system_matrix_);
        row_scaling_ = parameters_.row_scaling;
        row_scaling_->apply(system_matrix_, system_matrix_);
    }
    if (parameters_.col_scaling) {
        GKO_ASSERT_EQUAL_DIMENSIONS(parameters_.col_scaling, system_matrix_);
        col_scaling_ = parameters_.col_scaling;
        col_scaling_->rapply(system_matrix_, system_matrix_);
    }

    // If a reordering factory is provided, generate the reordering and
    // permute the system matrix accordingly.
    if (parameters_.reordering) {
        auto reordering = parameters_.reordering->generate(system_matrix_);
        permutation_array_ = reordering->get_permutation_array();
        system_matrix_ = as<Permutable<index_type>>(system_matrix_)
                             ->permute(&permutation_array_);
    }

    // Generate the inner operator with the scaled and reordered system
    // matrix. If none is provided, use the Identity.
    if (parameters_.inner_operator) {
        inner_operator_ = parameters_.inner_operator->generate(system_matrix_);
    } else {
        inner_operator_ = gko::matrix::Identity<value_type>::create(
            exec, this->get_size()[0]);
    }
}


template <typename ValueType, typename IndexType>
void ScaledReordered<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                       LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            auto exec = this->get_executor();
            this->set_cache_to(dense_b, dense_x);

            // Preprocess the input vectors before applying the inner operator.
            if (row_scaling_) {
                row_scaling_->apply(cache_.inner_b, cache_.intermediate);
                std::swap(cache_.inner_b, cache_.intermediate);
            }
            // Col scaling for x is only necessary if the inner operator uses an
            // initial guess. Otherwise x is overwritten anyway.
            if (col_scaling_ && inner_operator_->apply_uses_initial_guess()) {
                col_scaling_->inverse_apply(cache_.inner_x,
                                            cache_.intermediate);
                std::swap(cache_.inner_x, cache_.intermediate);
            }
            if (permutation_array_.get_size() > 0) {
                cache_.inner_b->row_permute(&permutation_array_,
                                            cache_.intermediate);
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
            dense_x->copy_from(cache_.inner_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void ScaledReordered<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                       const LinOp* b,
                                                       const LinOp* beta,
                                                       LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_SCALED_REORDERED(ValueType, IndexType) \
    class ScaledReordered<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCALED_REORDERED);


}  // namespace reorder
}  // namespace experimental
}  // namespace gko
