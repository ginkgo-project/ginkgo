// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_ell_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "reference/base/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Ell matrix format namespace.
 * @ref Ell
 * @ingroup batch_ell
 */
namespace batch_ell {


#include "reference/matrix/batch_ell_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const batch::matrix::Ell<ValueType, IndexType>* mat,
                  const batch::MultiVector<ValueType>* b,
                  batch::MultiVector<ValueType>* x)
{
    const auto b_ub = host::get_batch_struct(b);
    const auto x_ub = host::get_batch_struct(x);
    const auto mat_ub = host::get_batch_struct(mat);
#pragma omp parallel for
    for (size_type batch = 0; batch < x->get_num_batch_items(); ++batch) {
        const auto mat_item = batch::matrix::extract_batch_item(mat_ub, batch);
        const auto b_item = batch::extract_batch_item(b_ub, batch);
        const auto x_item = batch::extract_batch_item(x_ub, batch);
        simple_apply_kernel(mat_item, b_item, x_item);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_ELL_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_apply(std::shared_ptr<const DefaultExecutor> exec,
                    const batch::MultiVector<ValueType>* alpha,
                    const batch::matrix::Ell<ValueType, IndexType>* mat,
                    const batch::MultiVector<ValueType>* b,
                    const batch::MultiVector<ValueType>* beta,
                    batch::MultiVector<ValueType>* x)
{
    const auto b_ub = host::get_batch_struct(b);
    const auto x_ub = host::get_batch_struct(x);
    const auto mat_ub = host::get_batch_struct(mat);
    const auto alpha_ub = host::get_batch_struct(alpha);
    const auto beta_ub = host::get_batch_struct(beta);
#pragma omp parallel for
    for (size_type batch = 0; batch < x->get_num_batch_items(); ++batch) {
        const auto mat_item = batch::matrix::extract_batch_item(mat_ub, batch);
        const auto b_item = batch::extract_batch_item(b_ub, batch);
        const auto x_item = batch::extract_batch_item(x_ub, batch);
        const auto alpha_item = batch::extract_batch_item(alpha_ub, batch);
        const auto beta_item = batch::extract_batch_item(beta_ub, batch);
        advanced_apply_kernel(alpha_item.values[0], mat_item, b_item,
                              beta_item.values[0], x_item);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_ELL_ADVANCED_APPLY_KERNEL);


}  // namespace batch_ell
}  // namespace omp
}  // namespace kernels
}  // namespace gko
