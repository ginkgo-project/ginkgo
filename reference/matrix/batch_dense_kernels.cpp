// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_dense_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "reference/base/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Dense matrix format namespace.
 * @ref Dense
 * @ingroup batch_dense
 */
namespace batch_dense {


#include "reference/matrix/batch_dense_kernels.hpp.inc"


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const batch::matrix::Dense<ValueType>* mat,
                  const batch::MultiVector<ValueType>* b,
                  batch::MultiVector<ValueType>* x)
{
    const auto b_ub = host::get_batch_struct(b);
    const auto x_ub = host::get_batch_struct(x);
    const auto mat_ub = host::get_batch_struct(mat);
    for (size_type batch = 0; batch < x->get_num_batch_items(); ++batch) {
        const auto mat_item = batch::matrix::extract_batch_item(mat_ub, batch);
        const auto b_item = batch::extract_batch_item(b_ub, batch);
        const auto x_item = batch::extract_batch_item(x_ub, batch);
        simple_apply_kernel(mat_item, b_item, x_item);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void advanced_apply(std::shared_ptr<const DefaultExecutor> exec,
                    const batch::MultiVector<ValueType>* alpha,
                    const batch::matrix::Dense<ValueType>* mat,
                    const batch::MultiVector<ValueType>* b,
                    const batch::MultiVector<ValueType>* beta,
                    batch::MultiVector<ValueType>* x)
{
    const auto b_ub = host::get_batch_struct(b);
    const auto x_ub = host::get_batch_struct(x);
    const auto mat_ub = host::get_batch_struct(mat);
    const auto alpha_ub = host::get_batch_struct(alpha);
    const auto beta_ub = host::get_batch_struct(beta);
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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADVANCED_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const array<ValueType>* col_scale, const array<ValueType>* row_scale,
           batch::matrix::Dense<ValueType>* input)
{
    const auto col_scale_vals = col_scale->get_const_data();
    const auto row_scale_vals = row_scale->get_const_data();
    auto input_vals = input->get_values();
    const auto num_rows = static_cast<int>(input->get_common_size()[0]);
    const auto num_cols = static_cast<int>(input->get_common_size()[1]);
    const auto stride = input->get_common_size()[1];
    for (size_type batch_id = 0; batch_id < input->get_num_batch_items();
         ++batch_id) {
        const auto col_scale_b = col_scale_vals + num_cols * batch_id;
        const auto row_scale_b = row_scale_vals + num_rows * batch_id;
        const auto input_mat =
            input_vals + input->get_num_elements_per_item() * batch_id;
        scale(num_rows, num_cols, stride, col_scale_b, row_scale_b, input_mat);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);


template <typename ValueType>
void scale_add(std::shared_ptr<const DefaultExecutor> exec,
               const batch::MultiVector<ValueType>* alpha,
               const batch::matrix::Dense<ValueType>* mat,
               batch::matrix::Dense<ValueType>* input)
{
    const auto mat_ub = host::get_batch_struct(mat);
    const auto in_mat_ub = host::get_batch_struct(input);
    const auto alpha_ub = host::get_batch_struct(alpha);
    for (size_type batch_id = 0; batch_id < input->get_num_batch_items();
         ++batch_id) {
        const auto alpha_b = batch::extract_batch_item(alpha_ub, batch_id);
        const auto mat_b = batch::matrix::extract_batch_item(mat_ub, batch_id);
        const auto input_mat_b =
            batch::matrix::extract_batch_item(in_mat_ub, batch_id);
        scale_add_kernel(alpha_b.values[0], mat_b, input_mat_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_ADD_KERNEL);


template <typename ValueType>
void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec,
                         const batch::MultiVector<ValueType>* alpha,
                         const batch::MultiVector<ValueType>* beta,
                         batch::matrix::Dense<ValueType>* mat)
{
    const auto mat_ub = host::get_batch_struct(mat);
    const auto alpha_ub = host::get_batch_struct(alpha);
    const auto beta_ub = host::get_batch_struct(beta);
    for (size_type batch_id = 0; batch_id < mat->get_num_batch_items();
         ++batch_id) {
        const auto alpha_b = batch::extract_batch_item(alpha_ub, batch_id);
        const auto beta_b = batch::extract_batch_item(beta_ub, batch_id);
        const auto mat_b = batch::matrix::extract_batch_item(mat_ub, batch_id);
        add_scaled_identity_kernel(alpha_b.values[0], beta_b.values[0], mat_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_dense
}  // namespace reference
}  // namespace kernels
}  // namespace gko
