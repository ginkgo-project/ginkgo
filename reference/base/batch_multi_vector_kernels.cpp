// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/batch_multi_vector_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>


#include "core/base/batch_struct.hpp"
#include "reference/base/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The batch::MultiVector matrix format namespace.
 * @ref batch::MultiVector
 * @ingroup batch_multi_vector
 */
namespace batch_multi_vector {


#include "reference/base/batch_multi_vector_kernels.hpp.inc"


template <typename ValueType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const batch::MultiVector<ValueType>* alpha,
           batch::MultiVector<ValueType>* x)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto alpha_ub = host::get_batch_struct(alpha);
    for (size_type batch = 0; batch < x->get_num_batch_items(); ++batch) {
        const auto alpha_b = batch::extract_batch_item(alpha_ub, batch);
        const auto x_b = batch::extract_batch_item(x_ub, batch);
        scale_kernel(alpha_b, x_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const DefaultExecutor> exec,
                const batch::MultiVector<ValueType>* alpha,
                const batch::MultiVector<ValueType>* x,
                batch::MultiVector<ValueType>* y)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto alpha_ub = host::get_batch_struct(alpha);
    for (size_type batch = 0; batch < y->get_num_batch_items(); ++batch) {
        const auto alpha_b = batch::extract_batch_item(alpha_ub, batch);
        const auto x_b = batch::extract_batch_item(x_ub, batch);
        const auto y_b = batch::extract_batch_item(y_ub, batch);
        add_scaled_kernel(alpha_b, x_b, y_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_ADD_SCALED_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const DefaultExecutor> exec,
                 const batch::MultiVector<ValueType>* x,
                 const batch::MultiVector<ValueType>* y,
                 batch::MultiVector<ValueType>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto res_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < result->get_num_batch_items(); ++batch) {
        const auto res_b = batch::extract_batch_item(res_ub, batch);
        const auto x_b = batch::extract_batch_item(x_ub, batch);
        const auto y_b = batch::extract_batch_item(y_ub, batch);
        compute_dot_product_kernel(x_b, y_b, res_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const DefaultExecutor> exec,
                      const batch::MultiVector<ValueType>* x,
                      const batch::MultiVector<ValueType>* y,
                      batch::MultiVector<ValueType>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto res_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < result->get_num_batch_items(); ++batch) {
        const auto res_b = batch::extract_batch_item(res_ub, batch);
        const auto x_b = batch::extract_batch_item(x_ub, batch);
        const auto y_b = batch::extract_batch_item(y_ub, batch);
        compute_conj_dot_product_kernel(x_b, y_b, res_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_CONJ_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const DefaultExecutor> exec,
                   const batch::MultiVector<ValueType>* x,
                   batch::MultiVector<remove_complex<ValueType>>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto res_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < result->get_num_batch_items(); ++batch) {
        const auto res_b = batch::extract_batch_item(res_ub, batch);
        const auto x_b = batch::extract_batch_item(x_ub, batch);
        compute_norm2_kernel(x_b, res_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const batch::MultiVector<ValueType>* x,
          batch::MultiVector<ValueType>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto result_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < x->get_num_batch_items(); ++batch) {
        const auto result_b = batch::extract_batch_item(result_ub, batch);
        const auto x_b = batch::extract_batch_item(x_ub, batch);
        copy_kernel(x_b, result_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL);


}  // namespace batch_multi_vector
}  // namespace reference
}  // namespace kernels
}  // namespace gko
