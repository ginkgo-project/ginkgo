// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/base/batch_multi_vector_kernels.hpp"

#include <thrust/functional.h>
#include <thrust/transform.h>

#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "core/base/batch_multi_vector_kernels.hpp"
#include "core/base/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_multi_vector {


constexpr auto default_block_size = 256;


template <typename ValueType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const batch::MultiVector<ValueType>* alpha,
           batch::MultiVector<ValueType>* x)
{
    const auto num_blocks = x->get_num_batch_items();
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);
    if (alpha->get_common_size()[1] == 1) {
        batch_single_kernels::scale_kernel<<<num_blocks, default_block_size, 0,
                                             exec->get_stream()>>>(
            alpha_ub, x_ub,
            [] __device__(int row, int col, int stride) { return 0; });
    } else if (alpha->get_common_size() == x->get_common_size()) {
        batch_single_kernels::scale_kernel<<<num_blocks, default_block_size, 0,
                                             exec->get_stream()>>>(
            alpha_ub, x_ub, [] __device__(int row, int col, int stride) {
                return row * stride + col;
            });
    } else {
        batch_single_kernels::scale_kernel<<<num_blocks, default_block_size, 0,
                                             exec->get_stream()>>>(
            alpha_ub, x_ub,
            [] __device__(int row, int col, int stride) { return col; });
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
    const auto num_blocks = x->get_num_batch_items();
    const size_type nrhs = x->get_common_size()[1];
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    if (alpha->get_common_size()[1] == 1) {
        batch_single_kernels::add_scaled_kernel<<<
            num_blocks, default_block_size, 0, exec->get_stream()>>>(
            alpha_ub, x_ub, y_ub, [] __device__(int col) { return 0; });
    } else {
        batch_single_kernels::add_scaled_kernel<<<
            num_blocks, default_block_size, 0, exec->get_stream()>>>(
            alpha_ub, x_ub, y_ub, [] __device__(int col) { return col; });
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
    const auto num_blocks = x->get_num_batch_items();
    const auto num_rhs = x->get_common_size()[1];
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    const auto res_ub = get_batch_struct(result);
    batch_single_kernels::compute_gen_dot_product_kernel<<<
        num_blocks, default_block_size, 0, exec->get_stream()>>>(
        x_ub, y_ub, res_ub, [] __device__(auto val) { return val; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const DefaultExecutor> exec,
                      const batch::MultiVector<ValueType>* x,
                      const batch::MultiVector<ValueType>* y,
                      batch::MultiVector<ValueType>* result)
{
    const auto num_blocks = x->get_num_batch_items();
    const auto num_rhs = x->get_common_size()[1];
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    const auto res_ub = get_batch_struct(result);
    batch_single_kernels::compute_gen_dot_product_kernel<<<
        num_blocks, default_block_size, 0, exec->get_stream()>>>(
        x_ub, y_ub, res_ub, [] __device__(auto val) { return conj(val); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_CONJ_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const DefaultExecutor> exec,
                   const batch::MultiVector<ValueType>* x,
                   batch::MultiVector<remove_complex<ValueType>>* result)
{
    const auto num_blocks = x->get_num_batch_items();
    const auto num_rhs = x->get_common_size()[1];
    const auto x_ub = get_batch_struct(x);
    const auto res_ub = get_batch_struct(result);
    batch_single_kernels::compute_norm2_kernel<<<num_blocks, default_block_size,
                                                 0, exec->get_stream()>>>(
        x_ub, res_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const batch::MultiVector<ValueType>* x,
          batch::MultiVector<ValueType>* result)
{
    const auto num_blocks = x->get_num_batch_items();
    const auto result_ub = get_batch_struct(result);
    const auto x_ub = get_batch_struct(x);
    batch_single_kernels::
        copy_kernel<<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
            x_ub, result_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL);


}  // namespace batch_multi_vector
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
