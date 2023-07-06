/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/base/batch_multi_vector_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>


#include "core/base/batch_struct.hpp"
#include "hip/base/batch_struct.hip.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipblas_bindings.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The BatchMultiVector matrix format namespace.
 *
 * @ingroup batch_multi_vector
 */
namespace batch_multi_vector {


constexpr auto default_block_size = 256;
constexpr int sm_multiplier = 4;


#include "common/cuda_hip/base/batch_multi_vector_kernels.hpp.inc"


template <typename ValueType>
void scale(std::shared_ptr<const HipExecutor> exec,
           const BatchMultiVector<ValueType>* const alpha,
           BatchMultiVector<ValueType>* const x)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);
    hipLaunchKernelGGL(scale, dim3(num_blocks), dim3(default_block_size), 0, 0,
                       alpha_ub, x_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const HipExecutor> exec,
                const BatchMultiVector<ValueType>* const alpha,
                const BatchMultiVector<ValueType>* const x,
                BatchMultiVector<ValueType>* const y)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const size_type nrhs = x->get_size().at(0)[1];
    if (nrhs == 1) {
        const auto num_batch = x->get_num_batch_entries();
        const auto num_rows = x->get_size().at(0)[0];
        hipLaunchKernelGGL(
            single_add_scaled, dim3(num_blocks), dim3(default_block_size), 0, 0,
            num_batch, num_rows, as_hip_type(alpha->get_const_values()),
            as_hip_type(x->get_const_values()), as_hip_type(y->get_values()));
    } else {
        const auto alpha_ub = get_batch_struct(alpha);
        const auto x_ub = get_batch_struct(x);
        const auto y_ub = get_batch_struct(y);
        hipLaunchKernelGGL(add_scaled, dim3(num_blocks),
                           dim3(default_block_size), 0, 0, alpha_ub, x_ub,
                           y_ub);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_ADD_SCALED_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const HipExecutor> exec,
                 const BatchMultiVector<ValueType>* x,
                 const BatchMultiVector<ValueType>* y,
                 BatchMultiVector<ValueType>* result)
{
    const auto num_blocks = x->get_num_batch_entries();
    const auto num_rhs = x->get_size().at()[1];
    if (num_rhs == 1) {
        const auto num_rows = x->get_size().at()[0];
        hipLaunchKernelGGL(single_compute_dot_product, dim3(num_blocks),
                           dim3(default_block_size), 0, 0, num_blocks, num_rows,
                           as_hip_type(x->get_const_values()),
                           as_hip_type(y->get_const_values()),
                           as_hip_type(result->get_values()));
    } else {
        const auto x_ub = get_batch_struct(x);
        const auto y_ub = get_batch_struct(y);
        const auto res_ub = get_batch_struct(result);
        hipLaunchKernelGGL(compute_dot_product, dim3(num_blocks),
                           dim3(default_block_size), 0, 0, x_ub, y_ub, res_ub);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const HipExecutor> exec,
                   const BatchMultiVector<ValueType>* const x,
                   BatchMultiVector<remove_complex<ValueType>>* const result)
{
    const auto num_blocks = x->get_num_batch_entries();
    const auto num_rhs = x->get_size().at()[1];
    if (num_rhs == 1) {
        const auto num_rows = x->get_size().at()[0];
        hipLaunchKernelGGL(single_compute_norm2, dim3(num_blocks),
                           dim3(default_block_size), 0, 0, num_blocks, num_rows,
                           as_hip_type(x->get_const_values()),
                           as_hip_type(result->get_values()));
    } else {
        const auto x_ub = get_batch_struct(x);
        const auto res_ub = get_batch_struct(result);
        hipLaunchKernelGGL(compute_norm2, dim3(num_blocks),
                           dim3(default_block_size), 0, 0, x_ub, res_ub);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const BatchMultiVector<ValueType>* x,
          BatchMultiVector<ValueType>* result)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto result_ub = get_batch_struct(result);
    const auto x_ub = get_batch_struct(x);
    hipLaunchKernelGGL(copy, dim3(num_blocks), dim3(default_block_size), 0, 0,
                       x_ub, result_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL);


}  // namespace batch_multi_vector
}  // namespace hip
}  // namespace kernels
}  // namespace gko
