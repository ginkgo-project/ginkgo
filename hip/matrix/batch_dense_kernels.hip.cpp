/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/matrix/batch_dense_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/components/prefix_sum.hpp"
//#include "hip/base/config.hip.hpp"
#include "hip/base/hipblas_bindings.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"
//#include "hip/matrix/batch_struct.hip.hpp"

namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The BatchDense matrix format namespace.
 *
 * @ingroup batch_dense
 */
namespace batch_dense {


constexpr auto default_block_size = 512;
constexpr int sm_multiplier = 4;

//#include "common/matrix/batch_dense_kernels.hpp.inc"

template <typename ValueType>
void simple_apply(std::shared_ptr<const HipExecutor> exec,
                  const matrix::BatchDense<ValueType> *a,
                  const matrix::BatchDense<ValueType> *b,
                  matrix::BatchDense<ValueType> *c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const HipExecutor> exec,
           const matrix::BatchDense<ValueType> *alpha,
           const matrix::BatchDense<ValueType> *a,
           const matrix::BatchDense<ValueType> *b,
           const matrix::BatchDense<ValueType> *beta,
           matrix::BatchDense<ValueType> *c) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const HipExecutor> exec,
           const matrix::BatchDense<ValueType> *alpha,
           matrix::BatchDense<ValueType> *x) GKO_NOT_IMPLEMENTED;
// {
//     const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
//     const auto alpha_ub = get_batch_struct(alpha);
//     const auto x_ub = get_batch_struct(x);
//     hipLaunchKernelGGL(HIP_KERNEL_NAME(scale), dim3(num_blocks),
//                        dim3(default_block_size), 0, 0, alpha_ub, x_ub);
// }


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const HipExecutor> exec,
                const matrix::BatchDense<ValueType> *alpha,
                const matrix::BatchDense<ValueType> *x,
                matrix::BatchDense<ValueType> *y) GKO_NOT_IMPLEMENTED;
// {
//     const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
//     const auto alpha_ub = get_batch_struct(alpha);
//     const auto x_ub = get_batch_struct(x);
//     const auto y_ub = get_batch_struct(y);
//     hipLaunchKernelGGL(HIP_KERNEL_NAME(add_scaled), dim3(num_blocks),
//                        dim3(default_block_size), 0, 0, alpha_ub, x_ub, y_ub);
// }


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const HipExecutor> exec,
                     const matrix::BatchDense<ValueType> *alpha,
                     const matrix::Diagonal<ValueType> *x,
                     matrix::BatchDense<ValueType> *y) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const HipExecutor> exec,
                 const matrix::BatchDense<ValueType> *x,
                 const matrix::BatchDense<ValueType> *y,
                 matrix::BatchDense<ValueType> *result) GKO_NOT_IMPLEMENTED;
// {
//     const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
//     const auto x_ub = get_batch_struct(x);
//     const auto y_ub = get_batch_struct(y);
//     const auto res_ub = get_batch_struct(result);
//     hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_dot_product),
//     dim3(num_blocks),
//                        dim3(default_block_size), 0, 0, x_ub, y_ub, res_ub);
// }


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const HipExecutor> exec,
                   const matrix::BatchDense<ValueType> *x,
                   matrix::BatchDense<remove_complex<ValueType>> *result)
    GKO_NOT_IMPLEMENTED;
// {
//     const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
//     const auto x_ub = get_batch_struct(x);
//     const auto res_ub = get_batch_struct(result);
//     hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_norm2), dim3(num_blocks),
//                        dim3(default_block_size), 0, 0, x_ub, res_ub);
// }


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_batch_csr(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::BatchDense<ValueType> *source,
                          matrix::BatchCsr<ValueType, IndexType> *other)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_DENSE_CONVERT_TO_BATCH_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const HipExecutor> exec,
                    const matrix::BatchDense<ValueType> *source,
                    size_type *result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const HipExecutor> exec,
                               const matrix::BatchDense<ValueType> *source,
                               size_type *result) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const HipExecutor> exec,
                                const matrix::BatchDense<ValueType> *source,
                                Array<size_type> *result) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const HipExecutor> exec,
                          const matrix::BatchDense<ValueType> *source,
                          size_type *result, const size_type *stride_factor,
                          const size_type *slice_size) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const HipExecutor> exec,
               const matrix::BatchDense<ValueType> *orig,
               matrix::BatchDense<ValueType> *trans) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const HipExecutor> exec,
                    const matrix::BatchDense<ValueType> *orig,
                    matrix::BatchDense<ValueType> *trans) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL);


}  // namespace batch_dense
}  // namespace hip
}  // namespace kernels
}  // namespace gko
