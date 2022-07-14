/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include "core/matrix/batch_struct.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The BatchDense matrix format namespace.
 *
 * @ingroup batch_dense
 */
namespace batch_dense {


constexpr auto default_block_size = 256;
constexpr int sm_multiplier = 4;


#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"


template <typename ValueType>
void simple_apply(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::BatchDense<ValueType>* a,
                  const matrix::BatchDense<ValueType>* b,
                  matrix::BatchDense<ValueType>* c)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    if (b_ub.num_rhs > 1) {
        GKO_NOT_IMPLEMENTED;
    }
    mv<<<num_blocks, default_block_size>>>(a_ub, b_ub, c_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const matrix::BatchDense<ValueType>* alpha,
           const matrix::BatchDense<ValueType>* a,
           const matrix::BatchDense<ValueType>* b,
           const matrix::BatchDense<ValueType>* beta,
           matrix::BatchDense<ValueType>* c)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);
    if (b_ub.num_rhs > 1) {
        GKO_NOT_IMPLEMENTED;
    }
    advanced_mv<<<num_blocks, default_block_size>>>(alpha_ub, a_ub, b_ub,
                                                    beta_ub, c_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const CudaExecutor> exec,
           const matrix::BatchDense<ValueType>* const alpha,
           matrix::BatchDense<ValueType>* const x)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);
    scale<<<num_blocks, default_block_size>>>(alpha_ub, x_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const CudaExecutor> exec,
                const matrix::BatchDense<ValueType>* const alpha,
                const matrix::BatchDense<ValueType>* const x,
                matrix::BatchDense<ValueType>* const y)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const size_type nrhs = x->get_size().at(0)[1];
    if (nrhs == 1) {
        const auto num_batch = x->get_num_batch_entries();
        const auto num_rows = x->get_size().at(0)[0];
        single_add_scaled<<<num_blocks, default_block_size>>>(
            num_batch, num_rows, as_cuda_type(alpha->get_const_values()),
            as_cuda_type(x->get_const_values()), as_cuda_type(y->get_values()));
    } else {
        const auto alpha_ub = get_batch_struct(alpha);
        const auto x_ub = get_batch_struct(x);
        const auto y_ub = get_batch_struct(y);
        add_scaled<<<num_blocks, default_block_size>>>(alpha_ub, x_ub, y_ub);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scale(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::BatchDense<ValueType>* const alpha,
               const matrix::BatchDense<ValueType>* const x,
               const matrix::BatchDense<ValueType>* const beta,
               matrix::BatchDense<ValueType>* const y)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const size_type nrhs = x->get_size().at(0)[1];
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    add_scale<<<num_blocks, default_block_size>>>(alpha_ub, x_ub, beta_ub,
                                                  y_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALE_KERNEL);


template <typename ValueType>
void convergence_add_scaled(std::shared_ptr<const CudaExecutor> exec,
                            const matrix::BatchDense<ValueType>* const alpha,
                            const matrix::BatchDense<ValueType>* const x,
                            matrix::BatchDense<ValueType>* const y,
                            const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const CudaExecutor> exec,
                     const matrix::BatchDense<ValueType>* alpha,
                     const matrix::Diagonal<ValueType>* x,
                     matrix::BatchDense<ValueType>* y) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const CudaExecutor> exec,
                 const matrix::BatchDense<ValueType>* x,
                 const matrix::BatchDense<ValueType>* y,
                 matrix::BatchDense<ValueType>* result)
{
    const auto num_blocks = x->get_num_batch_entries();
    const auto num_rhs = x->get_size().at()[1];
    if (num_rhs == 1) {
        const auto num_rows = x->get_size().at()[0];
        single_compute_dot_product<<<num_blocks, default_block_size>>>(
            num_blocks, num_rows, as_cuda_type(x->get_const_values()),
            as_cuda_type(y->get_const_values()),
            as_cuda_type(result->get_values()));
    } else {
        const auto x_ub = get_batch_struct(x);
        const auto y_ub = get_batch_struct(y);
        const auto res_ub = get_batch_struct(result);
        compute_dot_product<<<num_blocks, default_block_size>>>(x_ub, y_ub,
                                                                res_ub);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void convergence_compute_dot(std::shared_ptr<const CudaExecutor> exec,
                             const matrix::BatchDense<ValueType>* x,
                             const matrix::BatchDense<ValueType>* y,
                             matrix::BatchDense<ValueType>* result,
                             const uint32& converged) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::BatchDense<ValueType>* const x,
                   matrix::BatchDense<remove_complex<ValueType>>* const result)
{
    const auto num_blocks = x->get_num_batch_entries();
    const auto num_rhs = x->get_size().at()[1];
    if (num_rhs == 1) {
        const auto num_rows = x->get_size().at()[0];
        single_compute_norm2<<<num_blocks, default_block_size>>>(
            num_blocks, num_rows, as_cuda_type(x->get_const_values()),
            as_cuda_type(result->get_values()));
    } else {
        const auto x_ub = get_batch_struct(x);
        const auto res_ub = get_batch_struct(result);
        compute_norm2<<<num_blocks, default_block_size>>>(x_ub, res_ub);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void convergence_compute_norm2(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::BatchDense<ValueType>* const x,
    matrix::BatchDense<remove_complex<ValueType>>* const result,
    const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_batch_csr(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::BatchDense<ValueType>* source,
                          matrix::BatchCsr<ValueType, IndexType>* other)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERT_TO_BATCH_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::BatchDense<ValueType>* source,
                    size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const CudaExecutor> exec,
                               const matrix::BatchDense<ValueType>* source,
                               size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const CudaExecutor> exec,
                                const matrix::BatchDense<ValueType>* source,
                                array<size_type>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const CudaExecutor> exec,
                          const matrix::BatchDense<ValueType>* source,
                          size_type* result, const size_type* stride_factor,
                          const size_type* slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const CudaExecutor> exec,
               const matrix::BatchDense<ValueType>* const orig,
               matrix::BatchDense<ValueType>* const trans)
{
    using cu_val_type = cuda_type<ValueType>;
    const size_type nbatch = orig->get_num_batch_entries();
    const size_type orig_stride = orig->get_stride().at();
    const size_type trans_stride = trans->get_stride().at();
    const int nrows = orig->get_size().at()[0];
    const int ncols = orig->get_size().at()[1];
    transpose<<<nbatch, default_block_size>>>(
        nrows, ncols, orig_stride, as_cuda_type(orig->get_const_values()),
        trans_stride, as_cuda_type(trans->get_values()),
        [] __device__(cu_val_type x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::BatchDense<ValueType>* orig,
                    matrix::BatchDense<ValueType>* trans)
{
    using cu_val_type = cuda_type<ValueType>;
    const size_type nbatch = orig->get_num_batch_entries();
    const size_type orig_stride = orig->get_stride().at();
    const size_type trans_stride = trans->get_stride().at();
    const int nrows = orig->get_size().at()[0];
    const int ncols = orig->get_size().at()[1];
    transpose<<<nbatch, default_block_size>>>(
        nrows, ncols, orig_stride, as_cuda_type(orig->get_const_values()),
        trans_stride, as_cuda_type(trans->get_values()),
        [] __device__(cu_val_type x) { return conj(x); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::BatchDense<ValueType>* x,
          matrix::BatchDense<ValueType>* result)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto result_ub = get_batch_struct(result);
    const auto x_ub = get_batch_struct(x);
    copy<<<num_blocks, default_block_size>>>(x_ub, result_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COPY_KERNEL);


template <typename ValueType>
void convergence_copy(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::BatchDense<ValueType>* x,
                      matrix::BatchDense<ValueType>* result,
                      const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COPY_KERNEL);


template <typename ValueType>
void batch_scale(std::shared_ptr<const CudaExecutor> exec,
                 const matrix::BatchDiagonal<ValueType>* const left_scale,
                 const matrix::BatchDiagonal<ValueType>* const rght_scale,
                 matrix::BatchDense<ValueType>* const vec_to_scale)
{
    if (!left_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    if (!rght_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    if (!vec_to_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;

    const auto stride = vec_to_scale->get_stride().at();
    const auto nrows = static_cast<int>(vec_to_scale->get_size().at()[0]);
    const auto nrhs = static_cast<int>(vec_to_scale->get_size().at()[1]);
    const auto nbatch = vec_to_scale->get_num_batch_entries();

    const int num_blocks = vec_to_scale->get_num_batch_entries();
    uniform_batch_scale<<<num_blocks, default_block_size>>>(
        nrows, stride, nrhs, nbatch,
        as_cuda_type(left_scale->get_const_values()),
        as_cuda_type(rght_scale->get_const_values()),
        as_cuda_type(vec_to_scale->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_BATCH_SCALE_KERNEL);


template <typename ValueType>
void add_scaled_identity(std::shared_ptr<const CudaExecutor> exec,
                         const matrix::BatchDense<ValueType>* const a,
                         const matrix::BatchDense<ValueType>* const b,
                         matrix::BatchDense<ValueType>* const mtx)
{
    if (!mtx->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const auto num_blocks = mtx->get_num_batch_entries();
    const auto nrows = static_cast<int>(mtx->get_size().at(0)[0]);
    const auto ncols = static_cast<int>(mtx->get_size().at(0)[1]);
    const auto stride = mtx->get_stride().at(0);
    const auto values = mtx->get_values();
    const auto alpha = a->get_const_values();
    const auto a_stride = a->get_stride().at(0);
    const auto b_stride = b->get_stride().at(0);
    const auto beta = b->get_const_values();
    add_scaled_identity<<<num_blocks, default_block_size>>>(
        num_blocks, nrows, ncols, stride, as_cuda_type(values), a_stride,
        as_cuda_type(alpha), b_stride, as_cuda_type(beta));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_dense
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
