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

#include "core/matrix/batch_dense_kernels.hpp"


#include <algorithm>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The BatchDense matrix format namespace.
 *
 * @ingroup batch_dense
 */
namespace batch_dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::BatchDense<ValueType>* a,
                  const matrix::BatchDense<ValueType>* b,
                  matrix::BatchDense<ValueType>* c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::BatchDense<ValueType>* alpha,
           const matrix::BatchDense<ValueType>* a,
           const matrix::BatchDense<ValueType>* b,
           const matrix::BatchDense<ValueType>* beta,
           matrix::BatchDense<ValueType>* c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::BatchDense<ValueType>* alpha,
           matrix::BatchDense<ValueType>* x) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const DpcppExecutor> exec,
                const matrix::BatchDense<ValueType>* alpha,
                const matrix::BatchDense<ValueType>* x,
                matrix::BatchDense<ValueType>* y) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scale(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::BatchDense<ValueType>* const alpha,
               const matrix::BatchDense<ValueType>* const x,
               const matrix::BatchDense<ValueType>* const beta,
               matrix::BatchDense<ValueType>* const y) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALE_KERNEL);


template <typename ValueType>
void convergence_add_scaled(std::shared_ptr<const DpcppExecutor> exec,
                            const matrix::BatchDense<ValueType>* alpha,
                            const matrix::BatchDense<ValueType>* x,
                            matrix::BatchDense<ValueType>* y,
                            const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const DpcppExecutor> exec,
                     const matrix::BatchDense<ValueType>* alpha,
                     const matrix::Diagonal<ValueType>* x,
                     matrix::BatchDense<ValueType>* y) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const DpcppExecutor> exec,
                 const matrix::BatchDense<ValueType>* x,
                 const matrix::BatchDense<ValueType>* y,
                 matrix::BatchDense<ValueType>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void convergence_compute_dot(std::shared_ptr<const DpcppExecutor> exec,
                             const matrix::BatchDense<ValueType>* x,
                             const matrix::BatchDense<ValueType>* y,
                             matrix::BatchDense<ValueType>* result,
                             const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_DOT_KERNEL);

template <typename ValueType>
void compute_norm2(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::BatchDense<ValueType>* x,
                   matrix::BatchDense<remove_complex<ValueType>>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void convergence_compute_norm2(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchDense<ValueType>* x,
    matrix::BatchDense<remove_complex<ValueType>>* result,
    const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_batch_csr(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::BatchDense<ValueType>* source,
                          matrix::BatchCsr<ValueType, IndexType>* other)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_DENSE_CONVERT_TO_BATCH_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::BatchDense<ValueType>* source,
                    size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const DpcppExecutor> exec,
                               const matrix::BatchDense<ValueType>* source,
                               size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const DpcppExecutor> exec,
                                const matrix::BatchDense<ValueType>* source,
                                array<size_type>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const DpcppExecutor> exec,
                          const matrix::BatchDense<ValueType>* source,
                          size_type* result, const size_type* stride_factor,
                          const size_type* slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::BatchDense<ValueType>* orig,
               matrix::BatchDense<ValueType>* trans) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::BatchDense<ValueType>* orig,
                    matrix::BatchDense<ValueType>* trans) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::BatchDense<ValueType>* x,
          matrix::BatchDense<ValueType>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COPY_KERNEL);


template <typename ValueType>
void convergence_copy(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::BatchDense<ValueType>* x,
                      matrix::BatchDense<ValueType>* result,
                      const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COPY_KERNEL);


template <typename ValueType>
void batch_scale(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::BatchDiagonal<ValueType>* left_diag,
                 const matrix::BatchDiagonal<ValueType>* right_diag,
                 matrix::BatchDense<ValueType>* x) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_BATCH_SCALE_KERNEL);


template <typename ValueType>
void add_scaled_identity(std::shared_ptr<const DpcppExecutor> exec,
                         const matrix::BatchDense<ValueType>* const a,
                         const matrix::BatchDense<ValueType>* const b,
                         matrix::BatchDense<ValueType>* const mtx)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
