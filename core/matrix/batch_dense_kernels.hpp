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

#ifndef GKO_CORE_MATRIX_BATCH_DENSE_KERNELS_HPP_
#define GKO_CORE_MATRIX_BATCH_DENSE_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_dense.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL(_type)         \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec, \
                      const matrix::BatchDense<_type>* a,          \
                      const matrix::BatchDense<_type>* b,          \
                      matrix::BatchDense<_type>* c)

#define GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL(_type)         \
    void apply(std::shared_ptr<const DefaultExecutor> exec, \
               const matrix::BatchDense<_type>* alpha,      \
               const matrix::BatchDense<_type>* a,          \
               const matrix::BatchDense<_type>* b,          \
               const matrix::BatchDense<_type>* beta,       \
               matrix::BatchDense<_type>* c)

#define GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL(_type)         \
    void scale(std::shared_ptr<const DefaultExecutor> exec, \
               const matrix::BatchDense<_type>* alpha,      \
               matrix::BatchDense<_type>* x)

#define GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL(_type)         \
    void add_scaled(std::shared_ptr<const DefaultExecutor> exec, \
                    const matrix::BatchDense<_type>* alpha,      \
                    const matrix::BatchDense<_type>* x,          \
                    matrix::BatchDense<_type>* y)

#define GKO_DECLARE_BATCH_DENSE_ADD_SCALE_KERNEL(_type)         \
    void add_scale(std::shared_ptr<const DefaultExecutor> exec, \
                   const matrix::BatchDense<_type>* alpha,      \
                   const matrix::BatchDense<_type>* x,          \
                   const matrix::BatchDense<_type>* beta,       \
                   matrix::BatchDense<_type>* y)

#define GKO_DECLARE_BATCH_DENSE_CONVERGENCE_ADD_SCALED_KERNEL(_type)         \
    void convergence_add_scaled(std::shared_ptr<const DefaultExecutor> exec, \
                                const matrix::BatchDense<_type>* alpha,      \
                                const matrix::BatchDense<_type>* x,          \
                                matrix::BatchDense<_type>* y,                \
                                const uint32& converged)

#define GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL(_type)         \
    void add_scaled_diag(std::shared_ptr<const DefaultExecutor> exec, \
                         const matrix::BatchDense<_type>* alpha,      \
                         const matrix::Diagonal<_type>* x,            \
                         matrix::BatchDense<_type>* y)

#define GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL(_type)         \
    void compute_dot(std::shared_ptr<const DefaultExecutor> exec, \
                     const matrix::BatchDense<_type>* x,          \
                     const matrix::BatchDense<_type>* y,          \
                     matrix::BatchDense<_type>* result)


#define GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_DOT_KERNEL(_type)         \
    void convergence_compute_dot(std::shared_ptr<const DefaultExecutor> exec, \
                                 const matrix::BatchDense<_type>* x,          \
                                 const matrix::BatchDense<_type>* y,          \
                                 matrix::BatchDense<_type>* result,           \
                                 const uint32& converged)

#define GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL(_type)         \
    void compute_norm2(std::shared_ptr<const DefaultExecutor> exec, \
                       const matrix::BatchDense<_type>* x,          \
                       matrix::BatchDense<remove_complex<_type>>* result)

#define GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_NORM2_KERNEL(_type) \
    void convergence_compute_norm2(                                     \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::BatchDense<_type>* x,                             \
        matrix::BatchDense<remove_complex<_type>>* result,              \
        const uint32& converged)


#define GKO_DECLARE_BATCH_DENSE_COPY_KERNEL(_type)         \
    void copy(std::shared_ptr<const DefaultExecutor> exec, \
              const matrix::BatchDense<_type>* x,          \
              matrix::BatchDense<_type>* result)

#define GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COPY_KERNEL(_type)         \
    void convergence_copy(std::shared_ptr<const DefaultExecutor> exec, \
                          const matrix::BatchDense<_type>* x,          \
                          matrix::BatchDense<_type>* result,           \
                          const uint32& converged)

#define GKO_DECLARE_BATCH_DENSE_CONVERT_TO_BATCH_CSR_KERNEL(_type, _prec)  \
    void convert_to_batch_csr(std::shared_ptr<const DefaultExecutor> exec, \
                              const matrix::BatchDense<_type>* source,     \
                              matrix::BatchCsr<_type, _prec>* other)

#define GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL(_type)         \
    void count_nonzeros(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::BatchDense<_type>* source,     \
                        size_type* result)

#define GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(_type) \
    void calculate_max_nnz_per_row(                                     \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::BatchDense<_type>* source, size_type* result)

#define GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL(_type) \
    void calculate_nonzeros_per_row(                                     \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::BatchDense<_type>* source, array<size_type>* result)

#define GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL(_type)  \
    void calculate_total_cols(                                      \
        std::shared_ptr<const DefaultExecutor> exec,                \
        const matrix::BatchDense<_type>* source, size_type* result, \
        const size_type* stride_factor, const size_type* slice_size)

#define GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL(_type)         \
    void transpose(std::shared_ptr<const DefaultExecutor> exec, \
                   const matrix::BatchDense<_type>* orig,       \
                   matrix::BatchDense<_type>* trans)

#define GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL(_type)         \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::BatchDense<_type>* orig,       \
                        matrix::BatchDense<_type>* trans)

#define GKO_DECLARE_BATCH_DENSE_BATCH_SCALE_KERNEL(ValueType)             \
    void batch_scale(std::shared_ptr<const DefaultExecutor> exec,         \
                     const matrix::BatchDiagonal<ValueType>* left_scale,  \
                     const matrix::BatchDiagonal<ValueType>* right_scale, \
                     matrix::BatchDense<ValueType>* vec_to_scale)

#define GKO_DECLARE_BATCH_DENSE_ADD_SCALED_IDENTITY_KERNEL(ValueType)     \
    void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec, \
                             const matrix::BatchDense<ValueType>* a,      \
                             const matrix::BatchDense<ValueType>* b,      \
                             matrix::BatchDense<ValueType>* mtx)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                           \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL(ValueType);                    \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL(ValueType);                           \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL(ValueType);                           \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL(ValueType);                      \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_ADD_SCALE_KERNEL(ValueType);                       \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_ADD_SCALED_KERNEL(ValueType);          \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL(ValueType);                 \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL(ValueType);                     \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_DOT_KERNEL(ValueType);         \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_DOT_KERNEL(ValueType);         \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL(ValueType);                   \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_NORM2_KERNEL(ValueType);       \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_COPY_KERNEL(ValueType);                            \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COPY_KERNEL(ValueType);                \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_BATCH_SCALE_KERNEL(ValueType);                     \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_BATCH_DENSE_CONVERT_TO_BATCH_CSR_KERNEL(ValueType, IndexType); \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL(ValueType);                  \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(ValueType);       \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL(ValueType);      \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL(ValueType);            \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL(ValueType);                       \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL(ValueType);                  \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_BATCH_SCALE_KERNEL(ValueType);                     \
    template <typename ValueType>                                              \
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_IDENTITY_KERNEL(ValueType)


namespace omp {
namespace batch_dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_dense
}  // namespace omp


namespace cuda {
namespace batch_dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_dense
}  // namespace cuda


namespace reference {
namespace batch_dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_dense
}  // namespace reference


namespace hip {
namespace batch_dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_dense
}  // namespace hip


namespace dpcpp {
namespace batch_dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_dense
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BATCH_DENSE_KERNELS_HPP_
