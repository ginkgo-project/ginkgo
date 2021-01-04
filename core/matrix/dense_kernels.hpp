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

#ifndef GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
#define GKO_CORE_MATRIX_DENSE_KERNELS_HPP_


#include <ginkgo/core/matrix/dense.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(_type)               \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec, \
                      const matrix::Dense<_type> *a,               \
                      const matrix::Dense<_type> *b, matrix::Dense<_type> *c)

#define GKO_DECLARE_DENSE_APPLY_KERNEL(_type)                                \
    void apply(std::shared_ptr<const DefaultExecutor> exec,                  \
               const matrix::Dense<_type> *alpha,                            \
               const matrix::Dense<_type> *a, const matrix::Dense<_type> *b, \
               const matrix::Dense<_type> *beta, matrix::Dense<_type> *c)

#define GKO_DECLARE_DENSE_SCALE_KERNEL(_type)               \
    void scale(std::shared_ptr<const DefaultExecutor> exec, \
               const matrix::Dense<_type> *alpha, matrix::Dense<_type> *x)

#define GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(_type)               \
    void add_scaled(std::shared_ptr<const DefaultExecutor> exec, \
                    const matrix::Dense<_type> *alpha,           \
                    const matrix::Dense<_type> *x, matrix::Dense<_type> *y)

#define GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL(_type)               \
    void add_scaled_diag(std::shared_ptr<const DefaultExecutor> exec, \
                         const matrix::Dense<_type> *alpha,           \
                         const matrix::Diagonal<_type> *x,            \
                         matrix::Dense<_type> *y)

#define GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(_type)               \
    void compute_dot(std::shared_ptr<const DefaultExecutor> exec, \
                     const matrix::Dense<_type> *x,               \
                     const matrix::Dense<_type> *y,               \
                     matrix::Dense<_type> *result)

#define GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL(_type)               \
    void compute_norm2(std::shared_ptr<const DefaultExecutor> exec, \
                       const matrix::Dense<_type> *x,               \
                       matrix::Dense<remove_complex<_type>> *result)

#define GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL(_type, _prec)        \
    void convert_to_coo(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Dense<_type> *source,          \
                        matrix::Coo<_type, _prec> *other)

#define GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL(_type, _prec)        \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Dense<_type> *source,          \
                        matrix::Csr<_type, _prec> *other)

#define GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL(_type, _prec)        \
    void convert_to_ell(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Dense<_type> *source,          \
                        matrix::Ell<_type, _prec> *other)

#define GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL(_type, _prec)        \
    void convert_to_hybrid(std::shared_ptr<const DefaultExecutor> exec, \
                           const matrix::Dense<_type> *source,          \
                           matrix::Hybrid<_type, _prec> *other)

#define GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL(_type, _prec)        \
    void convert_to_sellp(std::shared_ptr<const DefaultExecutor> exec, \
                          const matrix::Dense<_type> *source,          \
                          matrix::Sellp<_type, _prec> *other)

#define GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL(_type, _prec)        \
    void convert_to_sparsity_csr(std::shared_ptr<const DefaultExecutor> exec, \
                                 const matrix::Dense<_type> *source,          \
                                 matrix::SparsityCsr<_type, _prec> *other)

#define GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL(_type)               \
    void count_nonzeros(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Dense<_type> *source, size_type *result)

#define GKO_DECLARE_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(_type) \
    void calculate_max_nnz_per_row(                               \
        std::shared_ptr<const DefaultExecutor> exec,              \
        const matrix::Dense<_type> *source, size_type *result)

#define GKO_DECLARE_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL(_type) \
    void calculate_nonzeros_per_row(                               \
        std::shared_ptr<const DefaultExecutor> exec,               \
        const matrix::Dense<_type> *source, Array<size_type> *result)

#define GKO_DECLARE_DENSE_CALCULATE_TOTAL_COLS_KERNEL(_type)               \
    void calculate_total_cols(std::shared_ptr<const DefaultExecutor> exec, \
                              const matrix::Dense<_type> *source,          \
                              size_type *result, size_type stride_factor,  \
                              size_type slice_size)

#define GKO_DECLARE_DENSE_TRANSPOSE_KERNEL(_type)               \
    void transpose(std::shared_ptr<const DefaultExecutor> exec, \
                   const matrix::Dense<_type> *orig,            \
                   matrix::Dense<_type> *trans)

#define GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL(_type)               \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Dense<_type> *orig,            \
                        matrix::Dense<_type> *trans)

#define GKO_DECLARE_DENSE_ROW_GATHER_KERNEL(_vtype, _itype)      \
    void row_gather(std::shared_ptr<const DefaultExecutor> exec, \
                    const Array<_itype> *gather_indices,         \
                    const matrix::Dense<_vtype> *orig,           \
                    matrix::Dense<_vtype> *row_gathered)

#define GKO_DECLARE_DENSE_COLUMN_PERMUTE_KERNEL(_vtype, _itype)      \
    void column_permute(std::shared_ptr<const DefaultExecutor> exec, \
                        const Array<_itype> *permutation_indices,    \
                        const matrix::Dense<_vtype> *orig,           \
                        matrix::Dense<_vtype> *column_permuted)

#define GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL(_vtype, _itype)          \
    void inverse_row_permute(std::shared_ptr<const DefaultExecutor> exec, \
                             const Array<_itype> *permutation_indices,    \
                             const matrix::Dense<_vtype> *orig,           \
                             matrix::Dense<_vtype> *row_permuted)

#define GKO_DECLARE_DENSE_INV_COLUMN_PERMUTE_KERNEL(_vtype, _itype)          \
    void inverse_column_permute(std::shared_ptr<const DefaultExecutor> exec, \
                                const Array<_itype> *permutation_indices,    \
                                const matrix::Dense<_vtype> *orig,           \
                                matrix::Dense<_vtype> *column_permuted)

#define GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL(_vtype)              \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec, \
                          const matrix::Dense<_vtype> *orig,           \
                          matrix::Diagonal<_vtype> *diag)

#define GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL(_vtype)                    \
    void inplace_absolute_dense(std::shared_ptr<const DefaultExecutor> exec, \
                                matrix::Dense<_vtype> *source)

#define GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL(_vtype) \
    void outplace_absolute_dense(                          \
        std::shared_ptr<const DefaultExecutor> exec,       \
        const matrix::Dense<_vtype> *source,               \
        matrix::Dense<remove_complex<_vtype>> *result)

#define GKO_DECLARE_MAKE_COMPLEX_KERNEL(_vtype)                    \
    void make_complex(std::shared_ptr<const DefaultExecutor> exec, \
                      const matrix::Dense<_vtype> *source,         \
                      matrix::Dense<to_complex<_vtype>> *result)

#define GKO_DECLARE_GET_REAL_KERNEL(_vtype)                    \
    void get_real(std::shared_ptr<const DefaultExecutor> exec, \
                  const matrix::Dense<_vtype> *source,         \
                  matrix::Dense<remove_complex<_vtype>> *result)

#define GKO_DECLARE_GET_IMAG_KERNEL(_vtype)                    \
    void get_imag(std::shared_ptr<const DefaultExecutor> exec, \
                  const matrix::Dense<_vtype> *source,         \
                  matrix::Dense<remove_complex<_vtype>> *result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                        \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(ValueType);                       \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_APPLY_KERNEL(ValueType);                              \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_SCALE_KERNEL(ValueType);                              \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(ValueType);                         \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL(ValueType);                    \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(ValueType);                        \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL(ValueType);                      \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL(ValueType, IndexType); \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL(ValueType);                     \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(ValueType);          \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL(ValueType);         \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_CALCULATE_TOTAL_COLS_KERNEL(ValueType);               \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_TRANSPOSE_KERNEL(ValueType);                          \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL(ValueType);                     \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_ROW_GATHER_KERNEL(ValueType, IndexType);              \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_COLUMN_PERMUTE_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_INV_COLUMN_PERMUTE_KERNEL(ValueType, IndexType);      \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL(ValueType);                   \
    template <typename ValueType>                                           \
    GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL(ValueType);                   \
    template <typename ValueType>                                           \
    GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL(ValueType);                  \
    template <typename ValueType>                                           \
    GKO_DECLARE_MAKE_COMPLEX_KERNEL(ValueType);                             \
    template <typename ValueType>                                           \
    GKO_DECLARE_GET_REAL_KERNEL(ValueType);                                 \
    template <typename ValueType>                                           \
    GKO_DECLARE_GET_IMAG_KERNEL(ValueType)


namespace omp {
namespace dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace omp


namespace cuda {
namespace dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace cuda


namespace reference {
namespace dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace reference


namespace hip {
namespace dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace hip


namespace dpcpp {
namespace dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
