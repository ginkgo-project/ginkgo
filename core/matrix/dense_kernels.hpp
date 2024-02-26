// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
#define GKO_CORE_MATRIX_DENSE_KERNELS_HPP_


#include <ginkgo/core/matrix/dense.hpp>


#include <memory>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(_type)               \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec, \
                      const matrix::Dense<_type>* a,               \
                      const matrix::Dense<_type>* b, matrix::Dense<_type>* c)

#define GKO_DECLARE_DENSE_APPLY_KERNEL(_type)                                \
    void apply(std::shared_ptr<const DefaultExecutor> exec,                  \
               const matrix::Dense<_type>* alpha,                            \
               const matrix::Dense<_type>* a, const matrix::Dense<_type>* b, \
               const matrix::Dense<_type>* beta, matrix::Dense<_type>* c)

#define GKO_DECLARE_DENSE_COPY_KERNEL(_intype, _outtype)   \
    void copy(std::shared_ptr<const DefaultExecutor> exec, \
              const matrix::Dense<_intype>* input,         \
              matrix::Dense<_outtype>* output)

#define GKO_DECLARE_DENSE_FILL_KERNEL(_type)               \
    void fill(std::shared_ptr<const DefaultExecutor> exec, \
              matrix::Dense<_type>* mat, _type value)

#define GKO_DECLARE_DENSE_SCALE_KERNEL(_type, _scalar_type) \
    void scale(std::shared_ptr<const DefaultExecutor> exec, \
               const matrix::Dense<_scalar_type>* alpha,    \
               matrix::Dense<_type>* x)

#define GKO_DECLARE_DENSE_INV_SCALE_KERNEL(_type, _scalar_type) \
    void inv_scale(std::shared_ptr<const DefaultExecutor> exec, \
                   const matrix::Dense<_scalar_type>* alpha,    \
                   matrix::Dense<_type>* x)

#define GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(_type, _scalar_type) \
    void add_scaled(std::shared_ptr<const DefaultExecutor> exec, \
                    const matrix::Dense<_scalar_type>* alpha,    \
                    const matrix::Dense<_type>* x, matrix::Dense<_type>* y)

#define GKO_DECLARE_DENSE_SUB_SCALED_KERNEL(_type, _scalar_type) \
    void sub_scaled(std::shared_ptr<const DefaultExecutor> exec, \
                    const matrix::Dense<_scalar_type>* alpha,    \
                    const matrix::Dense<_type>* x, matrix::Dense<_type>* y)

#define GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL(_type)               \
    void add_scaled_diag(std::shared_ptr<const DefaultExecutor> exec, \
                         const matrix::Dense<_type>* alpha,           \
                         const matrix::Diagonal<_type>* x,            \
                         matrix::Dense<_type>* y)

#define GKO_DECLARE_DENSE_SUB_SCALED_DIAG_KERNEL(_type)               \
    void sub_scaled_diag(std::shared_ptr<const DefaultExecutor> exec, \
                         const matrix::Dense<_type>* alpha,           \
                         const matrix::Diagonal<_type>* x,            \
                         matrix::Dense<_type>* y)

#define GKO_DECLARE_DENSE_COMPUTE_DOT_DISPATCH_KERNEL(_type)               \
    void compute_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec, \
                              const matrix::Dense<_type>* x,               \
                              const matrix::Dense<_type>* y,               \
                              matrix::Dense<_type>* result, array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(_type)               \
    void compute_dot(std::shared_ptr<const DefaultExecutor> exec, \
                     const matrix::Dense<_type>* x,               \
                     const matrix::Dense<_type>* y,               \
                     matrix::Dense<_type>* result, array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_DISPATCH_KERNEL(_type)     \
    void compute_conj_dot_dispatch(                                   \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        const matrix::Dense<_type>* x, const matrix::Dense<_type>* y, \
        matrix::Dense<_type>* result, array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_KERNEL(_type)               \
    void compute_conj_dot(std::shared_ptr<const DefaultExecutor> exec, \
                          const matrix::Dense<_type>* x,               \
                          const matrix::Dense<_type>* y,               \
                          matrix::Dense<_type>* result, array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL(_type)                \
    void compute_norm2(std::shared_ptr<const DefaultExecutor> exec,  \
                       const matrix::Dense<_type>* x,                \
                       matrix::Dense<remove_complex<_type>>* result, \
                       array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_NORM2_DISPATCH_KERNEL(_type)                \
    void compute_norm2_dispatch(std::shared_ptr<const DefaultExecutor> exec,  \
                                const matrix::Dense<_type>* x,                \
                                matrix::Dense<remove_complex<_type>>* result, \
                                array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_NORM1_KERNEL(_type)                \
    void compute_norm1(std::shared_ptr<const DefaultExecutor> exec,  \
                       const matrix::Dense<_type>* x,                \
                       matrix::Dense<remove_complex<_type>>* result, \
                       array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_MEAN_KERNEL(_type)               \
    void compute_mean(std::shared_ptr<const DefaultExecutor> exec, \
                      const matrix::Dense<_type>* x,               \
                      matrix::Dense<_type>* result, array<char>& tmp)

#define GKO_DECLARE_DENSE_FILL_IN_MATRIX_DATA_KERNEL(_type, _prec)         \
    void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,  \
                             const device_matrix_data<_type, _prec>& data, \
                             matrix::Dense<_type>* output)

#define GKO_DECLARE_DENSE_COMPUTE_SQUARED_NORM2_KERNEL(_type)                \
    void compute_squared_norm2(std::shared_ptr<const DefaultExecutor> exec,  \
                               const matrix::Dense<_type>* x,                \
                               matrix::Dense<remove_complex<_type>>* result, \
                               array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_SQRT_KERNEL(_type)               \
    void compute_sqrt(std::shared_ptr<const DefaultExecutor> exec, \
                      matrix::Dense<_type>* data)

#define GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL(_type, _prec)        \
    void convert_to_coo(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Dense<_type>* source,          \
                        const int64* row_ptrs,                       \
                        matrix::Coo<_type, _prec>* other)

#define GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL(_type, _prec)        \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Dense<_type>* source,          \
                        matrix::Csr<_type, _prec>* other)

#define GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL(_type, _prec)        \
    void convert_to_ell(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Dense<_type>* source,          \
                        matrix::Ell<_type, _prec>* other)

#define GKO_DECLARE_DENSE_CONVERT_TO_FBCSR_KERNEL(_type, _prec)        \
    void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec, \
                          const matrix::Dense<_type>* source,          \
                          matrix::Fbcsr<_type, _prec>* other)

#define GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL(_type, _prec)        \
    void convert_to_hybrid(std::shared_ptr<const DefaultExecutor> exec, \
                           const matrix::Dense<_type>* source,          \
                           const int64* coo_row_ptrs,                   \
                           matrix::Hybrid<_type, _prec>* other)

#define GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL(_type, _prec)        \
    void convert_to_sellp(std::shared_ptr<const DefaultExecutor> exec, \
                          const matrix::Dense<_type>* source,          \
                          matrix::Sellp<_type, _prec>* other)

#define GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL(_type, _prec)        \
    void convert_to_sparsity_csr(std::shared_ptr<const DefaultExecutor> exec, \
                                 const matrix::Dense<_type>* source,          \
                                 matrix::SparsityCsr<_type, _prec>* other)

#define GKO_DECLARE_DENSE_COMPUTE_MAX_NNZ_PER_ROW_KERNEL(_type)               \
    void compute_max_nnz_per_row(std::shared_ptr<const DefaultExecutor> exec, \
                                 const matrix::Dense<_type>* source,          \
                                 size_type& result)

#define GKO_DECLARE_DENSE_COMPUTE_SLICE_SETS_KERNEL(_type)                 \
    void compute_slice_sets(std::shared_ptr<const DefaultExecutor> exec,   \
                            const matrix::Dense<_type>* source,            \
                            size_type slice_size, size_type stride_factor, \
                            size_type* slice_sets, size_type* slice_lengths)

#define GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL(_vtype, _itype)      \
    void count_nonzeros_per_row(std::shared_ptr<const DefaultExecutor> exec, \
                                const matrix::Dense<_vtype>* source,         \
                                _itype* result)

#define GKO_DECLARE_DENSE_COUNT_NONZERO_BLOCKS_PER_ROW_KERNEL(_vtype, _itype) \
    void count_nonzero_blocks_per_row(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::Dense<_vtype>* source, int block_size, _itype* result)

#define GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL_SIZE_T(_type) \
    GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL(_type, ::gko::size_type)

#define GKO_DECLARE_DENSE_TRANSPOSE_KERNEL(_type)               \
    void transpose(std::shared_ptr<const DefaultExecutor> exec, \
                   const matrix::Dense<_type>* orig,            \
                   matrix::Dense<_type>* trans)

#define GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL(_type)               \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Dense<_type>* orig,            \
                        matrix::Dense<_type>* trans)

#define GKO_DECLARE_DENSE_SYMM_SCALE_PERMUTE_KERNEL(_vtype, _itype)           \
    void symm_scale_permute(                                                  \
        std::shared_ptr<const DefaultExecutor> exec, const _vtype* scale,     \
        const _itype* permutation_indices, const matrix::Dense<_vtype>* orig, \
        matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_ROW_SCALE_PERMUTE_KERNEL(_vtype, _itype)            \
    void row_scale_permute(                                                   \
        std::shared_ptr<const DefaultExecutor> exec, const _vtype* scale,     \
        const _itype* permutation_indices, const matrix::Dense<_vtype>* orig, \
        matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_COL_SCALE_PERMUTE_KERNEL(_vtype, _itype)            \
    void col_scale_permute(                                                   \
        std::shared_ptr<const DefaultExecutor> exec, const _vtype* scale,     \
        const _itype* permutation_indices, const matrix::Dense<_vtype>* orig, \
        matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_INV_SYMM_SCALE_PERMUTE_KERNEL(_vtype, _itype)       \
    void inv_symm_scale_permute(                                              \
        std::shared_ptr<const DefaultExecutor> exec, const _vtype* scale,     \
        const _itype* permutation_indices, const matrix::Dense<_vtype>* orig, \
        matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_INV_ROW_SCALE_PERMUTE_KERNEL(_vtype, _itype)        \
    void inv_row_scale_permute(                                               \
        std::shared_ptr<const DefaultExecutor> exec, const _vtype* scale,     \
        const _itype* permutation_indices, const matrix::Dense<_vtype>* orig, \
        matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_INV_COL_SCALE_PERMUTE_KERNEL(_vtype, _itype)        \
    void inv_col_scale_permute(                                               \
        std::shared_ptr<const DefaultExecutor> exec, const _vtype* scale,     \
        const _itype* permutation_indices, const matrix::Dense<_vtype>* orig, \
        matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_NONSYMM_SCALE_PERMUTE_KERNEL(_vtype, _itype)        \
    void nonsymm_scale_permute(                                               \
        std::shared_ptr<const DefaultExecutor> exec, const _vtype* row_scale, \
        const _itype* row_permutation_indices, const _vtype* column_scale,    \
        const _itype* column_permutation_indices,                             \
        const matrix::Dense<_vtype>* orig, matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_INV_NONSYMM_SCALE_PERMUTE_KERNEL(_vtype, _itype)    \
    void inv_nonsymm_scale_permute(                                           \
        std::shared_ptr<const DefaultExecutor> exec, const _vtype* row_scale, \
        const _itype* row_permutation_indices, const _vtype* column_scale,    \
        const _itype* column_permutation_indices,                             \
        const matrix::Dense<_vtype>* orig, matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_SYMM_PERMUTE_KERNEL(_vtype, _itype)      \
    void symm_permute(std::shared_ptr<const DefaultExecutor> exec, \
                      const _itype* permutation_indices,           \
                      const matrix::Dense<_vtype>* orig,           \
                      matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_INV_SYMM_PERMUTE_KERNEL(_vtype, _itype)      \
    void inv_symm_permute(std::shared_ptr<const DefaultExecutor> exec, \
                          const _itype* permutation_indices,           \
                          const matrix::Dense<_vtype>* orig,           \
                          matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_NONSYMM_PERMUTE_KERNEL(_vtype, _itype)      \
    void nonsymm_permute(std::shared_ptr<const DefaultExecutor> exec, \
                         const _itype* row_permutation_indices,       \
                         const _itype* column_permutation_indices,    \
                         const matrix::Dense<_vtype>* orig,           \
                         matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_INV_NONSYMM_PERMUTE_KERNEL(_vtype, _itype)      \
    void inv_nonsymm_permute(std::shared_ptr<const DefaultExecutor> exec, \
                             const _itype* row_permutation_indices,       \
                             const _itype* column_permutation_indices,    \
                             const matrix::Dense<_vtype>* orig,           \
                             matrix::Dense<_vtype>* permuted)

#define GKO_DECLARE_DENSE_ROW_GATHER_KERNEL(_vtype, _otype, _itype) \
    void row_gather(std::shared_ptr<const DefaultExecutor> exec,    \
                    const _itype* gather_indices,                   \
                    const matrix::Dense<_vtype>* orig,              \
                    matrix::Dense<_otype>* row_collection)

#define GKO_DECLARE_DENSE_ADVANCED_ROW_GATHER_KERNEL(_vtype, _otype, _itype)  \
    void advanced_row_gather(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::Dense<_vtype>* alpha, const _itype* gather_indices,     \
        const matrix::Dense<_vtype>* orig, const matrix::Dense<_vtype>* beta, \
        matrix::Dense<_otype>* row_collection)

#define GKO_DECLARE_DENSE_COL_PERMUTE_KERNEL(_vtype, _itype)      \
    void col_permute(std::shared_ptr<const DefaultExecutor> exec, \
                     const _itype* permutation_indices,           \
                     const matrix::Dense<_vtype>* orig,           \
                     matrix::Dense<_vtype>* col_permuted)

#define GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL(_vtype, _itype)      \
    void inv_row_permute(std::shared_ptr<const DefaultExecutor> exec, \
                         const _itype* permutation_indices,           \
                         const matrix::Dense<_vtype>* orig,           \
                         matrix::Dense<_vtype>* row_permuted)

#define GKO_DECLARE_DENSE_INV_COL_PERMUTE_KERNEL(_vtype, _itype)      \
    void inv_col_permute(std::shared_ptr<const DefaultExecutor> exec, \
                         const _itype* permutation_indices,           \
                         const matrix::Dense<_vtype>* orig,           \
                         matrix::Dense<_vtype>* col_permuted)

#define GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL(_vtype)              \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec, \
                          const matrix::Dense<_vtype>* orig,           \
                          matrix::Diagonal<_vtype>* diag)

#define GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL(_vtype)                    \
    void inplace_absolute_dense(std::shared_ptr<const DefaultExecutor> exec, \
                                matrix::Dense<_vtype>* source)

#define GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL(_vtype) \
    void outplace_absolute_dense(                          \
        std::shared_ptr<const DefaultExecutor> exec,       \
        const matrix::Dense<_vtype>* source,               \
        matrix::Dense<remove_complex<_vtype>>* result)

#define GKO_DECLARE_MAKE_COMPLEX_KERNEL(_vtype)                    \
    void make_complex(std::shared_ptr<const DefaultExecutor> exec, \
                      const matrix::Dense<_vtype>* source,         \
                      matrix::Dense<to_complex<_vtype>>* result)

#define GKO_DECLARE_GET_REAL_KERNEL(_vtype)                    \
    void get_real(std::shared_ptr<const DefaultExecutor> exec, \
                  const matrix::Dense<_vtype>* source,         \
                  matrix::Dense<remove_complex<_vtype>>* result)

#define GKO_DECLARE_GET_IMAG_KERNEL(_vtype)                    \
    void get_imag(std::shared_ptr<const DefaultExecutor> exec, \
                  const matrix::Dense<_vtype>* source,         \
                  matrix::Dense<remove_complex<_vtype>>* result)

#define GKO_DECLARE_DENSE_ADD_SCALED_IDENTITY_KERNEL(_vtype, _scalar_type) \
    void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec,  \
                             const matrix::Dense<_scalar_type>* alpha,     \
                             const matrix::Dense<_scalar_type>* beta,      \
                             matrix::Dense<_vtype>* mtx)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                          \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(ValueType);                         \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_APPLY_KERNEL(ValueType);                                \
    template <typename InValueType, typename OutValueType>                    \
    GKO_DECLARE_DENSE_COPY_KERNEL(InValueType, OutValueType);                 \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_FILL_KERNEL(ValueType);                                 \
    template <typename ValueType, typename ScalarType>                        \
    GKO_DECLARE_DENSE_SCALE_KERNEL(ValueType, ScalarType);                    \
    template <typename ValueType, typename ScalarType>                        \
    GKO_DECLARE_DENSE_INV_SCALE_KERNEL(ValueType, ScalarType);                \
    template <typename ValueType, typename ScalarType>                        \
    GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(ValueType, ScalarType);               \
    template <typename ValueType, typename ScalarType>                        \
    GKO_DECLARE_DENSE_SUB_SCALED_KERNEL(ValueType, ScalarType);               \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL(ValueType);                      \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_SUB_SCALED_DIAG_KERNEL(ValueType);                      \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(ValueType);                          \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_DOT_DISPATCH_KERNEL(ValueType);                 \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_KERNEL(ValueType);                     \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_DISPATCH_KERNEL(ValueType);            \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL(ValueType);                        \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_NORM2_DISPATCH_KERNEL(ValueType);               \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_NORM1_KERNEL(ValueType);                        \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_MEAN_KERNEL(ValueType);                         \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType);       \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_SQUARED_NORM2_KERNEL(ValueType);                \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_SQRT_KERNEL(ValueType);                         \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL(ValueType, IndexType);            \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL(ValueType, IndexType);            \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL(ValueType, IndexType);            \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_CONVERT_TO_FBCSR_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL(ValueType, IndexType);   \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_MAX_NNZ_PER_ROW_KERNEL(ValueType);              \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_COMPUTE_SLICE_SETS_KERNEL(ValueType);                   \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_COUNT_NONZERO_BLOCKS_PER_ROW_KERNEL(ValueType,          \
                                                          IndexType);         \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_TRANSPOSE_KERNEL(ValueType);                            \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL(ValueType);                       \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_SYMM_PERMUTE_KERNEL(ValueType, IndexType);              \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_INV_SYMM_PERMUTE_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_NONSYMM_PERMUTE_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_INV_NONSYMM_PERMUTE_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename OutputType, typename IndexType>    \
    GKO_DECLARE_DENSE_ROW_GATHER_KERNEL(ValueType, OutputType, IndexType);    \
    template <typename ValueType, typename OutputType, typename IndexType>    \
    GKO_DECLARE_DENSE_ADVANCED_ROW_GATHER_KERNEL(ValueType, OutputType,       \
                                                 IndexType);                  \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_COL_PERMUTE_KERNEL(ValueType, IndexType);               \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_INV_COL_PERMUTE_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_SYMM_SCALE_PERMUTE_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_INV_SYMM_SCALE_PERMUTE_KERNEL(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_ROW_SCALE_PERMUTE_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_COL_SCALE_PERMUTE_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_INV_ROW_SCALE_PERMUTE_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_INV_COL_SCALE_PERMUTE_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_NONSYMM_SCALE_PERMUTE_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DENSE_INV_NONSYMM_SCALE_PERMUTE_KERNEL(ValueType, IndexType); \
    template <typename ValueType>                                             \
    GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL(ValueType);                     \
    template <typename ValueType>                                             \
    GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL(ValueType);                     \
    template <typename ValueType>                                             \
    GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL(ValueType);                    \
    template <typename ValueType>                                             \
    GKO_DECLARE_MAKE_COMPLEX_KERNEL(ValueType);                               \
    template <typename ValueType>                                             \
    GKO_DECLARE_GET_REAL_KERNEL(ValueType);                                   \
    template <typename ValueType>                                             \
    GKO_DECLARE_GET_IMAG_KERNEL(ValueType);                                   \
    template <typename ValueType, typename ScalarType>                        \
    GKO_DECLARE_DENSE_ADD_SCALED_IDENTITY_KERNEL(ValueType, ScalarType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(dense, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
