// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
#define GKO_CORE_MATRIX_DENSE_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/device_views.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(ValueType)           \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec, \
                      matrix::view::dense<const ValueType> a,      \
                      matrix::view::dense<const ValueType> b,      \
                      matrix::view::dense<ValueType> c)

#define GKO_DECLARE_DENSE_APPLY_KERNEL(ValueType)           \
    void apply(std::shared_ptr<const DefaultExecutor> exec, \
               matrix::view::dense<const ValueType> alpha,  \
               matrix::view::dense<const ValueType> a,      \
               matrix::view::dense<const ValueType> b,      \
               matrix::view::dense<const ValueType> beta,   \
               matrix::view::dense<ValueType> c)

#define GKO_DECLARE_DENSE_COPY_KERNEL(InValueType, OutValueType) \
    void copy(std::shared_ptr<const DefaultExecutor> exec,       \
              matrix::view::dense<const InValueType> input,      \
              matrix::view::dense<OutValueType> output)

#define GKO_DECLARE_DENSE_FILL_KERNEL(ValueType)           \
    void fill(std::shared_ptr<const DefaultExecutor> exec, \
              matrix::view::dense<ValueType> mat, ValueType value)

#define GKO_DECLARE_DENSE_SCALE_KERNEL(ValueType, ScalarType) \
    void scale(std::shared_ptr<const DefaultExecutor> exec,   \
               matrix::view::dense<const ScalarType> alpha,   \
               matrix::view::dense<ValueType> x)

#define GKO_DECLARE_DENSE_INV_SCALE_KERNEL(ValueType, ScalarType) \
    void inv_scale(std::shared_ptr<const DefaultExecutor> exec,   \
                   matrix::view::dense<const ScalarType> alpha,   \
                   matrix::view::dense<ValueType> x)

#define GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(ValueType, ScalarType) \
    void add_scaled(std::shared_ptr<const DefaultExecutor> exec,   \
                    matrix::view::dense<const ScalarType> alpha,   \
                    matrix::view::dense<const ValueType> x,        \
                    matrix::view::dense<ValueType> y)

#define GKO_DECLARE_DENSE_SUB_SCALED_KERNEL(ValueType, ScalarType) \
    void sub_scaled(std::shared_ptr<const DefaultExecutor> exec,   \
                    matrix::view::dense<const ScalarType> alpha,   \
                    matrix::view::dense<const ValueType> x,        \
                    matrix::view::dense<ValueType> y)

#define GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL(ValueType)           \
    void add_scaled_diag(std::shared_ptr<const DefaultExecutor> exec, \
                         matrix::view::dense<const ValueType> alpha,  \
                         const matrix::Diagonal<ValueType>* x,        \
                         matrix::view::dense<ValueType> y)

#define GKO_DECLARE_DENSE_SUB_SCALED_DIAG_KERNEL(ValueType)           \
    void sub_scaled_diag(std::shared_ptr<const DefaultExecutor> exec, \
                         matrix::view::dense<const ValueType> alpha,  \
                         const matrix::Diagonal<ValueType>* x,        \
                         matrix::view::dense<ValueType> y)

#define GKO_DECLARE_DENSE_COMPUTE_DOT_DISPATCH_KERNEL(ValueType)           \
    void compute_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec, \
                              matrix::view::dense<const ValueType> x,      \
                              matrix::view::dense<const ValueType> y,      \
                              matrix::view::dense<ValueType> result,       \
                              array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(ValueType)           \
    void compute_dot(std::shared_ptr<const DefaultExecutor> exec, \
                     matrix::view::dense<const ValueType> x,      \
                     matrix::view::dense<const ValueType> y,      \
                     matrix::view::dense<ValueType> result, array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_DISPATCH_KERNEL(ValueType) \
    void compute_conj_dot_dispatch(                                   \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        matrix::view::dense<const ValueType> x,                       \
        matrix::view::dense<const ValueType> y,                       \
        matrix::view::dense<ValueType> result, array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_KERNEL(ValueType)           \
    void compute_conj_dot(std::shared_ptr<const DefaultExecutor> exec, \
                          matrix::view::dense<const ValueType> x,      \
                          matrix::view::dense<const ValueType> y,      \
                          matrix::view::dense<ValueType> result,       \
                          array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL(ValueType)                     \
    void compute_norm2(std::shared_ptr<const DefaultExecutor> exec,           \
                       matrix::view::dense<const ValueType> x,                \
                       matrix::view::dense<remove_complex<ValueType>> result, \
                       array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_NORM2_DISPATCH_KERNEL(ValueType) \
    void compute_norm2_dispatch(                                   \
        std::shared_ptr<const DefaultExecutor> exec,               \
        matrix::view::dense<const ValueType> x,                    \
        matrix::view::dense<remove_complex<ValueType>> result,     \
        array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_NORM1_KERNEL(ValueType)                     \
    void compute_norm1(std::shared_ptr<const DefaultExecutor> exec,           \
                       matrix::view::dense<const ValueType> x,                \
                       matrix::view::dense<remove_complex<ValueType>> result, \
                       array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_MEAN_KERNEL(ValueType)           \
    void compute_mean(std::shared_ptr<const DefaultExecutor> exec, \
                      matrix::view::dense<const ValueType> x,      \
                      matrix::view::dense<ValueType> result, array<char>& tmp)

#define GKO_DECLARE_DENSE_FILL_IN_MATRIX_DATA_KERNEL(ValueType, _prec)         \
    void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,      \
                             const device_matrix_data<ValueType, _prec>& data, \
                             matrix::view::dense<ValueType> output)

#define GKO_DECLARE_DENSE_COMPUTE_SQUARED_NORM2_KERNEL(ValueType) \
    void compute_squared_norm2(                                   \
        std::shared_ptr<const DefaultExecutor> exec,              \
        matrix::view::dense<const ValueType> x,                   \
        matrix::view::dense<remove_complex<ValueType>> result,    \
        array<char>& tmp)

#define GKO_DECLARE_DENSE_COMPUTE_SQRT_KERNEL(ValueType)           \
    void compute_sqrt(std::shared_ptr<const DefaultExecutor> exec, \
                      matrix::view::dense<ValueType> data)

#define GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL(ValueType, IndexType) \
    void convert_to_coo(std::shared_ptr<const DefaultExecutor> exec,  \
                        matrix::view::dense<const ValueType> source,  \
                        const int64* row_ptrs,                        \
                        matrix::view::coo<ValueType, IndexType> other)

#define GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL(ValueType, IndexType) \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,  \
                        matrix::view::dense<const ValueType> source,  \
                        matrix::Csr<ValueType, IndexType>* other)

#define GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL(ValueType, IndexType) \
    void convert_to_ell(std::shared_ptr<const DefaultExecutor> exec,  \
                        matrix::view::dense<const ValueType> source,  \
                        matrix::view::ell<ValueType, IndexType> other)

#define GKO_DECLARE_DENSE_CONVERT_TO_FBCSR_KERNEL(ValueType, IndexType) \
    void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,  \
                          matrix::view::dense<const ValueType> source,  \
                          matrix::Fbcsr<ValueType, IndexType>* other)

#define GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL(ValueType, IndexType) \
    void convert_to_hybrid(std::shared_ptr<const DefaultExecutor> exec,  \
                           matrix::view::dense<const ValueType> source,  \
                           const int64* coo_row_ptrs,                    \
                           matrix::view::hybrid<ValueType, IndexType> other)

#define GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL(ValueType, IndexType) \
    void convert_to_sellp(std::shared_ptr<const DefaultExecutor> exec,  \
                          matrix::view::dense<const ValueType> source,  \
                          matrix::view::sellp<ValueType, IndexType> other)

#define GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL(ValueType, IndexType) \
    void convert_to_sparsity_csr(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        matrix::view::dense<const ValueType> source,                           \
        matrix::SparsityCsr<ValueType, IndexType>* other)

#define GKO_DECLARE_DENSE_COMPUTE_MAX_NNZ_PER_ROW_KERNEL(ValueType)           \
    void compute_max_nnz_per_row(std::shared_ptr<const DefaultExecutor> exec, \
                                 matrix::view::dense<const ValueType> source, \
                                 size_type& result)

#define GKO_DECLARE_DENSE_COMPUTE_SLICE_SETS_KERNEL(ValueType)             \
    void compute_slice_sets(std::shared_ptr<const DefaultExecutor> exec,   \
                            matrix::view::dense<const ValueType> source,   \
                            size_type slice_size, size_type stride_factor, \
                            size_type* slice_sets, size_type* slice_lengths)

#define GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL(ValueType, IndexType) \
    void count_nonzeros_per_row(std::shared_ptr<const DefaultExecutor> exec,  \
                                matrix::view::dense<const ValueType> source,  \
                                IndexType* result)

#define GKO_DECLARE_DENSE_COUNT_NONZERO_BLOCKS_PER_ROW_KERNEL(ValueType, \
                                                              IndexType) \
    void count_nonzero_blocks_per_row(                                   \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        matrix::view::dense<const ValueType> source, int block_size,     \
        IndexType* result)

#define GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL_SIZE_T(ValueType) \
    GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL(ValueType, ::gko::size_type)

#define GKO_DECLARE_DENSE_TRANSPOSE_KERNEL(ValueType)           \
    void transpose(std::shared_ptr<const DefaultExecutor> exec, \
                   matrix::view::dense<const ValueType> orig,   \
                   matrix::view::dense<ValueType> trans)

#define GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL(ValueType)           \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec, \
                        matrix::view::dense<const ValueType> orig,   \
                        matrix::view::dense<ValueType> trans)

#define GKO_DECLARE_DENSE_SYMM_SCALE_PERMUTE_KERNEL(ValueType, IndexType) \
    void symm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                            const ValueType* scale,                       \
                            const IndexType* permutation_indices,         \
                            matrix::view::dense<const ValueType> orig,    \
                            matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_ROW_SCALE_PERMUTE_KERNEL(ValueType, IndexType) \
    void row_scale_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                           const ValueType* scale,                       \
                           const IndexType* permutation_indices,         \
                           matrix::view::dense<const ValueType> orig,    \
                           matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_COL_SCALE_PERMUTE_KERNEL(ValueType, IndexType) \
    void col_scale_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                           const ValueType* scale,                       \
                           const IndexType* permutation_indices,         \
                           matrix::view::dense<const ValueType> orig,    \
                           matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_INV_SYMM_SCALE_PERMUTE_KERNEL(ValueType, IndexType) \
    void inv_symm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                                const ValueType* scale,                       \
                                const IndexType* permutation_indices,         \
                                matrix::view::dense<const ValueType> orig,    \
                                matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_INV_ROW_SCALE_PERMUTE_KERNEL(ValueType, IndexType) \
    void inv_row_scale_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                               const ValueType* scale,                       \
                               const IndexType* permutation_indices,         \
                               matrix::view::dense<const ValueType> orig,    \
                               matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_INV_COL_SCALE_PERMUTE_KERNEL(ValueType, IndexType) \
    void inv_col_scale_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                               const ValueType* scale,                       \
                               const IndexType* permutation_indices,         \
                               matrix::view::dense<const ValueType> orig,    \
                               matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_NONSYMM_SCALE_PERMUTE_KERNEL(ValueType, IndexType) \
    void nonsymm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                               const ValueType* row_scale,                   \
                               const IndexType* row_permutation_indices,     \
                               const ValueType* column_scale,                \
                               const IndexType* column_permutation_indices,  \
                               matrix::view::dense<const ValueType> orig,    \
                               matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_INV_NONSYMM_SCALE_PERMUTE_KERNEL(ValueType,         \
                                                           IndexType)         \
    void inv_nonsymm_scale_permute(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const ValueType* row_scale, const IndexType* row_permutation_indices, \
        const ValueType* column_scale,                                        \
        const IndexType* column_permutation_indices,                          \
        matrix::view::dense<const ValueType> orig,                            \
        matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_SYMM_PERMUTE_KERNEL(ValueType, IndexType) \
    void symm_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                      const IndexType* permutation_indices,         \
                      matrix::view::dense<const ValueType> orig,    \
                      matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_INV_SYMM_PERMUTE_KERNEL(ValueType, IndexType) \
    void inv_symm_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                          const IndexType* permutation_indices,         \
                          matrix::view::dense<const ValueType> orig,    \
                          matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_NONSYMM_PERMUTE_KERNEL(ValueType, IndexType) \
    void nonsymm_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                         const IndexType* row_permutation_indices,     \
                         const IndexType* column_permutation_indices,  \
                         matrix::view::dense<const ValueType> orig,    \
                         matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_INV_NONSYMM_PERMUTE_KERNEL(ValueType, IndexType) \
    void inv_nonsymm_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                             const IndexType* row_permutation_indices,     \
                             const IndexType* column_permutation_indices,  \
                             matrix::view::dense<const ValueType> orig,    \
                             matrix::view::dense<ValueType> permuted)

#define GKO_DECLARE_DENSE_ROW_GATHER_KERNEL(ValueType, OutputType, IndexType) \
    void row_gather(std::shared_ptr<const DefaultExecutor> exec,              \
                    const IndexType* gather_indices,                          \
                    matrix::view::dense<const ValueType> orig,                \
                    matrix::view::dense<OutputType> row_collection)

#define GKO_DECLARE_DENSE_ADVANCED_ROW_GATHER_KERNEL(ValueType, OutputType, \
                                                     IndexType)             \
    void advanced_row_gather(std::shared_ptr<const DefaultExecutor> exec,   \
                             matrix::view::dense<const ValueType> alpha,    \
                             const IndexType* gather_indices,               \
                             matrix::view::dense<const ValueType> orig,     \
                             matrix::view::dense<const ValueType> beta,     \
                             matrix::view::dense<OutputType> row_collection)

#define GKO_DECLARE_DENSE_COL_PERMUTE_KERNEL(ValueType, IndexType) \
    void col_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                     const IndexType* permutation_indices,         \
                     matrix::view::dense<const ValueType> orig,    \
                     matrix::view::dense<ValueType> col_permuted)

#define GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL(ValueType, IndexType) \
    void inv_row_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                         const IndexType* permutation_indices,         \
                         matrix::view::dense<const ValueType> orig,    \
                         matrix::view::dense<ValueType> row_permuted)

#define GKO_DECLARE_DENSE_INV_COL_PERMUTE_KERNEL(ValueType, IndexType) \
    void inv_col_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                         const IndexType* permutation_indices,         \
                         matrix::view::dense<const ValueType> orig,    \
                         matrix::view::dense<ValueType> col_permuted)

#define GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL(ValueType)           \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec, \
                          matrix::view::dense<const ValueType> orig,   \
                          matrix::Diagonal<ValueType>* diag)

#define GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL(ValueType)                 \
    void inplace_absolute_dense(std::shared_ptr<const DefaultExecutor> exec, \
                                matrix::view::dense<ValueType> source)

#define GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL(ValueType) \
    void outplace_absolute_dense(                             \
        std::shared_ptr<const DefaultExecutor> exec,          \
        matrix::view::dense<const ValueType> source,          \
        matrix::view::dense<remove_complex<ValueType>> result)

#define GKO_DECLARE_MAKE_COMPLEX_KERNEL(ValueType)                 \
    void make_complex(std::shared_ptr<const DefaultExecutor> exec, \
                      matrix::view::dense<const ValueType> source, \
                      matrix::view::dense<to_complex<ValueType>> result)

#define GKO_DECLARE_GET_REAL_KERNEL(ValueType)                 \
    void get_real(std::shared_ptr<const DefaultExecutor> exec, \
                  matrix::view::dense<const ValueType> source, \
                  matrix::view::dense<remove_complex<ValueType>> result)

#define GKO_DECLARE_GET_IMAG_KERNEL(ValueType)                 \
    void get_imag(std::shared_ptr<const DefaultExecutor> exec, \
                  matrix::view::dense<const ValueType> source, \
                  matrix::view::dense<remove_complex<ValueType>> result)

#define GKO_DECLARE_DENSE_ADD_SCALED_IDENTITY_KERNEL(ValueType, ScalarType) \
    void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec,   \
                             matrix::view::dense<const ScalarType> alpha,   \
                             matrix::view::dense<const ScalarType> beta,    \
                             matrix::view::dense<ValueType> mtx)


#define GKO_DECLARE_ALL_AS_TEMPLATES(_export_macro)                            \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(ValueType);            \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_APPLY_KERNEL(ValueType);                   \
    template <typename InValueType, typename OutValueType>                     \
    _export_macro GKO_DECLARE_DENSE_COPY_KERNEL(InValueType, OutValueType);    \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_FILL_KERNEL(ValueType);                    \
    template <typename ValueType, typename ScalarType>                         \
    _export_macro GKO_DECLARE_DENSE_SCALE_KERNEL(ValueType, ScalarType);       \
    template <typename ValueType, typename ScalarType>                         \
    _export_macro GKO_DECLARE_DENSE_INV_SCALE_KERNEL(ValueType, ScalarType);   \
    template <typename ValueType, typename ScalarType>                         \
    _export_macro GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(ValueType, ScalarType);  \
    template <typename ValueType, typename ScalarType>                         \
    _export_macro GKO_DECLARE_DENSE_SUB_SCALED_KERNEL(ValueType, ScalarType);  \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL(ValueType);         \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_SUB_SCALED_DIAG_KERNEL(ValueType);         \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(ValueType);             \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_DOT_DISPATCH_KERNEL(ValueType);    \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_KERNEL(ValueType);        \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_DISPATCH_KERNEL(          \
        ValueType);                                                            \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL(ValueType);           \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_NORM2_DISPATCH_KERNEL(ValueType);  \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_NORM1_KERNEL(ValueType);           \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_MEAN_KERNEL(ValueType);            \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_FILL_IN_MATRIX_DATA_KERNEL(ValueType,      \
                                                               IndexType);     \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_SQUARED_NORM2_KERNEL(ValueType);   \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_SQRT_KERNEL(ValueType);            \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL(ValueType,           \
                                                          IndexType);          \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL(ValueType,           \
                                                          IndexType);          \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL(ValueType,           \
                                                          IndexType);          \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_CONVERT_TO_FBCSR_KERNEL(ValueType,         \
                                                            IndexType);        \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL(ValueType,        \
                                                             IndexType);       \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL(ValueType,         \
                                                            IndexType);        \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL(ValueType,  \
                                                                   IndexType); \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_MAX_NNZ_PER_ROW_KERNEL(ValueType); \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_COMPUTE_SLICE_SETS_KERNEL(ValueType);      \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL(ValueType,   \
                                                                  IndexType);  \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_COUNT_NONZERO_BLOCKS_PER_ROW_KERNEL(       \
        ValueType, IndexType);                                                 \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_TRANSPOSE_KERNEL(ValueType);               \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL(ValueType);          \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_SYMM_PERMUTE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_INV_SYMM_PERMUTE_KERNEL(ValueType,         \
                                                            IndexType);        \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_NONSYMM_PERMUTE_KERNEL(ValueType,          \
                                                           IndexType);         \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_INV_NONSYMM_PERMUTE_KERNEL(ValueType,      \
                                                               IndexType);     \
    template <typename ValueType, typename OutputType, typename IndexType>     \
    _export_macro GKO_DECLARE_DENSE_ROW_GATHER_KERNEL(ValueType, OutputType,   \
                                                      IndexType);              \
    template <typename ValueType, typename OutputType, typename IndexType>     \
    _export_macro GKO_DECLARE_DENSE_ADVANCED_ROW_GATHER_KERNEL(                \
        ValueType, OutputType, IndexType);                                     \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_COL_PERMUTE_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL(ValueType,          \
                                                           IndexType);         \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_INV_COL_PERMUTE_KERNEL(ValueType,          \
                                                           IndexType);         \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_SYMM_SCALE_PERMUTE_KERNEL(ValueType,       \
                                                              IndexType);      \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_INV_SYMM_SCALE_PERMUTE_KERNEL(ValueType,   \
                                                                  IndexType);  \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_ROW_SCALE_PERMUTE_KERNEL(ValueType,        \
                                                             IndexType);       \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_COL_SCALE_PERMUTE_KERNEL(ValueType,        \
                                                             IndexType);       \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_INV_ROW_SCALE_PERMUTE_KERNEL(ValueType,    \
                                                                 IndexType);   \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_INV_COL_SCALE_PERMUTE_KERNEL(ValueType,    \
                                                                 IndexType);   \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_NONSYMM_SCALE_PERMUTE_KERNEL(ValueType,    \
                                                                 IndexType);   \
    template <typename ValueType, typename IndexType>                          \
    _export_macro GKO_DECLARE_DENSE_INV_NONSYMM_SCALE_PERMUTE_KERNEL(          \
        ValueType, IndexType);                                                 \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL(ValueType);        \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL(ValueType);        \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL(ValueType);       \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_MAKE_COMPLEX_KERNEL(ValueType);                  \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_GET_REAL_KERNEL(ValueType);                      \
    template <typename ValueType>                                              \
    _export_macro GKO_DECLARE_GET_IMAG_KERNEL(ValueType);                      \
    template <typename ValueType, typename ScalarType>                         \
    _export_macro GKO_DECLARE_DENSE_ADD_SCALED_IDENTITY_KERNEL(ValueType,      \
                                                               ScalarType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(dense, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
