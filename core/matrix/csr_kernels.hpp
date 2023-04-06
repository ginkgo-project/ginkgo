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

#ifndef GKO_CORE_MATRIX_CSR_KERNELS_HPP_
#define GKO_CORE_MATRIX_CSR_KERNELS_HPP_


#include <ginkgo/core/matrix/csr.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/kernel_declaration.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_CSR_SPMV_KERNEL(MatrixValueType, InputValueType, \
                                    OutputValueType, IndexType)      \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,           \
              const matrix::Csr<MatrixValueType, IndexType>* a,      \
              const matrix::Dense<InputValueType>* b,                \
              matrix::Dense<OutputValueType>* c)

#define GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL(MatrixValueType, InputValueType, \
                                             OutputValueType, IndexType)      \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,           \
                       const matrix::Dense<MatrixValueType>* alpha,           \
                       const matrix::Csr<MatrixValueType, IndexType>* a,      \
                       const matrix::Dense<InputValueType>* b,                \
                       const matrix::Dense<OutputValueType>* beta,            \
                       matrix::Dense<OutputValueType>* c)

#define GKO_DECLARE_CSR_SPGEMM_KERNEL(ValueType, IndexType)  \
    void spgemm(std::shared_ptr<const DefaultExecutor> exec, \
                const matrix::Csr<ValueType, IndexType>* a,  \
                const matrix::Csr<ValueType, IndexType>* b,  \
                matrix::Csr<ValueType, IndexType>* c)

#define GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL(ValueType, IndexType)  \
    void advanced_spgemm(std::shared_ptr<const DefaultExecutor> exec, \
                         const matrix::Dense<ValueType>* alpha,       \
                         const matrix::Csr<ValueType, IndexType>* a,  \
                         const matrix::Csr<ValueType, IndexType>* b,  \
                         const matrix::Dense<ValueType>* beta,        \
                         const matrix::Csr<ValueType, IndexType>* d,  \
                         matrix::Csr<ValueType, IndexType>* c)

#define GKO_DECLARE_CSR_SPGEAM_KERNEL(ValueType, IndexType)  \
    void spgeam(std::shared_ptr<const DefaultExecutor> exec, \
                const matrix::Dense<ValueType>* alpha,       \
                const matrix::Csr<ValueType, IndexType>* a,  \
                const matrix::Dense<ValueType>* beta,        \
                const matrix::Csr<ValueType, IndexType>* b,  \
                matrix::Csr<ValueType, IndexType>* c)

#define GKO_DECLARE_CSR_FILL_IN_DENSE_KERNEL(ValueType, IndexType)      \
    void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,     \
                       const matrix::Csr<ValueType, IndexType>* source, \
                       matrix::Dense<ValueType>* result)

#define GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL(ValueType, IndexType)      \
    void convert_to_ell(std::shared_ptr<const DefaultExecutor> exec,     \
                        const matrix::Csr<ValueType, IndexType>* source, \
                        matrix::Ell<ValueType, IndexType>* result)

#define GKO_DECLARE_CSR_CONVERT_TO_FBCSR_KERNEL(ValueType, IndexType)      \
    void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,     \
                          const matrix::Csr<ValueType, IndexType>* source, \
                          int block_size, array<IndexType>& row_ptrs,      \
                          array<IndexType>& col_idxs,                      \
                          array<ValueType>& values)

#define GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL(ValueType, IndexType)      \
    void convert_to_hybrid(std::shared_ptr<const DefaultExecutor> exec,     \
                           const matrix::Csr<ValueType, IndexType>* source, \
                           const int64* coo_row_ptrs,                       \
                           matrix::Hybrid<ValueType, IndexType>* result)

#define GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL(ValueType, IndexType)      \
    void convert_to_sellp(std::shared_ptr<const DefaultExecutor> exec,     \
                          const matrix::Csr<ValueType, IndexType>* source, \
                          matrix::Sellp<ValueType, IndexType>* result)

#define GKO_DECLARE_CSR_TRANSPOSE_KERNEL(ValueType, IndexType)    \
    void transpose(std::shared_ptr<const DefaultExecutor> exec,   \
                   const matrix::Csr<ValueType, IndexType>* orig, \
                   matrix::Csr<ValueType, IndexType>* trans)

#define GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType)    \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,   \
                        const matrix::Csr<ValueType, IndexType>* orig, \
                        matrix::Csr<ValueType, IndexType>* trans)

#define GKO_DECLARE_CSR_INV_SYMM_PERMUTE_KERNEL(ValueType, IndexType)    \
    void inv_symm_permute(std::shared_ptr<const DefaultExecutor> exec,   \
                          const IndexType* permutation_indices,          \
                          const matrix::Csr<ValueType, IndexType>* orig, \
                          matrix::Csr<ValueType, IndexType>* permuted)

#define GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL(ValueType, IndexType)    \
    void row_permute(std::shared_ptr<const DefaultExecutor> exec,   \
                     const IndexType* permutation_indices,          \
                     const matrix::Csr<ValueType, IndexType>* orig, \
                     matrix::Csr<ValueType, IndexType>* row_permuted)

#define GKO_DECLARE_CSR_INVERSE_ROW_PERMUTE_KERNEL(ValueType, IndexType)    \
    void inverse_row_permute(std::shared_ptr<const DefaultExecutor> exec,   \
                             const IndexType* permutation_indices,          \
                             const matrix::Csr<ValueType, IndexType>* orig, \
                             matrix::Csr<ValueType, IndexType>* row_permuted)

#define GKO_DECLARE_CSR_INVERSE_COLUMN_PERMUTE_KERNEL(ValueType, IndexType) \
    void inverse_column_permute(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const IndexType* permutation_indices,                               \
        const matrix::Csr<ValueType, IndexType>* orig,                      \
        matrix::Csr<ValueType, IndexType>* column_permuted)

#define GKO_DECLARE_INVERT_PERMUTATION_KERNEL(IndexType)             \
    void invert_permutation(                                         \
        std::shared_ptr<const DefaultExecutor> exec, size_type size, \
        const IndexType* permutation_indices, IndexType* inv_permutation)

#define GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_SPAN_KERNEL(ValueType, IndexType)  \
    void calculate_nonzeros_per_row_in_span(                                   \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::Csr<ValueType, IndexType>* source, const span& row_span, \
        const span& col_span, array<IndexType>* row_nnz)

#define GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_INDEX_SET_KERNEL(ValueType, \
                                                             IndexType) \
    void calculate_nonzeros_per_row_in_index_set(                       \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::Csr<ValueType, IndexType>* source,                \
        const gko::index_set<IndexType>& row_index_set,                 \
        const gko::index_set<IndexType>& col_index_set, IndexType* row_nnz)

#define GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_KERNEL(ValueType, IndexType)     \
    void compute_submatrix(std::shared_ptr<const DefaultExecutor> exec,     \
                           const matrix::Csr<ValueType, IndexType>* source, \
                           gko::span row_span, gko::span col_span,          \
                           matrix::Csr<ValueType, IndexType>* result)

#define GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_FROM_INDEX_SET_KERNEL(ValueType, \
                                                                 IndexType) \
    void compute_submatrix_from_index_set(                                  \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::Csr<ValueType, IndexType>* source,                    \
        const gko::index_set<IndexType>& row_index_set,                     \
        const gko::index_set<IndexType>& col_index_set,                     \
        matrix::Csr<ValueType, IndexType>* result)

#define GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX(ValueType, IndexType)         \
    void sort_by_column_index(std::shared_ptr<const DefaultExecutor> exec, \
                              matrix::Csr<ValueType, IndexType>* to_sort)

#define GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX(ValueType, IndexType) \
    void is_sorted_by_column_index(                                     \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::Csr<ValueType, IndexType>* to_check, bool* is_sorted)

#define GKO_DECLARE_CSR_EXTRACT_DIAGONAL(ValueType, IndexType)           \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,   \
                          const matrix::Csr<ValueType, IndexType>* orig, \
                          matrix::Diagonal<ValueType>* diag)

#define GKO_DECLARE_CSR_SCALE_KERNEL(ValueType, IndexType)  \
    void scale(std::shared_ptr<const DefaultExecutor> exec, \
               const matrix::Dense<ValueType>* alpha,       \
               matrix::Csr<ValueType, IndexType>* to_scale)

#define GKO_DECLARE_CSR_INV_SCALE_KERNEL(ValueType, IndexType)  \
    void inv_scale(std::shared_ptr<const DefaultExecutor> exec, \
                   const matrix::Dense<ValueType>* alpha,       \
                   matrix::Csr<ValueType, IndexType>* to_scale)

#define GKO_DECLARE_CSR_CHECK_DIAGONAL_ENTRIES_EXIST(ValueType, IndexType) \
    void check_diagonal_entries_exist(                                     \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::Csr<ValueType, IndexType>* mtx, bool& has_all_diags)

#define GKO_DECLARE_CSR_ADD_SCALED_IDENTITY_KERNEL(ValueType, IndexType)  \
    void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec, \
                             const matrix::Dense<ValueType>* alpha,       \
                             const matrix::Dense<ValueType>* beta,        \
                             matrix::Csr<ValueType, IndexType>* mtx)

#define GKO_DECLARE_CSR_BUILD_LOOKUP_OFFSETS_KERNEL(IndexType)               \
    void build_lookup_offsets(std::shared_ptr<const DefaultExecutor> exec,   \
                              const IndexType* row_ptrs,                     \
                              const IndexType* col_idxs, size_type num_rows, \
                              matrix::csr::sparsity_type allowed,            \
                              IndexType* storage_offsets)

#define GKO_DECLARE_CSR_BUILD_LOOKUP_KERNEL(IndexType)                        \
    void build_lookup(std::shared_ptr<const DefaultExecutor> exec,            \
                      const IndexType* row_ptrs, const IndexType* col_idxs,   \
                      size_type num_rows, matrix::csr::sparsity_type allowed, \
                      const IndexType* storage_offsets, int64* row_desc,      \
                      int32* storage)

#define GKO_DECLARE_CSR_BENCHMARK_LOOKUP_KERNEL(IndexType)               \
    void benchmark_lookup(std::shared_ptr<const DefaultExecutor> exec,   \
                          const IndexType* row_ptrs,                     \
                          const IndexType* col_idxs, size_type num_rows, \
                          const IndexType* storage_offsets,              \
                          const int64* row_desc, const int32* storage,   \
                          IndexType sample_size, IndexType* result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename MatrixValueType, typename InputValueType,           \
              typename OutputValueType, typename IndexType>                \
    GKO_DECLARE_CSR_SPMV_KERNEL(MatrixValueType, InputValueType,           \
                                OutputValueType, IndexType);               \
    template <typename MatrixValueType, typename InputValueType,           \
              typename OutputValueType, typename IndexType>                \
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL(MatrixValueType, InputValueType,  \
                                         OutputValueType, IndexType);      \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_SPGEMM_KERNEL(ValueType, IndexType);                   \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_SPGEAM_KERNEL(ValueType, IndexType);                   \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_FILL_IN_DENSE_KERNEL(ValueType, IndexType);            \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_CONVERT_TO_FBCSR_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_TRANSPOSE_KERNEL(ValueType, IndexType);                \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_INV_SYMM_PERMUTE_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL(ValueType, IndexType);              \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_INVERSE_ROW_PERMUTE_KERNEL(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_INVERSE_COLUMN_PERMUTE_KERNEL(ValueType, IndexType);   \
    template <typename IndexType>                                          \
    GKO_DECLARE_INVERT_PERMUTATION_KERNEL(IndexType);                      \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_SPAN_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_INDEX_SET_KERNEL(ValueType,        \
                                                         IndexType);       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_FROM_INDEX_SET_KERNEL(ValueType,    \
                                                             IndexType);   \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX(ValueType, IndexType);            \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_EXTRACT_DIAGONAL(ValueType, IndexType);                \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_SCALE_KERNEL(ValueType, IndexType);                    \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_INV_SCALE_KERNEL(ValueType, IndexType);                \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_CHECK_DIAGONAL_ENTRIES_EXIST(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_CSR_ADD_SCALED_IDENTITY_KERNEL(ValueType, IndexType);      \
    template <typename IndexType>                                          \
    GKO_DECLARE_CSR_BUILD_LOOKUP_OFFSETS_KERNEL(IndexType);                \
    template <typename IndexType>                                          \
    GKO_DECLARE_CSR_BUILD_LOOKUP_KERNEL(IndexType);                        \
    template <typename IndexType>                                          \
    GKO_DECLARE_CSR_BENCHMARK_LOOKUP_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(csr, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CSR_KERNELS_HPP_
