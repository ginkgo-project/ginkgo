// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_AMP_KERNELS_HPP_
#define GKO_CORE_MATRIX_AMP_KERNELS_HPP_


#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


constexpr int num_precs = matrix::AMP<default_precision, int32>::num_precisions;

#define GKO_DECLARE_AMP_SPMV_KERNEL(InputValueType, MatrixValueType, \
                                    OutputValueType, IndexType)      \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,           \
              const matrix::AMP<MatrixValueType, IndexType>* a,      \
              const matrix::Dense<InputValueType>* b,                \
              matrix::Dense<OutputValueType>* c)

#define GKO_DECLARE_AMP_ADVANCED_SPMV_KERNEL(InputValueType, MatrixValueType, \
                                             OutputValueType, IndexType)      \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,           \
                       const matrix::Dense<MatrixValueType>* alpha,           \
                       const matrix::AMP<MatrixValueType, IndexType>* a,      \
                       const matrix::Dense<InputValueType>* b,                \
                       const matrix::Dense<OutputValueType>* beta,            \
                       matrix::Dense<OutputValueType>* c)

#define GKO_DECLARE_AMP_GENERATE_CWISE_ELL_STEP1_KERNEL(ValueType, IndexType) \
    void generate_ell_rownorms_storage(                                       \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::Ell<ValueType, IndexType>* a,                           \
        std::array<int, num_precs>& max_nnz, array<ValueType>& rownorms)

#define GKO_DECLARE_AMP_GENERATE_ELL_SCATTER_BINS_KERNEL(ValueType, IndexType) \
    void generate_ell_scatter_bins(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::Ell<ValueType, IndexType>* a,                            \
        std::array<LinOp*, num_precs>& amat)

#define GKO_DECLARE_AMP_FILL_IN_DENSE_KERNEL(ValueType, IndexType)      \
    void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,     \
                       const matrix::AMP<ValueType, IndexType>* source, \
                       matrix::Dense<ValueType>* result)

#define GKO_DECLARE_AMP_EXTRACT_DIAGONAL_KERNEL(ValueType, IndexType)    \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,   \
                          const matrix::AMP<ValueType, IndexType>* orig, \
                          matrix::Diagonal<ValueType>* diag)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                        \
    template <typename InputValueType, typename MatrixValueType,            \
              typename OutputValueType, typename IndexType>                 \
    GKO_DECLARE_AMP_SPMV_KERNEL(InputValueType, MatrixValueType,            \
                                OutputValueType, IndexType);                \
    template <typename InputValueType, typename MatrixValueType,            \
              typename OutputValueType, typename IndexType>                 \
    GKO_DECLARE_AMP_ADVANCED_SPMV_KERNEL(InputValueType, MatrixValueType,   \
                                         OutputValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_AMP_GENERATE_CWISE_ELL_STEP1_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_AMP_GENERATE_ELL_SCATTER_BINS_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_AMP_FILL_IN_DENSE_KERNEL(ValueType, IndexType);             \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_AMP_EXTRACT_DIAGONAL_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(amp, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_AMP_KERNELS_HPP_
