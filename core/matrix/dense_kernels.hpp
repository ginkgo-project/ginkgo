// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <memory>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/device_views.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

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

#define GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL(ValueType, IndexType) \
    void convert_to_coo(std::shared_ptr<const DefaultExecutor> exec,  \
                        matrix::view::dense<const ValueType> source,  \
                        const int64* row_ptrs,                        \
                        matrix::view::coo<ValueType, IndexType> other)

#define GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL(ValueType, IndexType) \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,  \
                        matrix::view::dense<const ValueType> source,  \
                        matrix::view::csr<ValueType, IndexType> other)

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


#define GKO_DECLARE_ALL_AS_TEMPLATES                                        \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(ValueType);                       \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_APPLY_KERNEL(ValueType);                              \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_FBCSR_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL(ValueType, IndexType); \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_COMPUTE_MAX_NNZ_PER_ROW_KERNEL(ValueType);            \
    template <typename ValueType>                                           \
    GKO_DECLARE_DENSE_COMPUTE_SLICE_SETS_KERNEL(ValueType);                 \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_DENSE_COUNT_NONZERO_BLOCKS_PER_ROW_KERNEL(ValueType,        \
                                                          IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(dense, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko
