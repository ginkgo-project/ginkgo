// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_FIXED_COARSENING_KERNELS_HPP_
#define GKO_CORE_MULTIGRID_FIXED_COARSENING_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace fixed_coarsening {


#define GKO_DECLARE_FIXED_COARSENING_RENUMBER_KERNEL(IndexType) \
    void renumber(std::shared_ptr<const DefaultExecutor> exec,  \
                  const array<IndexType>& coarse_row,           \
                  array<IndexType>* coarse_map)

#define GKO_DECLARE_FIXED_COARSENING_BUILD_ROW_PTRS_KERNEL(IndexType) \
    void build_row_ptrs(std::shared_ptr<const DefaultExecutor> exec,  \
                        size_type original_nrows,                     \
                        const IndexType* original_row_ptrs,           \
                        const IndexType* original_col_idxs,           \
                        const array<IndexType>& coarse_rows,          \
                        const array<IndexType>& coarse_cols_map,      \
                        size_type new_nrows, IndexType* new_row_ptrs)

#define GKO_DECLARE_FIXED_COARSENING_MAP_TO_COARSE_KERNEL(ValueType,           \
                                                          IndexType)           \
    void map_to_coarse(                                                        \
        std::shared_ptr<const DefaultExecutor> exec, size_type original_nrows, \
        const IndexType* original_row_ptrs,                                    \
        const IndexType* original_col_idxs, const ValueType* original_values,  \
        const array<IndexType>& coarse_rows,                                   \
        const array<IndexType>& coarse_cols_map, size_type new_nrows,          \
        const IndexType* new_row_ptrs, IndexType* new_col_idxs,                \
        ValueType* new_values)


#define GKO_DECLARE_ALL_AS_TEMPLATES                               \
    template <typename IndexType>                                  \
    GKO_DECLARE_FIXED_COARSENING_RENUMBER_KERNEL(IndexType);       \
    template <typename IndexType>                                  \
    GKO_DECLARE_FIXED_COARSENING_BUILD_ROW_PTRS_KERNEL(IndexType); \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_FIXED_COARSENING_MAP_TO_COARSE_KERNEL(ValueType, IndexType)


}  // namespace fixed_coarsening


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(fixed_coarsening,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_FIXED_COARSENING_KERNELS_HPP_
