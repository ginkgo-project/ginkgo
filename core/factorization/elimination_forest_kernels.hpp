// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_ELIMINATION_FOREST_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_ELIMINATION_FOREST_KERNELS_HPP_


#include "core/factorization/elimination_forest.hpp"

#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SKELETON_TREE(IndexType)     \
    void compute_skeleton_tree(std::shared_ptr<const DefaultExecutor> exec, \
                               const IndexType* row_ptrs,                   \
                               const IndexType* cols, size_type size,       \
                               IndexType* out_row_ptrs, IndexType* out_cols)


#define GKO_DECLARE_ELIMINATION_FOREST_COMPUTE(IndexType)          \
    void compute(std::shared_ptr<const DefaultExecutor> exec,      \
                 const IndexType* row_ptrs, const IndexType* cols, \
                 size_type size,                                   \
                 gko::factorization::elimination_forest<IndexType>& forest)


#define GKO_DECLARE_ELIMINATION_FOREST_FROM_FACTOR(ValueType, IndexType) \
    void from_factor(                                                    \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* factors,                \
        gko::factorization::elimination_forest<IndexType>& forest)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                 \
    template <typename IndexType>                                    \
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SKELETON_TREE(IndexType); \
    template <typename IndexType>                                    \
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE(IndexType);               \
    template <typename ValueType, typename IndexType>                \
    GKO_DECLARE_ELIMINATION_FOREST_FROM_FACTOR(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(elimination_forest,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_ELIMINATION_FOREST_KERNELS_HPP_
