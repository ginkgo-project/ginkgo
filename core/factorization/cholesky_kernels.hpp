// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_CHOLESKY_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_CHOLESKY_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/kernel_declaration.hpp"
#include "core/components/range_minimum_query.hpp"
#include "core/factorization/elimination_forest.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_CHOLESKY_SYMBOLIC_POSTORDER(ValueType, IndexType)    \
    void symbolic_postorder(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* mtx,                    \
        const gko::factorization::elimination_forest<IndexType>& forest, \
        IndexType* postorder_cols, IndexType* postorder_lower_ends)


#define GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT(ValueType, IndexType)        \
    void symbolic_count(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* mtx,                    \
        const gko::factorization::elimination_forest<IndexType>& forest, \
        IndexType* row_nnz, array<IndexType>& tmp_storage)


#define GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT_LCA(ValueType, IndexType)    \
    void symbolic_count_lca(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* mtx,                    \
        const gko::factorization::elimination_forest<IndexType>& forest, \
        const IndexType* euler_walk_first,                               \
        const typename gko::range_minimum_query<IndexType>::view_type&   \
            lca_rmq,                                                     \
        IndexType* nz_path_lengths, IndexType* row_nnz,                  \
        array<IndexType>& tmp_storage)


#define GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE(ValueType, IndexType)    \
    void symbolic_factorize(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* mtx,                    \
        const gko::factorization::elimination_forest<IndexType>& forest, \
        matrix::Csr<ValueType, IndexType>* l_factor,                     \
        const array<IndexType>& tmp_storage)


#define GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE_FLATTENED(ValueType,     \
                                                          IndexType)     \
    void symbolic_factorize_flattened(                                   \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* mtx,                    \
        const gko::factorization::elimination_forest<IndexType>& forest, \
        const IndexType* nz_path_lengths,                                \
        matrix::Csr<ValueType, IndexType>* l_factor,                     \
        const array<IndexType>& tmp_storage)


#define GKO_DECLARE_CHOLESKY_INITIALIZE(ValueType, IndexType)                 \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,              \
                    const matrix::Csr<ValueType, IndexType>* mtx,             \
                    const IndexType* factor_lookup_offsets,                   \
                    const int64* factor_lookup_descs,                         \
                    const int32* factor_lookup_storage, IndexType* diag_idxs, \
                    IndexType* transpose_idxs,                                \
                    matrix::Csr<ValueType, IndexType>* factors)


#define GKO_DECLARE_CHOLESKY_FACTORIZE(ValueType, IndexType)                   \
    void factorize(std::shared_ptr<const DefaultExecutor> exec,                \
                   const IndexType* lookup_offsets, const int64* lookup_descs, \
                   const int32* lookup_storage, const IndexType* diag_idxs,    \
                   const IndexType* transpose_idxs,                            \
                   matrix::Csr<ValueType, IndexType>* factors,                 \
                   bool full_fillin, array<int>& tmp_storage)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                         \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT(ValueType, IndexType);               \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT_LCA(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE_FLATTENED(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_CHOLESKY_INITIALIZE(ValueType, IndexType);                   \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_CHOLESKY_FACTORIZE(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(cholesky, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_CHOLESKY_KERNELS_HPP_
