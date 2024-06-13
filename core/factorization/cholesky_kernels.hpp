// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_CHOLESKY_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_CHOLESKY_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/kernel_declaration.hpp"
#include "core/factorization/elimination_forest.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT(ValueType, IndexType)        \
    void symbolic_count(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* mtx,                    \
        const gko::factorization::elimination_forest<IndexType>& forest, \
        IndexType* row_nnz, array<IndexType>& tmp_storage)


#define GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE(ValueType, IndexType)    \
    void symbolic_factorize(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* mtx,                    \
        const gko::factorization::elimination_forest<IndexType>& forest, \
        matrix::Csr<ValueType, IndexType>* l_factor,                     \
        const array<IndexType>& tmp_storage)


#define GKO_DECLARE_CHOLESKY_FOREST_FROM_FACTOR(ValueType, IndexType) \
    void forest_from_factor(                                          \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        const matrix::Csr<ValueType, IndexType>* factors,             \
        gko::factorization::elimination_forest<IndexType>& forest)


#define GKO_DECLARE_CHOLESKY_INITIALIZE(ValueType, IndexType)                 \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,              \
                    const matrix::Csr<ValueType, IndexType>* mtx,             \
                    const IndexType* factor_lookup_offsets,                   \
                    const int64* factor_lookup_descs,                         \
                    const int32* factor_lookup_storage, IndexType* diag_idxs, \
                    IndexType* transpose_idxs,                                \
                    matrix::Csr<ValueType, IndexType>* factors)


#define GKO_DECLARE_CHOLESKY_FACTORIZE(ValueType, IndexType)             \
    void factorize(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const IndexType* lookup_offsets, const int64* lookup_descs,      \
        const int32* lookup_storage, const IndexType* diag_idxs,         \
        const IndexType* transpose_idxs,                                 \
        const gko::factorization::elimination_forest<IndexType>& forest, \
        matrix::Csr<ValueType, IndexType>* factors, array<int>& tmp_storage)


#define GKO_DECLARE_ALL_AS_TEMPLATES                               \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_CHOLESKY_FOREST_FROM_FACTOR(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_CHOLESKY_INITIALIZE(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_CHOLESKY_FACTORIZE(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(cholesky, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_CHOLESKY_KERNELS_HPP_
