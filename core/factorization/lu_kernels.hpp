// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_LU_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_LU_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_LU_INITIALIZE(ValueType, IndexType)                       \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,              \
                    const matrix::Csr<ValueType, IndexType>* mtx,             \
                    const IndexType* factor_lookup_offsets,                   \
                    const int64* factor_lookup_descs,                         \
                    const int32* factor_lookup_storage, IndexType* diag_idxs, \
                    matrix::Csr<ValueType, IndexType>* factors)


#define GKO_DECLARE_LU_FACTORIZE(ValueType, IndexType)                         \
    void factorize(std::shared_ptr<const DefaultExecutor> exec,                \
                   const IndexType* lookup_offsets, const int64* lookup_descs, \
                   const int32* lookup_storage, const IndexType* diag_idxs,    \
                   matrix::Csr<ValueType, IndexType>* factors,                 \
                   array<int>& tmp_storage)


#define GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE(IndexType)                  \
    void symbolic_factorize_simple(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const IndexType* row_ptrs, const IndexType* col_idxs,                 \
        const IndexType* factor_lookup_offsets,                               \
        const int64* factor_lookup_descs, const int32* factor_lookup_storage, \
        matrix::Csr<float, IndexType>* factors, IndexType* out_row_nnz)


#define GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE_FINALIZE(IndexType) \
    void symbolic_factorize_simple_finalize(                          \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        const matrix::Csr<float, IndexType>* factors, IndexType* col_idxs)


#define GKO_DECLARE_ALL_AS_TEMPLATES                      \
    template <typename ValueType, typename IndexType>     \
    GKO_DECLARE_LU_INITIALIZE(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>     \
    GKO_DECLARE_LU_FACTORIZE(ValueType, IndexType);       \
    template <typename IndexType>                         \
    GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE(IndexType); \
    template <typename IndexType>                         \
    GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE_FINALIZE(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(lu_factorization,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_LU_KERNELS_HPP_
