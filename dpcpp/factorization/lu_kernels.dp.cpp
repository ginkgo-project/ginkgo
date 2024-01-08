// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/lu_kernels.hpp"


#include <algorithm>
#include <memory>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/allocator.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The LU namespace.
 *
 * @ingroup factor
 */
namespace lu_factorization {


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                const IndexType* factor_lookup_offsets,
                const int64* factor_lookup_descs,
                const int32* factor_lookup_storage, IndexType* diag_idxs,
                matrix::Csr<ValueType, IndexType>* factors) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU_INITIALIZE);


template <typename ValueType, typename IndexType>
void factorize(std::shared_ptr<const DefaultExecutor> exec,
               const IndexType* lookup_offsets, const int64* lookup_descs,
               const int32* lookup_storage, const IndexType* diag_idxs,
               matrix::Csr<ValueType, IndexType>* factors,
               array<int>& tmp_storage) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU_FACTORIZE);


template <typename IndexType>
void symbolic_factorize_simple(std::shared_ptr<const DefaultExecutor> exec,
                               const IndexType* row_ptrs,
                               const IndexType* col_idxs,
                               const IndexType* lookup_offsets,
                               const int64* lookup_descs,
                               const int32* lookup_storage,
                               matrix::Csr<float, IndexType>* factors,
                               IndexType* out_row_nnz) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE);


template <typename IndexType>
void symbolic_factorize_simple_finalize(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<float, IndexType>* factors,
    IndexType* out_col_idxs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE_FINALIZE);


}  // namespace lu_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
