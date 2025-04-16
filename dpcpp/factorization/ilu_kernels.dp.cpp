// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ilu_kernels.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The ilu factorization namespace.
 *
 * @ingroup factor
 */
namespace ilu_factorization {


template <typename ValueType, typename IndexType>
void sparselib_ilu(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::Csr<ValueType, IndexType>* m) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILU_SPARSELIB_ILU_KERNEL);


template <typename ValueType, typename IndexType>
void factorize_on_both(std::shared_ptr<const DefaultExecutor> exec,
                       const IndexType* lookup_offsets,
                       const int64* lookup_descs, const int32* lookup_storage,
                       const IndexType* diag_idxs,
                       matrix::Csr<ValueType, IndexType>* factors,
                       const IndexType* matrix_lookup_offsets,
                       const int64* matrix_lookup_descs,
                       const int32* matrix_lookup_storage,
                       matrix::Csr<ValueType, IndexType>* matrix,
                       array<int>& tmp_storage) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILU_FACTORIZE_ON_BOTH_KERNEL);

}  // namespace ilu_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
