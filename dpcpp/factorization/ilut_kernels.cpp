// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ilut_kernels.hpp"

#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/allocator.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace ilut_factorization {


constexpr int default_block_size = 512;


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                matrix::Csr<ValueType, IndexType>* l_factor,
                const IndexType* l_lookup_offsets, const int64* l_lookup_descs,
                const int32* l_lookup_storage,
                matrix::Csr<ValueType, IndexType>* u_factor,
                const IndexType* u_lookup_offsets, const int64* u_lookup_descs,
                const int32* u_lookup_storage) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILUT_INITIALIZE_KERNEL);


template <typename ValueType, typename IndexType>
void compute_l_u_factors(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::Csr<ValueType, IndexType>* l, const IndexType* l_lookup_offsets,
    const int64* l_lookup_descs, const int32* l_lookup_storage,
    matrix::Csr<ValueType, IndexType>* u, const IndexType* u_lookup_offsets,
    const int64* u_lookup_descs, const int32* u_lookup_storage,
    array<int>& tmp_storage) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILUT_COMPUTE_LU_FACTORS_KERNEL);


}  // namespace ilut_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
