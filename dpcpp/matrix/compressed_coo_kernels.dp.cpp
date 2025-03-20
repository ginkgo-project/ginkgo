// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/compressed_coo_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
namespace compressed_coo {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::CompactRowCoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b,
          matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CRCOO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::CompactRowCompressedColumnCoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b,
          matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CRCOCOCOO_SPMV_KERNEL);


template <typename IndexType>
void idxs_to_bits(std::shared_ptr<const DefaultExecutor> exec,
                  const IndexType* idxs, size_type nnz, uint32* bits,
                  IndexType* ranks) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CRCOO_IDXS_TO_BITS_KERNEL);


template <typename IndexType>
void bits_to_idxs(std::shared_ptr<const DefaultExecutor> exec,
                  const uint32* bits, const IndexType* ranks, size_type nnz,
                  IndexType* idxs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CRCOO_BITS_TO_IDXS_KERNEL);


}  // namespace compressed_coo
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
