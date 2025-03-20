// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/bitvector_kernels.hpp"

#include "core/base/intrinsics.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace bitvector {


template <typename IndexType>
void compute_bits_and_ranks(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* indices,
    IndexType num_indices, IndexType size,
    typename device_bitvector<IndexType>::storage_type* bits,
    IndexType* ranks) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_BITVECTOR_COMPUTE_BITS_AND_RANKS_KERNEL);


}  // namespace bitvector
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
