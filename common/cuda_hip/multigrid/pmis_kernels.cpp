// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/components/atomic.hpp"
#include "common/unified/base/kernel_launch.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace pmis {


template <typename IndexType>
void add_at_indices(std::shared_ptr<const DefaultExecutor> exec, size_type num,
                    const IndexType* idxs, const IndexType* values,
                    IndexType* out)
{
    // idxs may repeat across neighbours, hence the atomic. The contention is
    // bounded by the number of neighbours that share a row.
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto idxs, auto values, auto out) {
            atomic_add(out + idxs[i], values ? values[i] : IndexType{1});
        },
        num, idxs, values, out);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PMIS_ADD_AT_INDICES);


}  // namespace pmis
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
