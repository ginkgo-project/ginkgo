// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "omp/components/atomic.hpp"

namespace gko {
namespace kernels {
namespace omp {
namespace pmis {


template <typename IndexType>
void add_at_indices(std::shared_ptr<const DefaultExecutor> exec, size_type num,
                    const IndexType* idxs, const IndexType* values,
                    IndexType* out)
{
    // idxs may repeat across neighbours, hence the atomic
#pragma omp parallel for
    for (size_type i = 0; i < num; i++) {
        atomic_add(out[idxs[i]], values ? values[i] : IndexType{1});
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PMIS_ADD_AT_INDICES);


}  // namespace pmis
}  // namespace omp
}  // namespace kernels
}  // namespace gko
