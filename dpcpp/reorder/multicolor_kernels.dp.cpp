// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/multicolor_kernels.hpp"

#include <algorithm>
#include <vector>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace multicolor {


template <typename IndexType>
void compute_permutation_csr(std::shared_ptr<const DpcppExecutor> exec,
                             const IndexType num_vertices,
                             const IndexType* const row_ptrs,
                             const IndexType* const col_idxs,
                             gko::array<IndexType>& color_ptrs,
                             IndexType* const permutation,
                             IndexType* const inv_permutation)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_MULTICOLOR_COMPUTE_PERMUTATION_CSR_KERNEL);


}  // namespace multicolor
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
