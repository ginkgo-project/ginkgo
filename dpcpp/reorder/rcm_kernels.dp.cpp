// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/rcm_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace rcm {


template <typename IndexType>
void compute_permutation(
    std::shared_ptr<const DpcppExecutor> exec, const IndexType num_vertices,
    const IndexType* const row_ptrs, const IndexType* const col_idxs,
    IndexType* const permutation, IndexType* const inv_permutation,
    const gko::reorder::starting_strategy strategy) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RCM_COMPUTE_PERMUTATION_KERNEL);


}  // namespace rcm
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
