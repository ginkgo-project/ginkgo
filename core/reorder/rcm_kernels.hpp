// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_REORDER_RCM_KERNELS_HPP_
#define GKO_CORE_REORDER_RCM_KERNELS_HPP_


#include <ginkgo/core/reorder/rcm.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_RCM_COMPUTE_PERMUTATION_KERNEL(IndexType)                \
    void compute_permutation(                                                \
        std::shared_ptr<const DefaultExecutor> exec, IndexType num_vertices, \
        const IndexType* row_ptrs, const IndexType* col_idxs,                \
        IndexType* permutation, IndexType* inv_permutation,                  \
        gko::reorder::starting_strategy strategy)

#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename IndexType>    \
    GKO_DECLARE_RCM_COMPUTE_PERMUTATION_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(rcm, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_REORDER_RCM_KERNELS_HPP_
