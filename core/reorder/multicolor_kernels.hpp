// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_REORDER_MULTICOLOR_KERNELS_HPP_
#define GKO_CORE_REORDER_MULTICOLOR_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_MULTICOLOR_COMPUTE_PERMUTATION_CSR_KERNEL(IndexType)     \
    void compute_permutation_csr(                                            \
        std::shared_ptr<const DefaultExecutor> exec, IndexType num_vertices, \
        const IndexType* row_ptrs, const IndexType* col_idxs,                \
        gko::array<IndexType>& color_ptrs, IndexType* permutation,           \
        IndexType* inv_permutation)

#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename IndexType>    \
    GKO_DECLARE_MULTICOLOR_COMPUTE_PERMUTATION_CSR_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(multicolor,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_REORDER_MULTICOLOR_KERNELS_HPP_
