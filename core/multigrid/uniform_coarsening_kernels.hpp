// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_UNIFORM_COARSENING_KERNELS_HPP_
#define GKO_CORE_MULTIGRID_UNIFORM_COARSENING_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/multigrid/uniform_coarsening.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace uniform_coarsening {

#define GKO_DECLARE_UNIFORM_COARSENING_FILL_RESTRICT_OP(ValueType, IndexType) \
    void fill_restrict_op(std::shared_ptr<const DefaultExecutor> exec,        \
                          const array<IndexType>* coarse_rows,                \
                          matrix::Csr<ValueType, IndexType>* restrict_op)

#define GKO_DECLARE_UNIFORM_COARSENING_FILL_INCREMENTAL_INDICES(IndexType)     \
    void fill_incremental_indices(std::shared_ptr<const DefaultExecutor> exec, \
                                  size_type coarse_skip,                       \
                                  array<IndexType>* coarse_rows)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_UNIFORM_COARSENING_FILL_RESTRICT_OP(ValueType, IndexType); \
    template <typename IndexType>                                          \
    GKO_DECLARE_UNIFORM_COARSENING_FILL_INCREMENTAL_INDICES(IndexType)


}  // namespace uniform_coarsening


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(uniform_coarsening,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_UNIFORM_COARSENING_KERNELS_HPP_
