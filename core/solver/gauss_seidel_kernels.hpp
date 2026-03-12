// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_GAUSS_SEIDEL_KERNELS_HPP_
#define GKO_CORE_SOLVER_GAUSS_SEIDEL_KERNELS_HPP_


#include <memory>
#include <vector>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace gssdl {


#define GKO_DECLARE_MULTICOLOR_FWD_GS_ELL_KERNEL(_vtype, _itype)         \
    void multicolor_fgs_ell(std::shared_ptr<const DefaultExecutor> exec, \
                            const std::vector<_itype>& color_ptrs,       \
                            const matrix::Ell<_vtype, _itype>* a,        \
                            const matrix::Dense<_vtype>* b,              \
                            matrix::Dense<_vtype>* x, bool first_iter,   \
                            array<stopping_status>* stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES                  \
    template <typename ValueType, typename IndexType> \
    GKO_DECLARE_MULTICOLOR_FWD_GS_ELL_KERNEL(ValueType, IndexType)


}  // namespace gssdl


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(gssdl, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GAUSS_SEIDEL_KERNELS_HPP_
