// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_DIRECT_KERNELS_HPP_
#define GKO_CORE_SOLVER_DIRECT_KERNELS_HPP_


#include <ginkgo/config.hpp>


#if GKO_HAVE_CUDSS


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/direct.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace direct {


#define GKO_DECLARE_DIRECT_GENERATE_KERNEL(_vtype, _itype)                    \
    void generate(std::shared_ptr<const DefaultExecutor> exec,                \
                  const matrix::Csr<_vtype, _itype>* matrix,                  \
                  std::shared_ptr<experimental::solver::direct_vendor_state>& \
                      solve_state,                                            \
                  const experimental::solver::vendor_parameters& params)


#define GKO_DECLARE_DIRECT_SOLVE_KERNEL(_vtype)                        \
    void solve(std::shared_ptr<const DefaultExecutor> exec,            \
               const experimental::solver::direct_vendor_state* state, \
               const matrix::Dense<_vtype>* b, matrix::Dense<_vtype>* x)


#define GKO_DECLARE_ALL_AS_TEMPLATES                          \
    template <typename ValueType, typename IndexType>         \
    GKO_DECLARE_DIRECT_GENERATE_KERNEL(ValueType, IndexType); \
    template <typename ValueType>                             \
    GKO_DECLARE_DIRECT_SOLVE_KERNEL(ValueType)


}  // namespace direct


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(direct, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_HAVE_CUDSS


#endif  // GKO_CORE_SOLVER_DIRECT_KERNELS_HPP_
