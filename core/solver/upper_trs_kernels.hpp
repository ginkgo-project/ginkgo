// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_UPPER_TRS_KERNELS_HPP_
#define GKO_CORE_SOLVER_UPPER_TRS_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace upper_trs {


#define GKO_DECLARE_UPPER_TRS_SHOULD_PERFORM_TRANSPOSE_KERNEL                  \
    void should_perform_transpose(std::shared_ptr<const DefaultExecutor> exec, \
                                  bool& do_transpose)


#define GKO_DECLARE_UPPER_TRS_GENERATE_KERNEL(_vtype, _itype)                 \
    void generate(std::shared_ptr<const DefaultExecutor> exec,                \
                  const matrix::Csr<_vtype, _itype>* matrix,                  \
                  std::shared_ptr<solver::SolveStruct>& solve_struct,         \
                  bool unit_diag, const solver::trisolve_algorithm algorithm, \
                  const size_type num_rhs)


#define GKO_DECLARE_UPPER_TRS_SOLVE_KERNEL(_vtype, _itype)                     \
    void solve(std::shared_ptr<const DefaultExecutor> exec,                    \
               const matrix::Csr<_vtype, _itype>* matrix,                      \
               const solver::SolveStruct* solve_struct, bool unit_diag,        \
               const solver::trisolve_algorithm algorithm,                     \
               matrix::Dense<_vtype>* trans_b, matrix::Dense<_vtype>* trans_x, \
               const matrix::Dense<_vtype>* b, matrix::Dense<_vtype>* x)


#define GKO_DECLARE_ALL_AS_TEMPLATES                          \
    GKO_DECLARE_UPPER_TRS_SHOULD_PERFORM_TRANSPOSE_KERNEL;    \
    template <typename ValueType, typename IndexType>         \
    GKO_DECLARE_UPPER_TRS_SOLVE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>         \
    GKO_DECLARE_UPPER_TRS_GENERATE_KERNEL(ValueType, IndexType)


}  // namespace upper_trs


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(upper_trs,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_UPPER_TRS_KERNELS_HPP_
