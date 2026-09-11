// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_LOWER_TRS_KERNELS_HPP_
#define GKO_CORE_SOLVER_LOWER_TRS_KERNELS_HPP_


#include <memory>
#include <optional>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace lower_trs {


#define GKO_DECLARE_LOWER_TRS_SHOULD_PERFORM_TRANSPOSE_KERNEL                  \
    void should_perform_transpose(std::shared_ptr<const DefaultExecutor> exec, \
                                  bool& do_transpose)


#define GKO_DECLARE_LOWER_TRS_GENERATE_KERNEL(ValueType, IndexType)           \
    void generate(std::shared_ptr<const DefaultExecutor> exec,                \
                  matrix::view::csr<const ValueType, const IndexType> matrix, \
                  std::shared_ptr<solver::SolveStruct>& solve_struct,         \
                  bool unit_diag, const solver::trisolve_algorithm algorithm, \
                  const size_type num_rhs)


#define GKO_DECLARE_LOWER_TRS_SOLVE_KERNEL(ValueType, IndexType)           \
    void solve(std::shared_ptr<const DefaultExecutor> exec,                \
               matrix::view::csr<const ValueType, const IndexType> matrix, \
               const solver::SolveStruct* solve_struct, bool unit_diag,    \
               const solver::trisolve_algorithm algorithm,                 \
               std::optional<matrix::view::dense<ValueType>> trans_b,      \
               std::optional<matrix::view::dense<ValueType>> trans_x,      \
               matrix::view::dense<const ValueType> b,                     \
               matrix::view::dense<ValueType> x)


#define GKO_DECLARE_ALL_AS_TEMPLATES                          \
    GKO_DECLARE_LOWER_TRS_SHOULD_PERFORM_TRANSPOSE_KERNEL;    \
    template <typename ValueType, typename IndexType>         \
    GKO_DECLARE_LOWER_TRS_SOLVE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>         \
    GKO_DECLARE_LOWER_TRS_GENERATE_KERNEL(ValueType, IndexType)


}  // namespace lower_trs


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(lower_trs,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_LOWER_TRS_KERNELS_HPP_
