// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/upper_trs_kernels.hpp"

#include <memory>
#include <optional>

#include <sycl/sycl.hpp>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "dpcpp/solver/common_trs_kernels.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The UPPER_TRS solver namespace.
 *
 * @ingroup upper_trs
 */
namespace upper_trs {


void should_perform_transpose(std::shared_ptr<const DpcppExecutor> exec,
                              bool& do_transpose)
{
    do_transpose = false;
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const DpcppExecutor> exec,
              matrix::view::csr<const ValueType, const IndexType> matrix,
              std::shared_ptr<solver::SolveStruct>& solve_struct,
              bool unit_diag, const solver::trisolve_algorithm algorithm,
              const size_type num_rhs)
{
    generate_kernel<ValueType, IndexType>(exec, matrix, solve_struct, num_rhs,
                                          true, unit_diag);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const DpcppExecutor> exec,
           matrix::view::csr<const ValueType, const IndexType> matrix,
           const solver::SolveStruct* solve_struct, bool unit_diag,
           const solver::trisolve_algorithm algorithm,
           std::optional<matrix::view::dense<ValueType>> trans_b,
           std::optional<matrix::view::dense<ValueType>> trans_x,
           matrix::view::dense<const ValueType> b,
           matrix::view::dense<ValueType> x)
{
    solve_kernel<ValueType, IndexType>(exec, matrix, solve_struct, b, x);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_SOLVE_KERNEL);


}  // namespace upper_trs
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
