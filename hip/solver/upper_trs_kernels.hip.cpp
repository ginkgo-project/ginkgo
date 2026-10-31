// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/upper_trs_kernels.hpp"

#include <memory>
#include <optional>

#include <hipsparse/hipsparse.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "hip/solver/common_trs_kernels.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The UPPER_TRS solver namespace.
 *
 */
namespace upper_trs {


void should_perform_transpose(std::shared_ptr<const HipExecutor> exec,
                              bool& do_transpose)
{
    should_perform_transpose_kernel(exec, do_transpose);
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const HipExecutor> exec,
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


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void solve(std::shared_ptr<const HipExecutor> exec,
           matrix::view::csr<const MatrixValueType, const IndexType> matrix,
           const solver::SolveStruct* solve_struct, bool unit_diag,
           const solver::trisolve_algorithm algorithm,
           std::optional<matrix::view::dense<InputValueType>> trans_b,
           std::optional<matrix::view::dense<OutputValueType>> trans_x,
           matrix::view::dense<const InputValueType> b,
           matrix::view::dense<OutputValueType> x)
{
    constexpr bool all_same_value_type =
        std::is_same<MatrixValueType, InputValueType>::value &&
        std::is_same<MatrixValueType, OutputValueType>::value;

    if (algorithm == solver::trisolve_algorithm::sparselib) {
        if constexpr (all_same_value_type) {
            solve_kernel<MatrixValueType, IndexType>(exec, matrix, solve_struct,
                                                     b, x);
            return;
        }
    } else if (algorithm == solver::trisolve_algorithm::syncfree) {
        sptrsv_naive_caching<true>(exec, matrix, unit_diag, b, x);
        return;
    }

    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_SOLVE_KERNEL);


}  // namespace upper_trs
}  // namespace hip
}  // namespace kernels
}  // namespace gko
