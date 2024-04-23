// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/lower_trs_kernels.hpp"

#include <memory>

#include <cuda.h>
#include <cusparse.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "cuda/solver/common_trs_kernels.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The LOWER_TRS solver namespace.
 *
 * @ingroup lower_trs
 */
namespace lower_trs {


void should_perform_transpose(std::shared_ptr<const CudaExecutor> exec,
                              bool& do_transpose)
{
    should_perform_transpose_kernel(exec, do_transpose);
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CudaExecutor> exec,
              const matrix::Csr<ValueType, IndexType>* matrix,
              std::shared_ptr<solver::SolveStruct>& solve_struct,
              bool unit_diag, const solver::trisolve_algorithm algorithm,
              const size_type num_rhs)
{
    if (algorithm == solver::trisolve_algorithm::sparselib) {
        generate_kernel<ValueType, IndexType>(exec, matrix, solve_struct,
                                              num_rhs, false, unit_diag);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_GENERATE_KERNEL);


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void solve(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Csr<MatrixValueType, IndexType>* matrix,
           const solver::SolveStruct* solve_struct, bool unit_diag,
           const solver::trisolve_algorithm algorithm,
           matrix::Dense<InputValueType>* trans_b,
           matrix::Dense<OutputValueType>* trans_x,
           const matrix::Dense<InputValueType>* b,
           matrix::Dense<OutputValueType>* x)
{
    if (algorithm == solver::trisolve_algorithm::sparselib) {
        solve_kernel(exec, matrix, solve_struct, trans_b, trans_x, b, x);
    } else {
        sptrsv_naive_caching<false>(exec, matrix, unit_diag, b, x);
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_SOLVE_KERNEL);


}  // namespace lower_trs
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
