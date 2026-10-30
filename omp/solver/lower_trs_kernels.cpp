// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/lower_trs_kernels.hpp"

#include <memory>
#include <optional>

#include <omp.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "core/base/mixed_precision_types.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/solver/common_trs_helpers.hpp"
#include "omp/components/atomic.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The LOWER_TRS solver namespace.
 *
 */
namespace lower_trs {


void should_perform_transpose(std::shared_ptr<const OmpExecutor> exec,
                              bool& do_transpose)
{
    do_transpose = false;
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const OmpExecutor> exec,
              matrix::view::csr<const ValueType, const IndexType> matrix,
              std::shared_ptr<solver::SolveStruct>& solve_struct,
              bool unit_diag, const solver::trisolve_algorithm algorithm,
              const size_type num_rhs)
{
    // This generate kernel is here to allow for a more sophisticated
    // implementation as for other executors. This kernel would perform the
    // "analysis" phase for the triangular matrix.
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_GENERATE_KERNEL);


/**
 * The parameters trans_x and trans_b are used only in the CUDA executor for
 * versions <=9.1 due to a limitation in the cssrsm_solve algorithm
 */
template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void solve(std::shared_ptr<const OmpExecutor> exec,
           matrix::view::csr<const MatrixValueType, const IndexType> matrix,
           const solver::SolveStruct* solve_struct, bool unit_diag,
           const solver::trisolve_algorithm algorithm,
           std::optional<matrix::view::dense<InputValueType>> trans_b,
           std::optional<matrix::view::dense<OutputValueType>> trans_x,
           matrix::view::dense<const InputValueType> b,
           matrix::view::dense<OutputValueType> x)
{
    using ArithmeticType =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    auto row_ptrs = matrix.row_ptrs;
    auto col_idxs = matrix.col_idxs;
    auto vals = matrix.values;
    const auto num_rows = matrix.size[0];
    const auto num_rhs = b.size[1];
    dense::fill(exec, x, nan<OutputValueType>());

#pragma omp parallel for schedule(monotonic : auto) collapse(2)
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type j = 0; j < num_rhs; ++j) {
            auto diag = one<ArithmeticType>();
            auto val = static_cast<ArithmeticType>(b(row, j));
            for (auto k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
                auto col = col_idxs[k];
                if (col < row) {
                    OutputValueType xval{};
                    while (is_nan_exact(xval = load_relaxed(&x(col, j)))) {
                    }
                    val = val - static_cast<ArithmeticType>(vals[k]) *
                                    static_cast<ArithmeticType>(xval);
                }
                if (col == row) {
                    diag = static_cast<ArithmeticType>(vals[k]);
                }
            }
            if (!unit_diag) {
                val /= diag;
            }
            if (is_nan_exact(val)) {
                val = zero<ArithmeticType>();
            }
            store_relaxed(&x(row, j), static_cast<OutputValueType>(val));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_SOLVE_KERNEL);


}  // namespace lower_trs
}  // namespace omp
}  // namespace kernels
}  // namespace gko
