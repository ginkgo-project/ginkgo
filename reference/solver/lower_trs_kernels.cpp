// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/lower_trs_kernels.hpp"


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The LOWER_TRS solver namespace.
 *
 * @ingroup lower_trs
 */
namespace lower_trs {


void should_perform_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                              bool& do_transpose)
{
    do_transpose = false;
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const ReferenceExecutor> exec,
              const matrix::Csr<ValueType, IndexType>* matrix,
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
 * versions <=9.1 due to a limitation in the cssrsm_solve algorithm and hence
 * here essentially unused.
 */
template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Csr<ValueType, IndexType>* matrix,
           const solver::SolveStruct* solve_struct, bool unit_diag,
           const solver::trisolve_algorithm algorithm,
           matrix::Dense<ValueType>*, matrix::Dense<ValueType>* trans_x,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x)
{
    auto row_ptrs = matrix->get_const_row_ptrs();
    auto col_idxs = matrix->get_const_col_idxs();
    auto vals = matrix->get_const_values();

    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        for (size_type row = 0; row < matrix->get_size()[0]; ++row) {
            auto diag = one<ValueType>();
            bool found_diag = false;
            x->at(row, j) = b->at(row, j);
            for (auto k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
                auto col = col_idxs[k];
                if (col < row) {
                    x->at(row, j) -= vals[k] * x->at(col, j);
                }
                if (col == row) {
                    diag = vals[k];
                    found_diag = true;
                }
            }
            if (!unit_diag) {
                GKO_ASSERT(found_diag);
                x->at(row, j) /= diag;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_SOLVE_KERNEL);


}  // namespace lower_trs
}  // namespace reference
}  // namespace kernels
}  // namespace gko
