/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/lower_trs_kernels.hpp"


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/base/mixed_precision_types.hpp"


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
template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void solve(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Csr<MatrixValueType, IndexType>* matrix,
           const solver::SolveStruct* solve_struct, bool unit_diag,
           const solver::trisolve_algorithm algorithm,
           matrix::Dense<InputValueType>* trans_b,
           matrix::Dense<OutputValueType>* trans_x,
           const matrix::Dense<InputValueType>* b,
           matrix::Dense<OutputValueType>* x)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;

    const auto row_ptrs = matrix->get_const_row_ptrs();
    const auto col_idxs = matrix->get_const_col_idxs();
    const auto vals = matrix->get_const_values();

    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        for (size_type row = 0; row < matrix->get_size()[0]; ++row) {
            auto diag = one<arithmetic_type>();
            bool found_diag = false;
            arithmetic_type result =
                static_cast<arithmetic_type>(b->at(row, j));
            for (auto k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
                auto col = col_idxs[k];
                if (col < row) {
                    result -=
                        static_cast<arithmetic_type>(vals[k]) *
                        static_cast<arithmetic_type>(x->at(col, j));
                }
                if (col == row) {
                    diag = static_cast<arithmetic_type>(vals[k]);
                    found_diag = true;
                }
            }
            if (!unit_diag) {
                GKO_ASSERT(found_diag);
                result /= diag;
            }
            x->at(row, j) = result;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_SOLVE_KERNEL);


}  // namespace lower_trs
}  // namespace reference
}  // namespace kernels
}  // namespace gko
