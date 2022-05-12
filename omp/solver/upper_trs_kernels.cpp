/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/solver/upper_trs_kernels.hpp"


#include <memory>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/base/mixed_precision_types.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The UPPER_TRS solver namespace.
 *
 * @ingroup upper_trs
 */
namespace upper_trs {


void should_perform_transpose(std::shared_ptr<const OmpExecutor> exec,
                              bool& do_transpose)
{
    do_transpose = false;
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const OmpExecutor> exec,
              const matrix::Csr<ValueType, IndexType>* matrix,
              std::shared_ptr<solver::SolveStruct>& solve_struct,
              const gko::size_type num_rhs)
{
    // This generate kernel is here to allow for a more sophisticated
    // implementation as for other executors. This kernel would perform the
    // "analysis" phase for the triangular matrix.
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_GENERATE_KERNEL);


/**
 * The parameters trans_x and trans_b are used only in the CUDA executor for
 * versions <=9.1 due to a limitation in the cssrsm_solve algorithm
 */
template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void solve(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Csr<MatrixValueType, IndexType>* matrix,
           const solver::SolveStruct* solve_struct,
           matrix::Dense<InputValueType>* trans_b,
           matrix::Dense<OutputValueType>* trans_x,
           const matrix::Dense<InputValueType>* b,
           matrix::Dense<OutputValueType>* x)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;

    const auto row_ptrs = matrix->get_const_row_ptrs();
    const auto col_idxs = matrix->get_const_col_idxs();

#pragma omp parallel for
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        for (size_type inv_row = 0; inv_row < matrix->get_size()[0];
             ++inv_row) {
            auto row = matrix->get_size()[0] - 1 - inv_row;
            arithmetic_type result =
                static_cast<arithmetic_type>(b->at(row, j)) /
                static_cast<arithmetic_type>(
                    matrix->get_const_values()[row_ptrs[row]]);
            for (auto k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
                auto col = col_idxs[k];
                if (col > row) {
                    result += -static_cast<arithmetic_type>(
                                  matrix->get_const_values()[k]) *
                              static_cast<arithmetic_type>(x->at(col, j)) /
                              static_cast<arithmetic_type>(
                                  matrix->get_const_values()[row_ptrs[row]]);
                }
            }
            x->at(row, j) = result;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_SOLVE_KERNEL);


}  // namespace upper_trs
}  // namespace omp
}  // namespace kernels
}  // namespace gko
