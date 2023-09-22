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


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "omp/components/atomic.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The LOWER_TRS solver namespace.
 *
 * @ingroup lower_trs
 */
namespace lower_trs {


void should_perform_transpose(std::shared_ptr<const OmpExecutor> exec,
                              bool& do_transpose)
{
    do_transpose = false;
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const OmpExecutor> exec,
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
 * versions <=9.1 due to a limitation in the cssrsm_solve algorithm
 */
template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Csr<ValueType, IndexType>* matrix,
           const solver::SolveStruct* solve_struct, bool unit_diag,
           const solver::trisolve_algorithm algorithm,
           matrix::Dense<ValueType>* trans_b, matrix::Dense<ValueType>* trans_x,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x)
{
    auto row_ptrs = matrix->get_const_row_ptrs();
    auto col_idxs = matrix->get_const_col_idxs();
    auto vals = matrix->get_const_values();
    const auto num_rows = matrix->get_size()[0];
    const auto num_rhs = b->get_size()[1];
    dense::fill(exec, x, nan<ValueType>());

#pragma omp parallel for schedule(monotonic : auto) collapse(2)
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type j = 0; j < num_rhs; ++j) {
            auto diag = one<ValueType>();
            auto val = b->at(row, j);
            for (auto k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
                auto col = col_idxs[k];
                if (col < row) {
                    ValueType xval{};
                    while (is_nan(xval = atomic_load(x->at(col, j)))) {
                    }
                    val -= vals[k] * xval;
                }
                if (col == row) {
                    diag = vals[k];
                }
            }
            if (!unit_diag) {
                val /= diag;
            }
            if (is_nan(val)) {
                val = zero<ValueType>();
            }
            atomic_store(x->at(row, j), val);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_SOLVE_KERNEL);


}  // namespace lower_trs
}  // namespace omp
}  // namespace kernels
}  // namespace gko
