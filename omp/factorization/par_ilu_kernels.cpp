/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/factorization/par_ilu_kernels.hpp"


#include <memory>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The parallel ILU factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilu_factorization {


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const OmpExecutor> exec,
                         size_type iterations,
                         const matrix::Coo<ValueType, IndexType> *system_matrix,
                         matrix::Csr<ValueType, IndexType> *l_factor,
                         matrix::Csr<ValueType, IndexType> *u_factor)
{
    // If `iterations` is set to `Auto`, we do 3 fix-point sweeps as
    // experiements indicate this works well for many problems.
    iterations = (iterations == 0) ? 3 : iterations;
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const auto row_idxs = system_matrix->get_const_row_idxs();
    const auto vals = system_matrix->get_const_values();
    const auto row_ptrs_l = l_factor->get_const_row_ptrs();
    const auto row_ptrs_u = u_factor->get_const_row_ptrs();
    const auto col_idxs_l = l_factor->get_const_col_idxs();
    const auto col_idxs_u = u_factor->get_const_col_idxs();
    auto vals_l = l_factor->get_values();
    auto vals_u = u_factor->get_values();
    for (size_type iter = 0; iter < iterations; ++iter) {
        // all elements in the incomplete factors are updated in parallel
#pragma omp parallel for
        for (size_type el = 0; el < system_matrix->get_num_stored_elements();
             ++el) {
            const auto row = row_idxs[el];
            const auto col = col_idxs[el];
            const auto val = vals[el];
            auto row_l = row_ptrs_l[row];
            auto row_u = row_ptrs_u[col];
            ValueType sum{val};
            ValueType last_operation{};
            while (row_l < row_ptrs_l[row + 1] && row_u < row_ptrs_u[col + 1]) {
                auto col_l = col_idxs_l[row_l];
                auto col_u = col_idxs_u[row_u];
                if (col_l == col_u) {
                    last_operation = vals_l[row_l] * vals_u[row_u];
                    sum -= last_operation;
                } else {
                    last_operation = zero<ValueType>();
                }
                if (col_l <= col_u) {
                    ++row_l;
                }
                if (col_u <= col_l) {
                    ++row_u;
                }
            }
            // The loop above calculates: sum = system_matrix(row, col) -
            // dot(l_factor(row, :), u_factor(:, col))
            sum += last_operation;  // undo the last operation

            if (row > col) {  // modify entry in L
                auto to_write = sum / vals_u[row_ptrs_u[col + 1] - 1];
                if (is_finite(to_write)) {
                    vals_l[row_l - 1] = to_write;
                }
            } else {  // modify entry in U
                auto to_write = sum;
                if (is_finite(to_write)) {
                    vals_u[row_u - 1] = to_write;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_COMPUTE_L_U_FACTORS_KERNEL);


}  // namespace par_ilu_factorization
}  // namespace omp
}  // namespace kernels
}  // namespace gko
