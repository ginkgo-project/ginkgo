/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The parallel ilu factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilu_factorization {


template <typename ValueType, typename IndexType>
void compute_nnz_l_u(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Csr<ValueType, IndexType> *system_matrix,
                     size_type *l_nnz, size_type *u_nnz)
{
    auto rowpts = system_matrix->get_const_row_ptrs();
    auto cols = system_matrix->get_const_col_idxs();
    *l_nnz = 0;
    *u_nnz = 0;
    for (size_type row = 0; row < system_matrix->get_size()[1]; ++row) {
        for (size_type el = rowpts[row]; el < rowpts[row + 1]; ++el) {
            size_type col = cols[el];
            if (col <= row) {
                ++(*l_nnz);
            }
            if (col >= row) {
                ++(*u_nnz);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_COMPUTE_NNZ_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l_u(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *system_matrix,
                    matrix::Csr<ValueType, IndexType> *csr_l,
                    matrix::Csr<ValueType, IndexType> *csr_u)
{
    const auto rowpts = system_matrix->get_const_row_ptrs();
    const auto cols = system_matrix->get_const_col_idxs();
    const auto vals = system_matrix->get_const_values();

    auto rowpts_l = csr_l->get_row_ptrs();
    auto cols_l = csr_l->get_col_idxs();
    auto vals_l = csr_l->get_values();

    auto rowpts_u = csr_u->get_row_ptrs();
    auto cols_u = csr_u->get_col_idxs();
    auto vals_u = csr_u->get_values();

    size_type current_index_l{};
    size_type current_index_u{};
    rowpts_l[current_index_l] = zero<IndexType>();
    rowpts_u[current_index_u] = zero<IndexType>();
    for (size_type row = 0; row < system_matrix->get_size()[1]; ++row) {
        for (size_type el = rowpts[row]; el < rowpts[row + 1]; ++el) {
            size_type col = cols[el];
            if (col < row) {
                cols_l[current_index_l] = col;
                vals_l[current_index_l] = vals[el];
                ++current_index_l;
            } else if (col == row) {
                // Update both L and U
                cols_l[current_index_l] = col;
                vals_l[current_index_l] = one<ValueType>();
                ++current_index_l;

                cols_u[current_index_u] = col;
                vals_u[current_index_u] = vals[el];
                ++current_index_u;
            } else {  // col > row
                cols_u[current_index_u] = col;
                vals_u[current_index_u] = vals[el];
                ++current_index_u;
            }
        }
        rowpts_l[row + 1] = current_index_l;
        rowpts_u[row + 1] = current_index_u;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_INITIALIZE_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const DefaultExecutor> exec,
                         unsigned int iterations,
                         const matrix::Coo<ValueType, IndexType> *system_matrix,
                         matrix::Csr<ValueType, IndexType> *l_factor,
                         matrix::Csr<ValueType, IndexType> *u_factor)
{
    const auto cols = system_matrix->get_const_col_idxs();
    const auto rows = system_matrix->get_const_row_idxs();
    const auto vals = system_matrix->get_const_values();
    const auto l_rowpts = l_factor->get_const_row_ptrs();
    const auto u_rowpts = u_factor->get_const_row_ptrs();
    const auto l_cols = l_factor->get_const_col_idxs();
    const auto u_cols = u_factor->get_const_col_idxs();
    auto l_vals = l_factor->get_values();
    auto u_vals = u_factor->get_values();
    for (decltype(iterations) iter = 0; iter < iterations; ++iter) {
        for (size_type el = 0; el < system_matrix->get_num_stored_elements();
             ++el) {
            ValueType sum{};
            ValueType tmp{};
            const auto row = rows[el];
            const auto col = cols[el];
            const auto val = vals[el];
            auto il = l_rowpts[row];
            auto iu = u_rowpts[col];
            while (il < l_rowpts[row + 1] && iu < u_rowpts[col + 1]) {
                tmp = zero<ValueType>();
                auto jl = l_cols[il];
                auto ju = u_cols[iu];
                tmp = (jl == ju) ? l_vals[il] * u_vals[iu] : tmp;
                sum = (jl == ju) ? sum - tmp : sum;
                il = (jl <= ju) ? il + 1 : il;
                iu = (jl >= ju) ? iu + 1 : iu;
            }
            sum += tmp;  // undo the last operation (it must be the last)
            // TODO: Check if this is the correct comparison
            if (iu > il) {  // modify entry in L
                l_vals[il - 1] = sum / u_vals[u_rowpts[col + 1] - 1];
            } else {  // modify entry in U
                u_vals[iu - 1] = sum;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_COMPUTE_L_U_FACTORS_KERNEL);


}  // namespace par_ilu_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
