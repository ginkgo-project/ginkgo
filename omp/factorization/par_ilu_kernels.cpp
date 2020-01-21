/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include <algorithm>
#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The parallel ILU factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilu_factorization {


template <bool IsSorted = false>
struct find_helper {
    template <typename ForwardIt, typename IndexType>
    static bool find(ForwardIt first, ForwardIt last, const IndexType &value)
    {
        return std::find(first, last, value) != last;
    }
};


template <>
struct find_helper<true> {
    template <typename ForwardIt, typename IndexType>
    static bool find(ForwardIt first, ForwardIt last, const IndexType &value)
    {
        return std::binary_search(first, last, value);
    }
};


template <bool IsSorted, typename ValueType, typename IndexType>
void find_missing_diagonal_elements(
    const matrix::Csr<ValueType, IndexType> *mtx,
    IndexType *elements_to_add_per_row, bool *changes_required)
{
    auto size = mtx->get_size();
    auto values = mtx->get_const_values();
    auto col_idxs = mtx->get_const_col_idxs();
    auto row_ptrs = mtx->get_const_row_ptrs();
    size_type row_limit{(size[0] < size[1]) ? size[0] : size[1]};
    bool local_change{false};
#pragma omp parallel for reduction(|| : local_change)
    for (IndexType row = 0; row < row_limit; ++row) {
        const auto *start_cols = col_idxs + row_ptrs[row];
        const auto *end_cols = col_idxs + row_ptrs[row + 1];
        if (find_helper<IsSorted>::find(start_cols, end_cols, row)) {
            elements_to_add_per_row[row] = 0;
        } else {
            elements_to_add_per_row[row] = 1;
            local_change = true;
        }
    }
    *changes_required = local_change;
}


template <typename ValueType, typename IndexType>
void add_missing_diagonal_elements(matrix::Csr<ValueType, IndexType> *mtx,
                                   ValueType *new_values,
                                   IndexType *new_col_idxs,
                                   const IndexType *row_ptrs_add)
{
    const auto num_rows = mtx->get_size()[0];
    const auto old_values = mtx->get_const_values();
    const auto old_col_idxs = mtx->get_const_col_idxs();
    auto row_ptrs = mtx->get_row_ptrs();
#pragma omp parallel for
    for (IndexType row = 0; row < num_rows; ++row) {
        const IndexType old_row_start{row_ptrs[row]};
        const IndexType old_row_end{row_ptrs[row + 1]};
        const IndexType new_row_start{row_ptrs_add[row] + old_row_start};
        const IndexType new_row_end{row_ptrs_add[row + 1] + old_row_end};

        row_ptrs[row] = new_row_start;
        // if no element needs to be added, do a simple copy
        if (new_row_end - new_row_start == old_row_end - old_row_start) {
            for (IndexType i = 0; i < new_row_end - new_row_start; ++i) {
                const IndexType new_idx = new_row_start + i;
                const IndexType old_idx = old_row_start + i;
                new_values[new_idx] = old_values[old_idx];
                new_col_idxs[new_idx] = old_col_idxs[old_idx];
            }
        } else {
            IndexType new_idx = new_row_start;
            bool diagonal_added{false};
            for (IndexType old_idx = old_row_start; old_idx < old_row_end;
                 ++old_idx, ++new_idx) {
                const auto col_idx = old_col_idxs[old_idx];
                if (!diagonal_added && row < col_idx) {
                    new_values[new_idx] = zero<ValueType>();
                    new_col_idxs[new_idx] = row;
                    ++new_idx;
                    diagonal_added = true;
                }
                new_values[new_idx] = old_values[old_idx];
                new_col_idxs[new_idx] = old_col_idxs[old_idx];
            }
            if (!diagonal_added) {
                new_values[new_idx] = zero<ValueType>();
                new_col_idxs[new_idx] = row;
                diagonal_added = true;
            }
        }
    }
    row_ptrs[num_rows] += row_ptrs_add[num_rows];
}


template <typename ValueType, typename IndexType>
void add_diagonal_elements(std::shared_ptr<const DefaultExecutor> exec,
                           matrix::Csr<ValueType, IndexType> *mtx,
                           bool is_sorted)
{
    auto mtx_size = mtx->get_size();
    size_type row_ptrs_size = mtx_size[0] + 1;
    Array<IndexType> row_ptrs_addition(exec, row_ptrs_size);
    bool needs_change{};
    if (is_sorted) {
        find_missing_diagonal_elements<true>(mtx, row_ptrs_addition.get_data(),
                                             &needs_change);
    } else {
        find_missing_diagonal_elements<false>(mtx, row_ptrs_addition.get_data(),
                                              &needs_change);
    }
    if (!needs_change) {
        return;
    }

    row_ptrs_addition.get_data()[row_ptrs_size - 1] = 0;
    prefix_sum(exec, row_ptrs_addition.get_data(), row_ptrs_size);

    auto new_num_elems = row_ptrs_addition.get_data()[row_ptrs_size - 1] +
                         mtx->get_num_stored_elements();
    Array<ValueType> new_values{exec, new_num_elems};
    Array<IndexType> new_col_idxs{exec, new_num_elems};
    add_missing_diagonal_elements(mtx, new_values.get_data(),
                                  new_col_idxs.get_data(),
                                  row_ptrs_addition.get_const_data());

    matrix::CsrBuilder<ValueType, IndexType> mtx_builder{mtx};
    mtx_builder.get_value_array() = std::move(new_values);
    mtx_builder.get_col_idx_array() = std::move(new_col_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_ADD_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *system_matrix,
    IndexType *l_row_ptrs, IndexType *u_row_ptrs)
{
    auto num_rows = system_matrix->get_size()[0];
    auto row_ptrs = system_matrix->get_const_row_ptrs();
    auto col_idxs = system_matrix->get_const_col_idxs();

// Calculate the NNZ per row first
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        size_type l_nnz{};
        size_type u_nnz{};
        bool has_diagonal{};
        for (size_type el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            size_type col = col_idxs[el];
            if (col <= row) {
                ++l_nnz;
            }
            if (col >= row) {
                ++u_nnz;
            }
            has_diagonal |= col == row;
        }
        l_row_ptrs[row] = l_nnz + !has_diagonal;
        u_row_ptrs[row] = u_nnz + !has_diagonal;
    }

    // Now, compute the prefix-sum, to get proper row_ptrs for L and U
    prefix_sum(exec, l_row_ptrs, num_rows + 1);
    prefix_sum(exec, u_row_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_INITIALIZE_ROW_PTRS_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l_u(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *system_matrix,
                    matrix::Csr<ValueType, IndexType> *csr_l,
                    matrix::Csr<ValueType, IndexType> *csr_u)
{
    const auto row_ptrs = system_matrix->get_const_row_ptrs();
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const auto vals = system_matrix->get_const_values();

    const auto row_ptrs_l = csr_l->get_const_row_ptrs();
    auto col_idxs_l = csr_l->get_col_idxs();
    auto vals_l = csr_l->get_values();

    const auto row_ptrs_u = csr_u->get_const_row_ptrs();
    auto col_idxs_u = csr_u->get_col_idxs();
    auto vals_u = csr_u->get_values();

#pragma omp parallel for
    for (size_type row = 0; row < system_matrix->get_size()[0]; ++row) {
        size_type current_index_l = row_ptrs_l[row];
        size_type current_index_u =
            row_ptrs_u[row] + 1;  // we treat the diagonal separately
        bool has_diagonal{};
        ValueType diag_val{};
        for (size_type el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            const auto col = col_idxs[el];
            const auto val = vals[el];
            if (col < row) {
                col_idxs_l[current_index_l] = col;
                vals_l[current_index_l] = val;
                ++current_index_l;
            } else if (col == row) {
                // save value for later
                has_diagonal = true;
                diag_val = val;
            } else {  // col > row
                col_idxs_u[current_index_u] = col;
                vals_u[current_index_u] = val;
                ++current_index_u;
            }
        }
        // if there was no diagonal entry, set it to one
        if (!has_diagonal) {
            diag_val = one<ValueType>();
        }
        // store diagonal entries
        size_type l_diag_idx = row_ptrs_l[row + 1] - 1;
        size_type u_diag_idx = row_ptrs_u[row];
        col_idxs_l[l_diag_idx] = row;
        col_idxs_u[u_diag_idx] = row;
        vals_l[l_diag_idx] = one<ValueType>();
        vals_u[u_diag_idx] = diag_val;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_INITIALIZE_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const DefaultExecutor> exec,
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
                if (isfinite(to_write)) {
                    vals_l[row_l - 1] = to_write;
                }
            } else {  // modify entry in U
                auto to_write = sum;
                if (isfinite(to_write)) {
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
