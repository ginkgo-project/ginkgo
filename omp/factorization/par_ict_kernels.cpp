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

#include "core/factorization/par_ict_kernels.hpp"


#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <unordered_set>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/utils.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"
#include "omp/components/csr_spgeam.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The parallel ICT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ict_factorization {


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *a,
                    matrix::Csr<ValueType, IndexType> *l,
                    const matrix::Coo<ValueType, IndexType> *)
{
    auto num_rows = a->get_size()[0];
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type l_nz = l_row_ptrs[row]; l_nz < l_row_ptrs[row + 1];
             ++l_nz) {
            auto col = l_col_idxs[l_nz];
            // find value from A
            auto a_begin = a_row_ptrs[row];
            auto a_end = a_row_ptrs[row + 1];
            auto a_nz_it =
                std::lower_bound(a_col_idxs + a_begin, a_col_idxs + a_end, col);
            auto a_nz = std::distance(a_col_idxs, a_nz_it);
            auto has_a = a_nz < a_end && a_col_idxs[a_nz] == col;
            auto a_val = has_a ? a_vals[a_nz] : zero<ValueType>();
            // accumulate l(row,:) * l(col,:) without the last entry l(col, col)
            ValueType sum{};
            IndexType lt_nz{};
            auto l_begin = l_row_ptrs[row];
            auto l_end = l_row_ptrs[row + 1];
            auto lt_begin = l_row_ptrs[col];
            auto lt_end = l_row_ptrs[col + 1];
            while (l_begin < l_end && lt_begin < lt_end) {
                auto l_col = l_col_idxs[l_begin];
                auto lt_row = l_col_idxs[lt_begin];
                if (l_col == lt_row && l_col < col) {
                    sum += l_vals[l_begin] * l_vals[lt_begin];
                }
                if (lt_row == row) {
                    lt_nz = lt_begin;
                }
                l_begin += (l_col <= lt_row);
                lt_begin += (lt_row <= l_col);
            }
            auto new_val = a_val - sum;
            if (row == col) {
                new_val = sqrt(new_val);
            } else {
                auto diag = l_vals[l_row_ptrs[col + 1] - 1];
                new_val = new_val / diag;
            }
            if (is_finite(new_val)) {
                l_vals[l_nz] = new_val;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ICT_COMPUTE_FACTOR_KERNEL);


template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *llt,
                    const matrix::Csr<ValueType, IndexType> *a,
                    const matrix::Csr<ValueType, IndexType> *l,
                    matrix::Csr<ValueType, IndexType> *l_new)
{
    auto num_rows = a->get_size()[0];
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_const_values();
    auto l_new_row_ptrs = l_new->get_row_ptrs();
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    // count nnz
    abstract_spgeam(
        a, llt, [](IndexType) { return IndexType{}; },
        [](IndexType row, IndexType col, ValueType, ValueType, IndexType &nnz) {
            nnz += col <= row;
        },
        [&](IndexType row, IndexType nnz) { l_new_row_ptrs[row] = nnz; });

    components::prefix_sum(exec, l_new_row_ptrs, num_rows + 1);

    // resize arrays
    auto l_nnz = l_new_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> l_builder{l_new};
    l_builder.get_col_idx_array().resize_and_reset(l_nnz);
    l_builder.get_value_array().resize_and_reset(l_nnz);
    auto l_new_col_idxs = l_new->get_col_idxs();
    auto l_new_vals = l_new->get_values();

    // accumulate non-zeros
    struct row_state {
        IndexType l_new_nz;
        IndexType l_old_begin;
        IndexType l_old_end;
    };
    abstract_spgeam(a, llt,
                    [&](IndexType row) {
                        row_state state{};
                        state.l_new_nz = l_new_row_ptrs[row];
                        state.l_old_begin = l_row_ptrs[row];
                        state.l_old_end = l_row_ptrs[row + 1];
                        return state;
                    },
                    [&](IndexType row, IndexType col, ValueType a_val,
                        ValueType llt_val, row_state &state) {
                        auto r_val = a_val - llt_val;
                        // load matching entry of L
                        auto l_col = checked_load(l_col_idxs, state.l_old_begin,
                                                  state.l_old_end, sentinel);
                        auto l_val =
                            checked_load(l_vals, state.l_old_begin,
                                         state.l_old_end, zero<ValueType>());
                        // load diagonal entry of L
                        auto diag = l_vals[l_row_ptrs[col + 1] - 1];
                        // if there is already an entry present, use that
                        // instead.
                        auto out_val = l_col == col ? l_val : r_val / diag;
                        // store output entries
                        if (row >= col) {
                            l_new_col_idxs[state.l_new_nz] = col;
                            l_new_vals[state.l_new_nz] = out_val;
                            state.l_new_nz++;
                        }
                        // advance entry of L if we used it
                        state.l_old_begin += (l_col == col);
                    },
                    [](IndexType, row_state) {});
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ICT_ADD_CANDIDATES_KERNEL);


}  // namespace par_ict_factorization
}  // namespace omp
}  // namespace kernels
}  // namespace gko
