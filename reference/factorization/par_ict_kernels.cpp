// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ict_kernels.hpp"


#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <unordered_set>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "reference/components/csr_spgeam.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The parallel ict factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ict_factorization {


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* a,
                    matrix::Csr<ValueType, IndexType>* l,
                    const matrix::Coo<ValueType, IndexType>*)
{
    auto num_rows = a->get_size()[0];
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();

    for (size_type row = 0; row < num_rows; ++row) {
        for (auto l_nz = l_row_ptrs[row]; l_nz < l_row_ptrs[row + 1]; ++l_nz) {
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
            IndexType lh_nz{};
            auto l_begin = l_row_ptrs[row];
            auto l_end = l_row_ptrs[row + 1];
            auto lh_begin = l_row_ptrs[col];
            auto lh_end = l_row_ptrs[col + 1];
            while (l_begin < l_end && lh_begin < lh_end) {
                auto l_col = l_col_idxs[l_begin];
                auto lh_row = l_col_idxs[lh_begin];
                if (l_col == lh_row && l_col < col) {
                    sum += l_vals[l_begin] * conj(l_vals[lh_begin]);
                }
                if (lh_row == row) {
                    lh_nz = lh_begin;
                }
                l_begin += (l_col <= lh_row);
                lh_begin += (lh_row <= l_col);
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
                    const matrix::Csr<ValueType, IndexType>* llh,
                    const matrix::Csr<ValueType, IndexType>* a,
                    const matrix::Csr<ValueType, IndexType>* l,
                    matrix::Csr<ValueType, IndexType>* l_new)
{
    auto num_rows = a->get_size()[0];
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_const_values();
    auto l_new_row_ptrs = l_new->get_row_ptrs();
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    // count nnz
    IndexType l_nnz{};
    abstract_spgeam(
        a, llh,
        [&](IndexType row) {
            l_new_row_ptrs[row] = l_nnz;
            return 0;
        },
        [&](IndexType row, IndexType col, ValueType, ValueType, int) {
            l_nnz += col <= row;
        },
        [](IndexType, int) {});
    l_new_row_ptrs[num_rows] = l_nnz;

    // resize arrays
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
    abstract_spgeam(
        a, llh,
        [&](IndexType row) {
            row_state state{};
            state.l_new_nz = l_new_row_ptrs[row];
            state.l_old_begin = l_row_ptrs[row];
            state.l_old_end = l_row_ptrs[row + 1];
            return state;
        },
        [&](IndexType row, IndexType col, ValueType a_val, ValueType llh_val,
            row_state& state) {
            auto r_val = a_val - llh_val;
            // load matching entry of L
            auto l_col = checked_load(l_col_idxs, state.l_old_begin,
                                      state.l_old_end, sentinel);
            auto l_val = checked_load(l_vals, state.l_old_begin,
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
}  // namespace reference
}  // namespace kernels
}  // namespace gko
