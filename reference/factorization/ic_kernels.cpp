// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ic_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The ic factorization namespace.
 *
 * @ingroup factor
 */
namespace ic_factorization {


template <typename ValueType, typename IndexType>
void compute(std::shared_ptr<const DefaultExecutor> exec,
             matrix::Csr<ValueType, IndexType>* m)
{
    vector<IndexType> diagonals{m->get_size()[0], -1, exec};
    const auto row_ptrs = m->get_const_row_ptrs();
    const auto col_idxs = m->get_const_col_idxs();
    const auto values = m->get_values();
    for (size_type row = 0; row < m->get_size()[0]; row++) {
        const auto begin = row_ptrs[row];
        const auto end = row_ptrs[row + 1];
        for (auto nz = begin; nz < end; nz++) {
            const auto col = col_idxs[nz];
            if (col == row) {
                diagonals[row] = nz;
            }
            if (col > row) {
                continue;
            }
            // accumulate l(row,:) * l(col,:) without the last entry l(col, col)
            ValueType sum{};
            auto l_idx = begin;
            const auto l_end = end;
            auto lh_idx = row_ptrs[col];
            const auto lh_end = row_ptrs[col + 1];
            while (l_idx < l_end && lh_idx < lh_end) {
                const auto l_col = col_idxs[l_idx];
                const auto lh_row = col_idxs[lh_idx];
                // only consider lower triangle of L
                if (max(l_col, lh_row) > row) {
                    break;
                }
                // ignore l(col, col)
                if (l_col == lh_row && l_col < col) {
                    sum += values[l_idx] * conj(values[lh_idx]);
                }
                l_idx += l_col <= lh_row ? 1 : 0;
                lh_idx += lh_row <= l_col ? 1 : 0;
            }
            if (row == col) {
                values[nz] = sqrt(values[nz] - sum);
            } else {
                GKO_ASSERT(diagonals[col] != -1);
                values[nz] = (values[nz] - sum) / values[diagonals[col]];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IC_COMPUTE_KERNEL);


}  // namespace ic_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
