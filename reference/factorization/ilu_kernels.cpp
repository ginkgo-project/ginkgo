// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ilu_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/math.hpp>


#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The ilu factorization namespace.
 *
 * @ingroup factor
 */
namespace ilu_factorization {


template <typename ValueType, typename IndexType>
void compute_lu(std::shared_ptr<const DefaultExecutor> exec,
                matrix::Csr<ValueType, IndexType>* m)
{
    vector<IndexType> diagonals{m->get_size()[0], -1, exec};
    const auto row_ptrs = m->get_const_row_ptrs();
    const auto col_idxs = m->get_const_col_idxs();
    const auto values = m->get_values();
    for (IndexType row = 0; row < static_cast<IndexType>(m->get_size()[0]);
         row++) {
        const auto begin = row_ptrs[row];
        const auto end = row_ptrs[row + 1];
        for (auto nz = begin; nz < end; nz++) {
            const auto col = col_idxs[nz];
            if (col == row) {
                diagonals[row] = nz;
            }
            auto value = values[nz];
            for (auto l_nz = begin; l_nz < end; l_nz++) {
                // for each lower triangular entry l_ik
                const auto l_col = col_idxs[l_nz];
                if (l_col >= min(row, col)) {
                    continue;
                }
                // find corresponding entry u_kj
                const auto u_begin_it = col_idxs + row_ptrs[l_col];
                const auto u_end_it = col_idxs + row_ptrs[l_col + 1];
                const auto u_it = std::lower_bound(u_begin_it, u_end_it, col);
                const auto u_nz = std::distance(col_idxs, u_it);
                if (u_it != u_end_it && *u_it == col) {
                    value -= values[l_nz] * values[u_nz];
                }
            }
            if (row <= col) {
                values[nz] = value;
            } else {
                GKO_ASSERT(diagonals[col] != -1);
                values[nz] = value / values[diagonals[col]];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILU_COMPUTE_LU_KERNEL);


}  // namespace ilu_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
