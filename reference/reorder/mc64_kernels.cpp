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

#include "core/reorder/mc64_kernels.hpp"


#include <algorithm>
#include <iterator>
#include <memory>
#include <queue>
#include <set>
#include <utility>
#include <vector>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace mc64 {


template <typename ValueType, typename IndexType>
void initialize_weights(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* mtx,
                        Array<remove_complex<ValueType>>& workspace)
{
    constexpr auto inf =
        std::numeric_limits<remove_complex<ValueType>>::infinity();
    const auto nnz = mtx->get_num_stored_elements();
    const auto num_rows = mtx->get_size()[0];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto values = mtx->get_const_values();
    auto weight = [](ValueType a) { return abs(a); };
    workspace.resize_and_reset(nnz + 2 * num_rows);
    auto weights = workspace.get_data();
    auto u = weights + nnz;
    auto v = u + num_rows;
    for (IndexType col = 0; col < num_rows; col++) {
        u[col] = inf;
        v[col] = zero<IndexType>();
    }

    for (IndexType row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        auto row_max = zero<remove_complex<ValueType>>();
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto w = abs(values[idx]);
            if (w > row_max) row_max = w;
        }

        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto c = weight(row_max) - weight(values[idx]);
            weights[idx] = c;
            const auto col = col_idxs[idx];
            if (c < u[col]) u[col] = c;
        }
    }

    // TODO: check if this really is not necessary
    /*for (IndexType row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        auto row_min = inf;
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto c = weights[idx] - u[col_idxs[idx]];
            if (c < row_min) row_min = c;
        }
        v[row] = row_min;
    }*/
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_INITIALIZE_WEIGHTS_KERNEL);


// Assume -1 in permutation and inv_permutation
template <typename ValueType, typename IndexType>
void initial_matching(std::shared_ptr<const DefaultExecutor> exec,
                      size_type num_rows, const IndexType* row_ptrs,
                      const IndexType* col_idxs,
                      const Array<ValueType>& workspace,
                      Array<IndexType>& permutation,
                      Array<IndexType>& inv_permutation,
                      std::list<IndexType>& unmatched_rows)
{
    const auto nnz = row_ptrs[num_rows];
    const auto c = workspace.get_const_data();
    const auto u = c + nnz;
    const auto v = u + num_rows;
    auto weight = [c, u, v](IndexType row, IndexType col, IndexType idx) {
        return c[idx] - u[col] - v[row];
    };
    auto p = permutation.get_data();
    auto ip = inv_permutation.get_data();

    // For each row, look for an unmatched column col for which weight(row, col)
    // = 0. If one is found, add the edge (row, col) to the matching and move on
    // to the next row.
    for (IndexType row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto col = col_idxs[idx];
            if (weight(row, col, idx) == zero<ValueType>() && ip[col] == -1) {
                p[row] = col;
                ip[col] = row;
                break;
            }
        }
        if (p[row] == -1) unmatched_rows.push_back(row);
    }

    // For remaining unmatched rows, look for a matched column with weight(row,
    // col) = 0 that is matched to another row, row_1. If there is another
    // column col_1 with weight(row_1, col_1) = 0 that is not yet matched,
    // replace the matched edge (row_1, col) with the two new matched edges
    // (row, col) and (row_1, col_1).
    auto it = unmatched_rows.begin();
    while (it != unmatched_rows.end()) {
        const auto row = *it;
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        bool found = false;
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto col = col_idxs[idx];
            if (weight(row, col, idx) == zero<ValueType>()) {
                const auto row_1 = ip[col];
                const auto row_1_begin = row_ptrs[row_1];
                const auto row_1_end = row_ptrs[row_1 + 1];
                for (IndexType idx_1 = row_1_begin; idx_1 < row_1_end;
                     idx_1++) {
                    const auto col_1 = col_idxs[idx_1];
                    if (weight(row_1, col_1, idx_1) == zero<ValueType>() &&
                        ip[col_1] == -1) {
                        p[row] = col;
                        ip[col] = row;
                        p[row_1] = col_1;
                        ip[col_1] = row_1;
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
        }
        if (found)
            it = unmatched_rows.erase(it);
        else
            it++;
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_INITIAL_MATCHING_KERNEL);


template <typename ValueType, typename IndexType>
void shortest_augmenting_path(std::shared_ptr<const DefaultExecutor> exec,
                              size_type num_rows, const IndexType* row_ptrs,
                              const IndexType* col_idxs,
                              Array<ValueType>& workspace,
                              Array<IndexType>& permutation,
                              Array<IndexType>& inv_permutation, IndexType root,
                              Array<IndexType>& parents)
{
    const auto nnz = row_ptrs[num_rows];
    auto c = workspace.get_data();
    auto u = c + nnz;
    auto v = u + num_rows;
    auto weight = [c, u, v](IndexType row, IndexType col, IndexType idx) {
        return c[idx] - u[col] - v[row];
    };
    auto p = permutation.get_data();
    auto ip = inv_permutation.get_data();
    parents.fill(-one<IndexType>());
    auto parents_ = parents.get_data();

    std::vector<ValueType> distance(num_rows, -one<ValueType>());
    auto cmp = [distance](IndexType a, IndexType b) {
        return (distance[b] == -one<IndexType>()) ||
               (distance[a] <= distance[b]);
    };
    std::set<IndexType> marked_cols;
    std::set<IndexType, decltype(cmp)> Q(cmp);
    ValueType lsp = 0;
    ValueType lsap = -1;
    IndexType jsap;

    auto row = root;

    while (true) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto col = col_idxs[idx];
            if (marked_cols.find(col) != marked_cols.end()) continue;
            ValueType dnew = lsp + weight(row, col, idx);
            if (lsap < 0 || dnew < lsap) {
                if (ip[col] == -1) {
                    lsap = dnew;
                    jsap = col;
                    parents_[col] = row;
                } else {
                    if (distance[col] == -1 || dnew < distance[col]) {
                        bool new_col = false;
                        if (distance[col] == -1) new_col = true;
                        distance[col] = dnew;
                        parents_[col] = row;
                        if (!new_col) Q.erase(col);
                        Q.insert(col);
                    }
                }
            }
        }

        if (Q.empty()) break;
        auto col_pos = Q.begin();
        auto col = *col_pos;
        lsp = distance[col];
        if (lsap >= 0 && lsap <= lsp) break;
        Q.erase(col_pos);
        marked_cols.insert(col);
        row = ip[col];
    }
    if (lsap != -1) {
        auto row = -1;
        auto col = -1;
        auto next_col = jsap;
        while (row != root) {
            col = next_col;
            row = parents_[col];
            next_col = p[row];
            p[row] = col;
            ip[col] = row;
        }

        for (auto marked_col : marked_cols) {
            u[marked_col] += distance[marked_col] - lsap;
        }

        for (size_type row = 0; row < num_rows; row++) {
            if (p[row] != -1) {
                auto col = p[row];
                auto idx = row_ptrs[row];
                while (col_idxs[idx] != col) idx++;
                v[row] = c[idx] - u[col];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_SHORTEST_AUGMENTING_PATH_KERNEL);
}  // namespace mc64
}  // namespace reference
}  // namespace kernels
}  // namespace gko
