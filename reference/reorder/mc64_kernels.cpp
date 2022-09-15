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


#include <cmath>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


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
                        array<remove_complex<ValueType>>& value_workspace,
                        gko::reorder::reordering_strategy strategy)
{
    constexpr auto inf =
        std::numeric_limits<remove_complex<ValueType>>::infinity();
    const auto nnz = mtx->get_num_stored_elements();
    const auto num_rows = mtx->get_size()[0];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto values = mtx->get_const_values();
    auto calculate_weight =
        strategy == gko::reorder::reordering_strategy::max_diagonal_sum
            ? [](ValueType a) { return abs(a); }
            : [](ValueType a) { return std::log2(abs(a)); };
    auto weights = value_workspace.get_data();
    auto dual_u = weights + nnz;
    auto distance = dual_u + num_rows;
    auto row_maxima = distance + num_rows;
    for (IndexType col = 0; col < num_rows; col++) {
        dual_u[col] = inf;
        distance[col] = inf;
    }

    for (IndexType row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        auto row_max = -inf;
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto weight = calculate_weight(values[idx]);
            weights[idx] = weight;
            if (weight > row_max) row_max = weight;
        }

        row_maxima[row] = row_max;

        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto weight = row_max - weights[idx];
            weights[idx] = weight;
            const auto col = col_idxs[idx];
            if (weight < dual_u[col]) dual_u[col] = weight;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_INITIALIZE_WEIGHTS_KERNEL);


// Assume -1 in permutation and inv_permutation
template <typename ValueType, typename IndexType>
void initial_matching(std::shared_ptr<const DefaultExecutor> exec,
                      size_type num_rows, const IndexType* row_ptrs,
                      const IndexType* col_idxs,
                      const array<ValueType>& value_workspace,
                      array<IndexType>& permutation,
                      array<IndexType>& inv_permutation,
                      array<IndexType>& index_workspace, ValueType tolerance)
{
    const auto nnz = row_ptrs[num_rows];
    const auto weights = value_workspace.get_const_data();
    const auto dual_u = weights + nnz;
    auto p = permutation.get_data();
    auto ip = inv_permutation.get_data();
    auto idxs = index_workspace.get_data() + 4 * num_rows;
    auto unmatched = idxs + num_rows;
    auto um_cnt = 0;

    // For each row, look for an unmatched column col for which weight(row, col)
    // = 0. If one is found, add the edge (row, col) to the matching and move on
    // to the next row.
    for (IndexType row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        bool matched = false;
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto col = col_idxs[idx];
            if (abs(weights[idx] - dual_u[col]) < tolerance && ip[col] == -1) {
                p[row] = col;
                ip[col] = row;
                idxs[row] = idx;
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Mark unmatched rows for later.
            unmatched[um_cnt++] = row;
        }
    }

    // For remaining unmatched rows, look for a matched column with weight(row,
    // col) = 0 that is matched to another row, row_1. If there is another
    // column col_1 with weight(row_1, col_1) = 0 that is not yet matched,
    // replace the matched edge (row_1, col) with the two new matched edges
    // (row, col) and (row_1, col_1).
    auto um = 0;
    auto row = unmatched[um];
    // If row == 0 we passed the last unmatched row and reached the
    // zero-initialized part of the array. Row 0 is always matched as the matrix
    // is assumed to be nonsingular and the previous loop starts with row 0.
    while (row != 0 && um < num_rows) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        bool found = false;
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto col = col_idxs[idx];
            if (abs(weights[idx] - dual_u[col]) < tolerance) {
                const auto row_1 = ip[col];
                const auto row_1_begin = row_ptrs[row_1];
                const auto row_1_end = row_ptrs[row_1 + 1];
                for (IndexType idx_1 = row_1_begin; idx_1 < row_1_end;
                     idx_1++) {
                    const auto col_1 = col_idxs[idx_1];
                    if (abs(weights[idx_1] - dual_u[col_1]) < tolerance &&
                        ip[col_1] == -1) {
                        p[row] = col;
                        ip[col] = row;
                        idxs[row] = idx;
                        p[row_1] = col_1;
                        ip[col_1] = row_1;
                        idxs[row_1] = idx_1;
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
        }
        if (found) {
            // Mark previously unmatched row as matched.
            unmatched[um] = -1;
        }
        row = unmatched[++um];
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_INITIAL_MATCHING_KERNEL);


template <typename ValueType, typename IndexType>
void shortest_augmenting_path(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_rows,
    const IndexType* row_ptrs, const IndexType* col_idxs,
    array<ValueType>& value_workspace, array<IndexType>& permutation,
    array<IndexType>& inv_permutation, IndexType root,
    array<IndexType>& index_workspace,
    addressable_priority_queue<ValueType, IndexType>& Q,
    std::vector<IndexType>& q_j, ValueType tolerance)
{
    constexpr auto inf = std::numeric_limits<ValueType>::infinity();
    const auto nnz = row_ptrs[num_rows];
    auto weights = value_workspace.get_data();
    auto dual_u = weights + nnz;
    auto distance = dual_u + num_rows;

    auto p = permutation.get_data();
    auto ip = inv_permutation.get_data();

    auto parents = index_workspace.get_data();
    // Handles to access and update entries in the addressable priority queue.
    auto handles = parents + num_rows;
    // Generation array to mark visited nodes.
    auto generation = handles + num_rows;
    // Set of marked columns whos shortest alternating paths and distances to
    // the root are known.
    auto marked_cols = generation + num_rows;
    // Indices of the nonzero entries corresponding to the matched column in
    // each matched row. So, if row i is matched to column j, W(i,j) is found
    // at weights[idxs[i]] where W is the weight matrix.
    auto idxs = marked_cols + num_rows;

    Q.reset();
    q_j.clear();

    // The length of the current path.
    ValueType lsp = inf;
    // The length of the currently shortest found augmenting path starting from
    // root.
    ValueType lsap = inf;
    // The column at the end of the currently shortest found augmenting path.
    IndexType jsap = -1;

    auto row = root;
    auto marked_counter = 0;

    const auto begin = row_ptrs[row];
    const auto end = row_ptrs[row + 1];

    // Look for matching candidates in the row corresponding to root.
    // As root is not yet matched, the corresponding entry in the dual
    // vector v is 0 so we do not have to compute it.
    for (IndexType idx = begin; idx < end; idx++) {
        const auto col = col_idxs[idx];
        const ValueType dnew = weights[idx] - dual_u[col];

        if (dnew < lsap) {
            if (ip[col] == -1) {
                // col is unmatched so we found an augmenting path.
                lsap = dnew;
                jsap = col;
                parents[col] = row;
            } else {
                distance[col] = dnew;
                parents[col] = row;
                generation[col] = num_rows + root;
                if (dnew < lsp) {
                    lsp = dnew;
                }
            }
        }
    }

    // Write the columns in the row corresponding to root with the
    // smallest distance into q_j, other columns with distance
    // smaller than lsap into the priority queue Q.
    for (IndexType idx = begin; idx < end; idx++) {
        const auto col = col_idxs[idx];
        const auto dist = distance[col];
        const auto gen = generation[col];
        if (dist < lsap && gen == num_rows + root) {
            if (abs(dist - lsp) < tolerance) {
                generation[col] = -num_rows - root;
                q_j.push_back(col);
            } else {
                generation[col] = root;
                handles[col] = Q.insert(dist, col);
            }
        }
    }

    while (true) {
        // Mark the column with the shortest known distance to the root
        // and proceed in its matched row. If both q_j and Q are empty
        // or if the current path becomes longer than the currently
        // shortest augmenting path, we are done.
        if (q_j.size() > 0) {
            // q_j is known to contain only entries with shortest known
            // distance to the root, so if it is not empty we do not
            // have to operate on the priority queue.
            if (lsap <= lsp) break;
            const auto col = q_j.back();
            q_j.pop_back();
            generation[col] = -root;
            marked_cols[marked_counter++] = col;
            row = ip[col];
        } else {
            if (Q.empty()) break;
            auto col = Q.min_val();
            while (generation[col] == -root && !Q.empty()) {
                // If col is already marked because it previously was in q_j
                // we have to disregard it.
                Q.pop_min();
                col = Q.min_val();
            }
            if (Q.empty()) break;
            lsp = distance[col];
            if (lsap <= lsp) break;
            generation[col] = -root;
            marked_cols[marked_counter++] = col;
            Q.pop_min();
            row = ip[col];
        }
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        // Compute the entry of the dual vector v corresponding to row.
        const auto dual_vi = p[row] == -1 ? zero<ValueType>()
                                          : weights[idxs[row]] - dual_u[p[row]];
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto col = col_idxs[idx];
            const auto gen = generation[col];

            // col is already marked. Note that root will never be 0 as this row
            // is guaranteed to already be part of the initial matching.
            if (gen == -root) continue;

            const ValueType dnew = lsp + weights[idx] - dual_u[col] - dual_vi;

            if (dnew < lsap) {
                if (ip[col] == -1) {
                    // col is unmatched so we found an augmenting path.
                    lsap = dnew;
                    jsap = col;
                    parents[col] = row;
                } else {
                    if ((gen != root || dnew < distance[col]) &&
                        gen != -num_rows - root) {
                        distance[col] = dnew;
                        parents[col] = row;
                        if (abs(dnew - lsp) < tolerance) {
                            // dnew is the shortest currently possible distance,
                            // so col can be put into q_j and be marked
                            // accordingly.
                            generation[col] = -num_rows - root;
                            q_j.push_back(col);
                        } else if (gen != root) {
                            // col was not encountered before.
                            generation[col] = root;
                            handles[col] = Q.insert(dnew, col);
                        } else {
                            // col was already encountered but with larger
                            // distance on a different path.
                            generation[col] = root;
                            Q.update_key(handles[col], dnew);
                        }
                    }
                }
            }
        }
    }
    if (lsap != inf) {
        IndexType col = jsap;
        // Update the matching along the shortest augmenting path.
        do {
            row = parents[col];
            ip[col] = row;
            auto idx = row_ptrs[row];
            while (col_idxs[idx] != col) idx++;
            idxs[row] = idx;
            std::swap(col, p[row]);
        } while (row != root);
        // Update the dual vector u.
        for (size_type i = 0; i < marked_counter; i++) {
            const auto col = marked_cols[i];
            dual_u[col] += distance[col] - lsap;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_SHORTEST_AUGMENTING_PATH_KERNEL);


template <typename ValueType, typename IndexType>
void compute_scaling(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Csr<ValueType, IndexType>* mtx,
                     const array<remove_complex<ValueType>>& value_workspace,
                     const array<IndexType>& permutation,
                     const array<IndexType>& index_workspace,
                     gko::reorder::reordering_strategy strategy,
                     gko::matrix::Diagonal<ValueType>* row_scaling,
                     gko::matrix::Diagonal<ValueType>* col_scaling)
{
    constexpr auto inf =
        std::numeric_limits<remove_complex<ValueType>>::infinity();
    const auto nnz = mtx->get_num_stored_elements();
    const auto num_rows = mtx->get_size()[0];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto values = mtx->get_const_values();
    const auto weights = value_workspace.get_const_data();
    const auto dual_u = weights + nnz;
    const auto row_maxima = dual_u + 2 * num_rows;
    const auto p = permutation.get_const_data();
    const auto idxs = index_workspace.get_const_data() + 4 * num_rows;
    auto rv = row_scaling->get_values();
    auto cv = col_scaling->get_values();

    if (strategy == gko::reorder::reordering_strategy::max_diagonal_product) {
        for (size_type i = 0; i < num_rows; i++) {
            const remove_complex<ValueType> u_val = std::exp2(dual_u[i]);
            const remove_complex<ValueType> v_val =
                std::exp2(weights[idxs[i]] - dual_u[p[i]] - row_maxima[i]);
            cv[i] = ValueType{u_val};
            rv[i] = ValueType{v_val};
        }
    } else {
        for (size_type i = 0; i < num_rows; i++) {
            cv[i] = 1.;
            rv[i] = 1.;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_COMPUTE_SCALING_KERNEL);

}  // namespace mc64
}  // namespace reference
}  // namespace kernels
}  // namespace gko
