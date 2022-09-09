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
#include <cmath>
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
                        array<remove_complex<ValueType>>& workspace,
                        gko::reorder::reordering_strategy strategy)
{
    constexpr auto inf =
        std::numeric_limits<remove_complex<ValueType>>::infinity();
    const auto nnz = mtx->get_num_stored_elements();
    const auto num_rows = mtx->get_size()[0];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto values = mtx->get_const_values();
    auto weight =
        strategy == gko::reorder::reordering_strategy::max_diagonal_sum
            ? [](ValueType a) { return abs(a); }
            : [](ValueType a) { return std::log2(abs(a)); };
    auto weights = workspace.get_data();
    auto u = weights + nnz;
    auto distance = u + num_rows;
    auto m = distance + num_rows;
    for (IndexType col = 0; col < num_rows; col++) {
        u[col] = inf;
        distance[col] = inf;
    }

    for (IndexType row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        auto row_max = -inf;
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto w = weight(values[idx]);
            weights[idx] = w;
            if (w > row_max) row_max = w;
        }

        m[row] = row_max;

        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto c = row_max - weights[idx];
            weights[idx] = c;
            const auto col = col_idxs[idx];
            if (c < u[col]) u[col] = c;
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
                      const array<ValueType>& workspace,
                      array<IndexType>& permutation,
                      array<IndexType>& inv_permutation,
                      array<IndexType>& parents)
{
    const auto nnz = row_ptrs[num_rows];
    const auto c = workspace.get_const_data();
    const auto u = c + nnz;
    auto p = permutation.get_data();
    auto ip = inv_permutation.get_data();
    auto idxs = parents.get_data() + 4 * num_rows;
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
            if (abs(c[idx] - u[col]) < 1e-14 && ip[col] == -1) {
                p[row] = col;
                ip[col] = row;
                idxs[row] = idx;
                matched = true;
                break;
            }
        }
        if (!matched) {
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
    while (row != 0 && um < num_rows) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        bool found = false;
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto col = col_idxs[idx];
            if (abs(c[idx] - u[col]) < 1e-14) {
                const auto row_1 = ip[col];
                const auto row_1_begin = row_ptrs[row_1];
                const auto row_1_end = row_ptrs[row_1 + 1];
                for (IndexType idx_1 = row_1_begin; idx_1 < row_1_end;
                     idx_1++) {
                    const auto col_1 = col_idxs[idx_1];
                    if (abs(c[idx_1] - u[col_1]) < 1e-14 && ip[col_1] == -1) {
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
    array<ValueType>& workspace, array<IndexType>& permutation,
    array<IndexType>& inv_permutation, IndexType root,
    array<IndexType>& parents,
    addressable_priority_queue<ValueType, IndexType, 2>& Q,
    std::vector<IndexType>& q_j)
{
    constexpr auto inf = std::numeric_limits<ValueType>::infinity();
    const auto nnz = row_ptrs[num_rows];
    auto c = workspace.get_data();
    auto u = c + nnz;
    auto distance = u + num_rows;

    auto p = permutation.get_data();
    auto ip = inv_permutation.get_data();

    auto parents_ = parents.get_data();
    auto handles = parents_ + num_rows;
    auto generation = handles + num_rows;
    auto marked_cols = generation + num_rows;
    auto idxs = marked_cols + num_rows;

    Q.reset();
    q_j.clear();

    ValueType lsp = inf;  // zero<ValueType>();
    ValueType lsap = inf;
    IndexType jsap = -1;

    auto row = root;
    auto marked_counter = 0;

    const auto begin = row_ptrs[row];
    const auto end = row_ptrs[row + 1];

    for (IndexType idx = begin; idx < end; idx++) {
        const auto col = col_idxs[idx];
        const ValueType dnew = c[idx] - u[col];

        if (dnew < lsap) {
            if (ip[col] == -1) {
                lsap = dnew;
                jsap = col;
                parents_[col] = row;
            } else {
                distance[col] = dnew;
                parents_[col] = row;
                generation[col] = num_rows + root;
                if (dnew < lsp) {
                    lsp = dnew;
                }
            }
        }
    }

    for (IndexType idx = begin; idx < end; idx++) {
        const auto col = col_idxs[idx];
        const auto dist = distance[col];
        const auto gen = generation[col];
        if (dist < lsap && gen == num_rows + root) {
            if (abs(dist - lsp) < 1e-14) {
                generation[col] = 2 * num_rows + root;
                q_j.push_back(col);
            } else {
                generation[col] = root;
                handles[col] = Q.insert(dist, col);
            }
        }
    }

    while (true) {
        if (q_j.size() > 0) {
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
                Q.pop_min();
                col = Q.min_val();
            }
            if (Q.empty()) break;
            lsp = distance[col];
            if (lsap <= lsp) break;
            generation[col] = -root;
            marked_cols[marked_counter++] = col;
            Q.pop_min();
            // while (Q.min_key() == lsp && !Q.empty()) {
            //     q_j.push_back(Q.min_val());
            //     Q.pop_min();
            // }
            row = ip[col];
        }
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        const auto vi = p[row] == -1 ? zero<ValueType>()
                                     : c[idxs[row]] - u[p[row]];  // v[row];
        for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto col = col_idxs[idx];
            const auto gen = generation[col];

            if (gen == -root) continue;

            const ValueType dnew = lsp + c[idx] - u[col] - vi;
            // if (col == 392960) std::cout << distance[col] << ", " << lsp <<
            // ", " << dnew << std::endl; if (dnew < lsp && abs(lsp - dnew) >
            // 1e-10){
            //     std::cout << root + num_rows << ", " << gen << ", " << jsap
            //     << ", " << col << std::endl; exit(1);
            // }
            if (dnew < lsap) {
                if (ip[col] == -1) {
                    lsap = dnew;
                    jsap = col;
                    parents_[col] = row;
                } else {
                    if ((gen != root || dnew < distance[col]) &&
                        gen != 2 * num_rows + root) {
                        distance[col] = dnew;
                        parents_[col] = row;
                        if (abs(dnew - lsp) < 1e-14) {
                            generation[col] = 2 * num_rows + root;
                            q_j.push_back(col);
                            // if (gen == root) {
                            //     Q.update_key(handles[col], -inf);
                            //     Q.pop_min();
                            // }
                        } else if (gen != root) {
                            // if (gen != root) {
                            generation[col] = root;  // num_rows + gen;
                            handles[col] = Q.insert(dnew, col);
                        } else {
                            generation[col] = root;  // num_rows + gen;
                            Q.update_key(handles[col], dnew);
                        }
                    }
                }
            }
        }
        /*for (IndexType idx = row_begin; idx < row_end; idx++) {
            const auto col = col_idxs[idx];
            const auto gen = generation[col];
            const auto dist = distance[col];
            if (dist < lsap) {
                generation[col] = root;
                if (dist == lsp) {
                    q_j.push_back(col);
                } else if ()
            }
        }*/
    }
    if (lsap != inf) {
        IndexType col = jsap;
        do {
            row = parents_[col];
            ip[col] = row;
            auto idx = row_ptrs[row];
            while (col_idxs[idx] != col) idx++;
            idxs[row] = idx;
            std::swap(col, p[row]);
        } while (row != root);
        for (size_type i = 0; i < marked_counter; i++) {
            const auto col = marked_cols[i];
            u[col] += distance[col] - lsap;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_SHORTEST_AUGMENTING_PATH_KERNEL);


template <typename ValueType, typename IndexType>
void compute_scaling(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Csr<ValueType, IndexType>* mtx,
                     const array<remove_complex<ValueType>>& workspace,
                     const array<IndexType>& permutation,
                     const array<IndexType>& parents,
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
    const auto weights = workspace.get_const_data();
    const auto u = weights + nnz;
    const auto m = u + 2 * num_rows;
    const auto p = permutation.get_const_data();
    const auto idxs = parents.get_const_data() + 4 * num_rows;
    auto rv = row_scaling->get_values();
    auto cv = col_scaling->get_values();

    if (strategy == gko::reorder::reordering_strategy::max_diagonal_product ||
        strategy ==
            gko::reorder::reordering_strategy::max_diagonal_product_fast) {
        // auto exp2_ = strategy ==
        // gko::reorder::reordering_strategy::max_diagonal_product_fast
        //     ? [](remove_complex<ValueType> a) { return fastexp2(a); }
        //     : [](remove_complex<ValueType> a) { return std::exp2(a); };
        for (size_type i = 0; i < num_rows; i++) {
            const remove_complex<ValueType> u_val = std::exp2(u[i]);
            const remove_complex<ValueType> v_val =
                weights[idxs[i]] - u[p[i]] - m[i];
            cv[i] = ValueType{u_val};
            rv[i] = ValueType{std::exp2(v_val)};
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
