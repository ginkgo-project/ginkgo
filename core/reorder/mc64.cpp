/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/reorder/mc64.hpp>


#include <chrono>
#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/addressable_pq.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace experimental {
namespace reorder {
namespace mc64 {


#define GKO_DECLARE_MC64_INITIALIZE_WEIGHTS(ValueType, IndexType) \
    void initialize_weights(                                      \
        const matrix::Csr<ValueType, IndexType>* mtx,             \
        array<remove_complex<ValueType>>& weights_array,          \
        array<remove_complex<ValueType>>& dual_u_array,           \
        array<remove_complex<ValueType>>& distance_array,         \
        array<remove_complex<ValueType>>& row_maxima_array,       \
        gko::experimental::reorder::mc64_strategy strategy)

template <typename ValueType, typename IndexType>
void initialize_weights(const matrix::Csr<ValueType, IndexType>* mtx,
                        array<remove_complex<ValueType>>& weights_array,
                        array<remove_complex<ValueType>>& dual_u_array,
                        array<remove_complex<ValueType>>& distance_array,
                        array<remove_complex<ValueType>>& row_maxima_array,
                        gko::experimental::reorder::mc64_strategy strategy)
{
    constexpr auto inf =
        std::numeric_limits<remove_complex<ValueType>>::infinity();
    const auto nnz = mtx->get_num_stored_elements();
    const auto num_rows = mtx->get_size()[0];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto values = mtx->get_const_values();
    auto weights = weights_array.get_data();
    auto dual_u = dual_u_array.get_data();
    auto distance = distance_array.get_data();
    auto row_maxima = row_maxima_array.get_data();
    dual_u_array.fill(inf);
    distance_array.fill(inf);
    auto run_computation = [&](auto calculate_weight) {
        for (IndexType row = 0; row < num_rows; row++) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            auto row_max = -inf;
            for (IndexType idx = row_begin; idx < row_end; idx++) {
                const auto weight = calculate_weight(values[idx]);
                weights[idx] = weight;
                row_max = std::max(weight, row_max);
            }

            row_maxima[row] = row_max;

            for (IndexType idx = row_begin; idx < row_end; idx++) {
                const auto weight = row_max - weights[idx];
                weights[idx] = weight;
                const auto col = col_idxs[idx];
                dual_u[col] = std::min(weight, dual_u[col]);
            }
        }
    };
    if (strategy ==
        gko::experimental::reorder::mc64_strategy::max_diagonal_sum) {
        run_computation([](ValueType a) { return abs(a); });
    } else {
        run_computation([](ValueType a) { return std::log2(abs(a)); });
    }
}


#define GKO_DECLARE_MC64_INITIAL_MATCHING(ValueType, IndexType)              \
    void initial_matching(                                                   \
        size_type num_rows, const IndexType* row_ptrs,                       \
        const IndexType* col_idxs, const array<ValueType>& weights_array,    \
        const array<ValueType>& dual_u_array, array<IndexType>& permutation, \
        array<IndexType>& inv_permutation,                                   \
        array<IndexType>& matched_idxs_array,                                \
        array<IndexType>& unmatched_rows_array, ValueType tolerance)

// Assume -1 in permutation and inv_permutation
template <typename ValueType, typename IndexType>
void initial_matching(
    size_type num_rows, const IndexType* row_ptrs, const IndexType* col_idxs,
    const array<ValueType>& weights_array, const array<ValueType>& dual_u_array,
    array<IndexType>& permutation, array<IndexType>& inv_permutation,
    array<IndexType>& matched_idxs_array,
    array<IndexType>& unmatched_rows_array, ValueType tolerance)
{
    const auto nnz = row_ptrs[num_rows];
    const auto weights = weights_array.get_const_data();
    const auto dual_u = dual_u_array.get_const_data();
    auto p = permutation.get_data();
    auto ip = inv_permutation.get_data();
    auto idxs = matched_idxs_array.get_data();
    auto unmatched = unmatched_rows_array.get_data();
    size_type um_count = 0;

    // In the following comments, w(row, col) will refer to the reduced weight
    // abs(weights(row, col) - dual_u(col)) where dual_u is a dual vector
    // needed for non-negativity of all weights.
    // For each row, look for an unmatched column col for which
    // w(row, col) < tolerance. If one is found, add the edge (row, col) to the
    // matching and move on to the next row.
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
            unmatched[um_count++] = row;
        }
    }

    // For remaining unmatched rows, look for a matched column with i
    // w(row, col) < tolerance that is matched to another row, row_1.
    // If there is another column col_1 with w(row_1, col_1) < tolerance
    // that is not yet matched, replace the matched edge (row_1, col)
    // with the two new matched edges (row, col) and (row_1, col_1).
    size_type um = 0;
    auto row = unmatched[um];
    // If row == 0 we passed the last unmatched row and reached the
    // zero-initialized part of the array. Row 0 is always matched as the matrix
    // is assumed to be nonsingular and the previous loop starts with row 0.
    while (row != 0 && um < num_rows) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        bool found = [&] {
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
                            return true;
                        }
                    }
                }
            }
            return false;
        }();
        if (found) {
            // Mark previously unmatched row as matched.
            unmatched[um] = -1;
        }
        row = unmatched[++um];
    }
}


#define GKO_DECLARE_MC64_SHORTEST_AUGMENTING_PATH(ValueType, IndexType)   \
    void shortest_augmenting_path(                                        \
        size_type num_rows, const IndexType* row_ptrs,                    \
        const IndexType* col_idxs, array<ValueType>& weights_array,       \
        array<ValueType>& dual_u_array, array<ValueType>& distance_array, \
        array<IndexType>& permutation, array<IndexType>& inv_permutation, \
        IndexType root, array<IndexType>& parents_array,                  \
        array<IndexType>& generation_array,                               \
        array<IndexType>& marked_cols_array,                              \
        array<IndexType>& matched_idxs_array,                             \
        addressable_priority_queue<ValueType, IndexType>& Q,              \
        std::vector<IndexType>& q_j, ValueType tolerance)

template <typename ValueType, typename IndexType>
void shortest_augmenting_path(
    size_type num_rows, const IndexType* row_ptrs, const IndexType* col_idxs,
    array<ValueType>& weights_array, array<ValueType>& dual_u_array,
    array<ValueType>& distance_array, array<IndexType>& permutation,
    array<IndexType>& inv_permutation, IndexType root,
    array<IndexType>& parents_array, array<IndexType>& generation_array,
    array<IndexType>& marked_cols_array, array<IndexType>& matched_idxs_array,
    addressable_priority_queue<ValueType, IndexType>& Q,
    std::vector<IndexType>& q_j, ValueType tolerance)
{
    constexpr auto inf = std::numeric_limits<ValueType>::infinity();
    const auto nnz = row_ptrs[num_rows];
    auto weights = weights_array.get_data();
    auto dual_u = dual_u_array.get_data();
    auto distance = distance_array.get_data();

    auto p = permutation.get_data();
    auto ip = inv_permutation.get_data();

    auto parents = parents_array.get_data();
    // Generation array to mark visited nodes.
    // It can take four states:
    //  - gen[col] = #rows + root: The distance to col is smaller than the
    //      length of the currently shortest augmenting path.
    //  - gen[col] = - #rows - root: The distance to col is within a tolerance
    //      of the currently shortest distance to the root. In this case, col
    //      is placed into the vector q_j holding the nodes with the shortest
    //      known distance to the root.
    //  - gen[col] = root: The distance to col is smaller than the length of
    //      the currently shortest augmenting path but larger than the currently
    //      shortest known distance to the root. In this case, col is placed
    //      into the priority queue Q.
    //  - gen[col] = - root: The shortest possible distance for col to the root
    //      has been found. If encountered again, col does not need to be
    //      considered another time.
    auto generation = generation_array.get_data();
    // Set of marked columns whose shortest alternating paths and distances to
    // the root are known.
    auto marked_cols = marked_cols_array.get_data();
    // Indices of the nonzero entries corresponding to the matched column in
    // each matched row. So, if row i is matched to column j, W(i,j) is found
    // at weights[idxs[i]] where W is the weight matrix.
    auto idxs = matched_idxs_array.get_data();

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
    IndexType marked_counter = 0;

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
                Q.insert(dist, col);
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
            if (lsap <= lsp) {
                break;
            }
            const auto col = q_j.back();
            q_j.pop_back();
            generation[col] = -root;
            marked_cols[marked_counter++] = col;
            row = ip[col];
        } else {
            if (Q.empty()) {
                break;
            }
            auto col = Q.min_val();
            while (generation[col] == -root && !Q.empty()) {
                // If col is already marked because it previously was in q_j
                // we have to disregard it.
                Q.pop_min();
                col = Q.min_val();
            }
            if (Q.empty()) {
                break;
            }
            lsp = distance[col];
            if (lsap <= lsp) {
                break;
            }
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
            if (gen == -root) {
                continue;
            }

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
                            Q.insert(dnew, col);
                        } else {
                            // col was already encountered but with larger
                            // distance on a different path.
                            generation[col] = root;
                            Q.update_key(dnew, col);
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
            while (col_idxs[idx] != col) {
                idx++;
            }
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


template <typename ValueType, typename IndexType>
void augment_matching(const matrix::Csr<ValueType, IndexType>* mtx,
                      array<remove_complex<ValueType>>& weights,
                      array<remove_complex<ValueType>>& dual_u,
                      array<remove_complex<ValueType>>& distance,
                      array<IndexType>& permutation,
                      array<IndexType>& inv_permutation,
                      array<IndexType>& unmatched_rows,
                      array<IndexType>& parents, array<IndexType>& generation,
                      array<IndexType>& marked_cols,
                      array<IndexType>& matched_idxs,
                      remove_complex<ValueType> tolerance)
{
    const auto host_exec = mtx->get_executor();
    const auto num_rows = mtx->get_size()[0];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    addressable_priority_queue<remove_complex<ValueType>, IndexType> Q{
        host_exec, num_rows};
    // For each row that is not contained in the initial matching, search for
    // an augmenting path, update the matching and compute the new entries
    // of the dual vectors.
    std::vector<IndexType> q_j{};
    const auto unmatched = unmatched_rows.get_data();
    size_type um = 0;
    auto root = unmatched[um];
    for (size_type um = 1; root != 0 && um < num_rows; um++) {
        if (root != -1) {
            mc64::shortest_augmenting_path(
                num_rows, row_ptrs, col_idxs, weights, dual_u, distance,
                permutation, inv_permutation, root, parents, generation,
                marked_cols, matched_idxs, Q, q_j, tolerance);
        }
        root = unmatched[um];
    }
}


#define GKO_DECLARE_MC64_COMPUTE_SCALING(ValueType, IndexType)              \
    void compute_scaling(                                                   \
        const matrix::Csr<ValueType, IndexType>* mtx,                       \
        const array<remove_complex<ValueType>>& weights_array,              \
        const array<remove_complex<ValueType>>& dual_u_array,               \
        const array<remove_complex<ValueType>>& row_maxima_array,           \
        const array<IndexType>& permutation,                                \
        const array<IndexType>& matched_idxs_array, mc64_strategy strategy, \
        ValueType* row_scaling, ValueType* col_scaling)

template <typename ValueType, typename IndexType>
void compute_scaling(const matrix::Csr<ValueType, IndexType>* mtx,
                     const array<remove_complex<ValueType>>& weights_array,
                     const array<remove_complex<ValueType>>& dual_u_array,
                     const array<remove_complex<ValueType>>& row_maxima_array,
                     const array<IndexType>& permutation,
                     const array<IndexType>& matched_idxs_array,
                     mc64_strategy strategy, ValueType* row_scaling,
                     ValueType* col_scaling)
{
    constexpr auto inf =
        std::numeric_limits<remove_complex<ValueType>>::infinity();
    const auto nnz = mtx->get_num_stored_elements();
    const auto num_rows = mtx->get_size()[0];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto values = mtx->get_const_values();
    const auto weights = weights_array.get_const_data();
    const auto dual_u = dual_u_array.get_const_data();
    const auto row_maxima = row_maxima_array.get_const_data();
    const auto p = permutation.get_const_data();
    const auto idxs = matched_idxs_array.get_const_data();

    if (strategy == mc64_strategy::max_diagonal_product) {
        for (size_type i = 0; i < num_rows; i++) {
            const remove_complex<ValueType> u_val = std::exp2(dual_u[i]);
            const remove_complex<ValueType> v_val =
                std::exp2(weights[idxs[i]] - dual_u[p[i]] - row_maxima[i]);
            col_scaling[i] = ValueType{u_val};
            row_scaling[i] = ValueType{v_val};
        }
    } else {
        for (size_type i = 0; i < num_rows; i++) {
            col_scaling[i] = 1.;
            row_scaling[i] = 1.;
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_INITIALIZE_WEIGHTS);
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_INITIAL_MATCHING);
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MC64_SHORTEST_AUGMENTING_PATH);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MC64_COMPUTE_SCALING);


}  // namespace mc64


namespace {


GKO_REGISTER_HOST_OPERATION(initialize_weights, mc64::initialize_weights);
GKO_REGISTER_HOST_OPERATION(initial_matching, mc64::initial_matching);
GKO_REGISTER_HOST_OPERATION(augment_matching, mc64::augment_matching);
GKO_REGISTER_HOST_OPERATION(compute_scaling, mc64::compute_scaling);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // namespace


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> Mc64<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product = std::unique_ptr<Composition<ValueType>>(
        static_cast<Composition<ValueType>*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename ValueType, typename IndexType>
Mc64<ValueType, IndexType>::Mc64(std::shared_ptr<const Executor> exec,
                                 const parameters_type& params)
    : EnablePolymorphicObject<Mc64, LinOpFactory>(std::move(exec)),
      parameters_{params}
{}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Mc64<ValueType, IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();
    const auto mtx =
        copy_and_convert_to<matrix_type>(host_exec, system_matrix.get());
    const auto num_rows = mtx->get_size()[0];
    const auto nnz = mtx->get_num_stored_elements();

    // Real valued arrays with space for:
    //     - nnz entries for weights
    //     - num_rows entries each for the dual vector u, distance information
    //       and the max weight per row
    array<remove_complex<ValueType>> weights{host_exec, nnz};
    array<remove_complex<ValueType>> dual_u{host_exec, num_rows};
    array<remove_complex<ValueType>> distance{host_exec, num_rows};
    array<remove_complex<ValueType>> row_maxima{host_exec, num_rows};
    // Zero initialized index arrays with space for n entries each for parent
    // information, priority queue handles, generation information, marked
    // columns, indices corresponding to matched columns in the according row
    // and still unmatched rows
    array<IndexType> parents{host_exec, num_rows};
    array<IndexType> generation{host_exec, num_rows};
    array<IndexType> marked_cols{host_exec, num_rows};
    array<IndexType> matched_idxs{host_exec, num_rows};
    array<IndexType> unmatched_rows{host_exec, num_rows};
    array<ValueType> row_scaling{host_exec, num_rows};
    array<ValueType> col_scaling{host_exec, num_rows};
    parents.fill(0);
    generation.fill(0);
    marked_cols.fill(0);
    matched_idxs.fill(0);
    unmatched_rows.fill(0);

    array<IndexType> permutation{host_exec, num_rows};
    array<IndexType> inv_permutation{host_exec, num_rows};
    permutation.fill(-one<IndexType>());
    inv_permutation.fill(-one<IndexType>());

    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();

    exec->run(make_initialize_weights(mtx.get(), weights, dual_u, distance,
                                      row_maxima, parameters_.strategy));

    // Compute an initial maximum matching from the nonzero entries for which
    // the reduced weight (W(i, j) - u(j) - v(i)) is zero. Here, W is the
    // weight matrix and u and v are the dual vectors. Note that v initially
    // only contains zeros and hence can still be ignored here.
    exec->run(make_initial_matching(
        num_rows, row_ptrs, col_idxs, weights, dual_u, permutation,
        inv_permutation, matched_idxs, unmatched_rows, parameters_.tolerance));

    exec->run(make_augment_matching(
        mtx.get(), weights, dual_u, distance, permutation, inv_permutation,
        unmatched_rows, parents, generation, marked_cols, matched_idxs,
        this->get_parameters().tolerance));

    exec->run(make_compute_scaling(
        mtx.get(), weights, dual_u, row_maxima, permutation, matched_idxs,
        parameters_.strategy, row_scaling.get_data(), col_scaling.get_data()));

    array<index_type> identity_permutation{exec, num_rows};
    exec->run(make_fill_seq_array(identity_permutation.get_data(), num_rows));

    using perm_type = gko::matrix::ScaledPermutation<ValueType, IndexType>;
    return result_type::create(
        perm_type::create(exec, std::move(row_scaling),
                          std::move(inv_permutation)),
        perm_type::create(exec, std::move(col_scaling),
                          std::move(identity_permutation)));
}


#define GKO_DECLARE_MC64(ValueType, IndexType) class Mc64<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MC64);


}  // namespace reorder
}  // namespace experimental
}  // namespace gko
