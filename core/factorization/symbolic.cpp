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

#include "core/factorization/symbolic.hpp"


#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/base/allocator.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"


namespace gko {
namespace factorization {
namespace {


GKO_REGISTER_OPERATION(cholesky_symbolic_count,
                       cholesky::cholesky_symbolic_count);
GKO_REGISTER_OPERATION(cholesky_symbolic,
                       cholesky::cholesky_symbolic_factorize);
GKO_REGISTER_OPERATION(prefix_sum, components::prefix_sum);
GKO_REGISTER_OPERATION(initialize, lu_factorization::initialize);
GKO_REGISTER_OPERATION(factorize, lu_factorization::factorize);
GKO_REGISTER_HOST_OPERATION(compute_elim_forest, compute_elim_forest);


}  // namespace


/** Computes the symbolic Cholesky factorization of the given matrix. */
template <typename ValueType, typename IndexType>
void symbolic_cholesky(
    const matrix::Csr<ValueType, IndexType>* mtx,
    std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors)
{
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    const auto exec = mtx->get_executor();
    const auto host_exec = exec->get_master();
    std::unique_ptr<elimination_forest<IndexType>> forest;
    exec->run(make_compute_elim_forest(mtx, forest));
    const auto num_rows = mtx->get_size()[0];
    array<IndexType> row_ptrs{exec, num_rows + 1};
    array<IndexType> tmp{exec};
    exec->run(
        make_cholesky_symbolic_count(mtx, *forest, row_ptrs.get_data(), tmp));
    exec->run(make_prefix_sum(row_ptrs.get_data(), num_rows + 1));
    const auto factor_nnz = static_cast<size_type>(
        exec->copy_val_to_host(row_ptrs.get_const_data() + num_rows));
    factors = matrix_type::create(
        exec, mtx->get_size(), array<ValueType>{exec, factor_nnz},
        array<IndexType>{exec, factor_nnz}, std::move(row_ptrs));
    exec->run(make_cholesky_symbolic(mtx, *forest, factors.get(), tmp));
    factors->sort_by_column_index();
    auto lt_factor = as<matrix_type>(factors->transpose());
    const auto scalar =
        initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec);
    const auto id = matrix::Identity<ValueType>::create(exec, num_rows);
    lt_factor->apply(scalar.get(), id.get(), scalar.get(), factors.get());
}


#define GKO_DECLARE_SYMBOLIC_CHOLESKY(ValueType, IndexType) \
    void symbolic_cholesky(                                 \
        const matrix::Csr<ValueType, IndexType>* mtx,       \
        std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SYMBOLIC_CHOLESKY);


/**
 * Computes the symbolic Cholesky factorization of the given matrix.
 * The implementation is based on fill1 algorithm introduced in Rose and Tarjan,
 * "Algorithmic Aspects of Vertex Elimination on Directed Graphs," SIAM J. Appl.
 * Math. 1978. and its formulation in Gaihre et. al,
 * "GSoFa: Scalable Sparse Symbolic LU Factorization on GPUs," arXiv 2021
 */
template <typename ValueType, typename IndexType>
void symbolic_lu(const matrix::Csr<ValueType, IndexType>* mtx,
                 std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors)
{
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    const auto exec = mtx->get_executor();
    const auto host_exec = exec->get_master();
    const auto num_rows = mtx->get_size()[0];
    const auto host_mtx = make_temporary_clone(host_exec, mtx);
    const auto in_row_ptrs = host_mtx->get_const_row_ptrs();
    const auto in_col_idxs = host_mtx->get_const_col_idxs();
    array<IndexType> host_out_row_ptr_array(host_exec, num_rows + 1);
    const auto out_row_ptrs = host_out_row_ptr_array.get_data();
    vector<IndexType> fill(num_rows, host_exec);
    vector<IndexType> out_col_idxs(host_exec);
    vector<IndexType> diags(num_rows, host_exec);
    deque<IndexType> frontier_queue(host_exec);
    for (IndexType row = 0; row < num_rows; row++) {
        out_row_ptrs[row] = out_col_idxs.size();
        fill[row] = row;
        // first copy over the original row and add all lower triangular entries
        // to the queue
        const auto row_begin = in_row_ptrs[row];
        const auto row_end = in_row_ptrs[row + 1];
        for (auto nz = row_begin; nz < row_end; nz++) {
            const auto col = in_col_idxs[nz];
            fill[col] = row;
            if (col < row) {
                frontier_queue.push_back(col);
            }
            out_col_idxs.push_back(col);
        }
        // then add fill-in for all queued entries
        while (!frontier_queue.empty()) {
            const auto frontier = frontier_queue.front();
            assert(frontier < row);
            frontier_queue.pop_front();
            // add the fill-in for this node from U
            const auto upper_begin = diags[frontier] + 1;
            const auto upper_end = out_row_ptrs[frontier + 1];
            for (auto nz = upper_begin; nz < upper_end; nz++) {
                const auto col = out_col_idxs[nz];
                if (fill[col] < row) {
                    fill[col] = row;
                    out_col_idxs.push_back(col);
                    // any fill-in on the lower triangle may introduce
                    // additional fill-in when eliminated, so we need to enqueue
                    // it as well.
                    if (col < row) {
                        frontier_queue.push_back(col);
                    }
                }
            }
        }
        // restore sorting and find diagonal entry to separate L and U
        const auto row_begin_it = out_col_idxs.begin() + out_row_ptrs[row];
        const auto row_end_it = out_col_idxs.end();
        std::sort(row_begin_it, row_end_it);
        auto row_diag_it = std::lower_bound(row_begin_it, row_end_it, row);
        // add diagonal if it's missing
        if (row_diag_it == row_end_it || *row_diag_it != row) {
            row_diag_it = out_col_idxs.insert(row_diag_it, row);
        }
        diags[row] = std::distance(out_col_idxs.begin(), row_diag_it);
    }
    const auto out_nnz = static_cast<size_type>(out_col_idxs.size());
    out_row_ptrs[num_rows] = out_nnz;
    array<IndexType> out_row_ptr_array{exec, std::move(host_out_row_ptr_array)};
    array<IndexType> out_col_idx_array{exec, out_nnz};
    array<ValueType> out_val_array{exec, out_nnz};
    exec->copy_from(host_exec.get(), out_nnz, out_col_idxs.data(),
                    out_col_idx_array.get_data());
    factors = matrix_type::create(
        exec, mtx->get_size(), std::move(out_val_array),
        std::move(out_col_idx_array), std::move(out_row_ptr_array));
}


#define GKO_DECLARE_SYMBOLIC_LU(ValueType, IndexType) \
    void symbolic_lu(                                 \
        const matrix::Csr<ValueType, IndexType>* mtx, \
        std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SYMBOLIC_LU);

}  // namespace factorization
}  // namespace gko
