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
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "core/reorder/mc64_kernels.hpp"


namespace gko {
namespace reorder {
namespace mc64 {
namespace {


GKO_REGISTER_OPERATION(initialize_weights, mc64::initialize_weights);
GKO_REGISTER_OPERATION(initial_matching, mc64::initial_matching);
GKO_REGISTER_OPERATION(shortest_augmenting_path,
                       mc64::shortest_augmenting_path);
GKO_REGISTER_OPERATION(compute_scaling, mc64::compute_scaling);


}  // anonymous namespace
}  // namespace mc64


template <typename ValueType, typename IndexType>
void Mc64<ValueType, IndexType>::generate(std::shared_ptr<const Executor>& exec,
                                          std::shared_ptr<LinOp> system_matrix)
{
    auto mtx = as<matrix_type>(system_matrix);
    size_type num_rows = mtx->get_size()[0];
    size_type nnz = mtx->get_num_stored_elements();

    // Real valued arrays with space for:
    //     - nnz entries for weights
    //     - num_rows entries each for the dual vector u, distance information
    //       and the max weight per row
    array<remove_complex<ValueType>> weights{exec, nnz};
    array<remove_complex<ValueType>> dual_u{exec, num_rows};
    array<remove_complex<ValueType>> distance{exec, num_rows};
    array<remove_complex<ValueType>> row_maxima{exec, num_rows};
    // Zero initialized index arrays with space for n entries each for parent
    // information, priority queue handles, generation information, marked
    // columns, indices corresponding to matched columns in the according row
    // and still unmatched rows
    array<IndexType> parents{exec, num_rows};
    array<IndexType> handles{exec, num_rows};
    array<IndexType> generation{exec, num_rows};
    array<IndexType> marked_cols{exec, num_rows};
    array<IndexType> matched_idxs{exec, num_rows};
    array<IndexType> unmatched_rows{exec, num_rows};
    parents.fill(0);
    handles.fill(0);
    generation.fill(0);
    marked_cols.fill(0);
    matched_idxs.fill(0);
    unmatched_rows.fill(0);

    array<IndexType> permutation{exec, num_rows};
    array<IndexType> inv_permutation{exec, num_rows};
    permutation.fill(-one<IndexType>());
    inv_permutation.fill(-one<IndexType>());

    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();

    exec->run(mc64::make_initialize_weights(mtx.get(), weights, dual_u,
                                            distance, row_maxima,
                                            parameters_.strategy));

    // Compute an initial maximum matching from the nonzero entries for which
    // the reduced weight (W(i, j) - u(j) - v(i)) is zero. Here, W is the
    // weight matrix and u and v are the dual vectors. Note that v initially
    // only contains zeros and hence can still be ignored here.
    exec->run(mc64::make_initial_matching(
        num_rows, row_ptrs, col_idxs, weights, dual_u, permutation,
        inv_permutation, matched_idxs, unmatched_rows, parameters_.tolerance));

    // For each row that is not contained in the initial matching, search for
    // an augmenting path, update the matching and compute the new entries
    // of the dual vectors.
    addressable_priority_queue<remove_complex<ValueType>, IndexType> Q{
        parameters_.deg_log2};
    std::vector<IndexType> q_j{};
    const auto unmatched = unmatched_rows.get_data();
    auto um = 0;
    auto root = unmatched[um];
    while (root != 0 && um < num_rows) {
        if (root != -1) {
            exec->run(mc64::make_shortest_augmenting_path(
                num_rows, row_ptrs, col_idxs, weights, dual_u, distance,
                permutation, inv_permutation, root, parents, handles,
                generation, marked_cols, matched_idxs, Q, q_j,
                parameters_.tolerance));
        }
        root = unmatched[++um];
    }

    permutation_ = std::move(share(
        PermutationMatrix::create(exec, system_matrix->get_size(),
                                  inv_permutation, gko::matrix::row_permute)));
    inv_permutation_ = std::move(share(PermutationMatrix::create(
        exec, system_matrix->get_size(), permutation, matrix::row_permute)));
    row_scaling_ = std::move(DiagonalMatrix::create(exec, num_rows));
    col_scaling_ = std::move(DiagonalMatrix::create(exec, num_rows));

    exec->run(mc64::make_compute_scaling(
        mtx.get(), weights, dual_u, row_maxima, permutation, matched_idxs,
        parameters_.strategy, row_scaling_.get(), col_scaling_.get()));
}


#define GKO_DECLARE_MC64(ValueType, IndexType) class Mc64<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MC64);


}  // namespace reorder
}  // namespace gko
