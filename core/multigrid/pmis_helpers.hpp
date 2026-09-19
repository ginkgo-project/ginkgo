// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_PMIS_HELPERS_HPP_
#define GKO_CORE_MULTIGRID_PMIS_HELPERS_HPP_


#include <memory>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/components/gather_kernels.hpp"
#include "core/multigrid/pmis_kernels.hpp"


namespace gko {
namespace multigrid {
namespace pmis {


GKO_REGISTER_OPERATION(classify_seed, pmis::classify_seed);
GKO_REGISTER_OPERATION(classify_select, pmis::classify_select);
GKO_REGISTER_OPERATION(classify_mark_fine, pmis::classify_mark_fine);
GKO_REGISTER_OPERATION(direct_interpolation_fill_coarse_rows,
                       pmis::direct_interpolation_fill_coarse_rows);
GKO_REGISTER_OPERATION(direct_interpolation_accumulate,
                       pmis::direct_interpolation_accumulate);
GKO_REGISTER_OPERATION(direct_interpolation_emit,
                       pmis::direct_interpolation_emit);
GKO_REGISTER_OPERATION(gather, components::gather);


// A single column block: a local matrix consists of one block holding the
// diagonal, a distributed one of a diagonal block plus an off-diagonal block
// starting at col_offset of the extended (local + halo) numbering. The
// strength graph is split in the same way and has no diagonal in either
// block.
template <typename MatrixType, typename IndexType>
struct column_block {
    const MatrixType* mtx;
    bool has_diagonal;
    IndexType col_offset;
};


/**
 * One selection round over every block: seed all unassigned nodes as coarse,
 * downgrade those that are outranked by a strong neighbour, then mark the
 * nodes that gained a new coarse strong neighbour as fine.
 *
 * @param exchange  called between the two phases, because mark_fine has to see
 *                  the C-points that the other blocks selected
 */
template <typename ValueType, typename IndexType, typename GlobalIndexType,
          typename ExchangeFn>
void classify_round(
    std::shared_ptr<const Executor> exec, size_type num_rows,
    const std::vector<column_block<matrix::SparsityCsr<ValueType, IndexType>,
                                   IndexType>>& blocks,
    const remove_complex<ValueType>* weight, const GlobalIndexType* global_idx,
    const int* status, int* new_status, ExchangeFn exchange)
{
    exec->run(pmis::make_classify_seed(num_rows, status, new_status));
    for (const auto& block : blocks) {
        exec->run(pmis::make_classify_select(weight, global_idx,
                                             block.col_offset, block.mtx,
                                             status, new_status));
    }
    exchange();
    for (const auto& block : blocks) {
        exec->run(pmis::make_classify_mark_fine(block.col_offset, block.mtx,
                                                new_status));
    }
}


/**
 * Fills the prolongation's column indices and values for the local rows. Every
 * block is accumulated before any is emitted, because alpha and beta sum over
 * the whole row. The emitted column indices are node indices in the extended
 * numbering, which the caller maps with a gather.
 */
template <typename ValueType, typename IndexType>
void fill_prolongation(
    std::shared_ptr<const Executor> exec, size_type num_rows,
    const std::vector<
        column_block<matrix::Csr<ValueType, IndexType>, IndexType>>& blocks,
    const remove_complex<ValueType>* row_maxabs,
    remove_complex<ValueType> strength_threshold, const int* status,
    const IndexType* prolong_row_ptrs, IndexType* prolong_col_idxs,
    ValueType* prolong_values)
{
    array<ValueType> pos{exec, num_rows};
    array<ValueType> pos_divisor{exec, num_rows};
    array<ValueType> neg{exec, num_rows};
    array<ValueType> neg_divisor{exec, num_rows};
    array<ValueType> diag{exec, num_rows};
    array<int> enable_pos{exec, num_rows};
    array<int> enable_neg{exec, num_rows};
    array<IndexType> cursor{exec, num_rows};
    pos.fill(zero<ValueType>());
    pos_divisor.fill(zero<ValueType>());
    neg.fill(zero<ValueType>());
    neg_divisor.fill(zero<ValueType>());
    diag.fill(zero<ValueType>());
    enable_pos.fill(0);
    enable_neg.fill(0);
    exec->copy_from(exec, num_rows, prolong_row_ptrs, cursor.get_data());
    const kernels::pmis::interpolation_workspace<ValueType, IndexType> ws{
        pos.get_data(),         pos_divisor.get_data(), neg.get_data(),
        neg_divisor.get_data(), diag.get_data(),        enable_pos.get_data(),
        enable_neg.get_data(),  cursor.get_data()};

    exec->run(pmis::make_direct_interpolation_fill_coarse_rows(
        num_rows, status, prolong_row_ptrs, prolong_col_idxs, prolong_values));
    for (const auto& block : blocks) {
        exec->run(pmis::make_direct_interpolation_accumulate(
            block.mtx, block.has_diagonal, block.col_offset, row_maxabs,
            strength_threshold, status, ws));
    }
    for (const auto& block : blocks) {
        exec->run(pmis::make_direct_interpolation_emit(
            block.mtx, block.has_diagonal, block.col_offset, row_maxabs,
            strength_threshold, status, ws, prolong_col_idxs, prolong_values));
    }
}


}  // namespace pmis
}  // namespace multigrid
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_PMIS_HELPERS_HPP_
