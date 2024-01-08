// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_COMPONENTS_CSR_SPGEAM_HPP_
#define GKO_REFERENCE_COMPONENTS_CSR_SPGEAM_HPP_


#include <limits>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/utils.hpp"


namespace gko {
namespace kernels {
namespace reference {


/**
 * Adds two (sorted) sparse matrices.
 *
 * Calls begin_cb(row) on each row to initialize row-local data
 * Calls entry_cb(row, col, a_val, b_val, local_data) on each output non-zero
 * Calls end_cb(row, local_data) on each row to finalize row-local data
 */
template <typename ValueType, typename IndexType, typename BeginCallback,
          typename EntryCallback, typename EndCallback>
void abstract_spgeam(const matrix::Csr<ValueType, IndexType>* a,
                     const matrix::Csr<ValueType, IndexType>* b,
                     BeginCallback begin_cb, EntryCallback entry_cb,
                     EndCallback end_cb)
{
    auto num_rows = a->get_size()[0];
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    auto b_vals = b->get_const_values();
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    for (size_type row = 0; row < num_rows; ++row) {
        auto a_begin = a_row_ptrs[row];
        auto a_end = a_row_ptrs[row + 1];
        auto b_begin = b_row_ptrs[row];
        auto b_end = b_row_ptrs[row + 1];
        auto total_size = (a_end - a_begin) + (b_end - b_begin);
        bool skip{};
        auto local_data = begin_cb(static_cast<IndexType>(row));
        for (IndexType i = 0; i < total_size; ++i) {
            if (skip) {
                skip = false;
                continue;
            }
            // load column indices or sentinel
            auto a_col = checked_load(a_col_idxs, a_begin, a_end, sentinel);
            auto b_col = checked_load(b_col_idxs, b_begin, b_end, sentinel);
            auto a_val =
                checked_load(a_vals, a_begin, a_end, zero<ValueType>());
            auto b_val =
                checked_load(b_vals, b_begin, b_end, zero<ValueType>());
            auto col = min(a_col, b_col);
            // callback
            entry_cb(row, col, a_col == col ? a_val : zero<ValueType>(),
                     b_col == col ? b_val : zero<ValueType>(), local_data);
            // advance indices
            a_begin += (a_col <= b_col);
            b_begin += (b_col <= a_col);
            skip = a_col == b_col;
        }
        end_cb(row, local_data);
    }
}


}  // namespace reference
}  // namespace kernels
}  // namespace gko


#endif  // GKO_REFERENCE_COMPONENTS_CSR_SPGEAM_HPP_
