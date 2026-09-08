// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/matrix_kernels.hpp"

#include <algorithm>
#include <vector>

#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "reference/distributed/partition_helpers.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void separate_diag_off_diag(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& diag_row_idxs,
    array<LocalIndexType>& diag_col_idxs, array<ValueType>& diag_values,
    array<LocalIndexType>& off_diag_row_idxs,
    array<GlobalIndexType>& off_diag_col_idxs,
    array<ValueType>& off_diag_values)
{
    using global_nonzero = matrix_data_entry<ValueType, GlobalIndexType>;
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();
    auto row_part_ids = row_partition->get_part_ids();
    auto col_part_ids = col_partition->get_part_ids();
    auto num_parts = row_partition->get_num_parts();

    vector<global_nonzero> diag_entries(exec);
    vector<global_nonzero> off_diag_entries(exec);
    size_type row_range_id = 0;
    size_type col_range_id = 0;
    for (size_type i = 0; i < input.get_num_stored_elements(); ++i) {
        auto global_row = input_row_idxs[i];
        row_range_id = find_range(global_row, row_partition, row_range_id);
        if (row_part_ids[row_range_id] == local_part) {
            auto global_col = input_col_idxs[i];
            col_range_id = find_range(global_col, col_partition, col_range_id);
            if (col_part_ids[col_range_id] == local_part) {
                diag_entries.push_back(
                    {map_to_local(global_row, row_partition, row_range_id),
                     map_to_local(global_col, col_partition, col_range_id),
                     input_vals[i]});
            } else {
                off_diag_entries.push_back(
                    {map_to_local(global_row, row_partition, row_range_id),
                     global_col, input_vals[i]});
            }
        }
    }

    // create diag matrix
    diag_row_idxs.resize_and_reset(diag_entries.size());
    diag_col_idxs.resize_and_reset(diag_entries.size());
    diag_values.resize_and_reset(diag_entries.size());
    for (size_type i = 0; i < diag_entries.size(); ++i) {
        const auto& entry = diag_entries[i];
        diag_row_idxs.get_data()[i] = entry.row;
        diag_col_idxs.get_data()[i] = entry.column;
        diag_values.get_data()[i] = entry.value;
    }

    // create off-diag matrix
    // copy off-diag data into row and value array
    // copy off-diag global column indices into temporary vector
    off_diag_row_idxs.resize_and_reset(off_diag_entries.size());
    off_diag_col_idxs.resize_and_reset(off_diag_entries.size());
    off_diag_values.resize_and_reset(off_diag_entries.size());
    for (size_type i = 0; i < off_diag_entries.size(); ++i) {
        const auto& entry = off_diag_entries[i];
        off_diag_row_idxs.get_data()[i] = entry.row;
        off_diag_col_idxs.get_data()[i] = entry.column;
        off_diag_values.get_data()[i] = entry.value;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SEPARATE_DIAG_OFF_DIAG);


template <typename ValueType, typename LocalIndexType>
void separate_local_nonlocal_columns(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<LocalIndexType>& row_idxs,
    const array<LocalIndexType>& col_idxs, const array<ValueType>& values,
    LocalIndexType num_local_cols, array<LocalIndexType>& diag_row_idxs,
    array<LocalIndexType>& diag_col_idxs, array<ValueType>& diag_values,
    array<LocalIndexType>& off_diag_row_idxs,
    array<LocalIndexType>& off_diag_col_idxs, array<ValueType>& off_diag_values)
{
    auto row_ptr = row_idxs.get_const_data();
    auto col_ptr = col_idxs.get_const_data();
    auto val_ptr = values.get_const_data();
    const auto nnz = col_idxs.get_size();

    // count
    size_type num_diag = 0;
    for (size_type i = 0; i < nnz; ++i) {
        if (col_ptr[i] < num_local_cols) {
            ++num_diag;
        }
    }
    const auto num_off = nnz - num_diag;

    diag_row_idxs.resize_and_reset(num_diag);
    diag_col_idxs.resize_and_reset(num_diag);
    diag_values.resize_and_reset(num_diag);
    off_diag_row_idxs.resize_and_reset(num_off);
    off_diag_col_idxs.resize_and_reset(num_off);
    off_diag_values.resize_and_reset(num_off);

    // fill (stable: preserves input order)
    size_type di = 0;
    size_type oi = 0;
    for (size_type i = 0; i < nnz; ++i) {
        if (col_ptr[i] < num_local_cols) {
            diag_row_idxs.get_data()[di] = row_ptr[i];
            diag_col_idxs.get_data()[di] = col_ptr[i];
            diag_values.get_data()[di] = val_ptr[i];
            ++di;
        } else {
            off_diag_row_idxs.get_data()[oi] = row_ptr[i];
            // shift into the non-local index space
            off_diag_col_idxs.get_data()[oi] = col_ptr[i] - num_local_cols;
            off_diag_values.get_data()[oi] = val_ptr[i];
            ++oi;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SEPARATE_LOCAL_NONLOCAL_COLUMNS);


template <typename LocalIndexType, typename GlobalIndexType>
void unique_nonlocal_columns(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<GlobalIndexType>& global_cols,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<GlobalIndexType>& nonlocal_cols)
{
    const auto n = global_cols.get_size();
    auto in = global_cols.get_const_data();
    auto col_part_ids = col_partition->get_part_ids();

    std::vector<GlobalIndexType> remote;
    remote.reserve(n);
    size_type col_range_id = 0;
    for (size_type i = 0; i < n; ++i) {
        col_range_id = find_range(in[i], col_partition, col_range_id);
        if (col_part_ids[col_range_id] != local_part) {
            remote.push_back(in[i]);
        }
    }
    std::sort(remote.begin(), remote.end());
    remote.erase(std::unique(remote.begin(), remote.end()), remote.end());

    nonlocal_cols.resize_and_reset(remote.size());
    std::copy(remote.begin(), remote.end(), nonlocal_cols.get_data());
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_UNIQUE_NONLOCAL_COLUMNS);


}  // namespace distributed_matrix
}  // namespace reference
}  // namespace kernels
}  // namespace gko
