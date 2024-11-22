// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/assembly_helpers_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace assembly_helpers {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void fill_send_buffers(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    comm_index_type local_part, const array<GlobalIndexType>& send_positions,
    const array<GlobalIndexType>& original_positions,
    array<GlobalIndexType>& send_row_idxs,
    array<GlobalIndexType>& send_col_idxs, array<ValueType>& send_values)
{
    auto num_entries = input.get_num_stored_elements();
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_values = input.get_const_values();

    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto in_rows, auto in_cols, auto in_vals,
                      auto in_pos, auto out_pos, auto out_rows, auto out_cols,
                      auto out_vals) {
            if (in_pos[i] >= 0) {
                out_rows[out_pos[i]] = in_rows[in_pos[i]];
                out_cols[out_pos[i]] = in_cols[in_pos[i]];
                out_vals[out_pos[i]] = in_vals[in_pos[i]];
            }
        },
        num_entries, input_row_idxs, input_col_idxs, input_values,
        original_positions.get_const_data(), send_positions.get_const_data(),
        send_row_idxs.get_data(), send_col_idxs.get_data(),
        send_values.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_FILL_SEND_BUFFERS);


}  // namespace assembly_helpers
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
