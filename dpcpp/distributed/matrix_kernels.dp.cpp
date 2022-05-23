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

#include "core/distributed/matrix_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>


#include "dpcpp/components/atomic.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_local_nonlocal(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& local_row_idxs,
    array<LocalIndexType>& local_col_idxs, array<ValueType>& local_values,
    array<LocalIndexType>& non_local_row_idxs,
    array<LocalIndexType>& non_local_col_idxs, array<ValueType>& non_local_values,
    array<LocalIndexType>& local_gather_idxs, array<comm_index_type> recv_sizes,
    array<GlobalIndexType>& local_to_global_ghost)
{
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();
    auto row_part_ids = row_partition->get_part_ids();
    auto col_part_ids = col_partition->get_part_ids();
    auto num_parts = static_cast<size_type>(row_partition->get_num_parts());
    const auto* row_range_bounds = row_partition->get_range_bounds();
    const auto* col_range_bounds = col_partition->get_range_bounds();
    const auto* row_range_starting_indices =
        row_partition->get_range_starting_indices();
    const auto* col_range_starting_indices =
        col_partition->get_range_starting_indices();
    const auto num_row_ranges = row_partition->get_num_ranges();
    const auto num_col_ranges = col_partition->get_num_ranges();
    const auto num_input_elements = input.get_num_elems();
    auto policy =
        oneapi::dpl::execution::make_device_policy(*exec->get_queue());

    // precompute the row and column range id of each input element
    Array<size_type> row_range_ids{exec, num_input_elements};
    oneapi::dpl::upper_bound(policy, row_range_bounds + 1,
                             row_range_bounds + num_row_ranges + 1,
                             input.get_const_row_idxs(),
                             input.get_const_row_idxs() + num_input_elements,
                             row_range_ids.get_data());
    Array<size_type> col_range_ids{exec, input.get_num_elems()};
    oneapi::dpl::upper_bound(policy, col_range_bounds + 1,
                             col_range_bounds + num_col_ranges + 1,
                             input.get_const_col_idxs(),
                             input.get_const_col_idxs() + num_input_elements,
                             col_range_ids.get_data());

    // count number of local<0> and non-local<1> elements
    auto range_ids_it = oneapi::dpl::make_zip_iterator(
        row_range_ids.get_const_data(), col_range_ids.get_const_data());
    auto reduce_it = oneapi::dpl::make_transform_iterator(
        range_ids_it, [local_part, row_part_ids, col_part_ids](const auto& t) {
            auto [row_range_id, col_range_id] = t;
            auto row_part = row_part_ids[row_range_id];
            auto col_part = col_part_ids[col_range_id];
            bool is_inner_entry =
                row_part == local_part && col_part == local_part;
            bool is_ghost_entry =
                row_part == local_part && col_part != local_part;
            return std::make_tuple(
                is_inner_entry ? size_type{1} : size_type{0},
                is_ghost_entry ? size_type{1} : size_type{0});
        });
    auto num_elements_pair = oneapi::dpl::reduce(
        policy, reduce_it, reduce_it + num_input_elements,
        std::tuple<size_type, size_type>{}, [](const auto& a, const auto& b) {
            return std::make_tuple(std::get<0>(a) + std::get<0>(b),
                                   std::get<1>(a) + std::get<1>(b));
        });
    auto num_local_elements = std::get<0>(num_elements_pair);
    auto num_non_local_elements = std::get<1>(num_elements_pair);

    // define global-to-local maps for row and column indices
    auto map_to_local_row = [row_range_bounds, row_range_starting_indices](
                                const GlobalIndexType row,
                                const size_type range_id) {
        return static_cast<LocalIndexType>(row - row_range_bounds[range_id]) +
               row_range_starting_indices[range_id];
    };
    auto map_to_local_col = [col_range_bounds, col_range_starting_indices](
                                const GlobalIndexType col,
                                const size_type range_id) {
        return static_cast<LocalIndexType>(col - col_range_bounds[range_id]) +
               col_range_starting_indices[range_id];
    };

    using input_type = std::tuple<GlobalIndexType, GlobalIndexType, ValueType,
                                  size_type, size_type>;
    auto input_it = oneapi::dpl::make_zip_iterator(
        input.get_const_row_idxs(), input.get_const_col_idxs(),
        input.get_const_values(), row_range_ids.get_const_data(),
        col_range_ids.get_const_data());

    // copy and transform local entries into arrays
    local_row_idxs.resize_and_reset(num_local_elements);
    local_col_idxs.resize_and_reset(num_local_elements);
    local_values.resize_and_reset(num_local_elements);
    auto local_local_it = oneapi::dpl::make_transform_iterator(
        input_it, [map_to_local_row, map_to_local_col](const input_type& t) {
            auto [row, col, value, row_rid, col_rid] = t;
            auto local_row = map_to_local_row(row, row_rid);
            auto local_col = map_to_local_col(col, col_rid);
            return std::make_tuple(local_row, local_col, value, row_rid,
                                   col_rid);
        });
    oneapi::dpl::copy_if(
        policy, local_local_it, local_local_it + input.get_num_elems(),
        oneapi::dpl::make_zip_iterator(
            local_row_idxs.get_data(), local_col_idxs.get_data(),
            local_values.get_data(), oneapi::dpl::discard_iterator(),
            oneapi::dpl::discard_iterator()),
        [local_part, row_part_ids, col_part_ids](const auto& t) {
            auto row_part = row_part_ids[std::get<3>(t)];
            auto col_part = col_part_ids[std::get<4>(t)];
            return row_part == local_part && col_part == local_part;
        });
    // copy and transform non-local entries into arrays. this keeps global column
    // indices, and also stores the column part id for each non-local entry in an
    // array
    non_local_row_idxs.resize_and_reset(num_non_local_elements);
    non_local_values.resize_and_reset(num_non_local_elements);
    Array<GlobalIndexType> non_local_global_col_idxs{exec, num_non_local_elements};
    Array<comm_index_type> non_local_col_part_ids{exec, num_non_local_elements};
    Array<size_type> non_local_col_range_ids{exec, num_non_local_elements};
    auto local_non_local_it = oneapi::dpl::make_transform_iterator(
        input_it, [map_to_local_row, col_part_ids](const input_type& t) {
            auto [row, col, value, row_rid, col_rid] = t;
            auto local_row = map_to_local_row(row, row_rid);
            return std::make_tuple(local_row, col, value, row_rid, col_rid,
                                   col_part_ids[col_rid]);
        });
    oneapi::dpl::copy_if(
        policy, local_non_local_it, local_non_local_it + input.get_num_elems(),
        oneapi::dpl::make_zip_iterator(
            non_local_row_idxs.get_data(), non_local_global_col_idxs.get_data(),
            non_local_values.get_data(), oneapi::dpl::discard_iterator(),
            non_local_col_range_ids.get_data(), non_local_col_part_ids.get_data()),
        [local_part, row_part_ids, col_part_ids](const auto& t) {
            auto row_part = row_part_ids[std::get<3>(t)];
            auto col_part = col_part_ids[std::get<4>(t)];
            return row_part == local_part && col_part == local_part;
        });

    // 1. sort global columns, part-id and range-id according to
    // their part-id and global columns
    // the previous `non_local_global_col_idxs` is not modify to
    // keep it consistent with the non_local row and values array
    array<GlobalIndexType> sorted_non_local_global_col_idxs{
        exec, non_local_global_col_idxs};
    auto key_it = oneapi::dpl::make_zip_iterator(
        non_local_col_part_ids.get_data(),
        sorted_non_local_global_col_idxs.get_data(),
        non_local_col_range_ids.get_data());
    oneapi::dpl::sort(policy, key_it, key_it + num_non_local_elements,
                      [](const auto& a, const auto& b) {
                          return std::tie(std::get<0>(a), std::get<1>(a)) <
                                 std::tie(std::get<0>(b), std::get<1>(b));
                      });

    // 2. remove duplicate columns, now the new column i has global index
    // non_local_global_col_idxs[i]
    auto non_local_global_col_idxs_begin =
        sorted_non_local_global_col_idxs.get_data();
    auto unique_it = oneapi::dpl::make_zip_iterator(
        non_local_global_col_idxs_begin, non_local_col_part_ids.get_data(),
        non_local_col_range_ids.get_data());
    auto unique_end =
        oneapi::dpl::unique(policy, unique_it, unique_it + num_non_local_elements,
                            [](const auto& a, const auto& b) {
                                return std::get<1>(a) == std::get<1>(b);
                            });
    auto num_non_local_cols =
        static_cast<size_type>(std::distance(unique_it, unique_end));

    // 2.5 copy unique_columns to local_to_global_ghost map
    local_to_global_ghost.resize_and_reset(num_non_local_cols);
    exec->copy(num_non_local_cols, non_local_global_col_idxs_begin,
               local_to_global_ghost.get_data());

    // 3. create mapping from unique_columns
    // since we don't have hash tables on GPUs I'm first sorting the non-local
    // global column indices and their new local index again by the global
    // column index. Then I'm using binary searches to find the new local column
    // index.
    Array<LocalIndexType> permutation{exec, num_non_local_cols};
    oneapi::dpl::copy(
        policy, oneapi::dpl::counting_iterator<LocalIndexType>(0),
        oneapi::dpl::counting_iterator<LocalIndexType>(num_non_local_cols),
        permutation.get_data());
    auto sorted_by_col_idx_it = oneapi::dpl::make_zip_iterator(
        non_local_global_col_idxs_begin, non_local_col_part_ids.get_data(),
        permutation.get_data());
    oneapi::dpl::sort(policy, sorted_by_col_idx_it,
                      sorted_by_col_idx_it + num_non_local_cols,
                      [](const auto& a, const auto& b) {
                          return std::get<0>(a) < std::get<0>(b);
                      });

    // 4. map column index of non-local entries to new columns
    non_local_col_idxs.resize_and_reset(num_non_local_elements);
    Array<size_type> lower_bounds{exec, num_non_local_elements};
    // I have to precompute the lower bounds because the calling binary
    // searches from the device does not work:
    // https://github.com/NVIDIA/oneapi::dpl/issues/1415
    oneapi::dpl::lower_bound(
        policy, non_local_global_col_idxs_begin,
        non_local_global_col_idxs_begin + num_non_local_cols,
        non_local_global_col_idxs.get_data(),
        non_local_global_col_idxs.get_data() + num_non_local_elements,
        lower_bounds.get_data());
    // auto permutation_data = permutation.get_data();
    oneapi::dpl::transform(policy, lower_bounds.get_data(),
                           lower_bounds.get_data() + num_non_local_elements,
                           non_local_col_idxs.get_data(),
                           [permutation_data = permutation.get_const_data()](
                               const size_type lower_bound) {
                               return permutation_data[lower_bound];
                           });

    // 5. compute gather idxs and recv_offsets
    local_gather_idxs.resize_and_reset(num_non_local_cols);
    auto transform_it = oneapi::dpl::make_zip_iterator(
        local_to_global_ghost.get_data(), non_local_col_range_ids.get_data());
    oneapi::dpl::transform(
        policy, transform_it, transform_it + num_non_local_cols,
        local_gather_idxs.get_data(), [map_to_local_col](const auto t) {
            return map_to_local_col(std::get<0>(t), std::get<1>(t));
        });
    auto recv_sizes_ptr = recv_sizes.get_data();
    oneapi::dpl::fill_n(policy, recv_sizes_ptr, num_parts, 0);
    oneapi::dpl::for_each_n(policy, non_local_col_part_ids.get_data(),
                            num_non_local_cols,
                            [recv_sizes_ptr](const size_type part) {
                                atomic_add(recv_sizes_ptr + part, 1);
                            });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_LOCAL_NONLOCAL);


}  // namespace distributed_matrix
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
