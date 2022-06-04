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


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_diag_offdiag(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& diag_row_idxs,
    array<LocalIndexType>& diag_col_idxs, array<ValueType>& diag_values,
    array<LocalIndexType>& offdiag_row_idxs,
    array<LocalIndexType>& offdiag_col_idxs, array<ValueType>& offdiag_values,
    array<LocalIndexType>& local_gather_idxs, comm_index_type* recv_sizes,
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

    // precompute the row and column range id of each input element
    array<size_type> row_range_ids{exec, num_input_elements};
    thrust::upper_bound(thrust::device, row_range_bounds + 1,
                        row_range_bounds + num_row_ranges + 1,
                        input.get_const_row_idxs(),
                        input.get_const_row_idxs() + num_input_elements,
                        row_range_ids.get_data());
    array<size_type> col_range_ids{exec, input.get_num_elems()};
    thrust::upper_bound(thrust::device, col_range_bounds + 1,
                        col_range_bounds + num_col_ranges + 1,
                        input.get_const_col_idxs(),
                        input.get_const_col_idxs() + num_input_elements,
                        col_range_ids.get_data());

    // count number of diag<0> and offdiag<1> elements
    auto range_ids_it = thrust::make_zip_iterator(thrust::make_tuple(
        row_range_ids.get_const_data(), col_range_ids.get_const_data()));
    auto num_elements_pair = thrust::transform_reduce(
        thrust::device, range_ids_it, range_ids_it + num_input_elements,
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            bool is_inner_entry =
                row_part == local_part && col_part == local_part;
            bool is_ghost_entry =
                row_part == local_part && col_part != local_part;
            return thrust::make_tuple(
                is_inner_entry ? size_type{1} : size_type{0},
                is_ghost_entry ? size_type{1} : size_type{0});
        },
        thrust::make_tuple(size_type{}, size_type{}),
        [] __host__ __device__(const thrust::tuple<size_type, size_type>& a,
                               const thrust::tuple<size_type, size_type>& b) {
            return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                      thrust::get<1>(a) + thrust::get<1>(b));
        });
    auto num_diag_elements = thrust::get<0>(num_elements_pair);
    auto num_offdiag_elements = thrust::get<1>(num_elements_pair);

    // define global-to-local maps for row and column indices
    auto map_to_local_row =
        [row_range_bounds, row_range_starting_indices] __host__ __device__(
            const GlobalIndexType row, const size_type range_id) {
            return static_cast<LocalIndexType>(row -
                                               row_range_bounds[range_id]) +
                   row_range_starting_indices[range_id];
        };
    auto map_to_local_col =
        [col_range_bounds, col_range_starting_indices] __host__ __device__(
            const GlobalIndexType col, const size_type range_id) {
            return static_cast<LocalIndexType>(col -
                                               col_range_bounds[range_id]) +
                   col_range_starting_indices[range_id];
        };

    using input_type = thrust::tuple<GlobalIndexType, GlobalIndexType,
                                     ValueType, size_type, size_type>;
    auto input_it = thrust::make_zip_iterator(thrust::make_tuple(
        input.get_const_row_idxs(), input.get_const_col_idxs(),
        input.get_const_values(), row_range_ids.get_const_data(),
        col_range_ids.get_const_data()));

    // copy and transform diag entries into arrays
    diag_row_idxs.resize_and_reset(num_diag_elements);
    diag_col_idxs.resize_and_reset(num_diag_elements);
    diag_values.resize_and_reset(num_diag_elements);
    auto local_diag_it = thrust::make_transform_iterator(
        input_it, [map_to_local_row, map_to_local_col] __host__ __device__(
                      const input_type& tuple) {
            auto local_row =
                map_to_local_row(thrust::get<0>(tuple), thrust::get<3>(tuple));
            auto local_col =
                map_to_local_col(thrust::get<1>(tuple), thrust::get<4>(tuple));
            return thrust::make_tuple(local_row, local_col,
                                      thrust::get<2>(tuple));
        });
    thrust::copy_if(
        thrust::device, local_diag_it, local_diag_it + input.get_num_elems(),
        range_ids_it,
        thrust::make_zip_iterator(thrust::make_tuple(diag_row_idxs.get_data(),
                                                     diag_col_idxs.get_data(),
                                                     diag_values.get_data())),
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            return row_part == local_part && col_part == local_part;
        });
    // copy and transform offdiag entries into arrays. this keeps global column
    // indices, and also stores the column part id for each offdiag entry in an
    // array
    offdiag_row_idxs.resize_and_reset(num_offdiag_elements);
    offdiag_values.resize_and_reset(num_offdiag_elements);
    array<GlobalIndexType> offdiag_global_col_idxs{exec, num_offdiag_elements};
    array<comm_index_type> offdiag_col_part_ids{exec, num_offdiag_elements};
    array<size_type> offdiag_col_range_ids{exec, num_offdiag_elements};
    auto local_offdiag_it = thrust::make_transform_iterator(
        input_it, [map_to_local_row, map_to_local_col,
                   col_part_ids] __host__ __device__(const input_type& tuple) {
            auto local_row =
                map_to_local_row(thrust::get<0>(tuple), thrust::get<3>(tuple));
            return thrust::make_tuple(
                local_row, thrust::get<1>(tuple), thrust::get<2>(tuple),
                col_part_ids[thrust::get<4>(tuple)], thrust::get<4>(tuple));
        });
    thrust::copy_if(
        thrust::device, local_offdiag_it,
        local_offdiag_it + input.get_num_elems(), range_ids_it,
        thrust::make_zip_iterator(thrust::make_tuple(
            offdiag_row_idxs.get_data(), offdiag_global_col_idxs.get_data(),
            offdiag_values.get_data(), offdiag_col_part_ids.get_data(),
            offdiag_col_range_ids.get_data())),
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            return row_part == local_part && col_part != local_part;
        });

    // 1. sort global columns, part-id and range-id according to
    // their part-id and global columns
    // the previous `offdiag_global_col_idxs` is not modify to
    // keep it consistent with the offdiag row and values array
    array<GlobalIndexType> sorted_offdiag_global_col_idxs{
        exec, offdiag_global_col_idxs};
    auto key_it = thrust::make_zip_iterator(
        thrust::make_tuple(offdiag_col_part_ids.get_data(),
                           sorted_offdiag_global_col_idxs.get_data()));
    thrust::sort_by_key(thrust::device, key_it, key_it + num_offdiag_elements,
                        offdiag_col_range_ids.get_data());

    // 2. remove duplicate columns, now the new column i has global index
    // offdiag_global_col_idxs[i]
    auto offdiag_global_col_idxs_begin =
        sorted_offdiag_global_col_idxs.get_data();
    auto offdiag_global_col_idxs_end = thrust::get<0>(thrust::unique_by_key(
        thrust::device, offdiag_global_col_idxs_begin,
        offdiag_global_col_idxs_begin + num_offdiag_elements,
        thrust::make_zip_iterator(
            thrust::make_tuple(offdiag_col_part_ids.get_data(),
                               offdiag_col_range_ids.get_data()))));
    auto num_offdiag_cols = static_cast<size_type>(thrust::distance(
        offdiag_global_col_idxs_begin, offdiag_global_col_idxs_end));

    // 2.5 copy unique_columns to local_to_global_ghost map
    local_to_global_ghost.resize_and_reset(num_offdiag_cols);
    exec->copy(num_offdiag_cols, offdiag_global_col_idxs_begin,
               local_to_global_ghost.get_data());

    // 3. create mapping from unique_columns
    // since we don't have hash tables on GPUs I'm first sorting the offdiag
    // global column indices and their new local index again by the global
    // column index. Then I'm using binary searches to find the new local column
    // index.
    array<LocalIndexType> permutation{exec, num_offdiag_cols};
    thrust::sequence(thrust::device, permutation.get_data(),
                     permutation.get_data() + num_offdiag_cols);
    thrust::sort_by_key(
        thrust::device, offdiag_global_col_idxs_begin,
        offdiag_global_col_idxs_begin + num_offdiag_cols,
        thrust::make_zip_iterator(thrust::make_tuple(
            offdiag_col_part_ids.get_data(), permutation.get_data())));

    // 4. map column index of offdiag entries to new columns
    offdiag_col_idxs.resize_and_reset(num_offdiag_elements);
    array<size_type> lower_bounds{exec, num_offdiag_elements};
    // I have to precompute the lower bounds because the calling binary
    // searches from the device does not work:
    // https://github.com/NVIDIA/thrust/issues/1415
    thrust::lower_bound(
        thrust::device, offdiag_global_col_idxs_begin,
        offdiag_global_col_idxs_begin + num_offdiag_cols,
        offdiag_global_col_idxs.get_data(),
        offdiag_global_col_idxs.get_data() + num_offdiag_elements,
        lower_bounds.get_data());
    auto permutation_data = permutation.get_data();
    thrust::transform(
        thrust::device, lower_bounds.get_data(),
        lower_bounds.get_data() + num_offdiag_elements,
        offdiag_col_idxs.get_data(),
        [permutation_data] __host__ __device__(const size_type lower_bound) {
            return permutation_data[lower_bound];
        });

    // 5. compute gather idxs and recv_offsets
    local_gather_idxs.resize_and_reset(num_offdiag_cols);
    auto transform_it = thrust::make_zip_iterator(thrust::make_tuple(
        local_to_global_ghost.get_data(), offdiag_col_range_ids.get_data()));
    thrust::transform(
        thrust::device, transform_it, transform_it + num_offdiag_cols,
        local_gather_idxs.get_data(),
        [map_to_local_col] __host__ __device__(
            const thrust::tuple<GlobalIndexType, size_type>& tuple) {
            return map_to_local_col(thrust::get<0>(tuple),
                                    thrust::get<1>(tuple));
        });
    thrust::fill_n(thrust::device, recv_sizes, num_parts, 0);
    thrust::for_each_n(thrust::device, offdiag_col_part_ids.get_data(),
                       num_offdiag_cols,
                       [recv_sizes] __device__(const size_type part) {
                           atomic_add(recv_sizes + part, 1);
                       });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_DIAG_OFFDIAG);
