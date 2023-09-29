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

template <typename ValueType, typename GlobalIndexType>
struct input_type {
    GlobalIndexType row;
    GlobalIndexType col;
    ValueType val;
    size_type row_range;
    size_type col_range;

    __forceinline__ __device__ __host__
    input_type(thrust::tuple<GlobalIndexType, GlobalIndexType, ValueType,
                             size_type, size_type>
                   t)
        : row(thrust::get<0>(t)),
          col(thrust::get<1>(t)),
          val(thrust::get<2>(t)),
          row_range(thrust::get<3>(t)),
          col_range(thrust::get<4>(t))
    {}
};

/**
 * Maps indices grouped by keys in a given range into the compact range [0, N),
 * where N is the sum of the number of unique indices in each group. Also
 * reorders the input keys and indices,
 *
 * Consider the following example with a single key:
 * ```
 * K = [0, 0, 0, 0]
 * I = [3, 2, 7, 7]
 * ```
 * then the output iterator will hold:
 * ```
 * O = [1, 0, 2, 2]
 * ```
 * If there is more than one key, the resulting indices for a group k will have
 * lower values than any indices in group k' with `k < k'`. Consider the
 * following example:
 * ```
 * K = [1, 1, 1, 0, 0, 0, 3, 3, 3]
 * I = [1, 2, 3, 5, 5, 6, 10, 9, 9]
 * ```
 * then the output will be
 * ```
 * O = [2, 3, 4, 0, 0, 1, 6, 5, 5]
 * ```
 * Additionally the input keys and indices are sorted and made unique, and the
 * end iterators of the unique range will be returned:
 * ```
 * K' = [0, 0, 1, 1, 1, 3, 3]
 * I' = [5, 6, 1, 2, 3, 9, 10]
 * ```
 * It is possible that different groups have the same indices. But the resulting
 * indices will have unique value ranges for each group. Consider the following
 * example:
 * ```
 * K = [0, 0, 1, 1]
 * I = [1, 2, 1, 2]
 * ```
 * then the output iterator will hold:
 * ```
 * O = [0, 1, 2, 3]
 * ```
 */
template <typename ExecType, typename KeyIt, typename IndexIt,
          typename OutputIt>
thrust::tuple<KeyIt, IndexIt, OutputIt> compress_indices_by_key(
    std::shared_ptr<const ExecType> exec, KeyIt keys_first, KeyIt keys_last,
    IndexIt indices, OutputIt out)
{
    using key_type = typename thrust::iterator_traits<KeyIt>::value_type;
    using index_type = typename thrust::iterator_traits<IndexIt>::value_type;
    using out_index_type =
        typename thrust::iterator_traits<OutputIt>::value_type;
    using diff_type = typename thrust::iterator_traits<KeyIt>::difference_type;

    auto policy = thrust_policy(exec);

    auto size = static_cast<size_type>(thrust::distance(keys_first, keys_last));
    array<key_type> original_keys{exec, size};
    array<index_type> original_indices{exec, size};
    auto original_keys_it = original_keys.get_data();
    auto original_indices_it = original_indices.get_data();
    thrust::copy(policy, keys_first, keys_last, original_keys_it);
    thrust::copy(policy, indices, indices + size, original_indices_it);

    auto combined_it =
        thrust::make_zip_iterator(thrust::make_tuple(keys_first, indices));

    // sort keys and indices into segment of indices with the same key
    thrust::sort(policy, combined_it, combined_it + size);

    // make indices unique in each segment
    auto unique_ends = thrust::unique(policy, combined_it, combined_it + size);
    auto unique_size =
        static_cast<diff_type>(thrust::distance(combined_it, unique_ends));

    // map input indices to compressed index space
    thrust::transform(
        policy, original_keys_it, original_keys_it + size, original_indices_it,
        out,
        [keys_first, indices, unique_size] __host__ __device__(
            const key_type& key, const index_type& index) {
            auto segment_start = binary_search(
                diff_type{0}, unique_size,
                [keys_first, key](auto i) { return keys_first[i] >= key; });
            auto segment_end = binary_search(
                diff_type{0}, unique_size,
                [keys_first, key](auto i) { return keys_first[i] > key; });
            auto segment_size = segment_end - segment_start;

            auto compressed_idx = binary_search(
                segment_start, segment_size,
                [indices, index](auto i) { return indices[i] >= index; });
            return static_cast<out_index_type>(compressed_idx);
        });

    return thrust::make_tuple(keys_first + unique_size, indices + unique_size,
                              out + size);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_local_nonlocal(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& local_row_idxs,
    array<LocalIndexType>& local_col_idxs, array<ValueType>& local_values,
    array<LocalIndexType>& non_local_row_idxs,
    array<LocalIndexType>& non_local_col_idxs,
    array<ValueType>& non_local_values,
    array<LocalIndexType>& local_gather_idxs,
    array<comm_index_type>& recv_sizes,
    array<GlobalIndexType>& non_local_to_global)
{
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

    auto policy = thrust_policy(exec);

    // precompute the row and column range id of each input element
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    array<size_type> row_range_ids{exec, num_input_elements};
    thrust::upper_bound(policy, row_range_bounds + 1,
                        row_range_bounds + num_row_ranges + 1, input_row_idxs,
                        input_row_idxs + num_input_elements,
                        row_range_ids.get_data());
    array<size_type> col_range_ids{exec, input.get_num_elems()};
    thrust::upper_bound(policy, col_range_bounds + 1,
                        col_range_bounds + num_col_ranges + 1, input_col_idxs,
                        input_col_idxs + num_input_elements,
                        col_range_ids.get_data());

    // count number of local<0> and non-local<1> elements
    auto range_ids_it = thrust::make_zip_iterator(thrust::make_tuple(
        row_range_ids.get_const_data(), col_range_ids.get_const_data()));
    auto num_elements_pair = thrust::transform_reduce(
        policy, range_ids_it, range_ids_it + num_input_elements,
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
    auto num_local_elements = thrust::get<0>(num_elements_pair);
    auto num_non_local_elements = thrust::get<1>(num_elements_pair);

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

    using input_type = input_type<ValueType, GlobalIndexType>;
    auto input_it = thrust::make_zip_iterator(thrust::make_tuple(
        input.get_const_row_idxs(), input.get_const_col_idxs(),
        input.get_const_values(), row_range_ids.get_const_data(),
        col_range_ids.get_const_data()));

    // copy and transform local entries into arrays
    local_row_idxs.resize_and_reset(num_local_elements);
    local_col_idxs.resize_and_reset(num_local_elements);
    local_values.resize_and_reset(num_local_elements);
    auto local_it = thrust::make_transform_iterator(
        input_it, [map_to_local_row, map_to_local_col] __host__ __device__(
                      const input_type input) {
            auto local_row = map_to_local_row(input.row, input.row_range);
            auto local_col = map_to_local_col(input.col, input.col_range);
            return thrust::make_tuple(local_row, local_col, input.val);
        });
    thrust::copy_if(
        policy, local_it, local_it + input.get_num_elems(), range_ids_it,
        thrust::make_zip_iterator(thrust::make_tuple(local_row_idxs.get_data(),
                                                     local_col_idxs.get_data(),
                                                     local_values.get_data())),
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            return row_part == local_part && col_part == local_part;
        });
    // copy and transform non-local entries into arrays. this keeps global
    // column indices, and also stores the column part id for each non-local
    // entry in an array
    non_local_row_idxs.resize_and_reset(num_non_local_elements);
    non_local_values.resize_and_reset(num_non_local_elements);
    array<GlobalIndexType> non_local_global_col_idxs{exec,
                                                     num_non_local_elements};
    array<comm_index_type> non_local_col_part_ids{exec, num_non_local_elements};
    array<size_type> non_local_col_range_ids{exec, num_non_local_elements};
    auto non_local_it = thrust::make_transform_iterator(
        input_it, [map_to_local_row,
                   col_part_ids] __host__ __device__(const input_type input) {
            auto local_row = map_to_local_row(input.row, input.row_range);
            return thrust::make_tuple(local_row, input.col, input.val,
                                      col_part_ids[input.col_range],
                                      input.col_range);
        });
    thrust::copy_if(
        policy, non_local_it, non_local_it + input.get_num_elems(),
        range_ids_it,
        thrust::make_zip_iterator(thrust::make_tuple(
            non_local_row_idxs.get_data(), non_local_global_col_idxs.get_data(),
            non_local_values.get_data(), non_local_col_part_ids.get_data(),
            non_local_col_range_ids.get_data())),
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            return row_part == local_part && col_part != local_part;
        });

    // compute compressed column indices for non-local cols
    non_local_col_idxs.resize_and_reset(num_non_local_elements);
    auto compress_key_it = thrust::make_zip_iterator(thrust::make_tuple(
        non_local_col_part_ids.get_data(), non_local_col_range_ids.get_data()));
    auto compressed_result = compress_indices_by_key(
        exec, compress_key_it, compress_key_it + num_non_local_elements,
        non_local_global_col_idxs.get_data(), non_local_col_idxs.get_data());
    auto num_non_local_cols =
        thrust::distance(compress_key_it, thrust::get<0>(compressed_result));

    // compute gather idxs and recv_sizes
    local_gather_idxs.resize_and_reset(num_non_local_cols);
    // keep until the mapping is removed from matrix
    non_local_to_global.resize_and_reset(num_non_local_cols);
    // map unique (global_col, range_id) to (remote_local_col, global_col)
    thrust::transform(
        policy, non_local_global_col_idxs.get_data(),
        non_local_global_col_idxs.get_data() + num_non_local_cols,
        non_local_col_range_ids.get_data(),
        thrust::make_zip_iterator(thrust::make_tuple(
            local_gather_idxs.get_data(), non_local_to_global.get_data())),
        [map_to_local_col] __host__ __device__(GlobalIndexType global_col,
                                               size_type range_id) {
            return thrust::make_tuple(map_to_local_col(global_col, range_id),
                                      global_col);
        });

    auto recv_sizes_ptr = recv_sizes.get_data();
    thrust::fill_n(policy, recv_sizes_ptr, num_parts, 0);
    thrust::for_each_n(policy, non_local_col_part_ids.get_data(),
                       num_non_local_cols,
                       [recv_sizes_ptr] __device__(const size_type part) {
                           atomic_add(recv_sizes_ptr + part, 1);
                       });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_LOCAL_NONLOCAL);
