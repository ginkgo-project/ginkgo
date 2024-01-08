// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace kernel {


template <int mat_blk_sz, int subwarp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void transpose_blocks(
    const IndexType nbnz, ValueType* const values)
{
    const auto total_subwarp_count =
        thread::get_subwarp_num_flat<subwarp_size, IndexType>();
    const IndexType begin_blk =
        thread::get_subwarp_id_flat<subwarp_size, IndexType>();

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<subwarp_size>(thread_block);
    const int sw_threadidx = subwarp_grp.thread_rank();

    constexpr int mat_blk_sz_2{mat_blk_sz * mat_blk_sz};
    constexpr int num_entries_per_thread{(mat_blk_sz_2 - 1) / subwarp_size + 1};
    ValueType orig_vals[num_entries_per_thread];

    for (auto ibz = begin_blk; ibz < nbnz; ibz += total_subwarp_count) {
        for (int i = sw_threadidx; i < mat_blk_sz_2; i += subwarp_size) {
            orig_vals[i / subwarp_size] = values[ibz * mat_blk_sz_2 + i];
        }
        subwarp_grp.sync();

        for (int i = 0; i < num_entries_per_thread; i++) {
            const int orig_pos = i * subwarp_size + sw_threadidx;
            if (orig_pos >= mat_blk_sz_2) {
                break;
            }
            const int orig_row = orig_pos % mat_blk_sz;
            const int orig_col = orig_pos / mat_blk_sz;
            const int new_pos = orig_row * mat_blk_sz + orig_col;
            values[ibz * mat_blk_sz_2 + new_pos] = orig_vals[i];
        }
        subwarp_grp.sync();
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void convert_to_csr(
    const IndexType* block_row_ptrs, const IndexType* block_col_idxs,
    const ValueType* blocks, IndexType* row_ptrs, IndexType* col_idxs,
    ValueType* values, size_type num_block_rows, int block_size)
{
    const auto block_row = thread::get_subwarp_id_flat<config::warp_size>();
    if (block_row >= num_block_rows) {
        return;
    }
    const auto block_begin = block_row_ptrs[block_row];
    const auto block_end = block_row_ptrs[block_row + 1];
    const auto num_blocks = block_end - block_begin;
    const auto first_row = block_row * block_size;
    const auto block_row_begin = block_begin * block_size * block_size;
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = warp.thread_rank();
    if (block_row == 0 && lane == 0) {
        row_ptrs[0] = 0;
    }
    for (auto i = lane; i < block_size; i += config::warp_size) {
        row_ptrs[first_row + i + 1] =
            block_row_begin + num_blocks * block_size * (i + 1);
    }
    for (IndexType i = lane; i < num_blocks * block_size * block_size;
         i += config::warp_size) {
        const auto local_row = i / (num_blocks * block_size);
        const auto local_block = (i % (num_blocks * block_size)) / block_size;
        const auto local_col = i % block_size;
        const auto block_idx = block_col_idxs[block_begin + local_block];
        // first nz of the row block + all prev rows + all prev blocks in row +
        // all previous cols in this block
        const auto out_idx = block_row_begin +
                             num_blocks * block_size * local_row +
                             local_block * block_size + local_col;
        col_idxs[out_idx] = block_idx * block_size + local_col;
        values[out_idx] =
            blocks[(block_begin + local_block) * block_size * block_size +
                   local_col * block_size + local_row];
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    const IndexType* block_row_ptrs, const IndexType* block_col_idxs,
    const ValueType* blocks, ValueType* values, size_type stride,
    size_type num_block_rows, int block_size)
{
    const auto block_row = thread::get_subwarp_id_flat<config::warp_size>();
    if (block_row >= num_block_rows) {
        return;
    }
    const auto block_begin = block_row_ptrs[block_row];
    const auto block_end = block_row_ptrs[block_row + 1];
    const auto num_blocks = block_end - block_begin;
    const auto block_row_begin = block_begin * block_size * block_size;
    const auto num_entries = num_blocks * block_size * block_size;
    const auto bs_sq = block_size * block_size;
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = warp.thread_rank();
    for (IndexType nz = lane; nz < num_entries; nz += config::warp_size) {
        const auto local_id = nz % bs_sq;
        const auto block = nz / bs_sq;
        const auto local_row = local_id % block_size;
        const auto local_col = local_id / block_size;
        const auto col =
            block_col_idxs[block + block_begin] * block_size + local_col;
        const auto row = block_row * block_size + local_row;
        values[row * stride + col] = blocks[nz + block_row_begin];
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         device_matrix_data<ValueType, IndexType>& data,
                         int block_size, array<IndexType>& block_row_ptr_array,
                         array<IndexType>& block_col_idx_array,
                         array<ValueType>& block_value_array)
{
    using tuple_type = thrust::tuple<IndexType, IndexType>;
    const auto nnz = data.get_num_stored_elements();
    const auto bs = block_size;
    auto block_row_ptrs = block_row_ptr_array.get_data();
    auto num_block_rows = block_row_ptr_array.get_size() - 1;
    if (nnz == 0) {
        components::fill_array(exec, block_row_ptrs, num_block_rows + 1,
                               IndexType{});
        block_col_idx_array.resize_and_reset(0);
        block_value_array.resize_and_reset(0);
        return;
    }
    auto in_rows = data.get_row_idxs();
    auto in_cols = data.get_col_idxs();
    auto in_vals = as_device_type(data.get_values());
    auto in_loc_it =
        thrust::make_zip_iterator(thrust::make_tuple(in_rows, in_cols));
    thrust::sort_by_key(thrust_policy(exec), in_loc_it, in_loc_it + nnz,
                        in_vals, [bs] __device__(tuple_type a, tuple_type b) {
                            return thrust::make_pair(thrust::get<0>(a) / bs,
                                                     thrust::get<1>(a) / bs) <
                                   thrust::make_pair(thrust::get<0>(b) / bs,
                                                     thrust::get<1>(b) / bs);
                        });
    // build block pattern
    auto adj_predicate = [bs, in_rows, in_cols, nnz] __device__(size_type i) {
        const auto a_block_row = i > 0 ? in_rows[i - 1] / bs : -1;
        const auto a_block_col = i > 0 ? in_cols[i - 1] / bs : -1;
        const auto b_block_row = in_rows[i] / bs;
        const auto b_block_col = in_cols[i] / bs;
        return (a_block_row != b_block_row) || (a_block_col != b_block_col);
    };
    auto iota = thrust::make_counting_iterator(size_type{});
    // count how many blocks we have by counting how often the block changes
    auto num_blocks = static_cast<size_type>(
        thrust::count_if(thrust_policy(exec), iota, iota + nnz, adj_predicate));
    // allocate storage
    array<IndexType> block_row_idx_array{exec, num_blocks};
    array<size_type> block_ptr_array{exec, num_blocks};
    block_col_idx_array.resize_and_reset(num_blocks);
    block_value_array.resize_and_reset(num_blocks * bs * bs);
    auto block_row_idxs = block_row_idx_array.get_data();
    auto block_col_idxs = block_col_idx_array.get_data();
    auto block_values = as_device_type(block_value_array.get_data());
    auto block_ptrs = block_ptr_array.get_data();
    // write (block_row, block_col, block_start_idx) tuples for each block
    thrust::copy_if(thrust_policy(exec), iota, iota + nnz, block_ptrs,
                    adj_predicate);
    auto block_output_it = thrust::make_zip_iterator(
        thrust::make_tuple(block_row_idxs, block_col_idxs));
    thrust::transform(
        thrust_policy(exec), block_ptrs, block_ptrs + num_blocks,
        block_output_it, [bs, in_rows, in_cols] __device__(size_type i) {
            return thrust::make_tuple(in_rows[i] / bs, in_cols[i] / bs);
        });
    // build row pointers from row indices
    components::convert_idxs_to_ptrs(exec, block_row_idx_array.get_const_data(),
                                     block_row_idx_array.get_size(),
                                     num_block_rows, block_row_ptrs);
    // fill in values
    components::fill_array(exec, block_value_array.get_data(),
                           num_blocks * bs * bs, zero<ValueType>());
    thrust::for_each_n(
        thrust_policy(exec), iota, num_blocks,
        [block_ptrs, nnz, num_blocks, bs, in_rows, in_cols, in_vals,
         block_values] __device__(size_type i) {
            const auto block_begin = block_ptrs[i];
            const auto block_end = i < num_blocks - 1 ? block_ptrs[i + 1] : nnz;
            for (auto nz = block_begin; nz < block_end; nz++) {
                block_values[i * bs * bs + (in_cols[nz] % bs) * bs +
                             (in_rows[nz] % bs)] = in_vals[nz];
            }
        });
}


namespace kernel {


template <typename ValueType, typename IndexType>
__global__ void __launch_bounds__(default_block_size)
    permute_transpose(const ValueType* __restrict__ in,
                      ValueType* __restrict__ out, int bs, size_type nnzb,
                      const IndexType* perm)
{
    const auto idx = thread::get_thread_id_flat();
    const auto block = idx / (bs * bs);
    const auto i = (idx % (bs * bs)) / bs;
    const auto j = idx % bs;
    if (block < nnzb) {
        out[block * bs * bs + j * bs + i] =
            in[perm[block] * bs * bs + i * bs + j];
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void fallback_transpose(const std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Fbcsr<ValueType, IndexType>* const input,
                        matrix::Fbcsr<ValueType, IndexType>* const output)
{
    const auto in_num_row_blocks = input->get_num_block_rows();
    const auto out_num_row_blocks = output->get_num_block_rows();
    const auto nnzb = output->get_num_stored_blocks();
    const auto bs = input->get_block_size();
    const auto in_row_ptrs = input->get_const_row_ptrs();
    const auto in_col_idxs = input->get_const_col_idxs();
    const auto in_vals = as_device_type(input->get_const_values());
    const auto out_row_ptrs = output->get_row_ptrs();
    const auto out_col_idxs = output->get_col_idxs();
    const auto out_vals = as_device_type(output->get_values());
    array<IndexType> out_row_idxs{exec, nnzb};
    array<IndexType> permutation{exec, nnzb};
    components::fill_seq_array(exec, permutation.get_data(), nnzb);
    components::convert_ptrs_to_idxs(exec, in_row_ptrs, in_num_row_blocks,
                                     out_col_idxs);
    exec->copy(nnzb, in_col_idxs, out_row_idxs.get_data());
    auto zip_it = thrust::make_zip_iterator(thrust::make_tuple(
        out_row_idxs.get_data(), out_col_idxs, permutation.get_data()));
    using tuple_type = thrust::tuple<IndexType, IndexType, IndexType>;
    thrust::sort(thrust_policy(exec), zip_it, zip_it + nnzb,
                 [] __device__(const tuple_type& a, const tuple_type& b) {
                     return thrust::tie(thrust::get<0>(a), thrust::get<1>(a)) <
                            thrust::tie(thrust::get<0>(b), thrust::get<1>(b));
                 });
    components::convert_idxs_to_ptrs(exec, out_row_idxs.get_data(), nnzb,
                                     out_num_row_blocks, out_row_ptrs);
    const auto grid_size = ceildiv(nnzb * bs * bs, default_block_size);
    if (grid_size > 0) {
        kernel::permute_transpose<<<grid_size, default_block_size, 0,
                                    exec->get_stream()>>>(
            in_vals, out_vals, bs, nnzb, permutation.get_const_data());
    }
}


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Fbcsr<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    constexpr auto warps_per_block = default_block_size / config::warp_size;
    const auto num_blocks =
        ceildiv(source->get_num_block_rows(), warps_per_block);
    if (num_blocks > 0) {
        kernel::fill_in_dense<<<num_blocks, default_block_size, 0,
                                exec->get_stream()>>>(
            source->get_const_row_ptrs(), source->get_const_col_idxs(),
            as_device_type(source->get_const_values()),
            as_device_type(result->get_values()), result->get_stride(),
            source->get_num_block_rows(), source->get_block_size());
    }
}


template <typename ValueType, typename IndexType>
void convert_to_csr(const std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType>* const source,
                    matrix::Csr<ValueType, IndexType>* const result)
{
    constexpr auto warps_per_block = default_block_size / config::warp_size;
    const auto num_blocks =
        ceildiv(source->get_num_block_rows(), warps_per_block);
    if (num_blocks > 0) {
        kernel::convert_to_csr<<<num_blocks, default_block_size, 0,
                                 exec->get_stream()>>>(
            source->get_const_row_ptrs(), source->get_const_col_idxs(),
            as_device_type(source->get_const_values()), result->get_row_ptrs(),
            result->get_col_idxs(), as_device_type(result->get_values()),
            source->get_num_block_rows(), source->get_block_size());
    }
}


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType>* const to_check,
    bool* const is_sorted)
{
    *is_sorted = true;
    auto gpu_array = array<bool>(exec, 1);
    // need to initialize the GPU value to true
    exec->copy_from(exec->get_master(), 1, is_sorted, gpu_array.get_data());
    auto block_size = default_block_size;
    const auto num_brows =
        static_cast<IndexType>(to_check->get_num_block_rows());
    const auto num_blocks = ceildiv(num_brows, block_size);
    if (num_blocks > 0) {
        kernel::
            check_unsorted<<<num_blocks, block_size, 0, exec->get_stream()>>>(
                to_check->get_const_row_ptrs(), to_check->get_const_col_idxs(),
                num_brows, gpu_array.get_data());
    }
    *is_sorted = get_element(gpu_array, 0);
}


template <typename ValueType, typename IndexType>
void sort_by_column_index(const std::shared_ptr<const DefaultExecutor> exec,
                          matrix::Fbcsr<ValueType, IndexType>* const to_sort)
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Fbcsr<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag) GKO_NOT_IMPLEMENTED;
