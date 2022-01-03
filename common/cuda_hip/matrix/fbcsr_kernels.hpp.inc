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


}  // namespace kernel


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(
    std::shared_ptr<const DefaultExecutor> exec,
    const Array<matrix_data_entry<ValueType, IndexType>>& data, int block_size,
    Array<IndexType>& row_ptr_array, Array<IndexType>& col_idx_array,
    Array<ValueType>& value_array)
{
    using entry = matrix_data_entry<ValueType, IndexType>;
    using device_entry = device_type<entry>;
    const auto nnz = data.get_num_elems();
    const auto bs = block_size;
    auto row_ptrs = row_ptr_array.get_data();
    auto num_rows = row_ptr_array.get_num_elems() - 1;
    if (nnz == 0) {
        components::fill_array(exec, row_ptrs, num_rows + 1, IndexType{});
        col_idx_array.resize_and_reset(0);
        value_array.resize_and_reset(0);
        return;
    }
    auto block_ordered_data = data;
    auto data_ptr = as_device_type(block_ordered_data.get_data());
    auto data_it = thrust::device_pointer_cast(data_ptr);
    thrust::stable_sort(thrust::device, data_it, data_it + nnz,
                        [bs] __device__(device_entry a, device_entry b) {
                            return thrust::make_pair(a.row / bs,
                                                     a.column / bs) <
                                   thrust::make_pair(b.row / bs, b.column / bs);
                        });
    // build block pattern
    auto adj_predicate = [bs, data_ptr, nnz] __device__(size_type i) {
        const auto a = i > 0 ? data_ptr[i - 1] : device_entry{-bs, -bs, {}};
        const auto b = data_ptr[i];
        return (a.row / bs != b.row / bs) || (a.column / bs != b.column / bs);
    };
    auto iota = thrust::make_counting_iterator(size_type{});
    // count how many blocks we have by counting how often the block changes
    auto num_blocks = static_cast<size_type>(
        thrust::count_if(thrust::device, iota, iota + nnz, adj_predicate));
    // allocate storage
    Array<IndexType> row_idx_array{exec, num_blocks};
    Array<size_type> block_ptr_array{exec, num_blocks};
    col_idx_array.resize_and_reset(num_blocks);
    value_array.resize_and_reset(num_blocks * bs * bs);
    auto row_idxs = row_idx_array.get_data();
    auto col_idxs = col_idx_array.get_data();
    auto values = as_device_type(value_array.get_data());
    auto block_ptrs = block_ptr_array.get_data();
    auto block_ptr_it = thrust::device_pointer_cast(block_ptrs);
    // write (block_row, block_col, block_start_idx) tuples for each block
    thrust::copy_if(thrust::device, iota, iota + nnz, block_ptr_it,
                    adj_predicate);
    auto block_output_it = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::device_pointer_cast(row_idxs),
                           thrust::device_pointer_cast(col_idxs)));
    thrust::transform(thrust::device, block_ptr_it, block_ptr_it + num_blocks,
                      block_output_it, [bs, data_ptr] __device__(size_type i) {
                          return thrust::make_tuple(data_ptr[i].row / bs,
                                                    data_ptr[i].column / bs);
                      });
    // build row pointers from row indices
    components::build_row_ptrs_from_idxs(exec, row_idx_array, num_rows,
                                         row_ptrs);
    // fill in values
    components::fill_array(exec, value_array.get_data(), num_blocks * bs * bs,
                           zero<ValueType>());
    thrust::for_each_n(
        thrust::device, iota, num_blocks,
        [block_ptrs, nnz, num_blocks, bs, data_ptr,
         values] __device__(size_type i) {
            const auto block_begin = block_ptrs[i];
            const auto block_end = i < num_blocks - 1 ? block_ptrs[i + 1] : nnz;
            for (auto nz = block_begin; nz < block_end; nz++) {
                const auto entry = data_ptr[nz];
                values[i * bs * bs + (entry.column % bs) * bs +
                       (entry.row % bs)] = fake_complex_unpack(entry.value);
            }
        });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_FILL_IN_MATRIX_DATA_KERNEL);
