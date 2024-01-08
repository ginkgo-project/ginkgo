// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace {


__global__
__launch_bounds__(default_num_warps* config::warp_size) void duplicate_array(
    const precision_reduction* __restrict__ source, size_type source_size,
    precision_reduction* __restrict__ dest, size_type dest_size)
{
    auto grid = group::this_grid();
    if (grid.thread_rank() >= dest_size) {
        return;
    }
    for (auto i = grid.thread_rank(); i < dest_size; i += grid.size()) {
        dest[i] = source[i % source_size];
    }
}


template <typename IndexType>
__global__ void compare_adjacent_rows(size_type num_rows, int32 max_block_size,
                                      const IndexType* __restrict__ row_ptrs,
                                      const IndexType* __restrict__ col_idx,
                                      bool* __restrict__ matching_next_row)
{
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto local_tid = warp.thread_rank();
    const auto warp_id = thread::get_subwarp_id_flat<config::warp_size>();

    if (warp_id >= num_rows - 1) {
        return;
    }

    const auto curr_row_start = row_ptrs[warp_id];
    const auto next_row_start = row_ptrs[warp_id + 1];
    const auto next_row_end = row_ptrs[warp_id + 2];

    const auto nz_this_row = next_row_end - next_row_start;
    const auto nz_prev_row = next_row_start - curr_row_start;

    if (nz_this_row != nz_prev_row) {
        matching_next_row[warp_id] = false;
        return;
    }
    size_type steps = ceildiv(nz_this_row, config::warp_size);
    for (size_type i = 0; i < steps; i++) {
        auto j = local_tid + i * config::warp_size;
        auto prev_col = (curr_row_start + j < next_row_start)
                            ? col_idx[curr_row_start + j]
                            : 0;
        auto this_col = (curr_row_start + j < next_row_start)
                            ? col_idx[next_row_start + j]
                            : 0;
        if (warp.any(prev_col != this_col)) {
            matching_next_row[warp_id] = false;
            return;
        }
    }
    matching_next_row[warp_id] = true;
}


template <typename IndexType>
__global__ void generate_natural_block_pointer(
    size_type num_rows, int32 max_block_size,
    const bool* __restrict__ matching_next_row,
    IndexType* __restrict__ block_ptrs, size_type* __restrict__ num_blocks_arr)
{
    block_ptrs[0] = 0;
    if (num_rows == 0) {
        return;
    }
    size_type num_blocks = 1;
    int32 current_block_size = 1;
    for (size_type i = 0; i < num_rows - 1; ++i) {
        if ((matching_next_row[i]) && (current_block_size < max_block_size)) {
            ++current_block_size;
        } else {
            block_ptrs[num_blocks] =
                block_ptrs[num_blocks - 1] + current_block_size;
            ++num_blocks;
            current_block_size = 1;
        }
    }
    block_ptrs[num_blocks] = block_ptrs[num_blocks - 1] + current_block_size;
    num_blocks_arr[0] = num_blocks;
}


template <typename IndexType>
__global__ void agglomerate_supervariables_kernel(
    int32 max_block_size, size_type num_natural_blocks,
    IndexType* __restrict__ block_ptrs, size_type* __restrict__ num_blocks_arr)
{
    num_blocks_arr[0] = 0;
    if (num_natural_blocks == 0) {
        return;
    }
    size_type num_blocks = 1;
    int32 current_block_size = block_ptrs[1] - block_ptrs[0];
    for (size_type i = 1; i < num_natural_blocks; ++i) {
        const int32 block_size = block_ptrs[i + 1] - block_ptrs[i];
        if (current_block_size + block_size <= max_block_size) {
            current_block_size += block_size;
        } else {
            block_ptrs[num_blocks] = block_ptrs[i];
            ++num_blocks;
            current_block_size = block_size;
        }
    }
    block_ptrs[num_blocks] = block_ptrs[num_natural_blocks];
    num_blocks_arr[0] = num_blocks;
}


template <bool conjugate, int max_block_size, int subwarp_size,
          int warps_per_block, typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block* config::warp_size)
    transpose_jacobi(const ValueType* __restrict__ blocks,
                     preconditioner::block_interleaved_storage_scheme<IndexType>
                         storage_scheme,
                     const IndexType* __restrict__ block_ptrs,
                     size_type num_blocks, ValueType* __restrict__ out_blocks)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];

    const auto block_ofs = storage_scheme.get_global_block_offset(block_id);
    const auto block_stride = storage_scheme.get_stride();
    const auto rank = subwarp.thread_rank();
    if (rank < block_size) {
        for (IndexType i = 0; i < block_size; ++i) {
            auto val = blocks[block_ofs + i * block_stride + rank];
            out_blocks[block_ofs + i + rank * block_stride] =
                conjugate ? conj(val) : val;
        }
    }
}


template <bool conjugate, int max_block_size, int subwarp_size,
          int warps_per_block, typename ValueType, typename IndexType>
__global__ void
__launch_bounds__(warps_per_block* config::warp_size) adaptive_transpose_jacobi(
    const ValueType* __restrict__ blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const precision_reduction* __restrict__ block_precisions,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks,
    ValueType* __restrict__ out_blocks)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];

    const auto block_stride = storage_scheme.get_stride();
    const auto rank = subwarp.thread_rank();
    GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
        ValueType, block_precisions[block_id],
        auto local_block =
            reinterpret_cast<const resolved_precision*>(
                blocks + storage_scheme.get_group_offset(block_id)) +
            storage_scheme.get_block_offset(block_id);
        auto local_out_block =
            reinterpret_cast<resolved_precision*>(
                out_blocks + storage_scheme.get_group_offset(block_id)) +
            storage_scheme.get_block_offset(block_id);
        for (int i = rank; i < block_size * block_size; i += subwarp_size) {
            int row = i % block_size;
            int col = i / block_size;
            auto val = local_block[row + col * block_stride];
            local_out_block[row * block_stride + col] =
                conjugate ? conj(val) : val;
        });
}


template <typename ValueType, typename IndexType>
size_type find_natural_blocks(std::shared_ptr<const DefaultExecutor> exec,
                              const matrix::Csr<ValueType, IndexType>* mtx,
                              int32 max_block_size,
                              IndexType* __restrict__ block_ptrs)
{
    array<size_type> nums(exec, 1);

    // FIXME: num_rows == 0 bug
    array<bool> matching_next_row(exec, mtx->get_size()[0] - 1);

    const auto block_size = config::warp_size;
    const auto grid_size =
        ceildiv(mtx->get_size()[0] * config::warp_size, block_size);

    if (grid_size > 0) {
        compare_adjacent_rows<<<grid_size, block_size, 0, exec->get_stream()>>>(
            mtx->get_size()[0], max_block_size, mtx->get_const_row_ptrs(),
            mtx->get_const_col_idxs(), matching_next_row.get_data());
    }
    generate_natural_block_pointer<<<1, 1, 0, exec->get_stream()>>>(
        mtx->get_size()[0], max_block_size, matching_next_row.get_const_data(),
        block_ptrs, nums.get_data());
    nums.set_executor(exec->get_master());
    return nums.get_const_data()[0];
}


template <typename IndexType>
inline size_type agglomerate_supervariables(
    std::shared_ptr<const DefaultExecutor> exec, int32 max_block_size,
    size_type num_natural_blocks, IndexType* block_ptrs)
{
    array<size_type> nums(exec, 1);

    agglomerate_supervariables_kernel<<<1, 1, 0, exec->get_stream()>>>(
        max_block_size, num_natural_blocks, block_ptrs, nums.get_data());

    nums.set_executor(exec->get_master());
    return nums.get_const_data()[0];
}


}  // namespace


void initialize_precisions(std::shared_ptr<const DefaultExecutor> exec,
                           const array<precision_reduction>& source,
                           array<precision_reduction>& precisions)
{
    const auto block_size = default_num_warps * config::warp_size;
    const auto grid_size =
        min(default_grid_size,
            static_cast<int32>(ceildiv(precisions.get_size(), block_size)));
    if (grid_size > 0) {
        duplicate_array<<<grid_size, block_size, 0, exec->get_stream()>>>(
            source.get_const_data(), source.get_size(), precisions.get_data(),
            precisions.get_size());
    }
}


template <typename ValueType, typename IndexType>
void find_blocks(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::Csr<ValueType, IndexType>* system_matrix,
                 uint32 max_block_size, size_type& num_blocks,
                 array<IndexType>& block_pointers)
{
    auto num_natural_blocks = find_natural_blocks(
        exec, system_matrix, max_block_size, block_pointers.get_data());
    num_blocks = agglomerate_supervariables(
        exec, max_block_size, num_natural_blocks, block_pointers.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_FIND_BLOCKS_KERNEL);


namespace {


template <bool conjugate, int warps_per_block, int max_block_size,
          typename ValueType, typename IndexType>
void transpose_jacobi(
    syn::value_list<int, max_block_size>,
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    const precision_reduction* block_precisions,
    const IndexType* block_pointers, const ValueType* blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    ValueType* out_blocks)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const auto grid_size =
        ceildiv(num_blocks, warps_per_block * blocks_per_warp);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (grid_size > 0) {
        if (block_precisions) {
            adaptive_transpose_jacobi<conjugate, max_block_size, subwarp_size,
                                      warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    as_device_type(blocks), storage_scheme, block_precisions,
                    block_pointers, num_blocks, as_device_type(out_blocks));
        } else {
            transpose_jacobi<conjugate, max_block_size, subwarp_size,
                             warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    as_device_type(blocks), storage_scheme, block_pointers,
                    num_blocks, as_device_type(out_blocks));
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_transpose_jacobi, transpose_jacobi);


}  // namespace


template <typename ValueType, typename IndexType>
void transpose_jacobi(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    uint32 max_block_size, const array<precision_reduction>& block_precisions,
    const array<IndexType>& block_pointers, const array<ValueType>& blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    array<ValueType>& out_blocks)
{
    select_transpose_jacobi(
        compiled_kernels(),
        [&](int compiled_block_size) {
            return max_block_size <= compiled_block_size;
        },
        syn::value_list<int, false, config::min_warps_per_block>(),
        syn::type_list<>(), exec, num_blocks, block_precisions.get_const_data(),
        block_pointers.get_const_data(), blocks.get_const_data(),
        storage_scheme, out_blocks.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose_jacobi(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    uint32 max_block_size, const array<precision_reduction>& block_precisions,
    const array<IndexType>& block_pointers, const array<ValueType>& blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    array<ValueType>& out_blocks)
{
    select_transpose_jacobi(
        compiled_kernels(),
        [&](int compiled_block_size) {
            return max_block_size <= compiled_block_size;
        },
        syn::value_list<int, true, config::min_warps_per_block>(),
        syn::type_list<>(), exec, num_blocks, block_precisions.get_const_data(),
        block_pointers.get_const_data(), blocks.get_const_data(),
        storage_scheme, out_blocks.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    const array<precision_reduction>& block_precisions,
    const array<IndexType>& block_pointers, const array<ValueType>& blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    ValueType* result_values, size_type result_stride) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_CONVERT_TO_DENSE_KERNEL);
