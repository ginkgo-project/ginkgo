// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/jacobi_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/extended_float.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Jacobi preconditioner namespace.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {
namespace {


// a total of 8 32-subgroup (256 threads)
constexpr int default_num_warps = 8;
// TODO: get a default_grid_size for dpcpp
// with current architectures, at most 32 warps can be scheduled per SM (and
// current GPUs have at most 84 SMs)
constexpr int default_grid_size = 32 * 32 * 128;


void duplicate_array(const precision_reduction* __restrict__ source,
                     size_type source_size,
                     precision_reduction* __restrict__ dest,
                     size_type dest_size, sycl::nd_item<3> item_ct1)
{
    auto grid = group::this_grid(item_ct1);
    if (grid.thread_rank() >= dest_size) {
        return;
    }
    for (auto i = grid.thread_rank(); i < dest_size; i += grid.size()) {
        dest[i] = source[i % source_size];
    }
}

void duplicate_array(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                     sycl::queue* queue, const precision_reduction* source,
                     size_type source_size, precision_reduction* dest,
                     size_type dest_size)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            duplicate_array(source, source_size, dest, dest_size, item_ct1);
        });
}


template <typename IndexType>
void compare_adjacent_rows(size_type num_rows, int32 max_block_size,
                           const IndexType* __restrict__ row_ptrs,
                           const IndexType* __restrict__ col_idx,
                           bool* __restrict__ matching_next_row,
                           sycl::nd_item<3> item_ct1)
{
    const auto warp = group::tiled_partition<config::warp_size>(
        group::this_thread_block(item_ct1));
    const auto local_tid = warp.thread_rank();
    const auto warp_id =
        thread::get_subwarp_id_flat<config::warp_size>(item_ct1);

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
void compare_adjacent_rows(dim3 grid, dim3 block,
                           size_type dynamic_shared_memory, sycl::queue* queue,
                           size_type num_rows, int32 max_block_size,
                           const IndexType* row_ptrs, const IndexType* col_idx,
                           bool* matching_next_row)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(config::warp_size)]] {
                                compare_adjacent_rows(
                                    num_rows, max_block_size, row_ptrs, col_idx,
                                    matching_next_row, item_ct1);
                            });
}


template <typename IndexType>
void generate_natural_block_pointer(size_type num_rows, int32 max_block_size,
                                    const bool* __restrict__ matching_next_row,
                                    IndexType* __restrict__ block_ptrs,
                                    size_type* __restrict__ num_blocks_arr)
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
void generate_natural_block_pointer(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    size_type num_rows, int32 max_block_size, const bool* matching_next_row,
    IndexType* block_ptrs, size_type* num_blocks_arr)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            generate_natural_block_pointer(num_rows, max_block_size,
                                           matching_next_row, block_ptrs,
                                           num_blocks_arr);
        });
}


template <typename IndexType>
void agglomerate_supervariables_kernel(int32 max_block_size,
                                       size_type num_natural_blocks,
                                       IndexType* __restrict__ block_ptrs,
                                       size_type* __restrict__ num_blocks_arr)
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

template <typename IndexType>
void agglomerate_supervariables_kernel(dim3 grid, dim3 block,
                                       size_type dynamic_shared_memory,
                                       sycl::queue* queue, int32 max_block_size,
                                       size_type num_natural_blocks,
                                       IndexType* block_ptrs,
                                       size_type* num_blocks_arr)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            agglomerate_supervariables_kernel(
                max_block_size, num_natural_blocks, block_ptrs, num_blocks_arr);
        });
}


template <bool conjugate, int max_block_size, int subwarp_size,
          int warps_per_block, typename ValueType, typename IndexType>
void transpose_jacobi(
    const ValueType* __restrict__ blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks,
    ValueType* __restrict__ out_blocks, sycl::nd_item<3> item_ct1)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>(item_ct1);
    const auto subwarp = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
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
void transpose_jacobi(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const ValueType* blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const IndexType* block_ptrs, size_type num_blocks, ValueType* out_blocks)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subwarp_size)]] {
                                transpose_jacobi<conjugate, max_block_size,
                                                 subwarp_size, warps_per_block>(
                                    blocks, storage_scheme, block_ptrs,
                                    num_blocks, out_blocks, item_ct1);
                            });
}


template <bool conjugate, int max_block_size, int subwarp_size,
          int warps_per_block, typename ValueType, typename IndexType>
void adaptive_transpose_jacobi(
    const ValueType* __restrict__ blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const precision_reduction* __restrict__ block_precisions,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks,
    ValueType* __restrict__ out_blocks, sycl::nd_item<3> item_ct1)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>(item_ct1);
    const auto subwarp = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];

    const auto block_stride = storage_scheme.get_stride();
    const auto rank = subwarp.thread_rank();
    if (rank < block_size) {
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
            for (IndexType i = 0; i < block_size; ++i) {
                auto val = local_block[i * block_stride + rank];
                local_out_block[i + rank * block_stride] =
                    conjugate ? conj(val) : val;
            });
    }
}

template <bool conjugate, int max_block_size, int subwarp_size,
          int warps_per_block, typename ValueType, typename IndexType>
void adaptive_transpose_jacobi(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const ValueType* blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const precision_reduction* block_precisions, const IndexType* block_ptrs,
    size_type num_blocks, ValueType* out_blocks)
{
    queue->parallel_for(
        sycl_nd_range(grid, block),
        [=](sycl::nd_item<3> item_ct1)
            [[sycl::reqd_sub_group_size(subwarp_size)]] {
                adaptive_transpose_jacobi<conjugate, max_block_size,
                                          subwarp_size, warps_per_block>(
                    blocks, storage_scheme, block_precisions, block_ptrs,
                    num_blocks, out_blocks, item_ct1);
            });
}


template <typename ValueType, typename IndexType>
size_type find_natural_blocks(std::shared_ptr<const DefaultExecutor> exec,
                              const matrix::Csr<ValueType, IndexType>* mtx,
                              int32 max_block_size,
                              IndexType* __restrict__ block_ptrs)
{
    array<size_type> nums(exec, 1);

    array<bool> matching_next_row(exec, mtx->get_size()[0] - 1);

    const dim3 block_size(config::warp_size, 1, 1);
    const dim3 grid_size(
        ceildiv(mtx->get_size()[0] * config::warp_size, block_size.x), 1, 1);
    compare_adjacent_rows(grid_size, block_size, 0, exec->get_queue(),
                          mtx->get_size()[0], max_block_size,
                          mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
                          matching_next_row.get_data());
    generate_natural_block_pointer(
        1, 1, 0, exec->get_queue(), mtx->get_size()[0], max_block_size,
        matching_next_row.get_const_data(), block_ptrs, nums.get_data());
    nums.set_executor(exec->get_master());
    return nums.get_const_data()[0];
}


template <typename IndexType>
inline size_type agglomerate_supervariables(
    std::shared_ptr<const DefaultExecutor> exec, int32 max_block_size,
    size_type num_natural_blocks, IndexType* block_ptrs)
{
    array<size_type> nums(exec, 1);

    agglomerate_supervariables_kernel(1, 1, 0, exec->get_queue(),
                                      max_block_size, num_natural_blocks,
                                      block_ptrs, nums.get_data());

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
        duplicate_array(grid_size, block_size, 0, exec->get_queue(),
                        source.get_const_data(), source.get_size(),
                        precisions.get_data(), precisions.get_size());
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
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (block_precisions) {
        adaptive_transpose_jacobi<conjugate, max_block_size, subwarp_size,
                                  warps_per_block>(
            grid_size, block_size, 0, exec->get_queue(), blocks, storage_scheme,
            block_precisions, block_pointers, num_blocks, out_blocks);
    } else {
        transpose_jacobi<conjugate, max_block_size, subwarp_size,
                         warps_per_block>(
            grid_size, block_size, 0, exec->get_queue(), blocks, storage_scheme,
            block_pointers, num_blocks, out_blocks);
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


}  // namespace jacobi
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
