// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/diagonal_block_manipulation.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "common/cuda_hip/components/warp_blas.hpp"
#include "core/base/extended_float.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
// generated header
#include "common/cuda_hip/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace jacobi {
namespace kernel {


template <int max_block_size, typename ReducedType, typename Group,
          typename ValueType, typename IndexType>
__device__ __forceinline__ bool validate_precision_reduction_feasibility(
    Group& __restrict__ group, IndexType block_size,
    ValueType* __restrict__ row, ValueType* __restrict__ work, size_type stride)
{
    using gko::detail::float_traits;
    // save original data and reduce precision
    if (group.thread_rank() < block_size) {
#pragma unroll
        for (int i = 0; i < max_block_size; ++i) {
            if (i < block_size) {
                work[i * stride + group.thread_rank()] = row[i];
                row[i] =
                    static_cast<ValueType>(static_cast<ReducedType>(row[i]));
            }
        }
    }

    // compute the condition number
    uint32 perm = group.thread_rank();
    uint32 trans_perm = perm;
    auto block_cond = compute_infinity_norm<max_block_size>(group, block_size,
                                                            block_size, row);
    auto succeeded = invert_block<max_block_size>(
        group, static_cast<uint32>(block_size), row, perm, trans_perm);
    block_cond *= compute_infinity_norm<max_block_size>(group, block_size,
                                                        block_size, row);

    // restore original data
    if (group.thread_rank() < block_size) {
#pragma unroll
        for (int i = 0; i < max_block_size; ++i) {
            if (i < block_size) {
                row[i] = work[i * stride + group.thread_rank()];
            }
        }
    }

    return succeeded && block_cond >= remove_complex<ValueType>{1.0} &&
           block_cond * static_cast<remove_complex<ValueType>>(
                            float_traits<remove_complex<ValueType>>::eps) <
               remove_complex<ValueType>{1e-3};
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block* config::warp_size) generate(
    size_type num_rows, const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values, ValueType* __restrict__ block_data,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto block = group::this_thread_block();
    ValueType row[max_block_size];
    __shared__ uninitialized_array<ValueType, max_block_size * warps_per_block>
        workspace;
    csr::extract_transposed_diag_blocks<max_block_size, warps_per_block>(
        block, config::warp_size / subwarp_size, row_ptrs, col_idxs, values,
        block_ptrs, num_blocks, row, 1,
        workspace + threadIdx.z * max_block_size);
    const auto subwarp = group::tiled_partition<subwarp_size>(block);
    if (block_id < num_blocks) {
        const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
        uint32 perm = subwarp.thread_rank();
        uint32 trans_perm = subwarp.thread_rank();
        invert_block<max_block_size>(subwarp, static_cast<uint32>(block_size),
                                     row, perm, trans_perm);
        copy_matrix<max_block_size, and_transpose>(
            subwarp, block_size, row, 1, perm, trans_perm,
            block_data + storage_scheme.get_global_block_offset(block_id),
            storage_scheme.get_stride());
    }
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void
__launch_bounds__(warps_per_block* config::warp_size) adaptive_generate(
    size_type num_rows, const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values, remove_complex<ValueType> accuracy,
    ValueType* __restrict__ block_data,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    remove_complex<ValueType>* __restrict__ conditioning,
    precision_reduction* __restrict__ block_precisions,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks)
{
    // extract blocks
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto block = group::this_thread_block();
    ValueType row[max_block_size];
    __shared__ uninitialized_array<ValueType, max_block_size * warps_per_block>
        workspace;
    csr::extract_transposed_diag_blocks<max_block_size, warps_per_block>(
        block, config::warp_size / subwarp_size, row_ptrs, col_idxs, values,
        block_ptrs, num_blocks, row, 1,
        workspace + threadIdx.z * max_block_size);

    // compute inverse and figure out the correct precision
    const auto subwarp = group::tiled_partition<subwarp_size>(block);
    const uint32 block_size =
        block_id < num_blocks ? block_ptrs[block_id + 1] - block_ptrs[block_id]
                              : 0;
    uint32 perm = subwarp.thread_rank();
    uint32 trans_perm = subwarp.thread_rank();
    auto prec_descriptor = ~uint32{};
    if (block_id < num_blocks) {
        auto block_cond = compute_infinity_norm<max_block_size>(
            subwarp, block_size, block_size, row);
        invert_block<max_block_size>(subwarp, block_size, row, perm,
                                     trans_perm);
        block_cond *= compute_infinity_norm<max_block_size>(subwarp, block_size,
                                                            block_size, row);
        conditioning[block_id] = block_cond;
        const auto prec = block_precisions[block_id];
        prec_descriptor =
            preconditioner::detail::precision_reduction_descriptor::singleton(
                prec);
        if (prec == precision_reduction::autodetect()) {
            using preconditioner::detail::get_supported_storage_reductions;
            prec_descriptor = get_supported_storage_reductions<ValueType>(
                accuracy, block_cond,
                [&subwarp, &block_size, &row, &block_data, &storage_scheme,
                 &block_id] {
                    using target = device_type<reduce_precision<ValueType>>;
                    return validate_precision_reduction_feasibility<
                        max_block_size, target>(
                        subwarp, block_size, row,
                        block_data +
                            storage_scheme.get_global_block_offset(block_id),
                        storage_scheme.get_stride());
                },
                [&subwarp, &block_size, &row, &block_data, &storage_scheme,
                 &block_id] {
                    using target = device_type<
                        reduce_precision<reduce_precision<ValueType>>>;
                    return validate_precision_reduction_feasibility<
                        max_block_size, target>(
                        subwarp, block_size, row,
                        block_data +
                            storage_scheme.get_global_block_offset(block_id),
                        storage_scheme.get_stride());
                });
        }
    }

    // make sure all blocks in the group have the same precision
    const auto warp = group::tiled_partition<config::warp_size>(block);
    const auto prec =
        preconditioner::detail::get_optimal_storage_reduction(reduce(
            warp, prec_descriptor, [](uint32 x, uint32 y) { return x & y; }));

    // store the block back into memory
    if (block_id < num_blocks) {
        block_precisions[block_id] = prec;
        GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
            ValueType, prec,
            copy_matrix<max_block_size, and_transpose>(
                subwarp, block_size, row, 1, perm, trans_perm,
                reinterpret_cast<device_type<resolved_precision>*>(
                    block_data + storage_scheme.get_group_offset(block_id)) +
                    storage_scheme.get_block_offset(block_id),
                storage_scheme.get_stride()));
    }
}


}  // namespace kernel


// clang-format off
#cmakedefine GKO_JACOBI_BLOCK_SIZE @GKO_JACOBI_BLOCK_SIZE@
// clang-format on
// make things easier for IDEs
#ifndef GKO_JACOBI_BLOCK_SIZE
#define GKO_JACOBI_BLOCK_SIZE 1
#endif


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void generate(syn::value_list<int, max_block_size>,
              std::shared_ptr<const DefaultExecutor> exec,
              const matrix::Csr<ValueType, IndexType>* mtx,
              remove_complex<ValueType> accuracy, ValueType* block_data,
              const preconditioner::block_interleaved_storage_scheme<IndexType>&
                  storage_scheme,
              remove_complex<ValueType>* conditioning,
              precision_reduction* block_precisions,
              const IndexType* block_ptrs, size_type num_blocks)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const auto grid_size =
        ceildiv(num_blocks, warps_per_block * blocks_per_warp);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (grid_size > 0) {
        if (block_precisions) {
            kernel::adaptive_generate<max_block_size, subwarp_size,
                                      warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    mtx->get_size()[0], mtx->get_const_row_ptrs(),
                    mtx->get_const_col_idxs(),
                    as_device_type(mtx->get_const_values()),
                    as_device_type(accuracy), as_device_type(block_data),
                    storage_scheme, as_device_type(conditioning),
                    block_precisions, block_ptrs, num_blocks);
        } else {
            kernel::generate<max_block_size, subwarp_size, warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    mtx->get_size()[0], mtx->get_const_row_ptrs(),
                    mtx->get_const_col_idxs(),
                    as_device_type(mtx->get_const_values()),
                    as_device_type(block_data), storage_scheme, block_ptrs,
                    num_blocks);
        }
    }
}


#define DECLARE_JACOBI_GENERATE_INSTANTIATION(ValueType, IndexType)          \
    void generate<config::min_warps_per_block, GKO_JACOBI_BLOCK_SIZE,        \
                  ValueType, IndexType>(                                     \
        syn::value_list<int, GKO_JACOBI_BLOCK_SIZE>,                         \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const matrix::Csr<ValueType, IndexType>*, remove_complex<ValueType>, \
        ValueType*,                                                          \
        const preconditioner::block_interleaved_storage_scheme<IndexType>&,  \
        remove_complex<ValueType>*, precision_reduction*, const IndexType*,  \
        size_type)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    DECLARE_JACOBI_GENERATE_INSTANTIATION);


}  // namespace jacobi
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
