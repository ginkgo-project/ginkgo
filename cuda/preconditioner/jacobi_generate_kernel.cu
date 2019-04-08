/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/preconditioner/jacobi_kernels.hpp"


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/extended_float.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/diagonal_block_manipulation.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/components/warp_blas.cuh"
#include "cuda/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Jacobi preconditioner namespace.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {
namespace kernel {


template <int max_block_size, typename ReducedType, typename Group,
          typename ValueType, typename IndexType>
__device__
#ifdef GINKGO_JACOBI_FULL_OPTIMIZATIONS
    __forceinline__
#else
    __noinline__
#endif
    bool
    validate_precision_reduction_feasibility(Group &__restrict__ group,
                                             IndexType block_size,
                                             ValueType *__restrict__ row,
                                             ValueType *__restrict__ work,
                                             size_type stride)
{
    using gko::detail::float_traits;
    // save original data and reduce precision
    if (group.thread_rank() < block_size) {
#pragma unroll
        for (auto i = 0u; i < max_block_size; ++i) {
            if (i >= block_size) {
                break;
            }
            work[i * stride + group.thread_rank()] = row[i];
            row[i] = static_cast<ValueType>(static_cast<ReducedType>(row[i]));
        }
    }

    // compute the condition number
    auto perm = group.thread_rank();
    auto trans_perm = perm;
    auto block_cond = compute_infinity_norm<max_block_size>(group, block_size,
                                                            block_size, row);
    auto succeeded =
        invert_block<max_block_size>(group, block_size, row, perm, trans_perm);
    block_cond *= compute_infinity_norm<max_block_size>(group, block_size,
                                                        block_size, row);

    // restore original data
    if (group.thread_rank() < block_size) {
#pragma unroll
        for (auto i = 0u; i < max_block_size; ++i) {
            if (i >= block_size) {
                break;
            }
            row[i] = work[i * stride + group.thread_rank()];
        }
    }

    return succeeded && block_cond >= 1.0 &&
           block_cond * float_traits<remove_complex<ValueType>>::eps < 1e-3;
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *cuda_config::warp_size)
    generate(size_type num_rows, const IndexType *__restrict__ row_ptrs,
             const IndexType *__restrict__ col_idxs,
             const ValueType *__restrict__ values,
             ValueType *__restrict__ block_data,
             preconditioner::block_interleaved_storage_scheme<IndexType>
                 storage_scheme,
             const IndexType *__restrict__ block_ptrs, size_type num_blocks)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto block = group::this_thread_block();
    ValueType row[max_block_size];
    __shared__ UninitializedArray<ValueType, max_block_size * warps_per_block>
        workspace;
    csr::extract_transposed_diag_blocks<max_block_size, warps_per_block>(
        block, cuda_config::warp_size / subwarp_size, row_ptrs, col_idxs,
        values, block_ptrs, num_blocks, row, 1,
        workspace + threadIdx.z * max_block_size);
    const auto subwarp = group::tiled_partition<subwarp_size>(block);
    if (block_id < num_blocks) {
        const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
        auto perm = subwarp.thread_rank();
        auto trans_perm = subwarp.thread_rank();
        invert_block<max_block_size>(subwarp, block_size, row, perm,
                                     trans_perm);
        copy_matrix<max_block_size, and_transpose>(
            subwarp, block_size, row, 1, perm, trans_perm,
            block_data + storage_scheme.get_global_block_offset(block_id),
            storage_scheme.get_stride());
    }
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void
__launch_bounds__(warps_per_block *cuda_config::warp_size) adaptive_generate(
    size_type num_rows, const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, remove_complex<ValueType> accuracy,
    ValueType *__restrict__ block_data,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    remove_complex<ValueType> *__restrict__ conditioning,
    precision_reduction *__restrict__ block_precisions,
    const IndexType *__restrict__ block_ptrs, size_type num_blocks)
{
    // extract blocks
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto block = group::this_thread_block();
    ValueType row[max_block_size];
    __shared__ UninitializedArray<ValueType, max_block_size * warps_per_block>
        workspace;
    csr::extract_transposed_diag_blocks<max_block_size, warps_per_block>(
        block, cuda_config::warp_size / subwarp_size, row_ptrs, col_idxs,
        values, block_ptrs, num_blocks, row, 1,
        workspace + threadIdx.z * max_block_size);

    // compute inverse and figure out the correct precision
    const auto subwarp = group::tiled_partition<subwarp_size>(block);
    const auto block_size =
        block_id < num_blocks ? block_ptrs[block_id + 1] - block_ptrs[block_id]
                              : 0;
    auto perm = subwarp.thread_rank();
    auto trans_perm = subwarp.thread_rank();
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
                    using target = reduce_precision<ValueType>;
                    return validate_precision_reduction_feasibility<
                        max_block_size, target>(
                        subwarp, block_size, row,
                        block_data +
                            storage_scheme.get_global_block_offset(block_id),
                        storage_scheme.get_stride());
                },
                [&subwarp, &block_size, &row, &block_data, &storage_scheme,
                 &block_id] {
                    using target =
                        reduce_precision<reduce_precision<ValueType>>;
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
    const auto warp = group::tiled_partition<cuda_config::warp_size>(block);
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
                reinterpret_cast<resolved_precision *>(
                    block_data + storage_scheme.get_group_offset(block_id)) +
                    storage_scheme.get_block_offset(block_id),
                storage_scheme.get_stride()));
    }
}


}  // namespace kernel


namespace {


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void generate(syn::value_list<int, max_block_size>,
              const matrix::Csr<ValueType, IndexType> *mtx,
              remove_complex<ValueType> accuracy, ValueType *block_data,
              const preconditioner::block_interleaved_storage_scheme<IndexType>
                  &storage_scheme,
              remove_complex<ValueType> *conditioning,
              precision_reduction *block_precisions,
              const IndexType *block_ptrs, size_type num_blocks)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = cuda_config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (block_precisions) {
        kernel::adaptive_generate<max_block_size, subwarp_size, warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                mtx->get_size()[0], mtx->get_const_row_ptrs(),
                mtx->get_const_col_idxs(),
                as_cuda_type(mtx->get_const_values()), as_cuda_type(accuracy),
                as_cuda_type(block_data), storage_scheme,
                as_cuda_type(conditioning), block_precisions, block_ptrs,
                num_blocks);
    } else {
        kernel::generate<max_block_size, subwarp_size, warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                mtx->get_size()[0], mtx->get_const_row_ptrs(),
                mtx->get_const_col_idxs(),
                as_cuda_type(mtx->get_const_values()), as_cuda_type(block_data),
                storage_scheme, block_ptrs, num_blocks);
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_generate, generate);


}  // namespace


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CudaExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size,
              remove_complex<ValueType> accuracy,
              const preconditioner::block_interleaved_storage_scheme<IndexType>
                  &storage_scheme,
              Array<remove_complex<ValueType>> &conditioning,
              Array<precision_reduction> &block_precisions,
              const Array<IndexType> &block_pointers, Array<ValueType> &blocks)
{
    select_generate(compiled_kernels(),
                    [&](int compiled_block_size) {
                        return max_block_size <= compiled_block_size;
                    },
                    syn::value_list<int, cuda_config::min_warps_per_block>(),
                    syn::type_list<>(), system_matrix, accuracy,
                    blocks.get_data(), storage_scheme, conditioning.get_data(),
                    block_precisions.get_data(),
                    block_pointers.get_const_data(), num_blocks);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_GENERATE_KERNEL);


}  // namespace jacobi
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
