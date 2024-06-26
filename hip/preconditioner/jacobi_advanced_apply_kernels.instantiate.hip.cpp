// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "core/base/extended_float.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/warp_blas.hip.hpp"
#include "hip/preconditioner/jacobi_common.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Jacobi preconditioner namespace.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {


#include "common/cuda_hip/preconditioner/jacobi_advanced_apply_kernels.hpp.inc"


// clang-format off
#cmakedefine GKO_JACOBI_BLOCK_SIZE @GKO_JACOBI_BLOCK_SIZE@
// clang-format on
// make things easier for IDEs
#ifndef GKO_JACOBI_BLOCK_SIZE
#define GKO_JACOBI_BLOCK_SIZE 1
#endif


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void advanced_apply(
    syn::value_list<int, max_block_size>,
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    const precision_reduction* block_precisions,
    const IndexType* block_pointers, const ValueType* blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    const ValueType* alpha, const ValueType* b, size_type b_stride,
    ValueType* x, size_type x_stride)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const auto grid_size =
        ceildiv(num_blocks, warps_per_block * blocks_per_warp);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (grid_size > 0) {
        if (block_precisions) {
            kernel::advanced_adaptive_apply<max_block_size, subwarp_size,
                                            warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    as_device_type(blocks), storage_scheme, block_precisions,
                    block_pointers, num_blocks, as_device_type(alpha),
                    as_device_type(b), b_stride, as_device_type(x), x_stride);
        } else {
            kernel::advanced_apply<max_block_size, subwarp_size,
                                   warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    as_device_type(blocks), storage_scheme, block_pointers,
                    num_blocks, as_device_type(alpha), as_device_type(b),
                    b_stride, as_device_type(x), x_stride);
        }
    }
}


#define DECLARE_JACOBI_ADVANCED_APPLY_INSTANTIATION(ValueType, IndexType)   \
    void advanced_apply<config::min_warps_per_block, GKO_JACOBI_BLOCK_SIZE, \
                        ValueType, IndexType>(                              \
        syn::value_list<int, GKO_JACOBI_BLOCK_SIZE>,                        \
        std::shared_ptr<const DefaultExecutor> exec, size_type,             \
        const precision_reduction*, const IndexType* block_pointers,        \
        const ValueType*,                                                   \
        const preconditioner::block_interleaved_storage_scheme<IndexType>&, \
        const ValueType*, const ValueType*, size_type, ValueType*, size_type)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    DECLARE_JACOBI_ADVANCED_APPLY_INSTANTIATION);


}  // namespace jacobi
}  // namespace hip
}  // namespace kernels
}  // namespace gko
