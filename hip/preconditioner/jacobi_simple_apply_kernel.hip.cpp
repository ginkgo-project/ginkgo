// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/jacobi_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/extended_float.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
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


#include "common/cuda_hip/preconditioner/jacobi_simple_apply_kernel.hpp.inc"


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void apply(syn::value_list<int, max_block_size>,
           std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
           const precision_reduction* block_precisions,
           const IndexType* block_pointers, const ValueType* blocks,
           const preconditioner::block_interleaved_storage_scheme<IndexType>&
               storage_scheme,
           const ValueType* b, size_type b_stride, ValueType* x,
           size_type x_stride);

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply, apply);


template <typename ValueType, typename IndexType>
void simple_apply(
    std::shared_ptr<const HipExecutor> exec, size_type num_blocks,
    uint32 max_block_size,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    const array<precision_reduction>& block_precisions,
    const array<IndexType>& block_pointers, const array<ValueType>& blocks,
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x)
{
    // TODO: write a special kernel for multiple RHS
    for (size_type col = 0; col < b->get_size()[1]; ++col) {
        select_apply(
            compiled_kernels(),
            [&](int compiled_block_size) {
                return max_block_size <= compiled_block_size;
            },
            syn::value_list<int, config::min_warps_per_block>(),
            syn::type_list<>(), exec, num_blocks,
            block_precisions.get_const_data(), block_pointers.get_const_data(),
            blocks.get_const_data(), storage_scheme,
            b->get_const_values() + col, b->get_stride(), x->get_values() + col,
            x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_SIMPLE_APPLY_KERNEL);


}  // namespace jacobi
}  // namespace hip
}  // namespace kernels
}  // namespace gko
