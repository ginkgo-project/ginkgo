/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/extended_float.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
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


#include "common/preconditioner/jacobi_simple_apply_kernel.hpp.inc"


namespace {


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void apply(syn::value_list<int, max_block_size>, size_type num_blocks,
           const precision_reduction *block_precisions,
           const IndexType *block_pointers, const ValueType *blocks,
           const preconditioner::block_interleaved_storage_scheme<IndexType>
               &storage_scheme,
           const ValueType *b, size_type b_stride, ValueType *x,
           size_type x_stride)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (block_precisions) {
        kernel::adaptive_apply<max_block_size, subwarp_size, warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                as_cuda_type(blocks), storage_scheme, block_precisions,
                block_pointers, num_blocks, as_cuda_type(b), b_stride,
                as_cuda_type(x), x_stride);
    } else {
        kernel::apply<max_block_size, subwarp_size, warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                as_cuda_type(blocks), storage_scheme, block_pointers,
                num_blocks, as_cuda_type(b), b_stride, as_cuda_type(x),
                x_stride);
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply, apply);


}  // namespace


template <typename ValueType, typename IndexType>
void simple_apply(
    std::shared_ptr<const CudaExecutor> exec, size_type num_blocks,
    uint32 max_block_size,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    const Array<precision_reduction> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x)
{
    // TODO: write a special kernel for multiple RHS
    for (size_type col = 0; col < b->get_size()[1]; ++col) {
        select_apply(
            compiled_kernels(),
            [&](int compiled_block_size) {
                return max_block_size <= compiled_block_size;
            },
            syn::value_list<int, config::min_warps_per_block>(),
            syn::type_list<>(), num_blocks, block_precisions.get_const_data(),
            block_pointers.get_const_data(), blocks.get_const_data(),
            storage_scheme, b->get_const_values() + col, b->get_stride(),
            x->get_values() + col, x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_SIMPLE_APPLY_KERNEL);


}  // namespace jacobi
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
