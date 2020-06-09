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


#include <hip/hip_runtime.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/extended_float.hpp"
#include "core/components/fill_array.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/diagonal_block_manipulation.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"
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


#include "common/preconditioner/jacobi_generate_kernel.hpp.inc"


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
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (block_precisions) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                kernel::adaptive_generate<max_block_size, subwarp_size,
                                          warps_per_block>),
            dim3(grid_size), dim3(block_size), 0, 0, mtx->get_size()[0],
            mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            as_hip_type(mtx->get_const_values()), as_hip_type(accuracy),
            as_hip_type(block_data), storage_scheme, as_hip_type(conditioning),
            block_precisions, block_ptrs, num_blocks);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel::generate<max_block_size, subwarp_size,
                                             warps_per_block>),
            dim3(grid_size), dim3(block_size), 0, 0, mtx->get_size()[0],
            mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            as_hip_type(mtx->get_const_values()), as_hip_type(block_data),
            storage_scheme, block_ptrs, num_blocks);
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_generate, generate);


}  // namespace


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const HipExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size,
              remove_complex<ValueType> accuracy,
              const preconditioner::block_interleaved_storage_scheme<IndexType>
                  &storage_scheme,
              Array<remove_complex<ValueType>> &conditioning,
              Array<precision_reduction> &block_precisions,
              const Array<IndexType> &block_pointers, Array<ValueType> &blocks)
{
    components::fill_array(exec, blocks.get_data(), blocks.get_num_elems(),
                           zero<ValueType>());
    select_generate(compiled_kernels(),
                    [&](int compiled_block_size) {
                        return max_block_size <= compiled_block_size;
                    },
                    syn::value_list<int, config::min_warps_per_block>(),
                    syn::type_list<>(), system_matrix, accuracy,
                    blocks.get_data(), storage_scheme, conditioning.get_data(),
                    block_precisions.get_data(),
                    block_pointers.get_const_data(), num_blocks);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_GENERATE_KERNEL);


}  // namespace jacobi
}  // namespace hip
}  // namespace kernels
}  // namespace gko
