/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
#include "core/components/fill_array_kernels.hpp"
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


#include "common/cuda_hip/preconditioner/jacobi_generate_kernel.hpp.inc"


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
                    as_hip_type(mtx->get_const_values()), as_hip_type(accuracy),
                    as_hip_type(block_data), storage_scheme,
                    as_hip_type(conditioning), block_precisions, block_ptrs,
                    num_blocks);
        } else {
            kernel::generate<max_block_size, subwarp_size, warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    mtx->get_size()[0], mtx->get_const_row_ptrs(),
                    mtx->get_const_col_idxs(),
                    as_hip_type(mtx->get_const_values()),
                    as_hip_type(block_data), storage_scheme, block_ptrs,
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
}  // namespace hip
}  // namespace kernels
}  // namespace gko
