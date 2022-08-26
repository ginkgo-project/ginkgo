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

#include "core/solver/async_richardson_kernels.hpp"


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The async_richardson solver namespace.
 *
 * @ingroup async_richardson
 */
namespace async_richardson {


constexpr int default_block_size = 4 * config::warp_size;

// Choose for different configuration
#define USE_DYNAMIC 1
#define DYNAMIC_OSCB 4
#define STATIC_SUBWARP_SIZE 1
#define USE_THREADFENCE 1


#if USE_DYNAMIC
// This is for dynamic implementation
#include "common/cuda_hip/solver/async_richardson_kernels.hpp.inc"
#else
// This is for static implementation
#include "common/cuda_hip/solver/async_richardson_kernels_static.hpp.inc"
#endif


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const std::string& check, int max_iters,
           const matrix::Dense<ValueType>* relaxation_factor,
           const matrix::Dense<ValueType>* second_factor,
           const matrix::Csr<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
#if USE_DYNAMIC
    int oscb = DYNAMIC_OSCB;
    constexpr int subwarp_size = 1;
    int v100 = 80 * oscb;  // V100 contains 80 SM
    auto num_subwarp = v100 * default_block_size / subwarp_size;
    int gridx = v100;
    if (num_subwarp > a->get_size()[0]) {
        gridx = a->get_size()[0] * subwarp_size / default_block_size;
    }
#else
    constexpr int subwarp_size = STATIC_SUBWARP_SIZE;
    int gridx = ceildiv(a->get_size()[0], default_block_size / subwarp_size);
#endif
    dim3 grid(gridx, b->get_size()[1]);
    if (check == "time") {
        subwarp_apply_time<subwarp_size><<<grid, default_block_size>>>(
            max_iters, a->get_size()[0], as_cuda_type(a->get_const_values()),
            a->get_const_col_idxs(), a->get_const_row_ptrs(),
            as_cuda_type(relaxation_factor->get_const_values()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
    } else if (check == "flow") {
        subwarp_apply_flow<subwarp_size><<<grid, default_block_size>>>(
            max_iters, a->get_size()[0], as_cuda_type(a->get_const_values()),
            a->get_const_col_idxs(), a->get_const_row_ptrs(),
            as_cuda_type(relaxation_factor->get_const_values()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
    } else if (check == "halfflow") {
        subwarp_apply_halfflow<subwarp_size><<<grid, default_block_size>>>(
            max_iters, a->get_size()[0], as_cuda_type(a->get_const_values()),
            a->get_const_col_idxs(), a->get_const_row_ptrs(),
            as_cuda_type(relaxation_factor->get_const_values()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
    } else {
        subwarp_apply<subwarp_size><<<grid, default_block_size>>>(
            max_iters, a->get_size()[0], as_cuda_type(a->get_const_values()),
            a->get_const_col_idxs(), a->get_const_row_ptrs(),
            as_cuda_type(relaxation_factor->get_const_values()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
    }
    // second_subwarp_apply<subwarp_size><<<grid, default_block_size>>>(
    //     max_iters, a->get_size()[0], as_cuda_type(a->get_const_values()),
    //     a->get_const_col_idxs(), a->get_const_row_ptrs(),
    //     as_cuda_type(relaxation_factor->get_const_values()),
    //     as_cuda_type(second_factor->get_const_values()),
    //     as_cuda_type(b->get_const_values()), b->get_stride(),
    //     as_cuda_type(c->get_values()), c->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ASYNC_RICHARDSON_APPLY_KERNEL);


}  // namespace async_richardson
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
