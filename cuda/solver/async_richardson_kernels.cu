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


#include "common/cuda_hip/solver/async_richardson_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::Dense<ValueType>* relaxation_factor,
           const matrix::Dense<ValueType>* second_factor,
           const matrix::Csr<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    constexpr int subwarp_size = 1;
    dim3 grid(ceildiv(a->get_size()[0], default_block_size / subwarp_size),
              b->get_size()[1]);
    std::cout << "Run" << std::endl;
    // subwarp_apply<subwarp_size><<<grid, default_block_size>>>(
    //     500, a->get_size()[0], as_cuda_type(a->get_const_values()),
    //     a->get_const_col_idxs(), a->get_const_row_ptrs(),
    //     as_cuda_type(relaxation_factor->get_const_values()),
    //     as_cuda_type(b->get_const_values()), b->get_stride(),
    //     as_cuda_type(c->get_values()), c->get_stride());
    second_subwarp_apply<subwarp_size><<<grid, default_block_size>>>(
        500, a->get_size()[0], as_cuda_type(a->get_const_values()),
        a->get_const_col_idxs(), a->get_const_row_ptrs(),
        as_cuda_type(relaxation_factor->get_const_values()),
        as_cuda_type(second_factor->get_const_values()),
        as_cuda_type(b->get_const_values()), b->get_stride(),
        as_cuda_type(c->get_values()), c->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ASYNC_RICHARDSON_APPLY_KERNEL);


}  // namespace async_richardson
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
