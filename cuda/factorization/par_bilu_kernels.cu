/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/factorization/par_bilu_kernels.hpp"


#include "core/components/fixed_block.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The parallel block-ILU factorization namespace.
 *
 * @ingroup factor
 */
namespace par_bilu_factorization {


constexpr int default_block_size{512};


#include "common/components/reduction.hpp.inc"
#include "common/components/warp_blas.hpp.inc"
#include "common/factorization/par_bilu_kernels.hpp.inc"


template <int mat_blk_sz, typename ValueType, typename IndexType>
void compute_bilu_factors_impl(
    std::shared_ptr<const CudaExecutor> exec, const int iterations,
    const matrix::Fbcsr<ValueType, IndexType> *const system_matrix,
    matrix::Fbcsr<ValueType, IndexType> *const l_factor,
    matrix::Fbcsr<ValueType, IndexType> *const u_factor)
{
    constexpr int subwarp_size = config::warp_size;
    const auto num_b_rows = system_matrix->get_num_block_rows();
    const dim3 block_size{default_block_size, 1, 1};
    const int num_sm = exec->get_num_multiprocessor();
    const dim3 grid_dim{static_cast<uint32>(2 * num_sm), 1, 1};
    for (int i = 0; i < iterations; ++i) {
        kernel::compute_bilu_factors_fbcsr_1<mat_blk_sz, subwarp_size>
            <<<grid_dim, block_size, 0, 0>>>(
                num_b_rows, as_cuda_type(system_matrix->get_const_row_ptrs()),
                as_cuda_type(system_matrix->get_const_col_idxs()),
                as_cuda_type(system_matrix->get_const_values()),
                as_cuda_type(l_factor->get_const_row_ptrs()),
                as_cuda_type(l_factor->get_const_col_idxs()),
                as_cuda_type(l_factor->get_values()),
                as_cuda_type(u_factor->get_const_row_ptrs()),
                as_cuda_type(u_factor->get_const_col_idxs()),
                as_cuda_type(u_factor->get_values()));
        // exec->synchronize();
    }
}

template <typename ValueType, typename IndexType>
void compute_bilu_factors(
    std::shared_ptr<const CudaExecutor> exec, int iterations,
    const matrix::Fbcsr<ValueType, IndexType> *const system_matrix,
    matrix::Fbcsr<ValueType, IndexType> *const l_factor,
    matrix::Fbcsr<ValueType, IndexType> *const u_factor)
{
    iterations = (iterations == -1) ? 10 : iterations;
    const int bs = system_matrix->get_block_size();
    if (bs == 2) {
        compute_bilu_factors_impl<2>(exec, iterations, system_matrix, l_factor,
                                     u_factor);
    } else if (bs == 3) {
        compute_bilu_factors_impl<3>(exec, iterations, system_matrix, l_factor,
                                     u_factor);
    } else if (bs == 4) {
        compute_bilu_factors_impl<4>(exec, iterations, system_matrix, l_factor,
                                     u_factor);
    } else if (bs == 7) {
        compute_bilu_factors_impl<7>(exec, iterations, system_matrix, l_factor,
                                     u_factor);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_BILU_FACTORS_FBCSR_KERNEL);


}  // namespace par_bilu_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
