// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/async_jacobi_kernels.hpp"

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
 * @brief The async_jacobi solver namespace.
 *
 * @ingroup async_jacobi
 */
namespace async_jacobi {


constexpr int default_block_size = 4 * config::warp_size;

#define NOSYNC static_assert(true, "avoid semi-colon warning")

// Choose for different configuration
#define USE_DYNAMIC 1
#define DYNAMIC_OSCB 4
#define SUBWARP_SIZE 1
#define USE_THREADFENCE 1
#define APPLY_SYNC NOSYNC
// APPLY_SYNC can use __syncwarp(), __syncthreads(), or NOSYNC


#if USE_DYNAMIC
// This is for dynamic implementation
#include "common/cuda_hip/solver/async_jacobi_kernels.hpp.inc"
#else
// This is for static implementation
#include "common/cuda_hip/solver/async_jacobi_kernels_static.hpp.inc"
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
    constexpr int subwarp_size = SUBWARP_SIZE;
    int num_blocks =
        exec->get_num_multiprocessor() * oscb;  // V100 contains 80 SM
    auto num_subwarp = num_blocks * default_block_size / subwarp_size;
    int gridx = num_blocks;
    if (num_subwarp > a->get_size()[0]) {
        gridx = a->get_size()[0] * subwarp_size / default_block_size;
    }
#else
    constexpr int subwarp_size = SUBWARP_SIZE;
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
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_ASYNC_JACOBI_APPLY_KERNEL);


}  // namespace async_jacobi
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
