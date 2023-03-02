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

#include "core/solver/batch_tridiagonal_solver_kernels.hpp"


#include "core/matrix/batch_struct.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/components/load_store.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"

namespace gko {
namespace kernels {
namespace cuda {
namespace batch_tridiagonal_solver {

namespace {

constexpr int default_subwarp_size = config::warp_size;
constexpr int default_block_size =
    128;  // found out by experimentally that 128 works the best

}  // namespace

namespace {

template <typename ValueType, typename Group>
__device__ void broadcast(Group& subwarp_grp, const int target_lane,
                          const ValueType& my_a, const ValueType& my_b,
                          const ValueType& my_c, const ValueType& my_d,
                          ValueType& piv_a, ValueType& piv_b, ValueType& piv_c,
                          ValueType& piv_d)
{
    piv_a = subwarp_grp.shfl(my_a, target_lane);
    piv_b = subwarp_grp.shfl(my_b, target_lane);
    piv_c = subwarp_grp.shfl(my_c, target_lane);
    piv_d = subwarp_grp.shfl(my_d, target_lane);
}

template <typename ValueType, typename Group>
__device__ void WM_step(Group& subwarp_grp, const int curr_group_size,
                        const int tile_size, ValueType& my_a, ValueType& my_b,
                        ValueType& my_c, ValueType& my_d)
{
    const int lane = subwarp_grp.thread_rank();
    const int curr_grp_idx = lane / curr_group_size;

    const bool is_left_grp = (curr_grp_idx % 2) == 0;
    ValueType piv_a, piv_b, piv_c, piv_d;
    ValueType my_f = zero<ValueType>();

    const int left_grp_last_lane =
        is_left_grp ? ((curr_grp_idx + 1) * curr_group_size) - 1
                    : (curr_grp_idx * curr_group_size) - 1;
    const int right_grp_first_lane = left_grp_last_lane + 1;

    // broadcast last equation of the left group
    broadcast(subwarp_grp, left_grp_last_lane, my_a, my_b, my_c, my_d, piv_a,
              piv_b, piv_c, piv_d);

    if (lane == right_grp_first_lane) {
        my_f = my_b;
    }

    // eliminate a of the right group
    if (!is_left_grp) {
        const ValueType mult = my_a / piv_b;
        my_a = -1 * piv_a * mult;
        my_d -= piv_d * mult;
        my_f -= piv_c * mult;
    }
    if (lane == right_grp_first_lane) {
        my_b = my_f;
    }

    // broadcast first equation of the right group
    broadcast(subwarp_grp, right_grp_first_lane, my_a, my_b, my_c, my_d, piv_a,
              piv_b, piv_c, piv_d);

    // eliminate c of the left group
    if (is_left_grp) {
        const ValueType mult = my_c / piv_b;
        my_a -= piv_a * mult;
        my_d -= piv_d * mult;
        my_c = -1 * piv_c * mult;
    }

    // eliminate fill-in of the right group except for its first row
    if (!is_left_grp && lane != right_grp_first_lane) {
        const ValueType mult = my_f / piv_b;
        my_a -= piv_a * mult;
        my_d -= piv_d * mult;
        my_c -= piv_c * mult;
    }
}

template <typename ValueType, typename Group>
__device__ void WM_phase(const int num_WM_steps, Group& subwarp_grp,
                         const int tile_size, ValueType& my_a, ValueType& my_b,
                         ValueType& my_c, ValueType& my_d, int& curr_group_size)
{
    for (int i = 0; i < num_WM_steps; i++) {
        WM_step(curr_group_size, tile_size, my_a, my_b, my_c, my_d);
        curr_group_size *= 2;
        subwarp_grp.sync();
    }
}

template <int subwarp_size, typename ValueType>
__global__ void WM_pGE_kernel_approach_1(const int num_WM_steps,
                                         const size_type nbatch,
                                         const int nrows, ValueType* const a,
                                         ValueType* const b, ValueType* const c,
                                         ValueType* const d, ValueType* const x)
{
    auto subwarpgrp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int subgrpwarp_id_in_grid =
        thread::get_subwarp_id_flat<subwarp_size, int>();
    const int total_num_subwarp_grps_in_grid =
        thread::get_subwarp_num_flat<subwarp_size, int>();
    const int id_within_warp = subwarpgrp.thread_rank();

    // a subwarp per matrix in the batch
    for (size_type batch_idx = subgrpwarp_id_in_grid; batch_idx < nbatch;
         batch_idx += total_num_subwarp_grps_in_grid) {
        // Approach: a thread in the subwarp handles one row of the matrix or to
        // be precise, a row in the matrix tile
        const auto tile_size = subwarp_size;
        const auto num_tiles = ceildiv(nrows, tile_size);
        const bool is_last_tile_similar = ((nrows % tile_size) == 0);
        assert(pow(2, num_WM_steps) <= tile_size);

        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            const int row_idx_st_tile = tile_id * tile_size;  // inclusive
            const int row_idx_end_tile =
                tile_id == num_tiles - 1
                    ? nrows
                    : (tile_id + 1) * tile_size;  // exclusive

            ValueType my_a;
            ValueType my_b;
            ValueType my_c;
            ValueType my_d;

            const int my_row_idx = row_idx_st_tile + id_within_warp;

            if (my_row_idx < row_idx_end_tile) {
                my_a = a[batch_idx * nrows + my_row_idx];
                my_b = b[batch_idx * nrows + my_row_idx];
                my_c = c[batch_idx * nrows + my_row_idx];
                my_d = d[batch_idx * nrows + my_row_idx];
                // coalesced accesses while reading data
            }

            // TODO: last tile - if non-similar ???

            // Phase-1 of the alogithm- WM phase
            int curr_group_size = 1;
            WM_phase(num_WM_steps, subwarpgrp, tile_size, my_a, my_b, my_c,
                     my_d, curr_group_size);
            // In each WM step, the adjacent groups are merged independently.

            // Phase-2 of the algorithm - Full Gaussean elimination of the
            // groups Now perform full Gaussean elimination on each group of the
            // transformed system to eliminate the bottom spikes
            // Forward_full_GE_phase();
        }
    }
}

}  // namespace


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           matrix::BatchTridiagonal<ValueType>* const tridiag_mat,
           matrix::BatchDense<ValueType>* const rhs,
           matrix::BatchDense<ValueType>* const x)
{
    const auto nbatch = tridiag_mat->get_num_batch_entries();
    const auto nrows = static_cast<int>(tridiag_mat->get_size().at(0)[0]);
    const auto nrhs = rhs->get_size().at(0)[1];
    assert(nrhs == 1);

    const int shared_size =
        gko::kernels::batch_tridiagonal_solver::local_memory_requirement<
            ValueType>(nrows, nrhs);

    const auto subwarpsize = default_subwarp_size;
    dim3 block(default_block_size);
    dim3 grid(ceildiv(nbatch * subwarpsize, default_block_size));

    const int num_WM_steps = 2;

    WM_pGE_kernel_approach_1<subwarpsize><<<grid, block, shared_size>>>(
        num_WM_steps, nbatch, nrows, tridiag_mat->get_sub_diagonal(),
        tridiag_mat->get_main_diagonal(), tridiag_mat->get_super_diagonal(),
        rhs->get_values(), x->get_values());

    GKO_CUDA_LAST_IF_ERROR_THROW;
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_TRIDIAGONAL_SOLVER_APPLY_KERNEL);


}  // namespace batch_tridiagonal_solver
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
