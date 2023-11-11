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

#include "core/solver/batch_band_solver_kernels.hpp"

#include <chrono>
#include <ginkgo/config.hpp>
#include <ginkgo/core/base/math.hpp>
#include "core/matrix/batch_struct.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/math.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"

namespace gko {
namespace kernels {
namespace cuda {
namespace batch_band_solver {

namespace {

template <typename ValueType>
__host__ __device__ bool is_matrix_in_shared_mem(const int N, const int KL,
                                                 const int KU)
{
    const int band_nrows = 2 * KL + KU + 1;
    const size_type storage_in_bytes = band_nrows * N * sizeof(ValueType);
    if (storage_in_bytes <= 20000) {
        return true;
    } else {
        return false;
    }
}

constexpr int default_block_size = 128;

// Block size for optimal performance - Found out by hit and trial
int inline get_thread_block_size_unblocked_banded(const int nrows)
{
    if (nrows <= 50) {
        return 64;
    } else if (nrows <= 100) {
        return 128;
    } else if (nrows <= 200) {
        return 256;
    } else {
        return 512;
    }
    // too many resources requested on launch -
    //  error for 1024 block size
}

int inline get_thread_block_size_blocked_banded(const int nrows)
{
    if (nrows <= 100) {
        return 64;
    } else if (nrows <= 200) {
        return 128;
    } else if (nrows <= 400) {
        return 256;
    } else {
        return 512;  // too many reources requested on launch - error for block
                     // size 1024
    }
}


// include all depedencies (note: do not remove this comment)
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/solver/batch_band_solver_kernels.hpp.inc"

}  // namespace


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::BatchBand<ValueType>* const band_mat,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x, const int workspace_size,
           ValueType* const workspace_ptr,
           const enum gko::solver::batch_band_solve_approach approach,
           const int blocked_solve_panel_size)
{
    const auto nbatch = band_mat->get_num_batch_entries();
    const auto nrows = static_cast<int>(band_mat->get_size().at(0)[0]);
    const auto nrhs = static_cast<int>(b->get_size().at(0)[1]);
    assert(nrhs == 1);
    const auto KL = static_cast<int>(band_mat->get_num_subdiagonals().at(0));
    const auto KU = static_cast<int>(band_mat->get_num_superdiagonals().at(0));

    assert(workspace_size >= band_mat->get_num_stored_elements());

    ValueType* const band_arr = workspace_ptr;
    exec->copy(band_mat->get_num_stored_elements(),
               band_mat->get_const_band_array(), band_arr);

    int shared_size = 0;
    if (is_matrix_in_shared_mem<ValueType>(
            nrows, KL,
            KU)) {  // TODO: Avoid the unnecessary workspace copy in this case:
                    // So either have a kernel prototype which accepts const
                    // band array pointer or use const cast.
        shared_size +=
            (band_mat->get_num_stored_elements() / nbatch) * sizeof(ValueType);
    }

    auto start = std::chrono::high_resolution_clock::now();


    if (approach == gko::solver::batch_band_solve_approach::unblocked ||
        (approach == gko::solver::batch_band_solve_approach::blocked &&
         blocked_solve_panel_size > KL)) {
        shared_size +=
            gko::kernels::batch_band_solver::local_memory_requirement<
                ValueType>(nrows, nrhs,
                           gko::solver::batch_band_solve_approach::unblocked);

        dim3 block(get_thread_block_size_unblocked_banded(nrows));
        dim3 grid(nbatch);

        band_solver_unblocked_kernel<config::warp_size>
            <<<grid, block, shared_size>>>(nbatch, nrows, KL, KU,
                                           as_cuda_type(band_arr),
                                           as_cuda_type(b->get_const_values()),
                                           as_cuda_type(x->get_values()));

    } else if (approach == gko::solver::batch_band_solve_approach::blocked) {
        shared_size +=
            gko::kernels::batch_band_solver::local_memory_requirement<
                ValueType>(nrows, nrhs,
                           gko::solver::batch_band_solve_approach::blocked,
                           blocked_solve_panel_size);

        dim3 block(get_thread_block_size_blocked_banded(nrows));
        dim3 grid(nbatch);

        const int subwarp_size = 8;
        band_solver_blocked_kernel<subwarp_size><<<grid, block, shared_size>>>(
            nbatch, nrows, KL, KU, blocked_solve_panel_size,
            as_cuda_type(band_arr), as_cuda_type(b->get_const_values()),
            as_cuda_type(x->get_values()));

    } else {
        GKO_NOT_IMPLEMENTED;
    }

    exec->synchronize();
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    double total_time_millisec =
        (double)(std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                       start))
            .count() /
        (double)1000;

    std::cout << "\nThe entire internal solve took " << total_time_millisec
              << " milliseconds." << std::endl;

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BAND_SOLVER_APPLY_KERNEL);

}  // namespace batch_band_solver
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
