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
    128;  // found by experiments that 128 works the best

}  // namespace

namespace {

template <int subwarpsize, typename ValueType>
__global__ void WM_pGE_kernel_approach_1(const size_type nbatch,
                                         const int nrows, ValueType* const a,
                                         ValueType* const b, ValueType* const c,
                                         ValueType* const d, ValueType* const x)
{}

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

    WM_pGE_kernel_approach_1<subwarpsize><<<grid, block, shared_size>>>(
        nbatch, nrows, tridiag_mat->get_sub_diagonal(),
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
