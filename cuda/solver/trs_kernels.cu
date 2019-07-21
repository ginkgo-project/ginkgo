/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/solver/trs_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The TRS solver namespace.
 *
 * @ingroup trs
 */
namespace trs {


constexpr int default_block_size = 512;


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void solve_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type x_stride, ValueType *__restrict__ x, ValueType *__restrict__ r,
    const ValueType *__restrict__ p, const ValueType *__restrict__ q,
    const ValueType *__restrict__ beta, const ValueType *__restrict__ rho,
    const stopping_status *__restrict__ stop_status) GKO_NOT_IMPLEMENTED;
//{
// TODO (script): change the code imported from solver/cg if needed
//    const auto tidx =
//        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
//    const auto row = tidx / stride;
//    const auto col = tidx % stride;
//
//    if (col >= num_cols || tidx >= num_rows * num_cols ||
//        stop_status[col].has_stopped()) {
//        return;
//    }
//    if (beta[col] != zero<ValueType>()) {
//        const auto tmp = rho[col] / beta[col];
//        x[row * x_stride + col] += tmp * p[tidx];
//        r[tidx] -= tmp * q[tidx];
//    }
//}


template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Csr<ValueType, IndexType> *matrix,
           const matrix::Dense<ValueType> *b,
           matrix::Dense<ValueType> *x) GKO_NOT_IMPLEMENTED;
//{
// TODO (script): change the code imported from solver/cg if needed
//    const dim3 block_size(default_block_size, 1, 1);
//    const dim3 grid_size(
//        ceildiv(p->get_size()[0] * p->get_stride(), block_size.x), 1, 1);
//
//    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
//        p->get_size()[0], p->get_size()[1], p->get_stride(), x->get_stride(),
//        as_cuda_type(x->get_values()), as_cuda_type(r->get_values()),
//        as_cuda_type(p->get_const_values()),
//        as_cuda_type(q->get_const_values()),
//        as_cuda_type(beta->get_const_values()),
//        as_cuda_type(rho->get_const_values()),
//        as_cuda_type(stop_status->get_const_data()));
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TRS_SOLVE_KERNEL);


}  // namespace trs
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
