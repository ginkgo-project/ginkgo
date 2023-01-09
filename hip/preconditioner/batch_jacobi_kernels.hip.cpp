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

#include "core/preconditioner/batch_jacobi_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/matrix/batch_struct.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/diagonal_block_manipulation.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"
#include "hip/components/warp_blas.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"

namespace gko {
namespace kernels {
namespace hip {


namespace batch_jacobi {


constexpr int default_block_size = 128;
// constexpr int sm_multiplier = 4;

#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_jacobi.hpp.inc"

template <typename ValueType>
void batch_jacobi_apply(std::shared_ptr<const gko::HipExecutor> exec,
                        const matrix::BatchEll<ValueType>* const a,
                        const matrix::BatchDense<ValueType>* const b,
                        matrix::BatchDense<ValueType>* const x)
{
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto x_ub = get_batch_struct(x);
    const size_type nbatch = a->get_num_batch_entries();
    const int shared_size =
        BatchJacobi<ValueType>::dynamic_work_size(a_ub.num_rows, a_ub.num_nnz) *
        sizeof(ValueType);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(batch_jacobi), dim3(nbatch),
                       dim3(default_block_size), shared_size, 0,
                       BatchJacobi<hip_type<ValueType>>(), a_ub, b_ub, x_ub);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_JACOBI_ELL_KERNEL);


template <typename ValueType>
void batch_jacobi_apply(std::shared_ptr<const gko::HipExecutor> exec,
                        const matrix::BatchCsr<ValueType>* const a,
                        const matrix::BatchDense<ValueType>* const b,
                        matrix::BatchDense<ValueType>* const x)
{
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto x_ub = get_batch_struct(x);
    const size_type nbatch = a->get_num_batch_entries();
    const int shared_size =
        BatchJacobi<ValueType>::dynamic_work_size(a_ub.num_rows, a_ub.num_nnz) *
        sizeof(ValueType);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(batch_jacobi), dim3(nbatch),
                       dim3(default_block_size), shared_size, 0,
                       BatchJacobi<hip_type<ValueType>>(), a_ub, b_ub, x_ub);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_JACOBI_KERNEL);

}  // namespace batch_jacobi
}  // namespace hip
}  // namespace kernels
}  // namespace gko
