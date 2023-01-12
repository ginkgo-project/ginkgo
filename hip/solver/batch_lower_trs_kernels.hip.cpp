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

#include "core/solver/batch_lower_trs_kernels.hpp"


#include "core/matrix/batch_struct.hpp"
#include "hip/base/exception.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/load_store.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"

namespace gko {
namespace kernels {
namespace hip {
namespace batch_lower_trs {

namespace {

constexpr int default_block_size = 256;

#include "common/cuda_hip/solver/batch_lower_trs_kernels.hpp.inc"
}  // namespace

template <typename BatchMatrixType, typename ValueType>
void call_apply_kernel(
    const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b_b,
    const gko::batch_dense::UniformBatch<ValueType>& x_b)
{
    const auto nbatch = a.num_batch;
    assert(b_b.num_rhs == 1);
    const int shared_size =
        gko::kernels::batch_lower_trs::local_memory_requirement<ValueType>(
            a.num_rows, b_b.num_rhs);

    hipLaunchKernelGGL(apply_kernel, nbatch, default_block_size, shared_size, 0,
                       a, b_b.values, x_b.values);

    GKO_HIP_LAST_IF_ERROR_THROW;
}


template <typename ValueType>
void dispatch_on_matrix_type(const BatchLinOp* const sys_mat,
                             const matrix::BatchDense<ValueType>* const b,
                             matrix::BatchDense<ValueType>* const x)
{
    namespace device = gko::kernels::hip;
    const auto b_b = device::get_batch_struct(b);
    const auto x_b = device::get_batch_struct(x);

    if (auto amat = dynamic_cast<const matrix::BatchCsr<ValueType>*>(sys_mat)) {
        auto m_b = device::get_batch_struct(amat);
        call_apply_kernel(m_b, b_b, x_b);

    } else if (auto amat =
                   dynamic_cast<const matrix::BatchEll<ValueType>*>(sys_mat)) {
        auto m_b = device::get_batch_struct(amat);
        call_apply_kernel(m_b, b_b, x_b);

    } else if (auto amat = dynamic_cast<const matrix::BatchDense<ValueType>*>(
                   sys_mat)) {
        auto m_b = device::get_batch_struct(amat);
        call_apply_kernel(m_b, b_b, x_b);
    } else {
        GKO_NOT_SUPPORTED(sys_mat);
    }
}


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const BatchLinOp* const sys_mat,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x)
{
    dispatch_on_matrix_type(sys_mat, b, x);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_LOWER_TRS_APPLY_KERNEL);


}  // namespace batch_lower_trs
}  // namespace hip
}  // namespace kernels
}  // namespace gko
