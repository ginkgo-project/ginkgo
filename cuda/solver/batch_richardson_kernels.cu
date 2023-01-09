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

#include "core/solver/batch_richardson_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "core/solver/batch_dispatch.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/kernel_config.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {


#define GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM 1
constexpr int default_block_size = 256;
constexpr int sm_multiplier = 4;


/**
 * @brief The batch Richardson solver namespace.
 *
 * @ingroup batch_rich
 */
namespace batch_rich {


#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
// include all depedencies (note: do not remove this comment)
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_ell_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/solver/batch_richardson_kernels.hpp.inc"


template <typename T>
using BatchRichardsonOptions =
    gko::kernels::batch_rich::BatchRichardsonOptions<T>;


template <typename CuValueType>
class KernelCaller {
public:
    using value_type = CuValueType;

    KernelCaller(std::shared_ptr<const CudaExecutor> exec,
                 const BatchRichardsonOptions<remove_complex<value_type>> opts)
        : exec_{exec}, opts_{opts}
    {}

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(LogType logger, const BatchMatrixType& a, PrecType prec,
                     const gko::batch_dense::UniformBatch<const value_type>& b,
                     const gko::batch_dense::UniformBatch<value_type>& x) const
    {
        using real_type = gko::remove_complex<value_type>;
        const size_type nbatch = a.num_batch;

        const int shared_size =
            gko::kernels::batch_rich::local_memory_requirement<value_type>(
                a.num_rows, b.num_rhs) +
            PrecType::dynamic_work_size(a.num_rows, a.num_nnz) *
                sizeof(value_type);

        apply_kernel<StopType><<<nbatch, default_block_size, shared_size>>>(
            opts_.max_its, opts_.residual_tol, opts_.relax_factor, logger, prec,
            a, b.values, x.values);

        GKO_CUDA_LAST_IF_ERROR_THROW;
    }

private:
    std::shared_ptr<const CudaExecutor> exec_;
    const BatchRichardsonOptions<remove_complex<value_type>> opts_;
};


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const BatchRichardsonOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a, const BatchLinOp* const prec,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    using cu_value_type = cuda_type<ValueType>;
    auto dispatcher = batch_solver::create_dispatcher<ValueType>(
        KernelCaller<cu_value_type>(exec, opts), opts, a, prec);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_RICHARDSON_APPLY_KERNEL);


}  // namespace batch_rich
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
