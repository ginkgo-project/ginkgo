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

#include "core/solver/batch_richardson_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "cuda/base/config.hpp"
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


#include "common/components/uninitialized_array.hpp.inc"
// include all depedencies (note: do not remove this comment)
#include "common/components/reduction.hpp.inc"
#include "common/log/batch_logger.hpp.inc"
#include "common/matrix/batch_csr_kernels.hpp.inc"
#include "common/matrix/batch_dense_kernels.hpp.inc"
#include "common/matrix/batch_vector_kernels.hpp.inc"
#include "common/preconditioner/batch_identity.hpp.inc"
#include "common/preconditioner/batch_jacobi.hpp.inc"
#include "common/solver/batch_richardson_kernels.hpp.inc"
#include "common/stop/batch_criteria.hpp.inc"


template <typename T>
using BatchRichardsonOptions =
    gko::kernels::batch_rich::BatchRichardsonOptions<T>;

#define BATCH_RICHARDSON_KERNEL_LAUNCH(_stoppertype, _prectype)         \
    apply_kernel<stop::_stoppertype<ValueType>>                         \
        <<<nbatch, default_block_size, shared_size>>>(                  \
            opts.max_its, opts.residual_tol, opts.relax_factor, logger, \
            _prectype<ValueType>(), a, bptr, xptr)

template <typename BatchMatrixType, typename LogType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const CudaExecutor> exec,
    const BatchRichardsonOptions<remove_complex<ValueType>> opts,
    LogType logger, const BatchMatrixType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x)
{
    using real_type = gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    if (b.num_rhs > 1) {
        GKO_NOT_IMPLEMENTED;
    }
    const ValueType *const bptr = b.values;
    ValueType *const xptr = x.values;

    gko::kernels::cuda::configure_shared_memory<ValueType>();

    int shared_size =
#if GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM
        gko::kernels::batch_rich::local_memory_requirement<ValueType>(
            a.num_rows, b.num_rhs);
#else
        0;
#endif

    if (opts.preconditioner == gko::preconditioner::batch::type::none) {
#if GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM
        shared_size +=
            BatchIdentity<ValueType>::dynamic_work_size(a.num_rows, a.num_nnz) *
            sizeof(ValueType);
#endif
        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            BATCH_RICHARDSON_KERNEL_LAUNCH(SimpleAbsResidual, BatchIdentity);
        } else {
            BATCH_RICHARDSON_KERNEL_LAUNCH(SimpleRelResidual, BatchIdentity);
        }
    } else if (opts.preconditioner ==
               gko::preconditioner::batch::type::jacobi) {
#if GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM
        shared_size +=
            BatchJacobi<ValueType>::dynamic_work_size(a.num_rows, a.num_nnz) *
            sizeof(ValueType);
#endif
        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            BATCH_RICHARDSON_KERNEL_LAUNCH(SimpleAbsResidual, BatchJacobi);
        } else {
            BATCH_RICHARDSON_KERNEL_LAUNCH(SimpleRelResidual, BatchJacobi);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const BatchRichardsonOptions<remove_complex<ValueType>> &opts,
           const BatchLinOp *const a,
           const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           log::BatchLogData<ValueType> &logdata)
{
    using cu_value_type = cuda_type<ValueType>;

    // batch_log::FinalLogger<remove_complex<ValueType>> logger(
    //     static_cast<int>(b->get_size().at(0)[1]), opts.max_its,
    //     logdata.res_norms->get_values(), logdata.iter_counts.get_data());
    batch_log::SimpleFinalLogger<remove_complex<ValueType>> logger(
        logdata.res_norms->get_values(), logdata.iter_counts.get_data());

    const gko::batch_dense::UniformBatch<cu_value_type> x_b =
        get_batch_struct(x);
    if (auto amat = dynamic_cast<const matrix::BatchCsr<ValueType> *>(a)) {
        auto m_b = get_batch_struct(amat);
        auto b_b = get_batch_struct(b);
        apply_impl(exec, opts, logger, m_b, b_b, x_b);
    } else {
        GKO_NOT_SUPPORTED(a);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_RICHARDSON_APPLY_KERNEL);


}  // namespace batch_rich
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
