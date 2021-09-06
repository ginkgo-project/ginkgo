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

#include "core/solver/batch_gmres_kernels.hpp"


#include <ginkgo/batch_config.hpp>
#include <ginkgo/core/base/math.hpp>


#include "cuda/base/config.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {


constexpr int default_block_size = 128;
constexpr int sm_multiplier = 4;

/**
 * @brief The batch Gmres solver namespace.
 *
 * @ingroup batch_gmres
 */
namespace batch_gmres {


#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
// include all depedencies (note: do not remove this comment)
#include "common/cuda_hip/components/reduction.hpp.inc"
#include "common/cuda_hip/log/batch_logger.hpp.inc"
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_identity.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_jacobi.hpp.inc"
#include "common/cuda_hip/solver/batch_gmres_kernels.hpp.inc"
#include "common/cuda_hip/stop/batch_criteria.hpp.inc"


template <typename T>
using BatchGmresOptions = gko::kernels::batch_gmres::BatchGmresOptions<T>;

#if GKO_CUDA_BATCH_GMRES_HAVE_NO_SHMEM

#define BATCH_GMRES_KERNEL_LAUNCH(_stoppertype, _prectype)                    \
    apply_kernel<stop::_stoppertype<ValueType>>                               \
        <<<nbatch, default_block_size>>>(global_gap, opts.max_its,            \
                                         opts.residual_tol, opts.restart_num, \
                                         logger, _prectype<ValueType>(), a,   \
                                         bptr, xptr, workspace.get_data())
#else


#define BATCH_GMRES_KERNEL_LAUNCH(_stoppertype, _prectype)                 \
    apply_kernel<stop::_stoppertype<ValueType>>                            \
        <<<nbatch, default_block_size, shared_size>>>(                     \
            global_gap, opts.max_its, opts.residual_tol, opts.restart_num, \
            logger, _prectype<ValueType>(), a, bptr, xptr)

#endif

template <typename BatchMatrixType, typename LogType, typename ValueType>
static void apply_impl(std::shared_ptr<const CudaExecutor> exec,
                       const BatchGmresOptions<remove_complex<ValueType>> opts,
                       LogType logger, const BatchMatrixType& a,
                       const gko::batch_dense::UniformBatch<const ValueType>& b,
                       const gko::batch_dense::UniformBatch<ValueType>& x)
{
    using real_type = gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    const ValueType* const bptr = b.values;
    ValueType* const xptr = x.values;

    static_assert(default_block_size >= 2 * config::warp_size,
                  "Need at least two warps per block!");

    int shared_size =
        gko::kernels::batch_gmres::local_memory_requirement<ValueType>(
            a.num_rows, b.num_rhs, opts.restart_num);
    auto nrhs = b.num_rhs;
    auto nrows = a.num_rows;
    auto restart = opts.restart_num;
    int global_gap = 6 * nrows * nrhs + 3 * restart * nrhs +
                     (restart + 1) * nrhs + restart * (restart + 1) * nrhs +
                     nrows * (restart + 1) * nrhs;
    auto workspace = gko::Array<ValueType>(exec);

    if (opts.preconditioner == gko::preconditioner::batch::type::none) {
        shared_size +=
            BatchIdentity<ValueType>::dynamic_work_size(a.num_rows, a.num_nnz) *
            sizeof(ValueType);
#if GKO_CUDA_BATCH_GMRES_HAVE_NO_SHMEM
        workspace = gko::Array<ValueType>(
            exec,
            static_cast<size_type>(shared_size * nbatch / sizeof(ValueType)));
#endif
        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            BATCH_GMRES_KERNEL_LAUNCH(SimpleAbsResidual, BatchIdentity);
        } else {
            BATCH_GMRES_KERNEL_LAUNCH(SimpleRelResidual, BatchIdentity);
        }
    } else if (opts.preconditioner ==
               gko::preconditioner::batch::type::jacobi) {
        shared_size +=
            BatchJacobi<ValueType>::dynamic_work_size(a.num_rows, a.num_nnz) *
            sizeof(ValueType);
#if GKO_CUDA_BATCH_GMRES_HAVE_NO_SHMEM
        workspace = gko::Array<ValueType>(
            exec,
            static_cast<size_type>(shared_size * nbatch / sizeof(ValueType)));
#endif
        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            BATCH_GMRES_KERNEL_LAUNCH(SimpleAbsResidual, BatchJacobi);
        } else {
            BATCH_GMRES_KERNEL_LAUNCH(SimpleRelResidual, BatchJacobi);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
    GKO_CUDA_LAST_IF_ERROR_THROW;
}


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const BatchGmresOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    using cu_value_type = cuda_type<ValueType>;

    batch_log::SimpleFinalLogger<remove_complex<ValueType>> logger(
        logdata.res_norms->get_values(), logdata.iter_counts.get_data());

    const gko::batch_dense::UniformBatch<cu_value_type> x_b =
        get_batch_struct(x);

    if (auto amat = dynamic_cast<const matrix::BatchCsr<ValueType>*>(a)) {
        // const gko::batch_csr::UniformBatch<cu_value_type> m_b =
        //     get_batch_struct(const_cast<matrix::BatchCsr<ValueType>
        //     *>(amat));
        const auto m_b = get_batch_struct(amat);
        const auto b_b = get_batch_struct(b);
        apply_impl(exec, opts, logger, m_b, b_b, x_b);
    } else {
        GKO_NOT_SUPPORTED(a);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_GMRES_APPLY_KERNEL);


}  // namespace batch_gmres
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
