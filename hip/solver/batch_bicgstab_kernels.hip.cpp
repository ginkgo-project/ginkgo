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

#include "core/solver/batch_bicgstab_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/math.hpp>


#include "hip/base/config.hip.hpp"
#include "hip/base/exception.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


constexpr int default_block_size = 256;
constexpr int sm_multiplier = 4;

/**
 * @brief The batch Bicgstab solver namespace.
 *
 * @ingroup batch_bicgstab
 */
namespace batch_bicgstab {

#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
// include all depedencies (note: do not remove this comment)
#include "common/cuda_hip/log/batch_logger.hpp.inc"
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_ell_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_identity.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_jacobi.hpp.inc"
#include "common/cuda_hip/solver/batch_bicgstab_kernels.hpp.inc"
#include "common/cuda_hip/stop/batch_criteria.hpp.inc"


template <typename BatchMatrixType>
int get_num_threads_per_block(std::shared_ptr<const HipExecutor> exec,
                              const int num_rows)
{
    int nwarps = num_rows / 4;
    if (nwarps < 2) {
        nwarps = 2;
    }
    constexpr int device_max_threads = 1024;
    const int num_regs_used_per_thread = 64;
    int max_regs_blk = 0;
    hipDeviceGetAttribute(&max_regs_blk, hipDeviceAttributeMaxRegistersPerBlock,
                          exec->get_device_id());
    const int max_threads_regs =
        ((max_regs_blk / num_regs_used_per_thread) / config::warp_size) *
        config::warp_size;
    const int max_threads = std::min(max_threads_regs, device_max_threads);
    return std::min(nwarps * static_cast<int>(config::warp_size), max_threads);
}

template <typename T>
using BatchBicgstabOptions =
    gko::kernels::batch_bicgstab::BatchBicgstabOptions<T>;

#define BATCH_BICGSTAB_KERNEL_LAUNCH(_stoppertype, _prectype)              \
    hipLaunchKernelGGL(                                                    \
        HIP_KERNEL_NAME(apply_kernel<stop::_stoppertype<ValueType>>),      \
        dim3(nbatch), dim3(block_size), shared_size, 0, shared_gap, sconf, \
        opts.max_its, opts.residual_tol, logger, _prectype(), a, b.values, \
        x.values, workspace.get_data())

template <typename PrecType, typename BatchMatrixType, typename LogType,
          typename ValueType>
static void apply_impl(
    std::shared_ptr<const HipExecutor> exec,
    const BatchBicgstabOptions<remove_complex<ValueType>> opts, LogType logger,
    const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x)
{
    using real_type = gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    const int shared_gap = ((a.num_rows - 1) / 8 + 1) * 8;
    static_assert(default_block_size >= 2 * config::warp_size,
                  "Need at least two warps!");

    int block_size = 256;
    if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
        block_size =
            get_num_threads_per_block<BatchMatrixType>(exec, a.num_rows);
    } else {
        block_size =
            get_num_threads_per_block<BatchMatrixType>(exec, a.num_rows);
    }
    assert(block_size >= 2 * config::warp_size);

    const size_t prec_size =
        PrecType::dynamic_work_size(shared_gap, a.num_nnz) * sizeof(ValueType);
    const int shmem_per_blk = exec->get_max_shared_memory_per_block();
    const auto sconf =
        gko::kernels::batch_bicgstab::compute_shared_storage<PrecType,
                                                             ValueType>(
            shmem_per_blk, shared_gap, a.num_nnz, b.num_rhs);
    const size_t shared_size = sconf.n_shared * shared_gap * sizeof(ValueType) +
                               (sconf.prec_shared ? prec_size : 0);
    auto workspace = gko::Array<ValueType>(
        exec, sconf.gmem_stride_bytes * nbatch / sizeof(ValueType));
    assert(sconf.gmem_stride_bytes % sizeof(ValueType) == 0);

    printf(" Bicgstab: vectors in shared memory = %d\n", sconf.n_shared);
    if (sconf.prec_shared) {
        printf(" Bicgstab: precondiioner is in shared memory.\n");
    }
    printf(" Bicgstab: vectors in global memory = %d\n", sconf.n_global);
    printf(" Hip: number of threads per warp = %d.\n", config::warp_size);
    printf(" Bicgstab: number of threads per block = %d.\n", block_size);
    if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
        BATCH_BICGSTAB_KERNEL_LAUNCH(SimpleAbsResidual, PrecType);
    } else {
        BATCH_BICGSTAB_KERNEL_LAUNCH(SimpleRelResidual, PrecType);
    }
    GKO_HIP_LAST_IF_ERROR_THROW;
}


template <typename MatrixType, typename OptsType, typename LogType,
          typename HipVT>
void dispatch_on_preconditioner(
    std::shared_ptr<const HipExecutor> exec, const OptsType& opts,
    const LogType& logger, const MatrixType* const amat,
    const batch_dense::UniformBatch<const HipVT>& b_b,
    const batch_dense::UniformBatch<HipVT>& x_b)
{
    auto m_b = get_batch_struct(amat);
    if (opts.preconditioner == gko::preconditioner::batch::type::none) {
        apply_impl<BatchIdentity<HipVT>>(exec, opts, logger, m_b, b_b, x_b);
    } else if (opts.preconditioner ==
               gko::preconditioner::batch::type::jacobi) {
        apply_impl<BatchJacobi<HipVT>>(exec, opts, logger, m_b, b_b, x_b);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void apply(std::shared_ptr<const HipExecutor> exec,
           const BatchBicgstabOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    using hip_value_type = hip_type<ValueType>;

    batch_log::SimpleFinalLogger<remove_complex<ValueType>> logger(
        logdata.res_norms->get_values(), logdata.iter_counts.get_data());

    const auto x_b = get_batch_struct(x);
    const auto b_b = get_batch_struct(b);

    if (auto amat = dynamic_cast<const matrix::BatchCsr<ValueType>*>(a)) {
        dispatch_on_preconditioner(exec, opts, logger, amat, b_b, x_b);
    } else if (auto amat =
                   dynamic_cast<const matrix::BatchEll<ValueType>*>(a)) {
        dispatch_on_preconditioner(exec, opts, logger, amat, b_b, x_b);
    } else {
        GKO_NOT_SUPPORTED(a);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL);


}  // namespace batch_bicgstab
}  // namespace hip
}  // namespace kernels
}  // namespace gko
