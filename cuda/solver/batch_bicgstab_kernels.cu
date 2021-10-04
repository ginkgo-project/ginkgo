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


#include <ginkgo/core/base/exception_helpers.hpp>
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


// NOTE: this default block size is not used for the main solver kernel.
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
#include "common/cuda_hip/components/reduction.hpp.inc"
#include "common/cuda_hip/log/batch_logger.hpp.inc"
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_ell_kernels.hpp.inc"
// TODO: remove batch dense include
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_identity.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_jacobi.hpp.inc"
#include "common/cuda_hip/solver/batch_bicgstab_kernels.hpp.inc"
#include "common/cuda_hip/stop/batch_criteria.hpp.inc"


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_num_threads_per_block(std::shared_ptr<const CudaExecutor> exec,
                              const int num_rows)
{
    int nwarps = num_rows / 4;
    if (nwarps < 2) {
        nwarps = 2;
    }
    constexpr int device_max_threads = 1024;
    cudaFuncAttributes funcattr;
    cudaFuncGetAttributes(
        &funcattr,
        apply_kernel<StopType, PrecType, LogType, BatchMatrixType, ValueType>);
    const int num_regs_used = funcattr.numRegs;
    int max_regs_blk = 0;
    cudaDeviceGetAttribute(&max_regs_blk, cudaDevAttrMaxRegistersPerBlock,
                           exec->get_device_id());
    const int max_threads_regs =
        ((max_regs_blk / num_regs_used) / config::warp_size) *
        config::warp_size;
    const int max_threads = std::min(max_threads_regs, device_max_threads);
    return std::min(nwarps * static_cast<int>(config::warp_size), max_threads);
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_max_dynamic_shared_memory(std::shared_ptr<const CudaExecutor> exec,
                                  const int required_cache_storage)
{
    int shmem_per_sm = 0;
    cudaDeviceGetAttribute(&shmem_per_sm,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                           exec->get_device_id());
    printf(" Max shared mem per SM = %d.\n", shmem_per_sm);
    int max_shared_pc =
        100 - static_cast<int>(static_cast<double>(required_cache_storage) /
                               shmem_per_sm * 100);
    if (max_shared_pc <= 0) {
        max_shared_pc = 1;
    }
    printf(" Max shared pc required = %d.\n", max_shared_pc);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaFuncSetAttribute(
        apply_kernel<StopType, PrecType, LogType, BatchMatrixType, ValueType>,
        cudaFuncAttributePreferredSharedMemoryCarveout, max_shared_pc - 1));
    cudaFuncAttributes funcattr;
    cudaFuncGetAttributes(
        &funcattr,
        apply_kernel<StopType, PrecType, LogType, BatchMatrixType, ValueType>);
    printf(" Max dyn. shared memory for batch bcgs = %d.\n",
           funcattr.maxDynamicSharedSizeBytes);
    return funcattr.maxDynamicSharedSizeBytes;
}


template <typename T>
using BatchBicgstabOptions =
    gko::kernels::batch_bicgstab::BatchBicgstabOptions<T>;

#define BATCH_BICGSTAB_KERNEL_LAUNCH(_stoppertype, _prectype)           \
    apply_kernel<stop::_stoppertype<ValueType>>                         \
        <<<nbatch, block_size, shared_size>>>(                          \
            shared_gap, sconf, opts.max_its, opts.residual_tol, logger, \
            _prectype(), a, b.values, x.values, workspace.get_data())

template <typename PrecType, typename BatchMatrixType, typename LogType,
          typename ValueType>
static void apply_impl(
    std::shared_ptr<const CudaExecutor> exec,
    const BatchBicgstabOptions<remove_complex<ValueType>> opts, LogType logger,
    const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x)
{
    using real_type = gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    const int shared_gap = ((a.num_rows - 1) / 8 + 1) * 8;

    // TODO: Add function to BatchCSR to return storage needed per matrix
    const int matrix_storage = (a.num_rows + 1) * sizeof(int) +
                               a.num_nnz * (sizeof(int) + sizeof(ValueType));
    int shmem_per_blk = 0;
    int block_size = 256;
    if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
        shmem_per_blk =
            get_max_dynamic_shared_memory<stop::SimpleAbsResidual<ValueType>,
                                          PrecType, LogType, BatchMatrixType,
                                          ValueType>(exec, matrix_storage);
        block_size =
            get_num_threads_per_block<stop::SimpleAbsResidual<ValueType>,
                                      PrecType, LogType, BatchMatrixType,
                                      ValueType>(exec, a.num_rows);
    } else {
        shmem_per_blk =
            get_max_dynamic_shared_memory<stop::SimpleRelResidual<ValueType>,
                                          PrecType, LogType, BatchMatrixType,
                                          ValueType>(exec, matrix_storage);
        block_size =
            get_num_threads_per_block<stop::SimpleRelResidual<ValueType>,
                                      PrecType, LogType, BatchMatrixType,
                                      ValueType>(exec, a.num_rows);
    }
    assert(block_size >= 2 * config::warp_size);

    const size_t prec_size =
        PrecType::dynamic_work_size(shared_gap, a.num_nnz) * sizeof(ValueType);
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
    printf(" Bicgstab: number of threads per block = %d.\n", block_size);

    if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
        BATCH_BICGSTAB_KERNEL_LAUNCH(SimpleAbsResidual, PrecType);
    } else {
        BATCH_BICGSTAB_KERNEL_LAUNCH(SimpleRelResidual, PrecType);
    }
    GKO_CUDA_LAST_IF_ERROR_THROW;
}


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const BatchBicgstabOptions<remove_complex<ValueType>>& opts,
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
        auto m_b = get_batch_struct(amat);
        auto b_b = get_batch_struct(b);
        if (opts.preconditioner == gko::preconditioner::batch::type::none) {
            apply_impl<BatchIdentity<cu_value_type>>(exec, opts, logger, m_b,
                                                     b_b, x_b);
        } else if (opts.preconditioner ==
                   gko::preconditioner::batch::type::jacobi) {
            apply_impl<BatchJacobi<cu_value_type>>(exec, opts, logger, m_b, b_b,
                                                   x_b);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    } else if (auto amat =
                   dynamic_cast<const matrix::BatchEll<ValueType>*>(a)) {
        auto m_b = get_batch_struct(amat);
        auto b_b = get_batch_struct(b);
        if (opts.preconditioner == gko::preconditioner::batch::type::none) {
            apply_impl<BatchIdentity<cu_value_type>>(exec, opts, logger, m_b,
                                                     b_b, x_b);
        } else if (opts.preconditioner ==
                   gko::preconditioner::batch::type::jacobi) {
            apply_impl<BatchJacobi<cu_value_type>>(exec, opts, logger, m_b, b_b,
                                                   x_b);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    } else {
        GKO_NOT_SUPPORTED(a);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL);


}  // namespace batch_bicgstab
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
