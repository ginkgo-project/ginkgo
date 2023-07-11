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

#include "core/solver/batch_gmres_kernels.hpp"


#include <ginkgo/batch_config.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/solver/batch_dispatch.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/kernel_config.cuh"
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
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_ell_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/solver/batch_gmres_kernels.hpp.inc"

int get_larger_power(int value, int guess = 64)
{
    return guess >= value ? guess : get_larger_power(value, guess << 1);
}

template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_num_threads_per_block(std::shared_ptr<const CudaExecutor> exec,
                              const int num_rows)
{
    int nwarps = num_rows / 4;
    if (nwarps < 2) {
        nwarps = 2;
    }
    const int min_block_size = 2 * config::warp_size;
    const int device_max_threads =
        ((std::max(num_rows, min_block_size)) / config::warp_size) *
        config::warp_size;
    cudaFuncAttributes funcattr;
    cudaFuncGetAttributes(&funcattr,
                          apply_kernel<StopType, 0, 0, PrecType, LogType,
                                       BatchMatrixType, ValueType>);
    const int num_regs_used = funcattr.numRegs;
    int max_regs_blk = 0;
    cudaDeviceGetAttribute(&max_regs_blk, cudaDevAttrMaxRegistersPerBlock,
                           exec->get_device_id());
    // FIXME: Using magic number, 1.1
    const int max_threads_regs =
        ((max_regs_blk /
          static_cast<int>((static_cast<double>(num_regs_used) * 1.1))) /
         config::warp_size) *
        config::warp_size;
    int max_threads = std::min(max_threads_regs, device_max_threads);
    max_threads = max_threads <= 1024 ? max_threads : 1024;
    return std::min(nwarps * static_cast<int>(config::warp_size), max_threads);
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_max_dynamic_shared_memory(std::shared_ptr<const CudaExecutor> exec,
                                  const size_type required_cache_storage)
{
    int shmem_per_sm = 0;
    cudaDeviceGetAttribute(&shmem_per_sm,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                           exec->get_device_id());
    // std::cerr << " Max shared mem per SM = " << shmem_per_sm << std::endl;
    // int max_shared_pc =
    //     100 - static_cast<int>(static_cast<double>(required_cache_storage) /
    //                            shmem_per_sm * 100);
    // if (max_shared_pc <= 0) {
    //     max_shared_pc = 1;
    // }
    // // std::cerr << " Max shared pc required = " << max_shared_pc <<
    // std::endl; GKO_ASSERT_NO_CUDA_ERRORS(cudaFuncSetAttribute(
    //     apply_kernel<StopType, 11, 1, PrecType, LogType, BatchMatrixType,
    //     ValueType>, cudaFuncAttributePreferredSharedMemoryCarveout,
    //     max_shared_pc - 1));
    // cudaFuncAttributes funcattr;
    // cudaFuncGetAttributes(
    //     &funcattr,
    //     apply_kernel<StopType, 11, 1, PrecType, LogType, BatchMatrixType,
    //     ValueType>);
    // std::cerr << " Max dyn. shared memory for batch bcgs = ",
    //        << funcattr.maxDynamicSharedSizeBytes << std::endl;
    // return funcattr.maxDynamicSharedSizeBytes;
    return shmem_per_sm;
}


template <typename T>
using BatchGmresOptions = gko::kernels::batch_gmres::BatchGmresOptions<T>;


template <typename CuValueType>
class KernelCaller {
public:
    using value_type = CuValueType;

    KernelCaller(std::shared_ptr<const CudaExecutor> exec,
                 const BatchGmresOptions<remove_complex<value_type>> opts)
        : exec_{exec}, opts_{opts}
    {}

    template <typename StopType, const int n_shared,
              const bool prec_shared_bool, typename PrecType, typename LogType,
              typename BatchMatrixType>
    void launch_apply_kernel(
        const gko::kernels::batch_gmres::StorageConfig& sconf, LogType& logger,
        PrecType& prec, const BatchMatrixType& a,
        const value_type* const __restrict__ b_values,
        value_type* const __restrict__ x_values,
        value_type* const __restrict__ workspace_data, const int& block_size,
        const size_t& shared_size) const
    {
        auto nrows = a.num_rows;

        apply_kernel<StopType, n_shared, prec_shared_bool>
            <<<a.num_batch, block_size, shared_size>>>(
                sconf, opts_.max_its, opts_.residual_tol, opts_.restart_num,
                logger, prec, a, b_values, x_values, workspace_data);
    }

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(LogType logger, const BatchMatrixType& a, PrecType prec,
                     const gko::batch_dense::UniformBatch<const value_type>& b,
                     const gko::batch_dense::UniformBatch<value_type>& x) const
    {
        using real_type = gko::remove_complex<value_type>;
        const size_type nbatch = a.num_batch;
        const auto restart = opts_.restart_num;
        constexpr int align_multiple = 8;
        const int shared_gap =
            ((a.num_rows - 1) / align_multiple + 1) * align_multiple;
        gko::kernels::cuda::configure_shared_memory_banks<value_type>();

        const int shmem_per_blk = 0;
        // get_max_dynamic_shared_memory<StopType, PrecType, LogType,
        //                               BatchMatrixType, value_type>(exec_,
        //                                                            0);

        const int block_size = 128;
        // get_num_threads_per_block<StopType, PrecType, LogType,
        //                           BatchMatrixType, value_type>(exec_,
        //                                                        a.num_rows);
        assert(block_size >= 2 * config::warp_size);

        const size_t prec_size =
            PrecType::dynamic_work_size(shared_gap, a.num_nnz);
        const size_t subspace_size = a.num_rows * (restart + 1);
        const size_t hess_size = restart * (restart + 2);
        const auto sconf =
            gko::kernels::batch_gmres::compute_shared_storage<PrecType,
                                                              value_type>(
                shmem_per_blk, shared_gap, a.num_nnz, b.num_rhs, restart);
        int num_main_vecs_shared = min(sconf.n_shared, 5);
        int num_rot_vecs_shared = min(sconf.n_shared - num_main_vecs_shared, 4);

        std::cout << "HERE  " << num_main_vecs_shared << " "
                  << num_rot_vecs_shared << std::endl;
        std::cout << "HERE  " << sconf.hess_shared << " "
                  << sconf.subspace_shared << " " << sconf.prec_shared
                  << std::endl;
        std::cout << sconf.gmem_stride_bytes << std::endl;

        const size_t shared_size =
            (num_main_vecs_shared * shared_gap +
             num_rot_vecs_shared * (restart + 1) +
             (sconf.prec_shared ? prec_size : 0) +
             (sconf.subspace_shared ? subspace_size : 0) +
             (sconf.hess_shared ? hess_size : 0)) *
            sizeof(value_type);
        auto workspace = gko::array<value_type>(
            exec_, sconf.gmem_stride_bytes * nbatch / sizeof(value_type));
        assert(sconf.gmem_stride_bytes % sizeof(value_type) == 0);

        value_type* const workspace_data = workspace.get_data();
        int n_shared = sconf.n_shared + int(sconf.hess_shared) +
                       int(sconf.subspace_shared);
        auto prec_shared_bool = sconf.prec_shared;

        // Template for calling launch_apply_kernel:
        // < StopType, n_shared, prec_shared_bool>
        if (prec_shared_bool) {
            switch (n_shared) {
            case 11:
                launch_apply_kernel<StopType, 11, 1>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 10:
                launch_apply_kernel<StopType, 10, 1>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 9:
                launch_apply_kernel<StopType, 9, 1>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            }
        } else {
            switch (n_shared) {
            case 0:
                launch_apply_kernel<StopType, 0, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 1:
                launch_apply_kernel<StopType, 1, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 2:
                launch_apply_kernel<StopType, 2, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 3:
                launch_apply_kernel<StopType, 3, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 4:
                launch_apply_kernel<StopType, 4, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 5:
                launch_apply_kernel<StopType, 5, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 6:
                launch_apply_kernel<StopType, 6, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 7:
                launch_apply_kernel<StopType, 7, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 8:
                launch_apply_kernel<StopType, 8, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 9:
                launch_apply_kernel<StopType, 9, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 10:
                launch_apply_kernel<StopType, 10, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            case 11:
                launch_apply_kernel<StopType, 11, 0>(
                    sconf, logger, prec, a, b.values, x.values, workspace_data,
                    block_size, shared_size);
                break;
            }
        }
        GKO_CUDA_LAST_IF_ERROR_THROW;
    }

private:
    std::shared_ptr<const CudaExecutor> exec_;
    const BatchGmresOptions<remove_complex<value_type>> opts_;
};


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const BatchGmresOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a, const BatchLinOp* const precon,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    using cu_value_type = cuda_type<ValueType>;
    auto dispatcher = batch_solver::create_dispatcher<ValueType>(
        KernelCaller<cu_value_type>(exec, opts), opts, a, precon);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_GMRES_APPLY_KERNEL);


}  // namespace batch_gmres
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
