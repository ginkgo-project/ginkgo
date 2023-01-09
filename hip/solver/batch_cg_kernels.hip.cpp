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

#include "core/solver/batch_cg_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "core/solver/batch_dispatch.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/exception.hip.hpp"
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
 * @brief The batch Cg solver namespace.
 *
 * @ingroup batch_cg
 */
namespace batch_cg {


#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
// include all depedencies (note: do not remove this comment)
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_ell_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/solver/batch_cg_kernels.hpp.inc"


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
    const int max_threads_regs = (max_regs_blk / num_regs_used_per_thread);
    const int max_threads = std::min(max_threads_regs, device_max_threads);
    return std::min(nwarps * static_cast<int>(config::warp_size), max_threads);
}


template <typename T>
using BatchCgOptions = gko::kernels::batch_cg::BatchCgOptions<T>;


template <typename DValueType>
class KernelCaller {
public:
    using value_type = DValueType;

    KernelCaller(std::shared_ptr<const HipExecutor> exec,
                 const BatchCgOptions<remove_complex<value_type>> opts)
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
        const int align_multiple = 8;
        const int shared_gap =
            ((a.num_rows - 1) / align_multiple + 1) * align_multiple;
        static_assert(default_block_size >= 2 * config::warp_size,
                      "Need at least two warps!");

        const int block_size =
            get_num_threads_per_block<BatchMatrixType>(exec_, a.num_rows);
        assert(block_size >= 2 * config::warp_size);
        const size_t prec_size =
            PrecType::dynamic_work_size(shared_gap, a.num_nnz) *
            sizeof(value_type);
        const int shmem_per_blk = exec_->get_max_shared_memory_per_block();
        // std::cerr << " Num shmem per block: " << shmem_per_blk << "\n";
        const auto sconf =
            gko::kernels::batch_cg::compute_shared_storage<PrecType,
                                                           value_type>(
                shmem_per_blk, shared_gap, a.num_nnz, b.num_rhs);
        const size_t shared_size =
            sconf.n_shared * shared_gap * sizeof(value_type) +
            (sconf.prec_shared ? prec_size : 0);
        auto workspace = gko::Array<value_type>(
            exec_, sconf.gmem_stride_bytes * nbatch / sizeof(value_type));
        assert(sconf.gmem_stride_bytes % sizeof(value_type) == 0);

        // std::cerr << " CG: vectors in shared memory = " << sconf.n_shared
        //           << " gmem stride bytes " << sconf.gmem_stride_bytes <<
        //           "\n";
        // if (sconf.prec_shared) {
        //     std::cerr << " CG: precondiioner is in shared memory.\n";
        // }
        // std::cerr << " CG: vectors in global memory = " << sconf.n_global
        //           << "\n Hip: number of threads per warp = "
        //           << config::warp_size
        //           << "\n CG: number of threads per block = " << block_size
        //           << "\n";
        hipLaunchKernelGGL(apply_kernel<StopType>, dim3(nbatch),
                           dim3(block_size), shared_size, 0, sconf,
                           opts_.max_its, opts_.residual_tol, logger, prec, a,
                           b.values, x.values, workspace.get_data());

        GKO_HIP_LAST_IF_ERROR_THROW;
    }

private:
    std::shared_ptr<const HipExecutor> exec_;
    const BatchCgOptions<remove_complex<value_type>> opts_;
};


template <typename ValueType>
void apply(std::shared_ptr<const HipExecutor> exec,
           const BatchCgOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a, const BatchLinOp* const precon,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    using d_value_type = hip_type<ValueType>;
    auto dispatcher = batch_solver::create_dispatcher<ValueType>(
        KernelCaller<d_value_type>(exec, opts), opts, a, precon);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG_APPLY_KERNEL);


}  // namespace batch_cg
}  // namespace hip
}  // namespace kernels
}  // namespace gko
