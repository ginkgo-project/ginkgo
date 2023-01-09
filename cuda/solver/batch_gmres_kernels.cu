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

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(LogType logger, const BatchMatrixType& a, PrecType prec,
                     const gko::batch_dense::UniformBatch<const value_type>& b,
                     const gko::batch_dense::UniformBatch<value_type>& x) const
    {
        using real_type = gko::remove_complex<value_type>;
        const size_type nbatch = a.num_batch;
        const value_type* const bptr = b.values;
        value_type* const xptr = x.values;

        static_assert(default_block_size >= 2 * config::warp_size,
                      "Need at least two warps per block!");

        const auto nrhs = b.num_rhs;
        const auto nrows = a.num_rows;
        const auto restart = opts_.restart_num;
        const int global_gap =
            6 * nrows * nrhs + 3 * restart * nrhs + (restart + 1) * nrhs +
            restart * (restart + 1) * nrhs + nrows * (restart + 1) * nrhs;
        auto workspace = gko::array<value_type>(exec_);

        const int shared_size =
            gko::kernels::batch_gmres::local_memory_requirement<value_type>(
                a.num_rows, b.num_rhs, opts_.restart_num) +
            PrecType::dynamic_work_size(a.num_rows, a.num_nnz) *
                sizeof(value_type);
#if GKO_CUDA_BATCH_GMRES_HAVE_NO_SHMEM
        workspace = gko::array<value_type>(
            exec_,
            static_cast<size_type>(shared_size * nbatch / sizeof(value_type)));
        apply_kernel<StopType><<<nbatch, default_block_size>>>(
            global_gap, opts_.max_its, opts_.residual_tol, opts_.restart_num,
            logger, prec, a, bptr, xptr, workspace.get_data());
#else
        apply_kernel<StopType><<<nbatch, default_block_size, shared_size>>>(
            global_gap, opts_.max_its, opts_.residual_tol, opts_.restart_num,
            logger, prec, a, bptr, xptr);
#endif
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
