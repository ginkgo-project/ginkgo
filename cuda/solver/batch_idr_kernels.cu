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

#include "core/solver/batch_idr_kernels.hpp"


#include <random>


#include <curand_kernel.h>


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

#define GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM 1
#define GKO_DEVICE_RAND_LIB curand
constexpr int default_block_size = 128;
constexpr int sm_multiplier = 4;

/**
 * @brief The batch Idr solver namespace.
 *
 * @ingroup batch_idr
 */
namespace batch_idr {


#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
// include all depedencies (note: do not remove this comment)
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_ell_kernels.hpp.inc"
// TODO: remove batch dense include
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/solver/batch_idr_kernels.hpp.inc"


template <typename T>
using BatchIdrOptions = gko::kernels::batch_idr::BatchIdrOptions<T>;

template <typename CuValueType>
class KernelCaller {
public:
    using value_type = CuValueType;

    KernelCaller(std::shared_ptr<const CudaExecutor> exec,
                 const BatchIdrOptions<remove_complex<value_type>> opts,
                 const value_type* const subspace_vectors)
        : exec_{exec}, opts_{opts}, subspace_vectors_{subspace_vectors}
    {}

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(LogType logger, const BatchMatrixType& a, PrecType prec,
                     const gko::batch_dense::UniformBatch<const value_type>& b,
                     const gko::batch_dense::UniformBatch<value_type>& x) const
    {
        using real_type = gko::remove_complex<value_type>;
        const size_type nbatch = a.num_batch;

        static_assert(default_block_size >= 2 * config::warp_size,
                      "Need at least two warps per block!");

        const int shared_size =
            gko::kernels::batch_idr::local_memory_requirement<value_type>(
                a.num_rows, b.num_rhs, opts_.subspace_dim_val) +
            PrecType::dynamic_work_size(a.num_rows, a.num_nnz) *
                sizeof(value_type);
        apply_kernel<StopType><<<nbatch, default_block_size, shared_size>>>(
            opts_.max_its, opts_.residual_tol, opts_.subspace_dim_val,
            opts_.kappa_val, opts_.to_use_smoothing, opts_.deterministic_gen,
            logger, prec, subspace_vectors_, a, b.values, x.values);
        GKO_CUDA_LAST_IF_ERROR_THROW;
    }

private:
    std::shared_ptr<const CudaExecutor> exec_;
    const BatchIdrOptions<remove_complex<value_type>> opts_;
    const value_type* const subspace_vectors_;
};

namespace {

template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<!is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution&& dist, Generator&& gen)
{
    return dist(gen);
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution&& dist, Generator&& gen)
{
    return ValueType(dist(gen), dist(gen));
}

}  // unnamed namespace


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const BatchIdrOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a, const BatchLinOp* const prec,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    using cu_value_type = cuda_type<ValueType>;

    if (opts.is_complex_subspace == true && !is_complex<ValueType>()) {
        GKO_NOT_IMPLEMENTED;
    }
    const gko::batch_dense::UniformBatch<cu_value_type> x_b =
        get_batch_struct(x);
    array<ValueType> arr(exec->get_master());
    if (opts.deterministic_gen) {
        arr.resize_and_reset(x_b.num_rows * opts.subspace_dim_val);
        auto dist =
            std::normal_distribution<remove_complex<ValueType>>(0.0, 1.0);
        const auto seed = 15;
        auto gen = std::ranlux48(seed);
        // WARNING: The same ranlux48 object MUST be used for all entries of
        //  the array or the IDR does not work for complex problems!
        for (int vec_index = 0; vec_index < opts.subspace_dim_val;
             vec_index++) {
            for (int row_index = 0; row_index < x_b.num_rows; row_index++) {
                ValueType val = get_rand_value<ValueType>(dist, gen);
                arr.get_data()[vec_index * x_b.num_rows + row_index] = val;
            }
        }
        arr.set_executor(exec);
    }
    const cu_value_type* const subspace_vectors_entry =
        opts.deterministic_gen ? as_cuda_type(arr.get_const_data()) : nullptr;

    auto dispatcher = batch_solver::create_dispatcher<ValueType>(
        KernelCaller<cu_value_type>(exec, opts, subspace_vectors_entry), opts,
        a, prec);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDR_APPLY_KERNEL);


}  // namespace batch_idr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
