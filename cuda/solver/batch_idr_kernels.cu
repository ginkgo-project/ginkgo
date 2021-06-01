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

#include "core/solver/batch_idr_kernels.hpp"

#include <random>
#include "curand_kernel.h"

#include <ginkgo/core/base/math.hpp>


#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {

#define GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM 1
constexpr int default_block_size = 128;
constexpr int sm_multiplier = 4;

/**
 * @brief The batch Idr solver namespace.
 *
 * @ingroup batch_idr
 */
namespace batch_idr {

#include "common/components/uninitialized_array.hpp.inc"


#include "common/log/batch_logger.hpp.inc"
#include "common/matrix/batch_csr_kernels.hpp.inc"
#include "common/matrix/batch_dense_kernels.hpp.inc"
#include "common/preconditioner/batch_identity.hpp.inc"
#include "common/preconditioner/batch_jacobi.hpp.inc"
#include "common/stop/batch_criteria.hpp.inc"

#include "common/solver/batch_idr_kernels.hpp.inc"

template <typename T>
using BatchIdrOptions = gko::kernels::batch_idr::BatchIdrOptions<T>;

template <typename BatchMatrixType, typename LogType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const CudaExecutor> exec,
    const BatchIdrOptions<remove_complex<ValueType>> opts, LogType logger,
    const BatchMatrixType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &left,
    const gko::batch_dense::UniformBatch<const ValueType> &right,
    const gko::batch_dense::UniformBatch<ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x,
    const gko::batch_dense::BatchEntry<const ValueType> &Subspace_vectors_entry)
{
    using real_type = gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;

    if (opts.preconditioner == gko::preconditioner::batch::type::none) {
        const int shared_size =
#if GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM
            gko::kernels::batch_idr::local_memory_requirement<ValueType>(
                a.num_rows, b.num_rhs, opts.subspace_dim_val) +
            BatchIdentity<ValueType>::dynamic_work_size(a.num_rows, a.num_nnz) *
                sizeof(ValueType);
#else
            0;
#endif

        apply_kernel<stop::AbsAndRelResidualMaxIter<ValueType>>
            <<<nbatch, default_block_size, shared_size>>>(
                opts.max_its, opts.abs_residual_tol, opts.rel_residual_tol,
                opts.subspace_dim_val, opts.kappa_val, opts.to_use_smoothing,
                opts.deterministic_gen, opts.tol_type, logger,
                BatchIdentity<ValueType>(), a, left, right, b, x,
                Subspace_vectors_entry);
    } else if (opts.preconditioner ==
               gko::preconditioner::batch::type::jacobi) {
        const int shared_size =
#if GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM
            gko::kernels::batch_idr::local_memory_requirement<ValueType>(
                a.num_rows, b.num_rhs, opts.subspace_dim_val) +
            BatchJacobi<ValueType>::dynamic_work_size(a.num_rows, a.num_nnz) *
                sizeof(ValueType);
#else
            0;
#endif


        apply_kernel<stop::AbsAndRelResidualMaxIter<ValueType>>
            <<<nbatch, default_block_size, shared_size>>>(
                opts.max_its, opts.abs_residual_tol, opts.rel_residual_tol,
                opts.subspace_dim_val, opts.kappa_val, opts.to_use_smoothing,
                opts.deterministic_gen, opts.tol_type, logger,
                BatchJacobi<ValueType>(), a, left, right, b, x,
                Subspace_vectors_entry);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<!is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution &&dist, Generator &&gen)
{
    return dist(gen);
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution &&dist, Generator &&gen)
{
    return ValueType(dist(gen), dist(gen));
}


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const BatchIdrOptions<remove_complex<ValueType>> &opts,
           const BatchLinOp *const a,
           const matrix::BatchDense<ValueType> *const left_scale,
           const matrix::BatchDense<ValueType> *const right_scale,
           const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           log::BatchLogData<ValueType> &logdata)
{
    using cu_value_type = cuda_type<ValueType>;

    if (opts.is_complex_subspace == true &&
        !is_complex<ValueType>())
    {
        GKO_NOT_IMPLEMENTED;
    }

    // For now, FinalLogger is the only one available
    batch_log::FinalLogger<remove_complex<ValueType>> logger(
        static_cast<int>(b->get_size().at(0)[1]), opts.max_its,
        logdata.res_norms->get_values(), logdata.iter_counts.get_data());

    const gko::batch_dense::UniformBatch<const cu_value_type> left_sb =
        maybe_null_batch_struct(left_scale);
    const gko::batch_dense::UniformBatch<const cu_value_type> right_sb =
        maybe_null_batch_struct(right_scale);
    const auto to_scale = left_sb.values || right_sb.values;
    if (to_scale) {
        if (!left_sb.values || !right_sb.values) {
            // one-sided scaling not implemented
            GKO_NOT_IMPLEMENTED;
        }
    }


    const gko::batch_dense::UniformBatch<cu_value_type> x_b =
        get_batch_struct(x);


    gko::batch_dense::BatchEntry<const cu_value_type> Subspace_vectors_entry;

    // std::unique_ptr<gko::matrix::Dense<ValueType>> Subspace_vectors;

    if (opts.deterministic_gen == true) {
        // Note: Usage of dense matrix here leads to compilation errors. So, for
        // the time being, use Array as an alternative.
        /*

        std::unique_ptr<gko::matrix::Dense<ValueType>> Subspace_vectors_cpu =
            gko::matrix::Dense<ValueType>::create(
                exec->get_master(),
                gko::dim<2>{x_b.num_rows * opts.subspace_dim_val, 1}, 1);

        auto dist =
            std::normal_distribution<remove_complex<ValueType>>(0.0, 1.0);
        auto seed = 15;
        auto gen = std::ranlux48(seed);

        for (int vec_index = 0; vec_index < opts.subspace_dim_val;
             vec_index++) {
            for (int row_index = 0; row_index < x_b.num_rows; row_index++) {
                ValueType val = get_rand_value<ValueType>(dist, gen);

                Subspace_vectors_cpu->at(vec_index * x_b.num_rows + row_index,
                                         0) = val;
            }
        }


        Subspace_vectors = gko::clone(exec, Subspace_vectors_cpu);


        Subspace_vectors_entry = {
            as_cuda_type(Subspace_vectors->get_const_values()),
            Subspace_vectors->get_stride(),
            static_cast<int>(Subspace_vectors->get_size()[0]),
            static_cast<int>(Subspace_vectors->get_size()[1])};

        */


        Array<ValueType> arr(exec->get_master(),
                             x_b.num_rows * opts.subspace_dim_val);

        auto dist =
            std::normal_distribution<remove_complex<ValueType>>(0.0, 1.0);

        auto seed = 15;
        auto gen = std::ranlux48(seed);

        for (int vec_index = 0; vec_index < opts.subspace_dim_val;
             vec_index++) {
            for (int row_index = 0; row_index < x_b.num_rows; row_index++) {
                ValueType val = get_rand_value<ValueType>(dist, gen);

                arr.get_data()[vec_index * x_b.num_rows + row_index] = val;
            }
        }


        arr.set_executor(exec);

        Subspace_vectors_entry = {
            as_cuda_type(arr.get_const_data()), 1,
            static_cast<int>(x_b.num_rows * opts.subspace_dim_val), 1};


    } else {
        Subspace_vectors_entry = {nullptr, 0, 0, 0};
    }


    if (auto amat = dynamic_cast<const matrix::BatchCsr<ValueType> *>(a)) {
        const gko::batch_csr::UniformBatch<cu_value_type> m_b =
            get_batch_struct(const_cast<matrix::BatchCsr<ValueType> *>(amat));

        const gko::batch_dense::UniformBatch<cu_value_type> b_b =
            get_batch_struct(const_cast<matrix::BatchDense<ValueType> *>(b));

        apply_impl(exec, opts, logger, m_b, left_sb, right_sb, b_b, x_b,
                   Subspace_vectors_entry);
    } else {
        GKO_NOT_SUPPORTED(a);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDR_APPLY_KERNEL);


}  // namespace batch_idr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
