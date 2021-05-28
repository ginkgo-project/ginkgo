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

template <typename StopType, typename PrecType, typename LogType,
          typename MatrixType, typename ValueType>
__global__ void apply_kernel(
    const int max_iter, const gko::remove_complex<ValueType> rel_tol,
    const gko::remove_complex<ValueType> relax_factor, LogType logger,
    PrecType prec, const MatrixType mat,
    const gko::batch_dense::UniformBatch<const ValueType> left_scale,
    const gko::batch_dense::UniformBatch<const ValueType> right_scale,
    const gko::batch_dense::UniformBatch<ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> x)
{
    static_assert(
        std::is_same<typename MatrixType::value_type, ValueType>::value,
        "Inconsistent value type of matrix type!");
    static_assert(std::is_same<typename PrecType::value_type, ValueType>::value,
                  "Inconsistent value type of preconditioner!");
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = mat.num_batch;
    const auto num_rows = mat.num_rows;
    const auto num_rhs = b.num_rhs;

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
#if GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM
        extern __shared__ char local_mem[];
        const auto residual = reinterpret_cast<ValueType *>(local_mem);
        ValueType *const delta_x = residual + num_rows * num_rhs;
        ValueType *const prec_work = residual + 2 * num_rows * num_rhs;
        const auto norms = reinterpret_cast<real_type *>(
            prec_work + PrecType::dynamic_work_size(num_rows, mat.num_nnz));
        real_type *const init_res_norm = norms + num_rhs;
#else
        constexpr int max_nrhs = batch_config<ValueType>::max_num_rhs;
        constexpr int max_nrows = batch_config<ValueType>::max_num_rows;
        if (max_nrows * max_nrhs < num_rows * num_rhs) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                printf(" Static vector size not enough!\n");
            }
            return;
        }
        __shared__ ValueType residual[max_nrows * max_nrhs];
        __shared__ ValueType delta_x[max_nrows * max_nrhs];
        __shared__ ValueType prec_work[PrecType::work_size];
        __shared__ real_type norms[max_nrhs];
        __shared__ real_type init_res_norm[max_nrhs];
#endif

        uint32 converged = 0;

        const auto left_b = gko::batch::batch_entry(left_scale, ibatch);
        const auto right_b = gko::batch::batch_entry(right_scale, ibatch);

        if (left_scale.values) {
            const auto a_b = gko::batch::batch_entry(mat, ibatch);
            const auto b_b = gko::batch::batch_entry(b, ibatch);
            batch_scale(left_b, right_b, a_b);
            batch_scale(left_b, b_b);
        }

        const auto a_b =
            gko::batch::batch_entry(gko::batch::to_const(mat), ibatch);
        const auto b_b =
            gko::batch::batch_entry(gko::batch::to_const(b), ibatch);
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        const gko::batch_dense::BatchEntry<ValueType> r_b{
            residual, static_cast<size_type>(num_rhs), num_rows, num_rhs};
        const gko::batch_dense::BatchEntry<ValueType> dx_b{
            delta_x, static_cast<size_type>(num_rhs), num_rows, num_rhs};

        // These need to be duplicated across threads
        const auto relax = static_cast<ValueType>(relax_factor);
        const gko::batch_dense::BatchEntry<const ValueType> relax_b{&relax, 1,
                                                                    1, 1};

        const gko::batch_dense::BatchEntry<real_type> norms_b{
            norms, static_cast<size_type>(num_rhs), 1, num_rhs};
        const gko::batch_dense::BatchEntry<real_type> init_res_norm_b{
            init_res_norm, static_cast<size_type>(num_rhs), 1, num_rhs};

        prec.generate(a_b, prec_work);

        // initial residual
        copy(b_b, r_b);
        compute_norm2<ValueType>(gko::batch::to_const(r_b), norms_b);
        copy(gko::batch::to_const(norms_b), init_res_norm_b);

        StopType stop(num_rhs, max_iter, rel_tol, converged,
                      init_res_norm_b.values);

        int iter = 0;
        while (true) {
            // r <- r - Adx  This causes instability!
            // adv_spmv_ker(static_cast<ValueType>(-1.0), a_b,
            //     gko::batch_dense::to_const(dx_b),
            //     static_cast<ValueType>(1.0), r_b);

            // r <- b - Ax
            copy(b_b, r_b);
            __syncthreads();
            advanced_spmv_kernel(static_cast<ValueType>(-1.0), a_b,
                                 gko::batch::to_const(x_b),
                                 static_cast<ValueType>(1.0), r_b);
            __syncthreads();

            compute_norm2<ValueType>(gko::batch::to_const(r_b), norms_b);
            __syncthreads();

            const bool all_converged =
                stop.check_converged(iter, norms, {NULL, 0, 0, 0}, converged);
            logger.log_iteration(ibatch, iter, norms, converged);
            if (all_converged) {
                break;
            }

            prec.apply(gko::batch::to_const(r_b), dx_b);
            __syncthreads();

            // zero out dx for rhs's which do not need to be updated,
            //  though this is unnecessary for this solver.
            for (int j = 0; j < num_rhs; j++) {
                const uint32 conv = converged & (1 << j);
                if (conv) {
                    for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
                        dx_b.values[i * dx_b.stride + j] = 0.0;
                    }
                }
            }
            __syncthreads();

            add_scaled(relax_b, gko::batch::to_const(dx_b), x_b);
            __syncthreads();
            iter++;
        }

        if (left_scale.values) {
            batch_scale(right_b, x_b);
        }
    }
}
