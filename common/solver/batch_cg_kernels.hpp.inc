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


namespace {


template <typename BatchMatrixType_entry, typename PrecType, typename ValueType>
__device__ __forceinline__ void initialize(
    const BatchMatrixType_entry &A_global_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &b_global_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_global_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_shared_entry,
    const PrecType &prec_shared,
    const gko::batch_dense::BatchEntry<ValueType> &z_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &p_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &rhs_norms_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &res_norms_shared_entry)
{
    // Compute norms of rhs
    compute_norm2<ValueType>(b_global_entry, rhs_norms_shared_entry);


    // copy x from global to shared memory
    for (int li = threadIdx.x;
         li < x_shared_entry.num_rows * x_shared_entry.num_rhs;
         li += blockDim.x) {
        int r = li / x_shared_entry.num_rhs;
        int c = li % x_shared_entry.num_rhs;

        x_shared_entry.values[r * x_shared_entry.stride + c] =
            x_global_entry.values[r * x_global_entry.stride + c];
    }


    // r = b
    for (int li = threadIdx.x;
         li < r_shared_entry.num_rows * r_shared_entry.num_rhs;
         li += blockDim.x) {
        int r = li / r_shared_entry.num_rhs;
        int c = li % r_shared_entry.num_rhs;

        r_shared_entry.values[r * r_shared_entry.stride + c] =
            b_global_entry.values[r * b_global_entry.stride + c];
    }

    __syncthreads();


    // r = b - A*x
    adv_spmv_ker(static_cast<ValueType>(-1.0), A_global_entry,
                 gko::batch::to_const(x_shared_entry),
                 static_cast<ValueType>(1.0), r_shared_entry);
    __syncthreads();

    compute_norm2<ValueType>(gko::batch::to_const(r_shared_entry),
                             res_norms_shared_entry);


    // z = precond * r
    prec_shared.apply(gko::batch::to_const(r_shared_entry), z_shared_entry);

    __syncthreads();


    // p = z
    for (int li = threadIdx.x;
         li < p_shared_entry.num_rows * p_shared_entry.num_rhs;
         li += blockDim.x) {
        int r = li / p_shared_entry.num_rhs;
        int c = li % p_shared_entry.num_rhs;

        p_shared_entry.values[r * p_shared_entry.stride + c] =
            z_shared_entry.values[r * z_shared_entry.stride + c];
    }


    // rho_old = r' * z
    compute_dot_product(gko::batch::to_const(r_shared_entry),
                        gko::batch::to_const(z_shared_entry),
                        rho_old_shared_entry);
}


template <typename ValueType>
__device__ __forceinline__ void compute_beta(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_new_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &beta_shared_entry,
    const uint32 &converged)
{
    for (int c = threadIdx.x; c < beta_shared_entry.num_rhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        beta_shared_entry.values[0 * beta_shared_entry.stride + c] =
            rho_new_shared_entry.values[0 * rho_new_shared_entry.stride + c] /
            rho_old_shared_entry.values[0 * rho_old_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_p(
    const gko::batch_dense::BatchEntry<const ValueType> &z_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &beta_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &p_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < p_shared_entry.num_rows * p_shared_entry.num_rhs;
         li += blockDim.x) {
        int r = li / p_shared_entry.num_rhs;
        int c = li % p_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        p_shared_entry.values[r * p_shared_entry.stride + c] =
            z_shared_entry.values[r * z_shared_entry.stride + c] +
            beta_shared_entry.values[0 * beta_shared_entry.stride + c] *
                p_shared_entry.values[r * p_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void compute_alpha(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &Ap_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_shared_entry,
    const uint32 &converged)
{
    const auto nrhs = rho_old_shared_entry.num_rhs;


    __shared__
        UninitializedArray<ValueType, batch_config<ValueType>::max_num_rhs>
            temp_sh;

    const gko::batch_dense::BatchEntry<ValueType> temp_shared_entry{
        temp_sh, static_cast<size_type>(nrhs), 1, nrhs};

    compute_dot_product<ValueType>(p_shared_entry, Ap_shared_entry,
                                   temp_shared_entry);
    __syncthreads();

    for (int c = threadIdx.x; c < alpha_shared_entry.num_rhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        alpha_shared_entry.values[c] =
            rho_old_shared_entry.values[c] / temp_shared_entry.values[c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x(
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < x_shared_entry.num_rows * x_shared_entry.num_rhs;
         li += blockDim.x) {
        int r = li / x_shared_entry.num_rhs;
        int c = li % x_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        x_shared_entry.values[r * x_shared_entry.stride + c] +=

            alpha_shared_entry.values[c] *
            p_shared_entry.values[r * p_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_r(
    const gko::batch_dense::BatchEntry<const ValueType> &Ap_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < r_shared_entry.num_rhs * r_shared_entry.num_rows;
         li += blockDim.x) {
        int r = li / r_shared_entry.num_rhs;
        int c = li % r_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        r_shared_entry.values[r * r_shared_entry.stride + c] -=
            alpha_shared_entry.values[0 * alpha_shared_entry.stride + c] *
            Ap_shared_entry.values[r * Ap_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void copy(
    const gko::batch_dense::BatchEntry<const ValueType> &src_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &dest_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < dest_shared_entry.num_rhs * dest_shared_entry.num_rows;
         li += blockDim.x) {
        int r = li / dest_shared_entry.num_rhs;
        int c = li % dest_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        dest_shared_entry.values[r * dest_shared_entry.stride + c] =
            src_shared_entry.values[r * src_shared_entry.stride + c];
    }
}


}  // namespace

// unnamed namespace


template <typename PrecType, typename StopType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(
    const int max_iter, const gko::remove_complex<ValueType> abs_tol,
    const gko::remove_complex<ValueType> rel_tol,
    const gko::stop::batch::ToleranceType tol_type, LogType logger,
    const BatchMatrixType a,
    const gko::batch_dense::UniformBatch<const ValueType> left,
    const gko::batch_dense::UniformBatch<const ValueType> right,
    const gko::batch_dense::UniformBatch<ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;

    // constexpr int max_nrhs = batch_config<ValueType>::max_num_rhs;

    if (batch_config<ValueType>::max_num_rows *
            batch_config<ValueType>::max_num_rhs <
        nrows * nrhs) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf(
                " Static vector size not enough!, allocated: %d , required: %d "
                "\n",
                batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs,
                nrows * nrhs);
        }
        return;
    }

    if (batch_config<ValueType>::max_num_rows < nrows) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf(
                " Compile time constant: max_num_rows < actual rows in "
                "matrix\n");
        }
        return;
    }

    if (batch_config<ValueType>::max_num_rhs < nrhs) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf(" Compile time constant: max_rhs < actual rhs in matrix!\n");
        }
        return;
    }


    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            r_sh;


        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            z_sh;

        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            p_sh;

        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            Ap_sh;


        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            x_sh;

        __shared__ UninitializedArray<ValueType, PrecType::work_size>
            prec_work_sh;


        uint32 converged = 0;

        const gko::batch_dense::BatchEntry<const ValueType> left_global_entry =
            gko::batch::batch_entry(left, ibatch);

        const gko::batch_dense::BatchEntry<const ValueType> right_global_entry =
            gko::batch::batch_entry(right, ibatch);


        if (left_global_entry.values) {
            const typename BatchMatrixType::entry_type A_global_entry =
                gko::batch::batch_entry(a, ibatch);
            const gko::batch_dense::BatchEntry<ValueType> b_global_entry =
                gko::batch::batch_entry(b, ibatch);
            batch_scale(left_global_entry, right_global_entry, A_global_entry);
            batch_scale(left_global_entry, b_global_entry);

            __syncthreads();
        }


        const auto A_global_entry =
            gko::batch::batch_entry(gko::batch::to_const(a), ibatch);

        const gko::batch_dense::BatchEntry<const ValueType> b_global_entry =
            gko::batch::batch_entry(gko::batch::to_const(b), ibatch);

        const gko::batch_dense::BatchEntry<ValueType> x_global_entry =
            gko::batch::batch_entry(x, ibatch);


        const gko::batch_dense::BatchEntry<ValueType> x_shared_entry{
            x_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> r_shared_entry{
            r_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> z_shared_entry{
            z_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> p_shared_entry{
            p_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> Ap_shared_entry{
            Ap_sh, static_cast<size_type>(nrhs), nrows, nrhs};


        __shared__
            UninitializedArray<ValueType, batch_config<ValueType>::max_num_rhs>
                rho_old_sh;
        const gko::batch_dense::BatchEntry<ValueType> rho_old_shared_entry{
            rho_old_sh, static_cast<size_type>(nrhs), 1, nrhs};


        __shared__
            UninitializedArray<ValueType, batch_config<ValueType>::max_num_rhs>
                rho_new_sh;
        const gko::batch_dense::BatchEntry<ValueType> rho_new_shared_entry{
            rho_new_sh, static_cast<size_type>(nrhs), 1, nrhs};


        __shared__
            UninitializedArray<ValueType, batch_config<ValueType>::max_num_rhs>
                alpha_sh;
        const gko::batch_dense::BatchEntry<ValueType> alpha_shared_entry{
            alpha_sh, static_cast<size_type>(nrhs), 1, nrhs};


        __shared__
            UninitializedArray<ValueType, batch_config<ValueType>::max_num_rhs>
                beta_sh;
        const gko::batch_dense::BatchEntry<ValueType> beta_shared_entry{
            beta_sh, static_cast<size_type>(nrhs), 1, nrhs};


        __shared__
            UninitializedArray<real_type, batch_config<ValueType>::max_num_rhs>
                norms_rhs_sh;
        const gko::batch_dense::BatchEntry<real_type> rhs_norms_shared_entry{
            norms_rhs_sh, static_cast<size_type>(nrhs), 1, nrhs};


        __shared__
            UninitializedArray<real_type, batch_config<ValueType>::max_num_rhs>
                norms_res_sh;
        const gko::batch_dense::BatchEntry<real_type> res_norms_shared_entry{
            norms_res_sh, static_cast<size_type>(nrhs), 1, nrhs};

        __shared__
            UninitializedArray<real_type, batch_config<ValueType>::max_num_rhs>
                norms_res_temp_sh;
        const gko::batch_dense::BatchEntry<real_type>
            res_norms_temp_shared_entry{norms_res_temp_sh,
                                        static_cast<size_type>(nrhs), 1, nrhs};


        // generate preconditioner
        PrecType prec_shared(A_global_entry, prec_work_sh);


        // initialization
        // compute b norms (precond b or what ?)
        // r = b - A*x
        // z = precond*r
        // compute residual norms (? precond res or what ?)
        // rho_old = r' * z (' is for hermitian transpose)
        // p = z
        initialize(A_global_entry, b_global_entry,
                   gko::batch::to_const(x_global_entry), x_shared_entry,
                   r_shared_entry, prec_shared, z_shared_entry,
                   rho_old_shared_entry, p_shared_entry, rhs_norms_shared_entry,
                   res_norms_shared_entry);


        __syncthreads();


        // stopping criterion object
        StopType stop(nrhs, max_iter, abs_tol, rel_tol,
                      static_cast<stop::tolerance>(tol_type), converged,
                      rhs_norms_shared_entry.values);

        int iter = -1;


        while (1) {
            ++iter;


            bool all_converged =
                stop.check_converged(iter, res_norms_shared_entry.values,
                                     {NULL, 0, 0, 0}, converged);


            logger.log_iteration(ibatch, iter, res_norms_shared_entry.values,
                                 converged);
            if (all_converged) {
                break;
            }

            // Ap = A * p
            spmv_ker(A_global_entry, gko::batch::to_const(p_shared_entry),
                     Ap_shared_entry);

            __syncthreads();

            // alpha = rho_old / (p' * Ap)
            compute_alpha(gko::batch::to_const(rho_old_shared_entry),
                          gko::batch::to_const(p_shared_entry),
                          gko::batch::to_const(Ap_shared_entry),
                          alpha_shared_entry, converged);

            __syncthreads();

            // x = x + alpha * p
            update_x(x_shared_entry, gko::batch::to_const(p_shared_entry),
                     gko::batch::to_const(alpha_shared_entry), converged);


            // r = r - alpha * Ap
            update_r(gko::batch::to_const(Ap_shared_entry),
                     gko::batch::to_const(alpha_shared_entry), r_shared_entry,
                     converged);

            __syncthreads();

            compute_norm2<ValueType>(
                gko::batch::to_const(r_shared_entry),
                res_norms_temp_shared_entry);  // residual norms
            __syncthreads();
            copy(gko::batch::to_const(res_norms_temp_shared_entry),
                 res_norms_shared_entry,
                 converged);  // update res norms only for those RHSs which have
                              // not yet converged.
            __syncthreads();

            // z = precond * r
            prec_shared.apply(gko::batch::to_const(r_shared_entry),
                              z_shared_entry);

            __syncthreads();

            // rho_new =  (r)' * (z)
            compute_dot_product<ValueType>(gko::batch::to_const(r_shared_entry),
                                           gko::batch::to_const(z_shared_entry),
                                           rho_new_shared_entry);

            __syncthreads();

            // beta = rho_new / rho_old
            compute_beta(gko::batch::to_const(rho_new_shared_entry),
                         gko::batch::to_const(rho_old_shared_entry),
                         beta_shared_entry, converged);

            __syncthreads();

            // p = z + beta * p
            update_p(gko::batch::to_const(z_shared_entry),
                     gko::batch::to_const(beta_shared_entry), p_shared_entry,
                     converged);

            __syncthreads();

            // rho_old = rho_new
            copy(gko::batch::to_const(rho_new_shared_entry),
                 rho_old_shared_entry, converged);

            __syncthreads();
        }

        if (left_global_entry.values) {
            batch_scale(right_global_entry, x_shared_entry);
        }

        __syncthreads();

        // copy x back to global memory
        for (int li = threadIdx.x;
             li < x_global_entry.num_rows * x_global_entry.num_rhs;
             li += blockDim.x) {
            int r = li / x_global_entry.num_rhs;
            int c = li % x_global_entry.num_rhs;

            x_global_entry.values[r * x_global_entry.stride + c] =
                x_shared_entry.values[r * x_shared_entry.stride + c];
        }

        __syncthreads();
    }
}
