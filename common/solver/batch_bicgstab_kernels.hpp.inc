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

template <typename BatchMatrixType_entry, typename ValueType>
__device__ __forceinline__ void initialize(
    const gko::batch_dense::BatchEntry<ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_old_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_shared_entry,
    const BatchMatrixType_entry &A_global_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &b_global_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_global_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_hat_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &p_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &v_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &rhs_norms_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &res_norms_shared_entry)
{
    for (int c = threadIdx.x; c < rho_old_shared_entry.num_rhs;
         c += blockDim.x) {
        rho_old_shared_entry.values[0 * rho_old_shared_entry.stride + c] =
            one<ValueType>();
        omega_old_shared_entry.values[0 * omega_old_shared_entry.stride + c] =
            one<ValueType>();
        alpha_shared_entry.values[0 * alpha_shared_entry.stride + c] =
            one<ValueType>();
    }


    // Compute norms of rhs
    batch_dense::compute_norm2<ValueType>(b_global_entry,
                                          rhs_norms_shared_entry);

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
    batch_adv_spmv_single(static_cast<ValueType>(-1.0), A_global_entry,
                          gko::batch_dense::to_const(x_shared_entry),
                          static_cast<ValueType>(1.0), r_shared_entry);
    __syncthreads();

    batch_dense::compute_norm2<ValueType>(
        gko::batch_dense::to_const(r_shared_entry), res_norms_shared_entry);


    for (int li = threadIdx.x;
         li < r_shared_entry.num_rows * r_shared_entry.num_rhs;
         li += blockDim.x) {
        int r = li / r_shared_entry.num_rhs;
        int c = li % r_shared_entry.num_rhs;

        r_hat_shared_entry.values[r * r_hat_shared_entry.stride + c] =
            r_shared_entry.values[r * r_shared_entry.stride + c];
        p_shared_entry.values[r * p_shared_entry.stride + c] =
            zero<ValueType>();
        v_shared_entry.values[r * v_shared_entry.stride + c] =
            zero<ValueType>();
    }
}


template <typename ValueType>
__device__ __forceinline__ void compute_beta(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_new_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_old_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &beta_shared_entry,
    const uint32 &converged)
{
    for (int c = threadIdx.x; c < beta_shared_entry.num_rhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        beta_shared_entry.values[0 * beta_shared_entry.stride + c] =
            (rho_new_shared_entry.values[0 * rho_new_shared_entry.stride + c] /
             rho_old_shared_entry.values[0 * rho_old_shared_entry.stride + c]) *
            (alpha_shared_entry.values[0 * alpha_shared_entry.stride + c] /
             omega_old_shared_entry
                 .values[0 * omega_old_shared_entry.stride + c]);
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_p(
    const gko::batch_dense::BatchEntry<const ValueType> &r_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &beta_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_old_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_shared_entry,
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
            r_shared_entry.values[r * r_shared_entry.stride + c] +
            beta_shared_entry.values[0 * beta_shared_entry.stride + c] *
                (p_shared_entry.values[r * p_shared_entry.stride + c] -
                 omega_old_shared_entry
                         .values[0 * omega_old_shared_entry.stride + c] *
                     v_shared_entry.values[r * v_shared_entry.stride + c]);
    }
}


template <typename ValueType>
__device__ __forceinline__ void compute_alpha(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_new_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_hat_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_shared_entry,
    const uint32 &converged)
{
    const auto nrhs = rho_new_shared_entry.num_rhs;


    __shared__
        UninitializedArray<ValueType, batch_config<ValueType>::max_num_rhs>
            temp_sh;

    const gko::batch_dense::BatchEntry<ValueType> temp_shared_entry{
        temp_sh, static_cast<size_type>(nrhs), 1, nrhs};

    batch_dense::compute_dot_product<ValueType>(
        r_hat_shared_entry, v_shared_entry, temp_shared_entry);
    __syncthreads();

    for (int c = threadIdx.x; c < alpha_shared_entry.num_rhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        alpha_shared_entry.values[c] =
            rho_new_shared_entry.values[c] / temp_shared_entry.values[c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_s(
    const gko::batch_dense::BatchEntry<const ValueType> &r_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &s_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < s_shared_entry.num_rows * s_shared_entry.num_rhs;
         li += blockDim.x) {
        int r = li / s_shared_entry.num_rhs;
        int c = li % s_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        s_shared_entry.values[r * s_shared_entry.stride + c] =
            r_shared_entry.values[r * r_shared_entry.stride + c] -
            alpha_shared_entry.values[0 * alpha_shared_entry.stride + c] *
                v_shared_entry.values[r * v_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void compute_omega_new(
    const gko::batch_dense::BatchEntry<const ValueType> &t_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_new_shared_entry,
    const uint32 &converged)
{
    const auto nrhs = omega_new_shared_entry.num_rhs;


    __shared__
        UninitializedArray<ValueType, batch_config<ValueType>::max_num_rhs>
            t_s_sh;
    const gko::batch_dense::BatchEntry<ValueType> t_s_shared_entry{
        t_s_sh, static_cast<size_type>(nrhs), 1, nrhs};


    __shared__
        UninitializedArray<ValueType, batch_config<ValueType>::max_num_rhs>
            t_t_sh;
    const gko::batch_dense::BatchEntry<ValueType> t_t_shared_entry{
        t_t_sh, static_cast<size_type>(nrhs), 1, nrhs};

    batch_dense::compute_dot_product<ValueType>(t_shared_entry, s_shared_entry,
                                                t_s_shared_entry);
    batch_dense::compute_dot_product<ValueType>(t_shared_entry, t_shared_entry,
                                                t_t_shared_entry);

    __syncthreads();

    for (int c = threadIdx.x; c < omega_new_shared_entry.num_rhs;
         c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        omega_new_shared_entry.values[c] =
            t_s_shared_entry.values[c] / t_t_shared_entry.values[c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x(
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_hat_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_hat_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_new_shared_entry,
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

        x_shared_entry.values[r * x_shared_entry.stride + c] =
            x_shared_entry.values[r * x_shared_entry.stride + c] +
            alpha_shared_entry.values[c] *
                p_hat_shared_entry.values[r * p_hat_shared_entry.stride + c] +
            omega_new_shared_entry.values[c] *
                s_hat_shared_entry.values[r * s_hat_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x_middle(
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_hat_shared_entry,
    const uint32 &converged_recent)
{
    for (int li = threadIdx.x;
         li < x_shared_entry.num_rows * x_shared_entry.num_rhs;
         li += blockDim.x) {
        int r = li / x_shared_entry.num_rhs;
        int c = li % x_shared_entry.num_rhs;

        const uint32 conv = converged_recent & (1 << c);

        if (conv) {
            x_shared_entry.values[r * x_shared_entry.stride + c] =
                x_shared_entry.values[r * x_shared_entry.stride + c] +
                alpha_shared_entry.values[c] *
                    p_hat_shared_entry
                        .values[r * p_hat_shared_entry.stride + c];
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_r(
    const gko::batch_dense::BatchEntry<const ValueType> &s_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_new_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &t_shared_entry,
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

        r_shared_entry.values[r * r_shared_entry.stride + c] =
            s_shared_entry.values[r * s_shared_entry.stride + c] -
            omega_new_shared_entry.values[c] *
                t_shared_entry.values[r * t_shared_entry.stride + c];
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


}  // unnamed namespace


template <typename PrecType, typename StopType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(
    const int max_iter, const gko::remove_complex<ValueType> abs_tol,
    const gko::remove_complex<ValueType> rel_tol,
    const gko::stop::batch::ToleranceType tol_type, LogType logger,
    const BatchMatrixType a,
    const gko::batch_dense::UniformBatch<const ValueType> b,
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
            r_hat_sh;

        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            p_sh;

        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            p_hat_sh;

        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            v_sh;

        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            s_sh;

        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            s_hat_sh;

        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            t_sh;

        __shared__ UninitializedArray<ValueType,
                                      batch_config<ValueType>::max_num_rows *
                                          batch_config<ValueType>::max_num_rhs>
            x_sh;

        __shared__ UninitializedArray<ValueType, PrecType::work_size>
            prec_work_sh;


        uint32 converged = 0;

        const typename BatchMatrixType::entry_type A_global_entry =
            gko::batch::batch_entry(a, ibatch);

        const gko::batch_dense::BatchEntry<const ValueType> b_global_entry =
            gko::batch::batch_entry(b, ibatch);

        const gko::batch_dense::BatchEntry<ValueType> x_global_entry =
            gko::batch::batch_entry(x, ibatch);


        const gko::batch_dense::BatchEntry<ValueType> x_shared_entry{
            x_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> r_shared_entry{
            r_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> r_hat_shared_entry{
            r_hat_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> p_shared_entry{
            p_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> p_hat_shared_entry{
            p_hat_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> v_shared_entry{
            v_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> s_shared_entry{
            s_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> s_hat_shared_entry{
            s_hat_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> t_shared_entry{
            t_sh, static_cast<size_type>(nrhs), nrows, nrhs};


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
                omega_old_sh;
        const gko::batch_dense::BatchEntry<ValueType> omega_old_shared_entry{
            omega_old_sh, static_cast<size_type>(nrhs), 1, nrhs};


        __shared__
            UninitializedArray<ValueType, batch_config<ValueType>::max_num_rhs>
                omega_new_sh;
        const gko::batch_dense::BatchEntry<ValueType> omega_new_shared_entry{
            omega_new_sh, static_cast<size_type>(nrhs), 1, nrhs};


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
        // rho_old = 1, omega_old = 1, alpha = 1
        // compute b norms
        // copy x from global to shared memory
        // r = b - A*x
        // compute residual norms
        // r_hat = r
        // p = 0
        // v = 0
        initialize(rho_old_shared_entry, omega_old_shared_entry,
                   alpha_shared_entry, A_global_entry, b_global_entry,
                   gko::batch_dense::to_const(x_global_entry), x_shared_entry,
                   r_shared_entry, r_hat_shared_entry, p_shared_entry,
                   v_shared_entry, rhs_norms_shared_entry,
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


            // rho_new =  < r_hat , r > = (r_hat)' * (r)
            batch_dense::compute_dot_product<ValueType>(
                gko::batch_dense::to_const(r_hat_shared_entry),
                gko::batch_dense::to_const(r_shared_entry),
                rho_new_shared_entry);


            __syncthreads();

            // beta = (rho_new / rho_old)*(alpha / omega_old)
            compute_beta(gko::batch_dense::to_const(rho_new_shared_entry),
                         gko::batch_dense::to_const(rho_old_shared_entry),
                         gko::batch_dense::to_const(alpha_shared_entry),
                         gko::batch_dense::to_const(omega_old_shared_entry),
                         beta_shared_entry, converged);

            __syncthreads();


            // p = r + beta*(p - omega_old * v)
            update_p(gko::batch_dense::to_const(r_shared_entry),
                     gko::batch_dense::to_const(beta_shared_entry),
                     gko::batch_dense::to_const(omega_old_shared_entry),
                     gko::batch_dense::to_const(v_shared_entry), p_shared_entry,
                     converged);

            __syncthreads();

            // p_hat = precond * p
            prec_shared.apply(gko::batch_dense::to_const(p_shared_entry),
                              p_hat_shared_entry);


            __syncthreads();

            // v = A * p_hat
            batch_spmv_single(A_global_entry,
                              gko::batch_dense::to_const(p_hat_shared_entry),
                              v_shared_entry);

            __syncthreads();

            // alpha = rho_new / < r_hat , v>
            compute_alpha(gko::batch_dense::to_const(rho_new_shared_entry),
                          gko::batch_dense::to_const(r_hat_shared_entry),
                          gko::batch_dense::to_const(v_shared_entry),
                          alpha_shared_entry, converged);

            __syncthreads();

            // s = r - alpha*v
            update_s(gko::batch_dense::to_const(r_shared_entry),
                     gko::batch_dense::to_const(alpha_shared_entry),
                     gko::batch_dense::to_const(v_shared_entry), s_shared_entry,
                     converged);

            __syncthreads();

            batch_dense::compute_norm2<ValueType>(
                gko::batch_dense::to_const(s_shared_entry),
                res_norms_temp_shared_entry);  // an estimate of residual norms
            __syncthreads();
            copy(gko::batch_dense::to_const(res_norms_temp_shared_entry),
                 res_norms_shared_entry,
                 converged);  // update res norms only for those RHSs which have
                              // not yet converged.

            __syncthreads();


            const uint32 converged_prev = converged;

            all_converged =
                stop.check_converged(iter, res_norms_shared_entry.values,
                                     {NULL, 0, 0, 0}, converged);

            // update x for the sytems (rhs) which converge at this point...  x
            // = x + alpha*p_hat

            // note bits could change from 0 to 1, not the other way round, so
            // we can use xor to get info about recent convergence...
            const uint32 converged_recent = converged_prev ^ converged;

            update_x_middle(x_shared_entry,
                            gko::batch_dense::to_const(alpha_shared_entry),
                            gko::batch_dense::to_const(p_hat_shared_entry),
                            converged_recent);

            __syncthreads();

            logger.log_iteration(ibatch, iter, res_norms_shared_entry.values,
                                 converged);

            if (all_converged) {
                break;
            }


            // s_hat = precond * s
            prec_shared.apply(gko::batch_dense::to_const(s_shared_entry),
                              s_hat_shared_entry);

            __syncthreads();


            // t = A * s_hat
            batch_spmv_single(A_global_entry,
                              gko::batch_dense::to_const(s_hat_shared_entry),
                              t_shared_entry);
            __syncthreads();

            // omega_new = <t,s> / <t,t>
            compute_omega_new(gko::batch_dense::to_const(t_shared_entry),
                              gko::batch_dense::to_const(s_shared_entry),
                              omega_new_shared_entry, converged);

            __syncthreads();

            // x = x + alpha*p_hat + omega_new*s_hat
            update_x(
                x_shared_entry, gko::batch_dense::to_const(p_hat_shared_entry),
                gko::batch_dense::to_const(s_hat_shared_entry),
                gko::batch_dense::to_const(alpha_shared_entry),
                gko::batch_dense::to_const(omega_new_shared_entry), converged);


            // r = s - omega_new*t
            update_r(gko::batch_dense::to_const(s_shared_entry),
                     gko::batch_dense::to_const(omega_new_shared_entry),
                     gko::batch_dense::to_const(t_shared_entry), r_shared_entry,
                     converged);

            __syncthreads();

            batch_dense::compute_norm2<ValueType>(
                gko::batch_dense::to_const(r_shared_entry),
                res_norms_temp_shared_entry);  // residual norms
            __syncthreads();
            copy(gko::batch_dense::to_const(res_norms_temp_shared_entry),
                 res_norms_shared_entry,
                 converged);  // update res norms only for those RHSs which have
                              // not yet converged.
            __syncthreads();

            // rho_old = rho_new
            // omega_old = omega_new
            copy(gko::batch_dense::to_const(rho_new_shared_entry),
                 rho_old_shared_entry, converged);
            copy(gko::batch_dense::to_const(omega_new_shared_entry),
                 omega_old_shared_entry, converged);

            __syncthreads();
        }


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
