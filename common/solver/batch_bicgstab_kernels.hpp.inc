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


template <typename ValueType>
__device__ __forceinline__ void copy(
    const gko::batch_dense::BatchEntry<const ValueType> &src_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &dest_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < dest_shared_entry.num_rhs * dest_shared_entry.num_rows;
         li += blockDim.x) {
        const int r = li / dest_shared_entry.num_rhs;
        const int c = li % dest_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        dest_shared_entry.values[r * dest_shared_entry.stride + c] =
            src_shared_entry.values[r * src_shared_entry.stride + c];
    }
}


template <typename BatchMatrixType_entry, typename ValueType>
__device__ __forceinline__ void initialize(
    const BatchMatrixType_entry &A_global_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &b_global_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_global_entry,

    const gko::batch_dense::BatchEntry<ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_shared_entry,

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
        rho_old_shared_entry.values[c] = one<ValueType>();
        omega_shared_entry.values[c] = one<ValueType>();
        alpha_shared_entry.values[c] = one<ValueType>();
    }


    // Compute norms of rhs
    compute_norm2<ValueType>(b_global_entry, rhs_norms_shared_entry);

    // copy x from global to shared memory
    copy(x_global_entry, x_shared_entry,
         0x00000000);  // TODO: Use simple copy function here to avoid uint32
                       // dependency or use static_cast<bitset_type>(0)


    // r = b
    copy(b_global_entry, r_shared_entry, 0x00000000);
    // TODO: Use simple copy function here to avoid uint32
    // dependency or use static_cast<bitset_type>(0)
    __syncthreads();

    // r = b - A*x
    advanced_spmv_kernel(static_cast<ValueType>(-1.0), A_global_entry,
                         gko::batch::to_const(x_shared_entry),
                         static_cast<ValueType>(1.0), r_shared_entry);
    __syncthreads();

    compute_norm2<ValueType>(gko::batch::to_const(r_shared_entry),
                             res_norms_shared_entry);

    copy(gko::batch::to_const(r_shared_entry), r_hat_shared_entry,
         0x00000000);  // TODO: Use simple copy function here to avoid uint32
                       // dependency or use static_cast<bitset_type>(0)

    for (int li = threadIdx.x;
         li < p_shared_entry.num_rows * p_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / p_shared_entry.num_rhs;
        const int c = li % p_shared_entry.num_rhs;

        p_shared_entry.values[r * p_shared_entry.stride + c] =
            zero<ValueType>();
        v_shared_entry.values[r * v_shared_entry.stride + c] =
            zero<ValueType>();
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_p(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_new_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &p_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < p_shared_entry.num_rows * p_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / p_shared_entry.num_rhs;
        const int c = li % p_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        const ValueType beta =
            (rho_new_shared_entry.values[c] / rho_old_shared_entry.values[c]) *
            (alpha_shared_entry.values[c] / omega_shared_entry.values[c]);

        p_shared_entry.values[r * p_shared_entry.stride + c] =
            r_shared_entry.values[r * r_shared_entry.stride + c] +
            beta * (p_shared_entry.values[r * p_shared_entry.stride + c] -
                    omega_shared_entry.values[c] *
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
    compute_dot_product<ValueType>(r_hat_shared_entry, v_shared_entry,
                                   alpha_shared_entry);
    __syncthreads();

    for (int c = threadIdx.x; c < alpha_shared_entry.num_rhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        alpha_shared_entry.values[c] =
            rho_new_shared_entry.values[c] / alpha_shared_entry.values[c];
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
        const int r = li / s_shared_entry.num_rhs;
        const int c = li % s_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        s_shared_entry.values[r * s_shared_entry.stride + c] =
            r_shared_entry.values[r * r_shared_entry.stride + c] -
            alpha_shared_entry.values[c] *
                v_shared_entry.values[r * v_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void compute_omega(
    const gko::batch_dense::BatchEntry<const ValueType> &t_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &temp_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_shared_entry,
    const uint32 &converged)
{
    compute_dot_product<ValueType>(t_shared_entry, s_shared_entry,
                                   omega_shared_entry);
    compute_dot_product<ValueType>(t_shared_entry, t_shared_entry,
                                   temp_shared_entry);

    __syncthreads();

    for (int c = threadIdx.x; c < omega_shared_entry.num_rhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        omega_shared_entry.values[c] /= temp_shared_entry.values[c];
    }
}

template <typename ValueType>
__device__ __forceinline__ void update_x_and_r(
    const gko::batch_dense::BatchEntry<const ValueType> &p_hat_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_hat_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &t_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < x_shared_entry.num_rows * x_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / x_shared_entry.num_rhs;
        const int c = li % x_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        const ValueType omega = omega_shared_entry.values[c];

        x_shared_entry.values[r * x_shared_entry.stride + c] =
            x_shared_entry.values[r * x_shared_entry.stride + c] +
            alpha_shared_entry.values[c] *
                p_hat_shared_entry.values[r * p_hat_shared_entry.stride + c] +
            omega *
                s_hat_shared_entry.values[r * s_hat_shared_entry.stride + c];

        r_shared_entry.values[r * r_shared_entry.stride + c] =
            s_shared_entry.values[r * s_shared_entry.stride + c] -
            omega * t_shared_entry.values[r * t_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x_middle(

    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_hat_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const uint32 &converged_recent)
{
    for (int li = threadIdx.x;
         li < x_shared_entry.num_rows * x_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / x_shared_entry.num_rhs;
        const int c = li % x_shared_entry.num_rhs;

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


}  // namespace

// unnamed namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(
    const int max_iter, const gko::remove_complex<ValueType> tol,
    LogType logger, PrecType prec_shared, const BatchMatrixType a,
    const gko::batch_dense::UniformBatch<const ValueType> left,
    const gko::batch_dense::UniformBatch<const ValueType> right,
    const gko::batch_dense::UniformBatch<ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;

    assert(batch_config<ValueType>::max_num_rhs >=
           nrhs);  // required for static allocation in stopping criterion

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
        extern __shared__ char local_mem_sh[];
        ValueType *const r_sh = reinterpret_cast<ValueType *>(local_mem_sh);
        ValueType *const r_hat_sh = r_sh + nrows * nrhs;
        ValueType *const p_sh = r_hat_sh + nrows * nrhs;
        ValueType *const p_hat_sh = p_sh + nrows * nrhs;
        ValueType *const v_sh = p_hat_sh + nrows * nrhs;
        ValueType *const s_sh = v_sh + nrows * nrhs;
        ValueType *const s_hat_sh = s_sh + nrows * nrhs;
        ValueType *const t_sh = s_hat_sh + nrows * nrhs;
        ValueType *const x_sh = t_sh + nrows * nrhs;
        ValueType *const prec_work_sh = x_sh + nrows * nrhs;
        ValueType *const rho_old_sh =
            prec_work_sh + PrecType::dynamic_work_size(nrows, a.num_nnz);
        ValueType *const rho_new_sh = rho_old_sh + nrhs;
        ValueType *const omega_sh = rho_new_sh + nrhs;
        ValueType *const alpha_sh = omega_sh + nrhs;
        ValueType *const temp_sh = alpha_sh + nrhs;
        real_type *const norms_rhs_sh =
            reinterpret_cast<real_type *>(temp_sh + nrhs);
        real_type *const norms_res_sh = norms_rhs_sh + nrhs;
        real_type *const norms_res_temp_sh = norms_res_sh + nrhs;


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

        const gko::batch_dense::BatchEntry<ValueType> rho_old_shared_entry{
            rho_old_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> rho_new_shared_entry{
            rho_new_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> omega_shared_entry{
            omega_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> alpha_shared_entry{
            alpha_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> temp_shared_entry{
            temp_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<real_type> rhs_norms_shared_entry{
            norms_rhs_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<real_type> res_norms_shared_entry{
            norms_res_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<real_type>
            res_norms_temp_shared_entry{norms_res_temp_sh,
                                        static_cast<size_type>(nrhs), 1, nrhs};

        // generate preconditioner
        prec_shared.generate(A_global_entry, prec_work_sh);


        // initialization
        // rho_old = 1, omega = 1, alpha = 1
        // compute b norms
        // copy x from global to shared memory
        // r = b - A*x
        // compute residual norms
        // r_hat = r
        // p = 0
        // v = 0
        initialize(A_global_entry, b_global_entry,
                   gko::batch::to_const(x_global_entry), rho_old_shared_entry,
                   omega_shared_entry, alpha_shared_entry, x_shared_entry,
                   r_shared_entry, r_hat_shared_entry, p_shared_entry,
                   v_shared_entry, rhs_norms_shared_entry,
                   res_norms_shared_entry);


        __syncthreads();


        // stopping criterion object
        StopType stop(nrhs, max_iter, tol, rhs_norms_shared_entry.values,
                      converged);

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
            compute_dot_product<ValueType>(
                gko::batch::to_const(r_hat_shared_entry),
                gko::batch::to_const(r_shared_entry), rho_new_shared_entry);


            __syncthreads();

            // beta = (rho_new / rho_old)*(alpha / omega)
            // p = r + beta*(p - omega * v)
            update_p(gko::batch::to_const(rho_new_shared_entry),
                     gko::batch::to_const(rho_old_shared_entry),
                     gko::batch::to_const(alpha_shared_entry),
                     gko::batch::to_const(omega_shared_entry),
                     gko::batch::to_const(r_shared_entry),
                     gko::batch::to_const(v_shared_entry), p_shared_entry,
                     converged);

            __syncthreads();

            // p_hat = precond * p
            prec_shared.apply(gko::batch::to_const(p_shared_entry),
                              p_hat_shared_entry);


            __syncthreads();

            // v = A * p_hat
            spmv_kernel(A_global_entry,
                        gko::batch::to_const(p_hat_shared_entry),
                        v_shared_entry);

            __syncthreads();

            // alpha = rho_new / < r_hat , v>
            compute_alpha(gko::batch::to_const(rho_new_shared_entry),
                          gko::batch::to_const(r_hat_shared_entry),
                          gko::batch::to_const(v_shared_entry),
                          alpha_shared_entry, converged);

            __syncthreads();

            // s = r - alpha*v
            update_s(gko::batch::to_const(r_shared_entry),
                     gko::batch::to_const(alpha_shared_entry),
                     gko::batch::to_const(v_shared_entry), s_shared_entry,
                     converged);

            __syncthreads();


            compute_norm2<ValueType>(
                gko::batch::to_const(s_shared_entry),
                res_norms_temp_shared_entry);  // an estimate of residual norms

            __syncthreads();


            copy(gko::batch::to_const(res_norms_temp_shared_entry),
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

            update_x_middle(gko::batch::to_const(alpha_shared_entry),
                            gko::batch::to_const(p_hat_shared_entry),
                            x_shared_entry, converged_recent);

            __syncthreads();

            logger.log_iteration(ibatch, iter, res_norms_shared_entry.values,
                                 converged);

            if (all_converged) {
                break;
            }


            // s_hat = precond * s
            prec_shared.apply(gko::batch::to_const(s_shared_entry),
                              s_hat_shared_entry);

            __syncthreads();


            // t = A * s_hat
            spmv_kernel(A_global_entry,
                        gko::batch::to_const(s_hat_shared_entry),
                        t_shared_entry);
            __syncthreads();

            // omega = <t,s> / <t,t>
            compute_omega(gko::batch::to_const(t_shared_entry),
                          gko::batch::to_const(s_shared_entry),
                          temp_shared_entry, omega_shared_entry, converged);

            __syncthreads();


            // x = x + alpha*p_hat + omega *s_hat
            // r = s - omega * t
            update_x_and_r(gko::batch::to_const(p_hat_shared_entry),
                           gko::batch::to_const(s_hat_shared_entry),
                           gko::batch::to_const(alpha_shared_entry),
                           gko::batch::to_const(omega_shared_entry),
                           gko::batch::to_const(s_shared_entry),
                           gko::batch::to_const(t_shared_entry), x_shared_entry,
                           r_shared_entry, converged);


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
        copy(gko::batch::to_const(x_shared_entry), x_global_entry);

        __syncthreads();
    }
}