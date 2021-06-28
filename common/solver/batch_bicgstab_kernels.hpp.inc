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
    const int num_rows, const BatchMatrixType_entry &A_global_entry,
    const ValueType *const b_global_entry,
    const ValueType *const x_global_entry, ValueType &rho_old, ValueType &omega,
    ValueType &alpha, ValueType *const x_shared_entry,
    ValueType *const r_shared_entry, ValueType *const r_hat_shared_entry,
    ValueType *const p_shared_entry, ValueType *const v_shared_entry,
    typename gko::remove_complex<ValueType> &rhs_norm,
    typename gko::remove_complex<ValueType> &res_norm)
{
    // constexpr int c = 0;
    // for (int c = threadIdx.x; c < rho_old_shared_entry.num_rhs;
    //     c += blockDim.x) {
    // rho_old_shared_entry.values[c] = one<ValueType>();
    // omega_shared_entry.values[c] = one<ValueType>();
    // alpha_shared_entry.values[c] = one<ValueType>();
    //}
    rho_old = one<ValueType>();
    omega = one<ValueType>();
    alpha = one<ValueType>();

    // Compute norms of rhs
    compute_norm2<ValueType>(num_rows, b_global_entry, rhs_norm);

    // copy x from global to shared memory
    // copy(num_rows, x_global_entry, x_shared_entry);
    // r = b
    // copy(num_rows, b_global_entry.values, r_shared_entry.values);
    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        x_shared_entry[iz] = x_global_entry[iz];
        r_shared_entry[iz] = b_global_entry[iz];
    }

    __syncthreads();

    // r = b - A*x
    single_advanced_spmv_kernel(static_cast<ValueType>(-1.0), A_global_entry,
                                x_shared_entry, static_cast<ValueType>(1.0),
                                r_shared_entry);
    __syncthreads();

    compute_norm2<ValueType>(num_rows, r_shared_entry, res_norm);

    // copy(r_shared_entry.num_rows, r_shared_entry.values,
    //     r_hat_shared_entry.values);

    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        r_hat_shared_entry[iz] = r_shared_entry[iz];
        p_shared_entry[iz] = zero<ValueType>();
        v_shared_entry[iz] = zero<ValueType>();
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_p(
    const int num_rows, const ValueType &rho_new, const ValueType &rho_old,
    const ValueType &alpha, const ValueType &omega,
    const ValueType *const r_shared_entry,
    const ValueType *const v_shared_entry,
    ValueType *const p_shared_entry /*,const uint32 &converged*/)
{
    // constexpr int c = 0;
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        // const int r = li / p_shared_entry.num_rhs;
        // const int c = li % p_shared_entry.num_rhs;

        // const uint32 conv = converged & (1 << c);

        // if (conv) {
        //    continue;
        //}

        const ValueType beta = (rho_new / rho_old) * (alpha / omega);

        // p_shared_entry.values[r * p_shared_entry.stride + c] =
        //    r_shared_entry.values[r * r_shared_entry.stride + c] +
        //    beta * (p_shared_entry.values[r * p_shared_entry.stride + c] -
        //            omega_shared_entry.values[c] *
        //                v_shared_entry.values[r * v_shared_entry.stride + c]);
        p_shared_entry[r] =
            r_shared_entry[r] +
            beta * (p_shared_entry[r] - omega * v_shared_entry[r]);
    }
}

template <typename ValueType>
__device__ __forceinline__ void compute_alpha(
    const int num_rows, const ValueType &rho_new,
    const ValueType *const r_hat_shared_entry,
    const ValueType *const v_shared_entry,
    ValueType &alpha /*,const uint32 &converged*/)
{
    // constexpr int c = 0;
    compute_dot_product<ValueType>(num_rows, r_hat_shared_entry, v_shared_entry,
                                   alpha);
    __syncthreads();

    // for (int c = threadIdx.x; c < alpha_shared_entry.num_rhs; c +=
    // blockDim.x) {
    //    //const uint32 conv = converged & (1 << c);

    //    //if (conv) {
    //    //    continue;
    //    //}
    //    alpha_shared_entry.values[c] =
    //        rho_new_shared_entry.values[c] / alpha_shared_entry.values[c];
    //}
    if (threadIdx.x == 0) {
        alpha = rho_new / alpha;
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_s(
    const int num_rows, const ValueType *const r_shared_entry,
    const ValueType &alpha, const ValueType *const v_shared_entry,
    ValueType *const s_shared_entry /*, const uint32 &converged*/)
{
    // constexpr int c = 0;
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        // const int r = li / s_shared_entry.num_rhs;
        // const int c = li % s_shared_entry.num_rhs;

        // const uint32 conv = converged & (1 << c);

        // if (conv) {
        //    continue;
        //}

        // s_shared_entry.values[r * s_shared_entry.stride + c] =
        //    r_shared_entry.values[r * r_shared_entry.stride + c] -
        //    alpha_shared_entry.values[c] *
        //        v_shared_entry.values[r * v_shared_entry.stride + c];
        s_shared_entry[r] = r_shared_entry[r] - alpha * v_shared_entry[r];
    }
}


template <typename ValueType>
__device__ __forceinline__ void compute_omega(
    const int num_rows, const ValueType *const t_shared_entry,
    const ValueType *const s_shared_entry, ValueType &temp,
    ValueType &omega /*, const uint32 &converged*/)
{
    compute_dot_product<ValueType>(num_rows, t_shared_entry, s_shared_entry,
                                   omega);
    compute_dot_product<ValueType>(num_rows, t_shared_entry, t_shared_entry,
                                   temp);

    __syncthreads();

    // for (int c = threadIdx.x; c < omega_shared_entry.num_rhs; c +=
    // blockDim.x) {
    //    //const uint32 conv = converged & (1 << c);

    //    //if (conv) {
    //    //    continue;
    //    //}
    //    omega_shared_entry.values[c] /= temp_shared_entry.values[c];
    //}
    if (threadIdx.x == 0) {
        omega /= temp;
    }
}

template <typename ValueType>
__device__ __forceinline__ void update_x_and_r(
    const int num_rows, const ValueType *const p_hat_shared_entry,
    const ValueType *const s_hat_shared_entry, const ValueType &alpha,
    const ValueType &omega, const ValueType *const s_shared_entry,
    const ValueType *const t_shared_entry, ValueType *const x_shared_entry,
    ValueType *const r_shared_entry /*, const uint32 &converged*/)
{
    // constexpr int c = 0;
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        // const int r = li / x_shared_entry.num_rhs;
        // const int c = li % x_shared_entry.num_rhs;

        // const uint32 conv = converged & (1 << c);

        // if (conv) {
        //    continue;
        //}

        // const ValueType omega = omega_shared_entry.values[c];

        // x_shared_entry.values[r * x_shared_entry.stride + c] =
        //    x_shared_entry.values[r * x_shared_entry.stride + c] +
        //    alpha_shared_entry.values[c] *
        //        p_hat_shared_entry.values[r * p_hat_shared_entry.stride + c] +
        //    omega *
        //        s_hat_shared_entry.values[r * s_hat_shared_entry.stride + c];

        // r_shared_entry.values[r * r_shared_entry.stride + c] =
        //    s_shared_entry.values[r * s_shared_entry.stride + c] -
        //    omega * t_shared_entry.values[r * t_shared_entry.stride + c];
        x_shared_entry[r] = x_shared_entry[r] + alpha * p_hat_shared_entry[r] +
                            omega * s_hat_shared_entry[r];

        r_shared_entry[r] = s_shared_entry[r] - omega * t_shared_entry[r];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x_middle(
    const int num_rows, const ValueType &alpha,
    const ValueType *const p_hat_shared_entry, ValueType *const x_shared_entry
    /*const uint32 &converged_recent*/)
{
    // constexpr int c = 0;
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        // const int r = li / x_shared_entry.num_rhs;
        // const int c = li % x_shared_entry.num_rhs;

        // const uint32 conv = converged_recent & (1 << c);

        // if (conv) {
        //    x_shared_entry.values[r * x_shared_entry.stride + c] =
        //        x_shared_entry.values[r * x_shared_entry.stride + c] +
        //        alpha_shared_entry.values[c] *
        //            p_hat_shared_entry
        //                .values[r * p_hat_shared_entry.stride + c];
        //}
        x_shared_entry[r] = x_shared_entry[r] + alpha * p_hat_shared_entry[r];
    }
}


}  // namespace

// unnamed namespace


template <typename StopType, typename PrecType, typename LogType,
          /*typename BatchMatrixType,*/ typename ValueType>
__global__ void apply_kernel(const int max_iter,
                             const gko::remove_complex<ValueType> tol,
                             LogType logger, PrecType prec_shared,
                             const size_type nbatch, const int num_rows,
                             const int shared_gap, const int num_nnz,
                             const ValueType *const __restrict__ avalues,
                             const int *const __restrict__ col_idxs,
                             const int *const __restrict__ row_ptrs,
                             const size_type stride, const int nrhs,
                             const ValueType *const __restrict__ b,
                             ValueType *const __restrict__ x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    // const auto nbatch = a.num_batch;
    // const auto nrows = a.num_rows;
    // const auto nrhs = b.num_rhs;

    const auto nrows = num_rows;

    // required for static allocation in stopping criterion
    assert(batch_config<ValueType>::max_num_rhs >= nrhs);

    const gko::batch_csr::UniformBatch<const ValueType> a{
        avalues, col_idxs, row_ptrs, nbatch, num_rows, num_nnz};

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
#ifdef GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM
        extern __shared__ char local_mem_sh[];
        ValueType *const r_sh = reinterpret_cast<ValueType *>(local_mem_sh);
        ValueType *const r_hat_sh = r_sh + shared_gap;
        ValueType *const p_sh = r_hat_sh + shared_gap;
        ValueType *const p_hat_sh = p_sh + shared_gap;
        ValueType *const v_sh = p_hat_sh + shared_gap;
        ValueType *const s_sh = v_sh + shared_gap;
        ValueType *const s_hat_sh = s_sh + shared_gap;
        ValueType *const t_sh = s_hat_sh + shared_gap;
        ValueType *const x_sh = t_sh + shared_gap;
        ValueType *const prec_work_sh = x_sh + shared_gap;

        // ValueType *const rho_old_sh =
        //    prec_work_sh + PrecType::dynamic_work_size(nrows, a.num_nnz);
        // ValueType *const rho_new_sh = rho_old_sh + nrhs;
        // ValueType *const omega_sh = rho_new_sh + nrhs;
        // ValueType *const alpha_sh = omega_sh + nrhs;
        // ValueType *const temp_sh = alpha_sh + nrhs;
        // real_type *const norms_rhs_sh =
        //    reinterpret_cast<real_type *>(temp_sh + nrhs);
        // real_type *const norms_res_sh = norms_rhs_sh + nrhs;
#else
        __shared__ ValueType r_sh[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType r_hat_sh[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType p_sh[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType p_hat_sh[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType s_sh[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType s_hat_sh[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType v_sh[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType t_sh[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType x_sh[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType
            prec_work_sh[batch_config<ValueType>::max_num_rows];
#endif
        __shared__ ValueType rho_old_sh[1];
        __shared__ ValueType rho_new_sh[1];
        __shared__ ValueType omega_sh[1];
        __shared__ ValueType alpha_sh[1];
        __shared__ ValueType temp_sh[1];
        __shared__ real_type norms_rhs_sh[1];
        __shared__ real_type norms_res_sh[1];

        // uint32 converged = 0;

        const auto A_global_entry =
            gko::batch::batch_entry(gko::batch::to_const(a), ibatch);

        // const gko::batch_dense::BatchEntry<const ValueType> b_global_entry =
        //    gko::batch::batch_entry(static_cast<const ValueType *>(b), stride,
        //                            nrows, nrhs, ibatch);

        const ValueType *const b_entry_ptr =
            gko::batch::batch_entry_ptr(b, 1, nrows, ibatch);
        ValueType *const x_gl_entry_ptr =
            gko::batch::batch_entry_ptr(x, 1, nrows, ibatch);

        const gko::batch_dense::BatchEntry<ValueType> x_global_entry =
            gko::batch::batch_entry(x, stride, nrows, nrhs, ibatch);

        // const gko::batch_dense::BatchEntry<ValueType> x_shared_entry{
        //    x_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> r_shared_entry{
        //    r_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> r_hat_shared_entry{
        //    r_hat_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> p_shared_entry{
        //    p_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> p_hat_shared_entry{
        //    p_hat_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> v_shared_entry{
        //    v_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> s_shared_entry{
        //    s_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> s_hat_shared_entry{
        //    s_hat_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> t_shared_entry{
        //    t_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> rho_old_shared_entry{
        //    rho_old_sh, static_cast<size_type>(nrhs), 1, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> rho_new_shared_entry{
        //    rho_new_sh, static_cast<size_type>(nrhs), 1, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> omega_shared_entry{
        //    omega_sh, static_cast<size_type>(nrhs), 1, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> alpha_shared_entry{
        //    alpha_sh, static_cast<size_type>(nrhs), 1, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> temp_shared_entry{
        //    temp_sh, static_cast<size_type>(nrhs), 1, nrhs};

        // const gko::batch_dense::BatchEntry<real_type> rhs_norms_shared_entry{
        //    norms_rhs_sh, static_cast<size_type>(nrhs), 1, nrhs};

        // const gko::batch_dense::BatchEntry<real_type> res_norms_shared_entry{
        //    norms_res_sh, static_cast<size_type>(nrhs), 1, nrhs};


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
        initialize(nrows, A_global_entry, b_entry_ptr, x_gl_entry_ptr,
                   rho_old_sh[0], omega_sh[0], alpha_sh[0], x_sh, r_sh,
                   r_hat_sh, p_sh, v_sh, norms_rhs_sh[0], norms_res_sh[0]);


        __syncthreads();


        // stopping criterion object
        // StopType stop(nrhs, max_iter, tol, rhs_norms_shared_entry.values,
        //              converged);

        // int iter = -1;
        // while (1) {
        //    ++iter;
        int iter = 0;
        for (; iter < max_iter; iter++) {
            // bool all_converged =
            //    stop.check_converged(iter, res_norms_shared_entry.values,
            //                         {NULL, 0, 0, 0}, converged);


            // logger.log_iteration(ibatch, iter, res_norms_shared_entry.values,
            //                     converged);
            // if (all_converged) {
            //    break;
            //}
            if (norms_res_sh[0] / norms_rhs_sh[0] < tol) {
                break;
            }


            // rho_new =  < r_hat , r > = (r_hat)' * (r)
            compute_dot_product<ValueType>(nrows, r_hat_sh, r_sh,
                                           rho_new_sh[0]);


            __syncthreads();

            // beta = (rho_new / rho_old)*(alpha / omega)
            // p = r + beta*(p - omega * v)
            update_p(nrows, rho_new_sh[0], rho_old_sh[0], alpha_sh[0],
                     omega_sh[0], r_sh, v_sh, p_sh /*,converged*/);

            __syncthreads();

            // p_hat = precond * p
            prec_shared.apply(nrows, p_sh, p_hat_sh);


            __syncthreads();

            // v = A * p_hat
            single_spmv_kernel(A_global_entry, p_hat_sh, v_sh);

            __syncthreads();

            // alpha = rho_new / < r_hat , v>
            compute_alpha(nrows, rho_new_sh[0], r_hat_sh, v_sh,
                          alpha_sh[0] /*, converged*/);

            __syncthreads();

            // s = r - alpha*v
            update_s(nrows, r_sh, alpha_sh[0], v_sh, s_sh /*, converged*/);

            __syncthreads();

            // an estimate of residual norms
            compute_norm2<ValueType>(nrows, s_sh, norms_res_sh[0]);

            __syncthreads();


#if 0
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
#endif

            if (norms_res_sh[0] / norms_rhs_sh[0] < tol) {
                update_x_middle(nrows, alpha_sh[0], p_hat_sh, x_sh);
                break;
            }


            // s_hat = precond * s
            prec_shared.apply(nrows, s_sh, s_hat_sh);

            __syncthreads();


            // t = A * s_hat
            single_spmv_kernel(A_global_entry, s_hat_sh, t_sh);
            // csr_spmv_kernel(nrows, avalues + ibatch*num_nnz, col_idxs,
            // row_ptrs, 	stride, nrhs, s_hat_sh, t_sh);
            __syncthreads();

            // omega = <t,s> / <t,t>
            compute_omega(nrows, t_sh, s_sh, temp_sh[0],
                          omega_sh[0] /*, converged*/);

            __syncthreads();


            // x = x + alpha*p_hat + omega *s_hat
            // r = s - omega * t
            update_x_and_r(nrows, p_hat_sh, s_hat_sh, alpha_sh[0], omega_sh[0],
                           s_sh, t_sh, x_sh, r_sh /*, converged*/);


            __syncthreads();

            compute_norm2<ValueType>(nrows, r_sh, norms_res_sh[0]);

            __syncthreads();

            // rho_old = rho_new
            // copy(nrows, rho_new_shared_entry.values,
            //     rho_old_shared_entry.values);
            if (threadIdx.x == 0) {
                rho_old_sh[0] = rho_new_sh[0];
            }

            __syncthreads();
        }

        // logger.log_iteration(ibatch, iter, norms_res_sh[0], 0);

        // copy x back to global memory
        copy(nrows, x_sh, x_gl_entry_ptr);

        __syncthreads();
    }
}
