/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const BatchMatrixType_entry& a_global_entry,
    const ValueType* const b_global_entry,
    const ValueType* const x_global_entry, ValueType& rho_old, ValueType& omega,
    ValueType& alpha, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry, ValueType* const r_hat_shared_entry,
    ValueType* const p_shared_entry, ValueType* const v_shared_entry,
    typename gko::remove_complex<ValueType>& rhs_norm,
    typename gko::remove_complex<ValueType>& res_norm)
{
    rho_old = one<ValueType>();
    omega = one<ValueType>();
    alpha = one<ValueType>();

    // copy x from global to shared memory
    // r = b
    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        x_shared_entry[iz] = x_global_entry[iz];
        r_shared_entry[iz] = b_global_entry[iz];
    }
    __syncthreads();

    // r = b - A*x
    single_advanced_matvec_kernel(static_cast<ValueType>(-1.0), a_global_entry,
                                  x_shared_entry, static_cast<ValueType>(1.0),
                                  r_shared_entry);
    __syncthreads();

    if (threadIdx.x / config::warp_size == 0) {
        single_compute_norm2(warp_grp, num_rows, r_shared_entry, res_norm);
    } else if (threadIdx.x / config::warp_size == 1) {
        // Compute norms of rhs
        single_compute_norm2(warp_grp, num_rows, b_global_entry, rhs_norm);
    }

    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        r_hat_shared_entry[iz] = r_shared_entry[iz];
        p_shared_entry[iz] = zero<ValueType>();
        v_shared_entry[iz] = zero<ValueType>();
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_p(
    const int num_rows, const ValueType& rho_new, const ValueType& rho_old,
    const ValueType& alpha, const ValueType& omega,
    const ValueType* const r_shared_entry,
    const ValueType* const v_shared_entry, ValueType* const p_shared_entry)
{
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        const ValueType beta = (rho_new / rho_old) * (alpha / omega);
        p_shared_entry[r] =
            r_shared_entry[r] +
            beta * (p_shared_entry[r] - omega * v_shared_entry[r]);
    }
}

template <typename ValueType>
__device__ __forceinline__ void compute_alpha(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const ValueType& rho_new, const ValueType* const r_hat_shared_entry,
    const ValueType* const v_shared_entry, ValueType& alpha)
{
    if (threadIdx.x / config::warp_size == 0) {
        single_compute_dot_product(warp_grp, num_rows, r_hat_shared_entry,
                                   v_shared_entry, alpha);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        alpha = rho_new / alpha;
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_s(const int num_rows,
                                         const ValueType* const r_shared_entry,
                                         const ValueType& alpha,
                                         const ValueType* const v_shared_entry,
                                         ValueType* const s_shared_entry)
{
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        s_shared_entry[r] = r_shared_entry[r] - alpha * v_shared_entry[r];
    }
}


template <typename ValueType>
__device__ __forceinline__ void compute_omega(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const ValueType* const t_shared_entry,
    const ValueType* const s_shared_entry, ValueType& temp, ValueType& omega)
{
    if (threadIdx.x / config::warp_size == 0) {
        single_compute_dot_product(warp_grp, num_rows, t_shared_entry,
                                   s_shared_entry, omega);
    } else if (threadIdx.x / config::warp_size == 1) {
        single_compute_dot_product(warp_grp, num_rows, t_shared_entry,
                                   t_shared_entry, temp);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        omega /= temp;
    }
}

template <typename ValueType>
__device__ __forceinline__ void update_x_and_r(
    const int num_rows, const ValueType* const p_hat_shared_entry,
    const ValueType* const s_hat_shared_entry, const ValueType& alpha,
    const ValueType& omega, const ValueType* const s_shared_entry,
    const ValueType* const t_shared_entry, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry)
{
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        x_shared_entry[r] = x_shared_entry[r] + alpha * p_hat_shared_entry[r] +
                            omega * s_hat_shared_entry[r];
        r_shared_entry[r] = s_shared_entry[r] - omega * t_shared_entry[r];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x_middle(
    const int num_rows, const ValueType& alpha,
    const ValueType* const p_hat_shared_entry, ValueType* const x_shared_entry)
{
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        x_shared_entry[r] = x_shared_entry[r] + alpha * p_hat_shared_entry[r];
    }
}


}  // namespace

// unnamed namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(
    const gko::kernels::batch_bicgstab::StorageConfig sconf, const int max_iter,
    const gko::remove_complex<ValueType> tol, LogType logger,
    PrecType prec_shared, const BatchMatrixType a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ x,
    ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = a.num_batch;
    const auto nrows = a.num_rows;

    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto warp_grp = group::tiled_partition<tile_size>(thread_block);

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
        // int gmem_offset = ibatch * (padded_vec_len * (10 - num_sh_vecs));
        const int gmem_offset =
            ibatch * sconf.gmem_stride_bytes / sizeof(ValueType);
        extern __shared__ char local_mem_sh[];

        ValueType* p_hat_sh;
        ValueType* s_hat_sh;
        ValueType* p_sh;
        ValueType* s_sh;
        ValueType* r_sh;
        ValueType* r_hat_sh;
        ValueType* v_sh;
        ValueType* t_sh;
        ValueType* x_sh;
        ValueType* prec_work_sh;

        if (sconf.n_shared >= 1) {
            p_hat_sh = reinterpret_cast<ValueType*>(local_mem_sh);
        } else {
            p_hat_sh = workspace + gmem_offset;
        }
        if (sconf.n_shared == 1) {
            s_hat_sh = workspace + gmem_offset;
        } else {
            s_hat_sh = p_hat_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 2) {
            v_sh = workspace + gmem_offset;
        } else {
            v_sh = s_hat_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 3) {
            t_sh = workspace + gmem_offset;
        } else {
            t_sh = v_sh + sconf.padded_vec_len;
        }
        if (!sconf.prec_shared && sconf.n_shared == 4) {
            prec_work_sh = workspace + gmem_offset;
        } else {
            prec_work_sh = t_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 4 && sconf.prec_shared) {
            p_sh = workspace + gmem_offset;
        } else {
            p_sh = prec_work_sh + PrecType::dynamic_work_size(nrows, a.num_nnz);
        }
        if (sconf.n_shared == 5) {
            s_sh = workspace + gmem_offset;
        } else {
            s_sh = p_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 6) {
            r_sh = workspace + gmem_offset;
        } else {
            r_sh = s_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 7) {
            r_hat_sh = workspace + gmem_offset;
        } else {
            r_hat_sh = r_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 8) {
            x_sh = workspace + gmem_offset;
        } else {
            x_sh = r_hat_sh + sconf.padded_vec_len;
        }

        __shared__ uninitialized_array<ValueType, 1> rho_old_sh;
        __shared__ uninitialized_array<ValueType, 1> rho_new_sh;
        __shared__ uninitialized_array<ValueType, 1> omega_sh;
        __shared__ uninitialized_array<ValueType, 1> alpha_sh;
        __shared__ uninitialized_array<ValueType, 1> temp_sh;
        __shared__ real_type norms_rhs_sh[1];
        __shared__ real_type norms_res_sh[1];

        const auto a_global_entry = gko::batch::batch_entry(a, ibatch);
        const ValueType* const b_entry_ptr =
            gko::batch::batch_entry_ptr(b, 1, nrows, ibatch);
        ValueType* const x_gl_entry_ptr =
            gko::batch::batch_entry_ptr(x, 1, nrows, ibatch);

        // generate preconditioner
        prec_shared.generate(ibatch, a_global_entry, prec_work_sh);

        // initialization
        // rho_old = 1, omega = 1, alpha = 1
        // compute b norms
        // copy x from global to shared memory
        // r = b - A*x
        // compute residual norms
        // r_hat = r
        // p = 0
        // v = 0
        initialize(warp_grp, nrows, a_global_entry, b_entry_ptr, x_gl_entry_ptr,
                   rho_old_sh[0], omega_sh[0], alpha_sh[0], x_sh, r_sh,
                   r_hat_sh, p_sh, v_sh, norms_rhs_sh[0], norms_res_sh[0]);
        __syncthreads();

        // stopping criterion object
        StopType stop(tol, norms_rhs_sh);

        int iter = 0;
        for (; iter < max_iter; iter++) {
            if (stop.check_converged(norms_res_sh)) {
                break;
            }

            // rho_new =  < r_hat , r > = (r_hat)' * (r)
            if (threadIdx.x / config::warp_size == 0) {
                single_compute_dot_product(warp_grp, nrows, r_hat_sh, r_sh,
                                           rho_new_sh[0]);
            }
            __syncthreads();

            // beta = (rho_new / rho_old)*(alpha / omega)
            // p = r + beta*(p - omega * v)
            update_p(nrows, rho_new_sh[0], rho_old_sh[0], alpha_sh[0],
                     omega_sh[0], r_sh, v_sh, p_sh);
            __syncthreads();

            // p_hat = precond * p
            prec_shared.apply(nrows, p_sh, p_hat_sh);
            __syncthreads();

            // v = A * p_hat
            single_matvec_kernel(a_global_entry, p_hat_sh, v_sh);
            __syncthreads();

            // alpha = rho_new / < r_hat , v>
            compute_alpha(warp_grp, nrows, rho_new_sh[0], r_hat_sh, v_sh,
                          alpha_sh[0] /*, converged*/);
            __syncthreads();

            // s = r - alpha*v
            update_s(nrows, r_sh, alpha_sh[0], v_sh, s_sh /*, converged*/);
            __syncthreads();

            // an estimate of residual norms
            if (threadIdx.x / config::warp_size == 0) {
                single_compute_norm2(warp_grp, nrows, s_sh, norms_res_sh[0]);
            }
            __syncthreads();

            // if (norms_res_sh[0] / norms_rhs_sh[0] < tol) {
            if (stop.check_converged(norms_res_sh)) {
                update_x_middle(nrows, alpha_sh[0], p_hat_sh, x_sh);
                break;
            }

            // s_hat = precond * s
            prec_shared.apply(nrows, s_sh, s_hat_sh);
            __syncthreads();

            // t = A * s_hat
            single_matvec_kernel(a_global_entry, s_hat_sh, t_sh);
            __syncthreads();

            // omega = <t,s> / <t,t>
            compute_omega(warp_grp, nrows, t_sh, s_sh, temp_sh[0],
                          omega_sh[0] /*, converged*/);
            __syncthreads();

            // x = x + alpha*p_hat + omega *s_hat
            // r = s - omega * t
            update_x_and_r(nrows, p_hat_sh, s_hat_sh, alpha_sh[0], omega_sh[0],
                           s_sh, t_sh, x_sh, r_sh /*, converged*/);
            __syncthreads();

            if (threadIdx.x / config::warp_size == 0) {
                single_compute_norm2(warp_grp, nrows, r_sh, norms_res_sh[0]);
            }
            //__syncthreads();

            if (threadIdx.x == blockDim.x - 1) {
                rho_old_sh[0] = rho_new_sh[0];
            }
            __syncthreads();
        }

        logger.log_iteration(ibatch, iter, norms_res_sh[0]);

        // copy x back to global memory
        single_copy(nrows, x_sh, x_gl_entry_ptr);
        __syncthreads();
    }
}
