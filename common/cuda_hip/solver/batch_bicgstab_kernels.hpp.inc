// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename Group, typename BatchMatrixType_entry, typename ValueType>
__device__ __forceinline__ void initialize(
    Group subgroup, const int num_rows, const BatchMatrixType_entry& mat_entry,
    const ValueType* const b_global_entry,
    const ValueType* const x_global_entry, ValueType& rho_old, ValueType& omega,
    ValueType& alpha, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry, ValueType* const r_hat_shared_entry,
    ValueType* const p_shared_entry, ValueType* const p_hat_shared_entry,
    ValueType* const v_shared_entry,
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
    advanced_apply(static_cast<ValueType>(-1.0), mat_entry, x_shared_entry,
                   static_cast<ValueType>(1.0), r_shared_entry);
    __syncthreads();

    if (threadIdx.x / config::warp_size == 0) {
        single_rhs_compute_norm2(subgroup, num_rows, r_shared_entry, res_norm);
    } else if (threadIdx.x / config::warp_size == 1) {
        // Compute norms of rhs
        single_rhs_compute_norm2(subgroup, num_rows, b_global_entry, rhs_norm);
    }
    __syncthreads();

    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        r_hat_shared_entry[iz] = r_shared_entry[iz];
        p_shared_entry[iz] = zero<ValueType>();
        p_hat_shared_entry[iz] = zero<ValueType>();
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
    const ValueType beta = (rho_new / rho_old) * (alpha / omega);
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        p_shared_entry[r] =
            r_shared_entry[r] +
            beta * (p_shared_entry[r] - omega * v_shared_entry[r]);
    }
}

template <typename Group, typename ValueType>
__device__ __forceinline__ void compute_alpha(
    Group subgroup, const int num_rows, const ValueType& rho_new,
    const ValueType* const r_hat_shared_entry,
    const ValueType* const v_shared_entry, ValueType& alpha)
{
    if (threadIdx.x / config::warp_size == 0) {
        single_rhs_compute_conj_dot(subgroup, num_rows, r_hat_shared_entry,
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


template <typename Group, typename ValueType>
__device__ __forceinline__ void compute_omega(
    Group subgroup, const int num_rows, const ValueType* const t_shared_entry,
    const ValueType* const s_shared_entry, ValueType& temp, ValueType& omega)
{
    if (threadIdx.x / config::warp_size == 0) {
        single_rhs_compute_conj_dot(subgroup, num_rows, t_shared_entry,
                                    s_shared_entry, omega);
    } else if (threadIdx.x / config::warp_size == 1) {
        single_rhs_compute_conj_dot(subgroup, num_rows, t_shared_entry,
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


template <typename StopType, int n_shared, bool prec_shared_bool,
          typename PrecType, typename LogType, typename BatchMatrixType,
          typename ValueType>
__global__ void apply_kernel(
    const gko::kernels::batch_bicgstab::storage_config sconf,
    const int max_iter, const gko::remove_complex<ValueType> tol,
    LogType logger, PrecType prec_shared, const BatchMatrixType mat,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ x,
    ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto num_batch_items = mat.num_batch_items;
    const auto num_rows = mat.num_rows;

    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto subgroup = group::tiled_partition<tile_size>(thread_block);

    for (int batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        const int gmem_offset =
            batch_id * sconf.gmem_stride_bytes / sizeof(ValueType);
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

        if (n_shared >= 1) {
            p_hat_sh = reinterpret_cast<ValueType*>(local_mem_sh);
        } else {
            p_hat_sh = workspace + gmem_offset;
        }
        if (n_shared == 1) {
            s_hat_sh = workspace + gmem_offset;
        } else {
            s_hat_sh = p_hat_sh + sconf.padded_vec_len;
        }
        if (n_shared == 2) {
            v_sh = workspace + gmem_offset;
        } else {
            v_sh = s_hat_sh + sconf.padded_vec_len;
        }
        if (n_shared == 3) {
            t_sh = workspace + gmem_offset;
        } else {
            t_sh = v_sh + sconf.padded_vec_len;
        }
        if (n_shared == 4) {
            p_sh = workspace + gmem_offset;
        } else {
            p_sh = t_sh + sconf.padded_vec_len;
        }
        if (n_shared == 5) {
            s_sh = workspace + gmem_offset;
        } else {
            s_sh = p_sh + sconf.padded_vec_len;
        }
        if (n_shared == 6) {
            r_sh = workspace + gmem_offset;
        } else {
            r_sh = s_sh + sconf.padded_vec_len;
        }
        if (n_shared == 7) {
            r_hat_sh = workspace + gmem_offset;
        } else {
            r_hat_sh = r_sh + sconf.padded_vec_len;
        }
        if (n_shared == 8) {
            x_sh = workspace + gmem_offset;
        } else {
            x_sh = r_hat_sh + sconf.padded_vec_len;
        }
        if (!prec_shared_bool && n_shared == 9) {
            prec_work_sh = workspace + gmem_offset;
        } else {
            prec_work_sh = x_sh + sconf.padded_vec_len;
        }

        __shared__ uninitialized_array<ValueType, 1> rho_old_sh;
        __shared__ uninitialized_array<ValueType, 1> rho_new_sh;
        __shared__ uninitialized_array<ValueType, 1> omega_sh;
        __shared__ uninitialized_array<ValueType, 1> alpha_sh;
        __shared__ uninitialized_array<ValueType, 1> temp_sh;
        __shared__ real_type norms_rhs_sh[1];
        __shared__ real_type norms_res_sh[1];

        const auto mat_entry =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const ValueType* const b_entry_ptr =
            gko::batch::multi_vector::batch_item_ptr(b, 1, num_rows, batch_id);
        ValueType* const x_gl_entry_ptr =
            gko::batch::multi_vector::batch_item_ptr(x, 1, num_rows, batch_id);

        // generate preconditioner
        prec_shared.generate(batch_id, mat_entry, prec_work_sh);

        // initialization
        // rho_old = 1, omega = 1, alpha = 1
        // compute b norms
        // copy x from global to shared memory
        // r = b - A*x
        // compute residual norms
        // r_hat = r
        // p = 0
        // p_hat = 0
        // v = 0
        initialize(subgroup, num_rows, mat_entry, b_entry_ptr, x_gl_entry_ptr,
                   rho_old_sh[0], omega_sh[0], alpha_sh[0], x_sh, r_sh,
                   r_hat_sh, p_sh, p_hat_sh, v_sh, norms_rhs_sh[0],
                   norms_res_sh[0]);
        __syncthreads();

        // stopping criterion object
        StopType stop(tol, norms_rhs_sh);

        int iter = 0;
        for (; iter < max_iter; iter++) {
            if (stop.check_converged(norms_res_sh)) {
                logger.log_iteration(batch_id, iter, norms_res_sh[0]);
                break;
            }

            // rho_new =  < r_hat , r > = (r_hat)' * (r)
            if (threadIdx.x / config::warp_size == 0) {
                single_rhs_compute_conj_dot(subgroup, num_rows, r_hat_sh, r_sh,
                                            rho_new_sh[0]);
            }
            __syncthreads();

            // beta = (rho_new / rho_old)*(alpha / omega)
            // p = r + beta*(p - omega * v)
            update_p(num_rows, rho_new_sh[0], rho_old_sh[0], alpha_sh[0],
                     omega_sh[0], r_sh, v_sh, p_sh);
            __syncthreads();

            // p_hat = precond * p
            prec_shared.apply(num_rows, p_sh, p_hat_sh);
            __syncthreads();

            // v = A * p_hat
            simple_apply(mat_entry, p_hat_sh, v_sh);
            __syncthreads();

            // alpha = rho_new / < r_hat , v>
            compute_alpha(subgroup, num_rows, rho_new_sh[0], r_hat_sh, v_sh,
                          alpha_sh[0]);
            __syncthreads();

            // s = r - alpha*v
            update_s(num_rows, r_sh, alpha_sh[0], v_sh, s_sh);
            __syncthreads();

            // an estimate of residual norms
            if (threadIdx.x / config::warp_size == 0) {
                single_rhs_compute_norm2(subgroup, num_rows, s_sh,
                                         norms_res_sh[0]);
            }
            __syncthreads();

            // if (norms_res_sh[0] / norms_rhs_sh[0] < tol) {
            if (stop.check_converged(norms_res_sh)) {
                update_x_middle(num_rows, alpha_sh[0], p_hat_sh, x_sh);
                logger.log_iteration(batch_id, iter, norms_res_sh[0]);
                break;
            }

            // s_hat = precond * s
            prec_shared.apply(num_rows, s_sh, s_hat_sh);
            __syncthreads();

            // t = A * s_hat
            simple_apply(mat_entry, s_hat_sh, t_sh);
            __syncthreads();

            // omega = <t,s> / <t,t>
            compute_omega(subgroup, num_rows, t_sh, s_sh, temp_sh[0],
                          omega_sh[0]);
            __syncthreads();

            // x = x + alpha*p_hat + omega *s_hat
            // r = s - omega * t
            update_x_and_r(num_rows, p_hat_sh, s_hat_sh, alpha_sh[0],
                           omega_sh[0], s_sh, t_sh, x_sh, r_sh);
            __syncthreads();

            if (threadIdx.x / config::warp_size == 0) {
                single_rhs_compute_norm2(subgroup, num_rows, r_sh,
                                         norms_res_sh[0]);
            }
            //__syncthreads();

            if (threadIdx.x == blockDim.x - 1) {
                rho_old_sh[0] = rho_new_sh[0];
            }
            __syncthreads();
        }

        logger.log_iteration(batch_id, iter, norms_res_sh[0]);

        // copy x back to global memory
        single_rhs_copy(num_rows, x_sh, x_gl_entry_ptr);
        __syncthreads();
    }
}
