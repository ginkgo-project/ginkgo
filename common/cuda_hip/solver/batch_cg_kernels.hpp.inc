// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename Group, typename BatchMatrixType_entry, typename PrecType,
          typename ValueType>
__device__ __forceinline__ void initialize(
    Group subgroup, const int num_rows, const BatchMatrixType_entry& mat_entry,
    const ValueType* const b_global_entry,
    const ValueType* const x_global_entry, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry, const PrecType& prec_shared,
    ValueType* const z_shared_entry, ValueType& rho_old_shared_entry,
    ValueType* const p_shared_entry,
    typename gko::remove_complex<ValueType>& rhs_norms_sh)
{
    // copy x from global to shared memory
    // r = b
    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        x_shared_entry[iz] = x_global_entry[iz];
        r_shared_entry[iz] = b_global_entry[iz];
    }
    __syncthreads();

    // r = b - A*x
    gko::kernels::GKO_DEVICE_NAMESPACE::batch_single_kernels::advanced_apply(
        static_cast<ValueType>(-1.0), mat_entry, x_shared_entry,
        static_cast<ValueType>(1.0), r_shared_entry);
    __syncthreads();

    // z = precond * r
    prec_shared.apply(num_rows, r_shared_entry, z_shared_entry);
    __syncthreads();

    if (threadIdx.x / config::warp_size == 0) {
        // Compute norms of rhs
        gko::kernels::GKO_DEVICE_NAMESPACE::batch_single_kernels::
            single_rhs_compute_norm2(subgroup, num_rows, b_global_entry,
                                     rhs_norms_sh);
    } else if (threadIdx.x / config::warp_size == 1) {
        // rho_old = r' * z
        gko::kernels::GKO_DEVICE_NAMESPACE::batch_single_kernels::
            single_rhs_compute_conj_dot(subgroup, num_rows, r_shared_entry,
                                        z_shared_entry, rho_old_shared_entry);
    }

    // p = z
    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        p_shared_entry[iz] = z_shared_entry[iz];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_p(const int num_rows,
                                         const ValueType& rho_new_shared_entry,
                                         const ValueType& rho_old_shared_entry,
                                         const ValueType* const z_shared_entry,
                                         ValueType* const p_shared_entry)
{
    for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
        const ValueType beta = rho_new_shared_entry / rho_old_shared_entry;
        p_shared_entry[li] = z_shared_entry[li] + beta * p_shared_entry[li];
    }
}


template <typename Group, typename ValueType>
__device__ __forceinline__ void update_x_and_r(
    Group subgroup, const int num_rows, const ValueType& rho_old_shared_entry,
    const ValueType* const p_shared_entry,
    const ValueType* const Ap_shared_entry, ValueType& alpha_shared_entry,
    ValueType* const x_shared_entry, ValueType* const r_shared_entry)
{
    if (threadIdx.x / config::warp_size == 0) {
        gko::kernels::GKO_DEVICE_NAMESPACE::batch_single_kernels::
            single_rhs_compute_conj_dot(subgroup, num_rows, p_shared_entry,
                                        Ap_shared_entry, alpha_shared_entry);
    }
    __syncthreads();

    for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
        const ValueType alpha = rho_old_shared_entry / alpha_shared_entry;
        x_shared_entry[li] += alpha * p_shared_entry[li];
        r_shared_entry[li] -= alpha * Ap_shared_entry[li];
    }
}


template <typename StopType, const int n_shared, const bool prec_shared_bool,
          typename PrecType, typename LogType, typename BatchMatrixType,
          typename ValueType>
__global__ void apply_kernel(const gko::kernels::batch_cg::storage_config sconf,
                             const int max_iter,
                             const gko::remove_complex<ValueType> tol,
                             LogType logger, PrecType prec_shared,
                             const BatchMatrixType mat,
                             const ValueType* const __restrict__ b,
                             ValueType* const __restrict__ x,
                             ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto num_batch_items = mat.num_batch_items;
    const auto num_rows = mat.num_rows;

    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto subgroup = group::tiled_partition<tile_size>(thread_block);

    for (size_type batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        const int gmem_offset =
            batch_id * sconf.gmem_stride_bytes / sizeof(ValueType);
        extern __shared__ char local_mem_sh[];

        ValueType* r_sh;
        ValueType* z_sh;
        ValueType* p_sh;
        ValueType* Ap_sh;
        ValueType* x_sh;
        ValueType* prec_work_sh;

        if (n_shared >= 1) {
            r_sh = reinterpret_cast<ValueType*>(local_mem_sh);
        } else {
            r_sh = workspace + gmem_offset;
        }
        if (n_shared == 1) {
            z_sh = workspace + gmem_offset;
        } else {
            z_sh = r_sh + sconf.padded_vec_len;
        }
        if (n_shared == 2) {
            p_sh = workspace + gmem_offset;
        } else {
            p_sh = z_sh + sconf.padded_vec_len;
        }
        if (n_shared == 3) {
            Ap_sh = workspace + gmem_offset;
        } else {
            Ap_sh = p_sh + sconf.padded_vec_len;
        }
        if (n_shared == 4) {
            x_sh = workspace + gmem_offset;
        } else {
            x_sh = Ap_sh + sconf.padded_vec_len;
        }
        if (!prec_shared_bool && n_shared == 5) {
            prec_work_sh = workspace + gmem_offset;
        } else {
            prec_work_sh = x_sh + sconf.padded_vec_len;
        }

        __shared__ uninitialized_array<ValueType, 1> rho_old_sh;
        __shared__ uninitialized_array<ValueType, 1> rho_new_sh;
        __shared__ uninitialized_array<ValueType, 1> alpha_sh;
        __shared__ real_type norms_rhs_sh[1];
        __shared__ real_type norms_res_sh[1];

        const auto mat_entry =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const ValueType* const b_global_entry =
            gko::batch::multi_vector::batch_item_ptr(b, 1, num_rows, batch_id);
        ValueType* const x_global_entry =
            gko::batch::multi_vector::batch_item_ptr(x, 1, num_rows, batch_id);

        // generate preconditioner
        prec_shared.generate(batch_id, mat_entry, prec_work_sh);

        // initialization
        // compute b norms
        // r = b - A*x
        // z = precond*r
        // rho_old = r' * z (' is for hermitian transpose)
        // p = z
        initialize(subgroup, num_rows, mat_entry, b_global_entry,
                   x_global_entry, x_sh, r_sh, prec_shared, z_sh, rho_old_sh[0],
                   p_sh, norms_rhs_sh[0]);
        __syncthreads();

        // stopping criterion object
        StopType stop(tol, norms_rhs_sh);

        int iter = 0;
        for (; iter < max_iter; iter++) {
            norms_res_sh[0] = sqrt(abs(rho_old_sh[0]));
            __syncthreads();
            if (stop.check_converged(norms_res_sh)) {
                logger.log_iteration(batch_id, iter, norms_res_sh[0]);
                break;
            }

            // Ap = A * p
            gko::kernels::GKO_DEVICE_NAMESPACE::batch_single_kernels::
                simple_apply(mat_entry, p_sh, Ap_sh);
            __syncthreads();

            // alpha = rho_old / (p' * Ap)
            // x = x + alpha * p
            // r = r - alpha * Ap
            update_x_and_r(subgroup, num_rows, rho_old_sh[0], p_sh, Ap_sh,
                           alpha_sh[0], x_sh, r_sh);
            __syncthreads();

            // z = precond * r
            prec_shared.apply(num_rows, r_sh, z_sh);
            __syncthreads();

            if (threadIdx.x / config::warp_size == 0) {
                // rho_new =  (r)' * (z)
                gko::kernels::GKO_DEVICE_NAMESPACE::batch_single_kernels::
                    single_rhs_compute_conj_dot(subgroup, num_rows, r_sh, z_sh,
                                                rho_new_sh[0]);
            }
            __syncthreads();

            // beta = rho_new / rho_old
            // p = z + beta * p
            update_p(num_rows, rho_new_sh[0], rho_old_sh[0], z_sh, p_sh);
            __syncthreads();

            // rho_old = rho_new
            if (threadIdx.x == 0) {
                rho_old_sh[0] = rho_new_sh[0];
            }
            __syncthreads();
        }

        logger.log_iteration(batch_id, iter, norms_res_sh[0]);

        // copy x back to global memory
        gko::kernels::GKO_DEVICE_NAMESPACE::batch_single_kernels::
            single_rhs_copy(num_rows, x_sh, x_global_entry);
        __syncthreads();
    }
}
