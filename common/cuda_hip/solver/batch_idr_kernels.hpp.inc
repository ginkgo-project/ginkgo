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


namespace {


template <typename ValueType>
__device__ __forceinline__ void orthonormalize_subspace_vectors(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const int subspace_dim, ValueType* const subspace_vecs_sh,
    ValueType* const temp_shared_entry,
    typename gko::remove_complex<ValueType>* const tmp_norms_sh)
{
    using real_type = typename gko::remove_complex<ValueType>;

    for (int i = 0; i < subspace_dim; i++) {
        ValueType* const p_i_sh = subspace_vecs_sh + i * num_rows;
        ValueType* const w_i_sh = temp_shared_entry;

        // w_i = p_i
        for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
            w_i_sh[iz] = p_i_sh[iz];
        }

        for (int j = 0; j < i; j++) {
            // w_i = w_i - proj(p_i) on w_j that is w_i = w_i - (< w_j , p_i >
            // /< w_j , w_j > ) * w_j
            __syncthreads();

            ValueType* const w_j_sh = subspace_vecs_sh + j * num_rows;

            __shared__ uninitialized_array<ValueType, 1> mul_sh;
            if (threadIdx.x / config::warp_size == 0) {
                single_compute_dot_product(warp_grp, num_rows, w_j_sh, p_i_sh,
                                           mul_sh[0]);
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                mul_sh[0] /=
                    static_cast<ValueType>(tmp_norms_sh[j] * tmp_norms_sh[j]);
                mul_sh[0] *= -one<ValueType>();
            }
            __syncthreads();

            single_add_scaled(num_rows, mul_sh[0], w_j_sh, w_i_sh);
        }
        __syncthreads();

        if (threadIdx.x / config::warp_size == 0) {
            single_compute_norm2(warp_grp, num_rows, w_i_sh, tmp_norms_sh[i]);
        }
        // p_i = w_i
        single_copy(num_rows, w_i_sh, p_i_sh);
        __syncthreads();
    }

    // e_k = w_k / || w_k ||  for k = 0, 1, ..., subspace_dim -1
    for (int tid = threadIdx.x; tid < subspace_dim * num_rows;
         tid += blockDim.x) {
        const int row_index = tid % num_rows;
        const int vec_index = tid / num_rows;
        subspace_vecs_sh[vec_index * num_rows + row_index] /=
            static_cast<ValueType>(tmp_norms_sh[vec_index]);
    }
}

template <typename BatchMatrixEntry, typename ValueType>
__device__ __forceinline__ void initialize(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const int subspace_dim, const bool deterministic, const bool smoothing,
    const BatchMatrixEntry& A_global_entry,
    const ValueType* const b_global_entry,
    const ValueType* const x_global_entry,
    const ValueType* const subspace_vecs_global,
    ValueType* const x_shared_entry, ValueType* const r_shared_entry,
    ValueType* const G_shared_entry, ValueType* const U_shared_entry,
    ValueType* const M_shared_entry, ValueType* const subspace_vecs_sh,
    ValueType* const xs_shared_entry, ValueType* const rs_shared_entry,
    ValueType& omega_shared_entry,
    typename gko::remove_complex<ValueType>& rhs_norms_shared_entry,
    typename gko::remove_complex<ValueType>& res_norms_shared_entry,
    ValueType* const temp_shared_entry,
    typename gko::remove_complex<ValueType>* const tmp_norms_sh)
{
    using real_type = gko::remove_complex<ValueType>();

    // copy x from global to shared memory
    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        x_shared_entry[iz] = x_global_entry[iz];
        r_shared_entry[iz] = b_global_entry[iz];
    }
    __syncthreads();

    // r = b - A*x
    single_advanced_matvec_kernel(static_cast<ValueType>(-1.0), A_global_entry,
                                  x_shared_entry, static_cast<ValueType>(1.0),
                                  r_shared_entry);
    __syncthreads();

    if (threadIdx.x / config::warp_size == 0) {
        // compute residual norms
        single_compute_norm2(warp_grp, num_rows, r_shared_entry,
                             res_norms_shared_entry);
    } else if (threadIdx.x / config::warp_size == 1) {
        // Compute norms of rhs
        single_compute_norm2(warp_grp, num_rows, b_global_entry,
                             rhs_norms_shared_entry);
    }
    if (threadIdx.x == 0) {
        // omega = 1
        omega_shared_entry = one<ValueType>();
    }

    // M = Identity
    for (int tid = threadIdx.x; tid < subspace_dim * subspace_dim;
         tid += blockDim.x) {
        const int row_index = tid / (subspace_dim);
        const int col_index = tid % subspace_dim;

        if (row_index == col_index) {
            M_shared_entry[row_index * subspace_dim + col_index] =
                one<ValueType>();
        } else {
            M_shared_entry[row_index * subspace_dim + col_index] =
                zero<ValueType>();
        }
    }

    // G = zero
    // U = zero
    for (int tid = threadIdx.x; tid < num_rows * subspace_dim;
         tid += blockDim.x) {
        const int vec_index = tid / num_rows;
        const int row_index = tid % num_rows;
        G_shared_entry[vec_index * num_rows + row_index] = zero<ValueType>();
        U_shared_entry[vec_index * num_rows + row_index] = zero<ValueType>();
    }

    if (smoothing) {
        // xs = x
        // rs = r
        for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
            xs_shared_entry[iz] = x_shared_entry[iz];
            rs_shared_entry[iz] = r_shared_entry[iz];
        }
    }

    curandState_t state;
    curand_init(0, threadIdx.x, 0, &state);  // TODO: Have a random seed value

    // initialize Subspace_vectors
    for (int li = threadIdx.x; li < num_rows * subspace_dim; li += blockDim.x) {
        const int vec_index = li / num_rows;
        const int row_index = li % num_rows;
        if (deterministic) {
            subspace_vecs_sh[vec_index * num_rows + row_index] =
                subspace_vecs_global[vec_index * num_rows + row_index];
        } else {
            ValueType val = zero<ValueType>();
            if (is_complex<ValueType>()) {
                // TODO: Random generation of complex number
                // real_type re = curand_normal(&state);
                // real_type im = curand_normal(&state);
                // val = ValueType{ re,im };
                val = static_cast<ValueType>(curand_normal(&state));
            } else {
                val = static_cast<ValueType>(curand_normal(&state));
            }
            subspace_vecs_sh[vec_index * num_rows + row_index] = val;
        }
    }
    __syncthreads();

    orthonormalize_subspace_vectors(warp_grp, num_rows, subspace_dim,
                                    subspace_vecs_sh, temp_shared_entry,
                                    tmp_norms_sh);
}


template <typename ValueType>
__device__ __forceinline__ void update_f(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const int subspace_dim, const ValueType* const subspace_vecs_sh,
    const ValueType* const r_shared_entry, ValueType* const f_shared_entry)
{
    constexpr auto tile_size = config::warp_size;
    const auto warp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_warps_per_block = ceildiv(blockDim.x, tile_size);

    for (int igrp = warp_grp_id; igrp < subspace_dim;
         igrp += num_warps_per_block) {
        single_compute_dot_product(warp_grp, num_rows,
                                   subspace_vecs_sh + igrp * num_rows,
                                   r_shared_entry, f_shared_entry[igrp]);
    }
}

template <typename ValueType>
__device__ __forceinline__ void update_c(const int num_rows,
                                         const int subspace_dim,
                                         const ValueType* const M_shared_entry,
                                         const ValueType* const f_shared_entry,
                                         ValueType* const c_shared_entry)
{
    const auto mstride = subspace_dim;
    // upper triangular solve
    // solve top to bottom
    for (int row_index = 0; row_index < subspace_dim; row_index++) {
        ValueType temp_sum = zero<ValueType>();

        for (int col_index = 0; col_index < row_index; col_index++) {
            temp_sum += M_shared_entry[row_index * mstride + col_index] *
                        c_shared_entry[col_index];
        }

        c_shared_entry[row_index] =
            (f_shared_entry[row_index] - temp_sum) /
            M_shared_entry[row_index * mstride + row_index];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_v(const int num_rows,
                                         const int subspace_dim,
                                         const ValueType* const G_shared_entry,
                                         const ValueType* const c_shared_entry,
                                         const ValueType* const r_shared_entry,
                                         ValueType* const v_shared_entry,
                                         const size_type k)
{
    for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
        v_shared_entry[li] = r_shared_entry[li];

        for (int vec_index = k; vec_index < subspace_dim; vec_index++) {
            v_shared_entry[li] -= c_shared_entry[vec_index] *
                                  G_shared_entry[vec_index * num_rows + li];
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_u_k(
    const int num_rows, const int subspace_dim,
    const ValueType& omega_shared_entry, const ValueType* const c_shared_entry,
    const ValueType* const v_shared_entry,
    const ValueType* const U_shared_entry, const size_type k,
    ValueType* const helper_shared_entry, ValueType* const u_k_shared_entry)
{
    for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
        helper_shared_entry[li] = omega_shared_entry * v_shared_entry[li];

        for (int vec_index = k; vec_index < subspace_dim; vec_index++) {
            helper_shared_entry[li] +=
                c_shared_entry[vec_index] *
                U_shared_entry[vec_index * num_rows + li];
        }
    }
    __syncthreads();

    single_copy(num_rows, helper_shared_entry, u_k_shared_entry);
}


template <typename ValueType>
__device__ __forceinline__ void update_g_k_and_u_k(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const int subspace_dim, const int k, const ValueType* const G_shared_entry,
    const ValueType* const U_shared_entry,
    const ValueType* const subspace_vecs_sh,
    const ValueType* const M_shared_entry, ValueType& alpha_shared_entry,
    ValueType* const g_k_shared_entry, ValueType* const u_k_shared_entry)
{
    for (int i = 0; i <= k - 1; i++) {
        // alpha = (p_i * g_k)/M(i,i)
        constexpr auto tile_size = config::warp_size;
        const auto warp_grp_id = static_cast<int>(threadIdx.x / tile_size);
        const int num_warps_per_block = ceildiv(blockDim.x, tile_size);

        if (warp_grp_id == 0) {
            single_compute_dot_product(warp_grp, num_rows,
                                       subspace_vecs_sh + i * num_rows,
                                       g_k_shared_entry, alpha_shared_entry);
            if (warp_grp.thread_rank() == 0) {
                alpha_shared_entry /= M_shared_entry[i * subspace_dim + i];
            }
        }
        __syncthreads();

        // g_k = g_k - alpha * g_i
        // u_k = u_k - alpha * u_i
        for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
            g_k_shared_entry[li] -=
                alpha_shared_entry * G_shared_entry[i * num_rows + li];
            u_k_shared_entry[li] -=
                alpha_shared_entry * U_shared_entry[i * num_rows + li];
        }
        __syncthreads();
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_M(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const int subspace_dim, const int k,
    const ValueType* const g_k_shared_entry,
    const ValueType* const subspace_vecs_sh, ValueType* const M_shared_entry)
{
    // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1

    constexpr auto tile_size = config::warp_size;
    const auto warp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_warps_per_block = ceildiv(blockDim.x, tile_size);

    // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1
    for (int ivec = warp_grp_id; ivec < subspace_dim - k;
         ivec += num_warps_per_block) {
        single_compute_dot_product(
            warp_grp, num_rows, subspace_vecs_sh + ivec * num_rows,
            g_k_shared_entry, M_shared_entry[ivec * subspace_dim + k]);
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_r_and_x_inner_loop(
    const int num_rows, const ValueType* const g_k_shared_entry,
    const ValueType* const u_k_shared_entry, const ValueType& beta_sh,
    ValueType* const r_shared_entry, ValueType* const x_shared_entry)
{
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        r_shared_entry[row] -= beta_sh * g_k_shared_entry[row];
        x_shared_entry[row] += beta_sh * u_k_shared_entry[row];
    }
}


template <typename ValueType>
__device__ __forceinline__ void smoothing_operation(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const ValueType* const x_shared_entry,
    const ValueType* const r_shared_entry, ValueType& gamma_shared_entry,
    ValueType* const t_shared_entry, ValueType* const xs_shared_entry,
    ValueType* const rs_shared_entry,
    gko::remove_complex<ValueType>& norms_t_shared_entry)
{
    // t = rs - r
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        t_shared_entry[row] = rs_shared_entry[row] - r_shared_entry[row];
    }
    __syncthreads();

    // gamma = (t * rs)/(t * t)
    if (threadIdx.x / config::warp_size == 0) {
        single_compute_dot_product(warp_grp, num_rows, t_shared_entry,
                                   rs_shared_entry, gamma_shared_entry);
    } else if (threadIdx.x / config::warp_size == 1) {
        single_compute_norm2(warp_grp, num_rows, t_shared_entry,
                             norms_t_shared_entry);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        gamma_shared_entry /=
            static_cast<ValueType>(norms_t_shared_entry * norms_t_shared_entry);
    }
    __syncthreads();

    // rs = rs - gamma*(rs - r)
    // xs = xs - gamma*(xs - x)
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        rs_shared_entry[row] =
            (one<ValueType>() - gamma_shared_entry) * rs_shared_entry[row] +
            gamma_shared_entry * r_shared_entry[row];

        xs_shared_entry[row] =
            (one<ValueType>() - gamma_shared_entry) * xs_shared_entry[row] +
            gamma_shared_entry * x_shared_entry[row];
    }
}

template <typename ValueType>
__device__ __forceinline__ void compute_omega(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const ValueType* const t_shared_entry,
    const ValueType* const r_shared_entry, ValueType& rho_shared_entry,
    ValueType& t_r_dot_shared_entry,
    gko::remove_complex<ValueType>& norms_t_shared_entry,
    gko::remove_complex<ValueType>& norms_r_shared_entry,
    ValueType& omega_shared_entry, const gko::remove_complex<ValueType> kappa)
{
    if (threadIdx.x / config::warp_size == 0) {
        single_compute_dot_product(warp_grp, num_rows, t_shared_entry,
                                   r_shared_entry, t_r_dot_shared_entry);
    } else if (threadIdx.x / config::warp_size == 1) {
        single_compute_norm2(warp_grp, num_rows, t_shared_entry,
                             norms_t_shared_entry);
        single_compute_norm2(warp_grp, num_rows, r_shared_entry,
                             norms_r_shared_entry);
    }
    __syncthreads();

    // omega = ( t * r )/ (t * t)
    // rho = (t * r ) /(||t|| * || r||)
    if (threadIdx.x == 0) {
        omega_shared_entry =
            t_r_dot_shared_entry /
            static_cast<ValueType>(norms_t_shared_entry * norms_t_shared_entry);
    } else if (threadIdx.x == config::warp_size) {
        rho_shared_entry =
            t_r_dot_shared_entry /
            static_cast<ValueType>(norms_t_shared_entry * norms_r_shared_entry);
    }
    __syncthreads();

    // if |rho| < kappa
    //      omega = omega * kappa / |rho|
    // end if
    if (threadIdx.x == 0) {
        if (abs(rho_shared_entry) < kappa) {
            omega_shared_entry *= kappa / abs(rho_shared_entry);
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_r_and_x_outer_loop(
    const int num_rows, const ValueType* const t_shared_entry,
    const ValueType* const v_shared_entry, const ValueType& omega_shared_entry,
    ValueType* const r_shared_entry, ValueType* const x_shared_entry)
{
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        r_shared_entry[row] -= omega_shared_entry * t_shared_entry[row];
        x_shared_entry[row] += omega_shared_entry * v_shared_entry[row];
    }
}


}  // unnamed namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(
    const int max_iter, const gko::remove_complex<ValueType> tol,
    const int subspace_dim, const gko::remove_complex<ValueType> kappa,
    const bool smoothing, const bool deterministic, LogType logger,
    PrecType prec_shared,
    const ValueType* const __restrict__ subspace_vecs_global,
    const BatchMatrixType a, const ValueType* const __restrict__ b,
    ValueType* const __restrict__ x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = a.num_batch;
    const int nrows = a.num_rows;

    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto warp_grp = group::tiled_partition<tile_size>(thread_block);

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
        extern __shared__ char local_mem_sh[];
        ValueType* const r_sh = reinterpret_cast<ValueType*>(local_mem_sh);
        ValueType* const t_sh = r_sh + nrows;
        ValueType* const v_sh = t_sh + nrows;
        ValueType* const x_sh = v_sh + nrows;
        ValueType* const xs_sh = x_sh + nrows;
        ValueType* const rs_sh = xs_sh + nrows;
        ValueType* const helper_sh = rs_sh + nrows;
        ValueType* const f_sh = helper_sh + nrows;
        ValueType* const c_sh = f_sh + subspace_dim;

        /* P = [ p_0 , p_1 , ... , p_(subspace_dim - 1) ] , subspace S is the
         * left null space of matrix P to store subspace defining vectors: p_i ,
         * i = 0, ..., subspace_dim -1 , we use a matrix named Subspace_vectors
         * storage:row-major order , subspace vectors
         * are stored in a single col. one after the other-(matrix
         * Subspace_vectors on paper)(to have efficient memory accesses).  And
         * to get p_i : that is ith subspace vector : p_i_entry{
         * &Subspace_vectors[i* Subspace_vectors_entry.stride * nrows],
         * Subspace_vectors_entry.stride , nrows, 1 }; So, effectively the cols.
         * are stored contiguously in memory one after the other as
         * Subspace_vectors_entry.stride = 1
         */
        ValueType* const subspace_vecs_sh = c_sh + subspace_dim;

        /* to store vectors: g_i , i = 0, ..., subspace_dim -1, we use matrix G
         * storage:row-major order , vectors corr. to each rhs
         * are stored in a single col. one after the other-(matrix G on
         * paper)(to have efficient memory accesses). And to get g_i : that is
         * ith  vector for each rhs: g_i_entry{  &G[i* G_entry.stride * nrows],
         * G_entry.stride , nrows, nrhs}; So if nrhs=1, effectively the cols.
         * are stored contiguously in memory one after the other.
         */
        ValueType* const G_sh = subspace_vecs_sh + nrows * subspace_dim;

        /* to store vectors: u_i , i = 0, ..., subspace_dim -1 , we use matrix U
         * storage:row-major order , vectors corr. to each rhs
         * are stored in a single col. one after the other-(matrix U on
         * paper)(to have efficient memory accesses). And to get u_i : that is
         * ith  vector for each rhs: u_i_entry{  &U[i* U_entry.stride * nrows],
         * U_entry.stride , nrows, nrhs}; So if nrhs=1, effectively the cols.
         * are stored contiguously in memory one after the other.
         */
        ValueType* const U_sh = G_sh + nrows * subspace_dim;

        /* storage:row-major ,  entry (i,j) for different RHSs are placed one
         * after the other in a row - when drawn on paper-(to have efficient
         * memory accesses), (and the same is true for actual storage as the
         * storage order is row-major) to get entry (i,j) for rhs: rhs_k ,
         * scalar_M_i_j_for_rhs_k =  M[M_entry.stride*i + j*nrhs  + rhs_k ]
         */
        ValueType* const M_sh = U_sh + nrows * subspace_dim;

        ValueType* const prec_work_sh = M_sh + subspace_dim * subspace_dim;
        ValueType* const tempv_sh =
            prec_work_sh + PrecType::dynamic_work_size(nrows, a.num_nnz);
        // This is one for each subspace vector
        const auto norms_tmp_sh =
            reinterpret_cast<real_type*>(tempv_sh + nrows);

        __shared__ uninitialized_array<ValueType, 1> omega_sh;
        __shared__ uninitialized_array<ValueType, 1> temp1_sh;
        __shared__ uninitialized_array<ValueType, 1> temp2_sh;
        __shared__ uninitialized_array<real_type, 1> norms_t_sh;
        __shared__ uninitialized_array<real_type, 1> norms_r_sh;
        __shared__ real_type norms_rhs_sh[1];
        __shared__ real_type norms_res_sh[1];

        const auto A_global_entry = gko::batch::batch_entry(a, ibatch);
        const auto b_global_entry =
            gko::batch::batch_entry_ptr(b, 1, nrows, ibatch);
        const auto x_global_entry =
            gko::batch::batch_entry_ptr(x, 1, nrows, ibatch);

        // generate preconditioner
        prec_shared.generate(ibatch, A_global_entry, prec_work_sh);

        // initialization
        // compute b norms
        // r = b - A*x
        // compute residual norms
        // initialize G, U with zeroes
        // M = Identity
        // xs = x and rs = r if smoothing is enabled
        // initialize (either random numbers or deterministically) and
        // orthonormalize Subspace_vectors omega = 1
        initialize(warp_grp, nrows, subspace_dim, deterministic, smoothing,
                   A_global_entry, b_global_entry, x_global_entry,
                   subspace_vecs_global, x_sh, r_sh, G_sh, U_sh, M_sh,
                   subspace_vecs_sh, xs_sh, rs_sh, omega_sh[0], norms_rhs_sh[0],
                   norms_res_sh[0], tempv_sh, norms_tmp_sh);
        __syncthreads();

        // stopping criterion object
        StopType stop(tol, norms_rhs_sh);

        int outer_iter = 0;

        for (; outer_iter < max_iter; outer_iter++) {
            if (stop.check_converged(norms_res_sh)) {
                break;
            }

            // f = HermitianTranspose(P) * r
            update_f(warp_grp, nrows, subspace_dim, subspace_vecs_sh, r_sh,
                     f_sh);
            __syncthreads();

            for (int k = 0; k < subspace_dim; k++) {
                ValueType* const u_k_sh = U_sh + k * nrows;
                ValueType* const g_k_sh = G_sh + k * nrows;

                // solve c from Mc = f (Lower Triangular solve)
                update_c(nrows, subspace_dim, M_sh, f_sh, c_sh);
                __syncthreads();

                // v = r - ( c(k) * g_k  +  c(k+1) * g_(k+1)  + ...  +
                // c(subspace_dim - 1) * g_(subspace_dim - 1))
                update_v(nrows, subspace_dim, G_sh, c_sh, r_sh, v_sh, k);
                __syncthreads();

                // helper = v
                single_copy(nrows, v_sh, helper_sh);
                __syncthreads();

                // v = precond * helper
                prec_shared.apply(nrows, helper_sh, v_sh);
                __syncthreads();

                // u_k = omega * v + (c(k) * u_k  +  c(k+1) * u_(k+1) + ...  +
                // c(subspace_dim - 1) * u_(subspace_dim - 1) )
                update_u_k(nrows, subspace_dim, omega_sh[0], c_sh, v_sh, U_sh,
                           k, helper_sh, u_k_sh);
                __syncthreads();

                // g_k = A * u_k
                single_matvec_kernel(A_global_entry, u_k_sh, g_k_sh);
                __syncthreads();

                // for i = 0 to k-1
                //     alpha = (p_i * g_k)/M(i,i)
                //     g_k = g_k - alpha * g_i
                //     u_k = u_k - alpha * u_i
                // end
                ValueType& alpha_sh = temp1_sh[0];
                update_g_k_and_u_k(warp_grp, nrows, subspace_dim, k, G_sh, U_sh,
                                   subspace_vecs_sh, M_sh, alpha_sh, g_k_sh,
                                   u_k_sh);
                __syncthreads();

                // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1
                update_M(warp_grp, nrows, subspace_dim, k, g_k_sh,
                         subspace_vecs_sh, M_sh);
                __syncthreads();

                // beta = f(k)/M(k,k)
                ValueType& beta_sh = temp1_sh[0];
                if (threadIdx.x == 0) {
                    beta_sh = f_sh[k] / M_sh[k * subspace_dim + k];
                }
                __syncthreads();

                // r = r - beta * g_k
                // x = x + beta * u_k
                update_r_and_x_inner_loop(nrows, g_k_sh, u_k_sh, beta_sh, r_sh,
                                          x_sh);
                __syncthreads();

                if (smoothing == true) {
                    ValueType& gamma_sh = temp2_sh[0];
                    smoothing_operation(warp_grp, nrows, x_sh, r_sh, gamma_sh,
                                        t_sh, xs_sh, rs_sh, norms_t_sh[0]);
                }

                // if k + 1 <= subspace_dim - 1
                //     f(i) = 0 , where i = 0,...,k
                //     f(i) = f(i) - beta * M(i,k) ,where i = k + 1, ... ,
                //     subspace_dim -1
                // end if
                if (k + 1 <= subspace_dim - 1) {
                    for (int row = threadIdx.x; row < subspace_dim;
                         row += blockDim.x) {
                        if (row <= k) {
                            f_sh[row] = zero<ValueType>();
                        } else {
                            f_sh[row] -= beta_sh * M_sh[row * subspace_dim + k];
                        }
                    }
                }
                __syncthreads();
            }
            __syncthreads();

            // v = precond * r
            prec_shared.apply(nrows, r_sh, v_sh);
            __syncthreads();

            // t = A *v
            single_matvec_kernel(A_global_entry, v_sh, t_sh);
            __syncthreads();

            // omega = ( t * r )/ (t * t)
            // rho = (t * r ) /(||t|| * || r||)
            // if |rho| < kappa
            //      omega = omega * kappa / |rho|
            // end if
            ValueType& t_r_dot_sh = temp1_sh[0];
            ValueType& rho_sh = temp2_sh[0];
            compute_omega(warp_grp, nrows, t_sh, r_sh, rho_sh, t_r_dot_sh,
                          norms_t_sh[0], norms_r_sh[0], omega_sh[0], kappa);
            __syncthreads();

            // r = r - omega * t
            // x = x + omega * v
            update_r_and_x_outer_loop(nrows, t_sh, v_sh, omega_sh[0], r_sh,
                                      x_sh);
            __syncthreads();

            if (smoothing == true) {
                ValueType& gamma_sh = temp2_sh[0];
                smoothing_operation(warp_grp, nrows, x_sh, r_sh, gamma_sh, t_sh,
                                    xs_sh, rs_sh, norms_t_sh[0]);
                __syncthreads();
                if (threadIdx.x / config::warp_size == 0) {
                    single_compute_norm2(warp_grp, nrows, rs_sh,
                                         norms_res_sh[0]);
                }
            } else {
                if (threadIdx.x / config::warp_size == 0) {
                    single_compute_norm2(warp_grp, nrows, r_sh,
                                         norms_res_sh[0]);
                }
            }
            __syncthreads();
        }

        if (smoothing == true) {
            for (int i = threadIdx.x; i < nrows; i += blockDim.x) {
                x_sh[i] = xs_sh[i];
                r_sh[i] = rs_sh[i];
            }
        }
        __syncthreads();

        logger.log_iteration(ibatch, outer_iter, norms_res_sh[0]);

        // copy x back to global memory
        single_copy(nrows, x_sh, x_global_entry);
    }
}
