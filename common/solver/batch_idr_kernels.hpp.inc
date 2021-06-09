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
__device__ __forceinline__ void orthonormalize_subspace_vectors(
    const gko::batch_dense::BatchEntry<ValueType>
        &Subspace_vectors_shared_entry,
    const int num_rows, const int subspace_dim,
    const gko::batch_dense::BatchEntry<ValueType>
        &temp_for_single_rhs_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &tmp_norms_shared_entry)
{
    using real_type = typename gko::remove_complex<ValueType>;

    for (int i = 0; i < subspace_dim; i++) {
        const gko::batch_dense::BatchEntry<ValueType> p_i_shared_entry{
            &Subspace_vectors_shared_entry
                 .values[i * num_rows * Subspace_vectors_shared_entry.stride],
            Subspace_vectors_shared_entry.stride, num_rows, 1};
        const gko::batch_dense::BatchEntry<ValueType> &w_i_shared_entry =
            temp_for_single_rhs_shared_entry;

        // w_i = p_i
        copy(gko::batch::to_const(p_i_shared_entry), w_i_shared_entry);

        for (int j = 0; j < i; j++) {
            // w_i = w_i - proj(p_i) on w_j that is w_i = w_i - (< w_j , p_i >
            // /< w_j , w_j > ) * w_j

            __syncthreads();

            const gko::batch_dense::BatchEntry<ValueType> w_j_shared_entry{
                &Subspace_vectors_shared_entry
                     .values[j * num_rows *
                             Subspace_vectors_shared_entry.stride],
                Subspace_vectors_shared_entry.stride, num_rows, 1};

            __shared__ UninitializedArray<ValueType, 1> mul_sh;
            const gko::batch_dense::BatchEntry<ValueType> mul_shared_entry{
                mul_sh, 1, 1, 1};
            compute_dot_product(gko::batch::to_const(w_j_shared_entry),
                                gko::batch::to_const(p_i_shared_entry),
                                mul_shared_entry);
            __syncthreads();

            if (threadIdx.x == 0) {
                mul_shared_entry.values[0] /=
                    static_cast<ValueType>(tmp_norms_shared_entry.values[j] *
                                           tmp_norms_shared_entry.values[j]);
                mul_shared_entry.values[0] *= -one<ValueType>();
            }
            __syncthreads();

            add_scaled(gko::batch::to_const(mul_shared_entry),
                       gko::batch::to_const(w_j_shared_entry),
                       w_i_shared_entry);
        }

        __syncthreads();

        // p_i = w_i
        copy(gko::batch::to_const(w_i_shared_entry), p_i_shared_entry);
        compute_norm2(gko::batch::to_const(w_i_shared_entry),
                      gko::batch_dense::BatchEntry<real_type>{
                          &tmp_norms_shared_entry.values[i], 1, 1, 1});
        __syncthreads();
    }


    // e_k = w_k / || w_k ||  for k = 0, 1, ..., subspace_dim -1

    for (int tid = threadIdx.x; tid < subspace_dim * num_rows;
         tid += blockDim.x) {
        const int row_index = tid % num_rows;
        const int vec_index = tid / num_rows;
        Subspace_vectors_shared_entry
            .values[vec_index * num_rows *
                        Subspace_vectors_shared_entry.stride +
                    row_index * Subspace_vectors_shared_entry.stride] /=
            static_cast<ValueType>(tmp_norms_shared_entry.values[vec_index]);
    }
}

template <typename BatchMatrixType_entry, typename ValueType>
__device__ __forceinline__ void initialize(
    const BatchMatrixType_entry &A_global_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &b_global_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_global_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &G_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &U_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &M_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType>
        &Subspace_vectors_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType>
        &Subspace_vectors_global_entry,
    const bool deterministic,
    const gko::batch_dense::BatchEntry<ValueType> &xs_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rs_shared_entry,
    const bool smoothing,
    const gko::batch_dense::BatchEntry<ValueType> &omega_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &rhs_norms_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &res_norms_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType>
        &temp_for_single_rhs_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &tmp_norms_shared_entry)
{
    using real_type = gko::remove_complex<ValueType>();

    const int subspace_dim = M_shared_entry.num_rows;
    const int nrhs = x_global_entry.num_rhs;
    const int nrows = x_global_entry.num_rows;

    // Compute norms of rhs
    compute_norm2<ValueType>(b_global_entry, rhs_norms_shared_entry);


    // copy x from global to shared memory
    copy(x_global_entry, x_shared_entry);
    copy(b_global_entry, r_shared_entry);


    __syncthreads();


    // r = b - A*x
    advanced_spmv_kernel(static_cast<ValueType>(-1.0), A_global_entry,
                         gko::batch::to_const(x_shared_entry),
                         static_cast<ValueType>(1.0), r_shared_entry);
    __syncthreads();

    // compute residual norms
    compute_norm2<ValueType>(gko::batch::to_const(r_shared_entry),
                             res_norms_shared_entry);


    // omega = 1
    for (int rhs = threadIdx.x; rhs < nrhs; rhs += blockDim.x) {
        omega_shared_entry.values[rhs] = one<ValueType>();
    }


    // M = Identity
    for (int tid = threadIdx.x; tid < subspace_dim * nrhs * subspace_dim;
         tid += blockDim.x) {
        const int row_index = tid / (subspace_dim * nrhs);
        const int col_index = (tid % (subspace_dim * nrhs)) / nrhs;
        const int rhs_index = (tid % (subspace_dim * nrhs)) % nrhs;

        if (row_index == col_index) {
            M_shared_entry.values[row_index * M_shared_entry.stride +
                                  col_index * nrhs + rhs_index] =
                one<ValueType>();
        } else {
            M_shared_entry.values[row_index * M_shared_entry.stride +
                                  col_index * nrhs + rhs_index] =
                zero<ValueType>();
        }
    }

    // G = zero
    // U = zero
    for (int tid = threadIdx.x; tid < nrows * subspace_dim * nrhs;
         tid += blockDim.x) {
        const int vec_index = tid / (nrhs * nrows);
        const int row_index = (tid % (nrhs * nrows)) / nrhs;
        const int rhs_index = (tid % (nrhs * nrows)) % nrhs;

        G_shared_entry.values[vec_index * nrows * G_shared_entry.stride +
                              row_index * G_shared_entry.stride + rhs_index] =
            zero<ValueType>();
        U_shared_entry.values[vec_index * nrows * U_shared_entry.stride +
                              row_index * U_shared_entry.stride + rhs_index] =
            zero<ValueType>();
    }

    if (smoothing == true) {
        // xs = x
        // rs = r
        copy(gko::batch::to_const(x_shared_entry), xs_shared_entry);
        copy(gko::batch::to_const(r_shared_entry), rs_shared_entry);
    }

    curandState_t state;
    curand_init(0, threadIdx.x, 0, &state);  // TODO: Have a random seed value
    /* curand_init (
        unsigned long long seed, unsigned long long sequence,
        unsigned long long offset, curandState_t *state)  */


    // initialize Subspace_vectors
    for (int li = threadIdx.x; li < nrows * subspace_dim; li += blockDim.x) {
        const int vec_index = li / nrows;
        const int row_index = li % nrows;

        if (deterministic == true) {
            Subspace_vectors_shared_entry
                .values[vec_index * nrows *
                            Subspace_vectors_shared_entry.stride +
                        row_index * Subspace_vectors_shared_entry.stride] =
                Subspace_vectors_global_entry
                    .values[vec_index * nrows *
                                Subspace_vectors_global_entry.stride +
                            row_index * Subspace_vectors_global_entry.stride];
        } else {
            ValueType val = zero<ValueType>();


            if (is_complex<ValueType>() == true) {
                // TODO: Random generation of complex number

                // real_type re = curand_normal(&state);
                // real_type im = curand_normal(&state);
                // val = ValueType{ re,im };

                val = static_cast<ValueType>(curand_normal(&state));
            } else {
                val = static_cast<ValueType>(curand_normal(&state));
            }

            Subspace_vectors_shared_entry
                .values[vec_index * nrows *
                            Subspace_vectors_shared_entry.stride +
                        row_index * Subspace_vectors_shared_entry.stride] = val;
        }
    }

    __syncthreads();

    orthonormalize_subspace_vectors(
        Subspace_vectors_shared_entry, nrows, subspace_dim,
        temp_for_single_rhs_shared_entry, tmp_norms_shared_entry);
}


template <typename ValueType>
__device__ __forceinline__ void update_f(
    const gko::batch_dense::BatchEntry<const ValueType>
        &Subspace_vectors_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_shared_entry,
    const size_type subspace_dim,
    const gko::batch_dense::BatchEntry<ValueType> &f_shared_entry,
    const uint32 &converged)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();

    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);


    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);


    for (int vec_and_rhs_combination = subwarp_grp_id;
         vec_and_rhs_combination < subspace_dim * r_shared_entry.num_rhs;
         vec_and_rhs_combination += num_subwarp_grps_per_block) {
        const int vec = vec_and_rhs_combination / r_shared_entry.num_rhs;

        const int rhs = vec_and_rhs_combination % r_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << rhs);

        if (conv) {
            continue;
        }

        ValueType val = zero<ValueType>();

        for (int r = subwarp_grp.thread_rank(); r < r_shared_entry.num_rows;
             r += subwarp_grp.size()) {
            val += conj(Subspace_vectors_shared_entry
                            .values[vec * r_shared_entry.num_rows *
                                        Subspace_vectors_shared_entry.stride +
                                    r * Subspace_vectors_shared_entry.stride]) *
                   r_shared_entry.values[r * r_shared_entry.stride + rhs];
        }

        // subwarp_grp level reduction
        for (int i = subwarp_grp.size() / 2; i > 0; i /= 2) {
            val += subwarp_grp.shfl_down(val, i);
        }

        if (subwarp_grp.thread_rank() == 0) {
            f_shared_entry.values[vec * f_shared_entry.stride + rhs] = val;
        }
    }
}

template <typename ValueType>
__device__ __forceinline__ void update_c(
    const gko::batch_dense::BatchEntry<const ValueType> &M_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &f_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &c_shared_entry,
    const uint32 &converged)
{
    const auto subspace_dim = M_shared_entry.num_rows;
    const auto nrhs = f_shared_entry.num_rhs;
    // upper triangular solve
    // solve top to bottom


    for (int rhs_index = threadIdx.x; rhs_index < nrhs;
         rhs_index += blockDim.x) {
        const uint32 conv = converged & (1 << rhs_index);

        if (conv) {
            continue;
        }

        for (int row_index = 0; row_index < subspace_dim; row_index++) {
            ValueType temp_sum = zero<ValueType>();

            for (int col_index = 0; col_index < row_index; col_index++) {
                temp_sum +=
                    M_shared_entry.values[row_index * M_shared_entry.stride +
                                          col_index * nrhs + rhs_index] *
                    c_shared_entry
                        .values[col_index * c_shared_entry.stride + rhs_index];
            }

            c_shared_entry
                .values[row_index * c_shared_entry.stride + rhs_index] =
                (f_shared_entry
                     .values[row_index * f_shared_entry.stride + rhs_index] -
                 temp_sum) /
                M_shared_entry.values[row_index * M_shared_entry.stride +
                                      row_index * nrhs + rhs_index];
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_v(
    const gko::batch_dense::BatchEntry<const ValueType> &G_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &c_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &v_shared_entry,
    const size_type k, const uint32 &converged)
{
    const auto subspace_dim = c_shared_entry.num_rows;


    for (int li = threadIdx.x;
         li < v_shared_entry.num_rows * v_shared_entry.num_rhs;
         li += blockDim.x) {
        const int row_index = li / v_shared_entry.num_rhs;
        const int rhs_index = li % v_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << rhs_index);

        if (conv) {
            continue;
        }

        v_shared_entry.values[row_index * v_shared_entry.stride + rhs_index] =
            r_shared_entry
                .values[row_index * r_shared_entry.stride + rhs_index];

        for (int vec_index = k; vec_index < subspace_dim; vec_index++) {
            v_shared_entry
                .values[row_index * v_shared_entry.stride + rhs_index] -=
                c_shared_entry
                    .values[vec_index * c_shared_entry.stride + rhs_index] *
                G_shared_entry
                    .values[vec_index * r_shared_entry.num_rows *
                                G_shared_entry.stride +
                            row_index * G_shared_entry.stride + rhs_index];
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_u_k(
    const gko::batch_dense::BatchEntry<const ValueType> &omega_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &c_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &U_shared_entry,
    const size_type k,
    const gko::batch_dense::BatchEntry<ValueType> &helper_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &u_k_shared_entry,
    const uint32 &converged)
{
    const auto subspace_dim = c_shared_entry.num_rows;
    const auto nrows = v_shared_entry.num_rows;
    const auto nrhs = v_shared_entry.num_rhs;

    for (int li = threadIdx.x;
         li < helper_shared_entry.num_rows * helper_shared_entry.num_rhs;
         li += blockDim.x) {
        const int row_index = li / nrhs;
        const int rhs_index = li % nrhs;

        const uint32 conv = converged & (1 << rhs_index);

        if (conv) {
            continue;
        }

        helper_shared_entry
            .values[row_index * helper_shared_entry.stride + rhs_index] =
            omega_shared_entry.values[rhs_index] *
            v_shared_entry
                .values[row_index * v_shared_entry.stride + rhs_index];

        for (int vec_index = k; vec_index < subspace_dim; vec_index++) {
            helper_shared_entry
                .values[row_index * helper_shared_entry.stride + rhs_index] +=
                c_shared_entry
                    .values[vec_index * c_shared_entry.stride + rhs_index] *
                U_shared_entry
                    .values[vec_index * nrows * U_shared_entry.stride +
                            row_index * U_shared_entry.stride + rhs_index];
        }
    }

    __syncthreads();

    copy(gko::batch::to_const(helper_shared_entry), u_k_shared_entry,
         converged);
}


template <typename ValueType>
__device__ __forceinline__ void update_g_k_and_u_k(
    const size_type k,
    const gko::batch_dense::BatchEntry<const ValueType> &G_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &U_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType>
        &Subspace_vectors_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &M_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &g_k_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &u_k_shared_entry,
    const uint32 &converged)
{
    const auto nrows = g_k_shared_entry.num_rows;
    const auto nrhs = g_k_shared_entry.num_rhs;


    for (int i = 0; i <= static_cast<int>(k) - 1; i++) {
        // alpha = (p_i * g_k)/M(i,i)

        constexpr auto tile_size = config::warp_size;

        auto thread_block = group::this_thread_block();

        auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);

        const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);

        const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);


        for (int rhs_index = subwarp_grp_id;
             rhs_index < g_k_shared_entry.num_rhs;
             rhs_index += num_subwarp_grps_per_block) {
            const uint32 conv = converged & (1 << rhs_index);

            if (conv) {
                continue;
            }

            ValueType val = zero<ValueType>();

            for (int r = subwarp_grp.thread_rank();
                 r < g_k_shared_entry.num_rows; r += subwarp_grp.size()) {
                val +=
                    conj(
                        Subspace_vectors_shared_entry
                            .values[i * g_k_shared_entry.num_rows *
                                        Subspace_vectors_shared_entry.stride +
                                    r * Subspace_vectors_shared_entry.stride]) *
                    g_k_shared_entry
                        .values[r * g_k_shared_entry.stride + rhs_index];
            }

            // subwarp_grp level reduction
            for (int j = subwarp_grp.size() / 2; j > 0; j /= 2) {
                val += subwarp_grp.shfl_down(val, j);
            }

            if (subwarp_grp.thread_rank() == 0) {
                alpha_shared_entry.values[rhs_index] =
                    val / M_shared_entry.values[i * M_shared_entry.stride +
                                                i * nrhs + rhs_index];
            }
        }

        __syncthreads();

        // g_k = g_k - alpha * g_i
        // u_k = u_k - alpha * u_i
        for (int li = threadIdx.x;
             li < g_k_shared_entry.num_rows * g_k_shared_entry.num_rhs;
             li += blockDim.x) {
            const int row_index = li / nrhs;
            const int rhs_index = li % nrhs;

            const uint32 conv = converged & (1 << rhs_index);

            if (conv) {
                continue;
            }

            const ValueType alpha = alpha_shared_entry.values[rhs_index];

            g_k_shared_entry
                .values[row_index * g_k_shared_entry.stride + rhs_index] -=
                alpha *
                G_shared_entry
                    .values[i * nrows * G_shared_entry.stride +
                            row_index * G_shared_entry.stride + rhs_index];

            u_k_shared_entry
                .values[row_index * u_k_shared_entry.stride + rhs_index] -=
                alpha *
                U_shared_entry
                    .values[i * nrows * U_shared_entry.stride +
                            row_index * U_shared_entry.stride + rhs_index];
        }

        __syncthreads();
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_M(
    const gko::batch_dense::BatchEntry<const ValueType> &g_k_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType>
        &Subspace_vectors_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &M_shared_entry,
    const size_type k, const uint32 &converged)
{
    const auto subspace_dim = M_shared_entry.num_rows;
    const auto nrhs = g_k_shared_entry.num_rhs;

    // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1

    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();

    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);


    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);

    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1

    for (int i_and_rhs_combination = subwarp_grp_id;
         i_and_rhs_combination < (subspace_dim - k) * nrhs;
         i_and_rhs_combination += num_subwarp_grps_per_block) {
        const int i = (i_and_rhs_combination / nrhs) + k;
        const int rhs = i_and_rhs_combination % nrhs;

        const uint32 conv = converged & (1 << rhs);

        if (conv) {
            continue;
        }

        ValueType val = zero<ValueType>();

        for (int r = subwarp_grp.thread_rank(); r < g_k_shared_entry.num_rows;
             r += subwarp_grp.size()) {
            val += conj(Subspace_vectors_shared_entry
                            .values[i * g_k_shared_entry.num_rows *
                                        Subspace_vectors_shared_entry.stride +
                                    r * Subspace_vectors_shared_entry.stride]) *
                   g_k_shared_entry.values[r * g_k_shared_entry.stride + rhs];
        }

        // subwarp_grp level reduction
        for (int j = subwarp_grp.size() / 2; j > 0; j /= 2) {
            val += subwarp_grp.shfl_down(val, j);
        }

        if (subwarp_grp.thread_rank() == 0) {
            M_shared_entry.values[i * M_shared_entry.stride + k * nrhs + rhs] =
                val;
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_r_and_x_inner_loop(
    const gko::batch_dense::BatchEntry<const ValueType> &g_k_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &u_k_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &beta_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < g_k_shared_entry.num_rows * g_k_shared_entry.num_rhs;
         li += blockDim.x) {
        const int row = li / g_k_shared_entry.num_rhs;
        const int rhs = li % g_k_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << rhs);

        if (conv) {
            continue;
        }

        const ValueType beta = beta_shared_entry.values[rhs];

        r_shared_entry.values[row * r_shared_entry.stride + rhs] -=
            beta * g_k_shared_entry.values[row * g_k_shared_entry.stride + rhs];

        x_shared_entry.values[row * x_shared_entry.stride + rhs] +=
            beta * u_k_shared_entry.values[row * u_k_shared_entry.stride + rhs];
    }
}


template <typename ValueType>
__device__ __forceinline__ void smoothing_operation(
    const gko::batch_dense::BatchEntry<const ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &gamma_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &t_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &xs_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rs_shared_entry,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>
        &norms_t_shared_entry,
    const uint32 &converged)
{
    // t = rs - r
    for (int li = threadIdx.x;
         li < t_shared_entry.num_rows * t_shared_entry.num_rhs;
         li += blockDim.x) {
        const int row = li / t_shared_entry.num_rhs;
        const int rhs = li % t_shared_entry.num_rhs;

        t_shared_entry.values[row * t_shared_entry.stride + rhs] =
            rs_shared_entry.values[row * rs_shared_entry.stride + rhs] -
            r_shared_entry.values[row * r_shared_entry.stride + rhs];
    }

    __syncthreads();


    // gamma = (t * rs)/(t * t)
    compute_dot_product(gko::batch::to_const(t_shared_entry),
                        gko::batch::to_const(rs_shared_entry),
                        gamma_shared_entry, converged);
    compute_norm2(gko::batch::to_const(t_shared_entry), norms_t_shared_entry,
                  converged);

    __syncthreads();

    for (int rhs = threadIdx.x; rhs < t_shared_entry.num_rhs;
         rhs += blockDim.x) {
        gamma_shared_entry.values[rhs] /=
            static_cast<ValueType>(norms_t_shared_entry.values[rhs] *
                                   norms_t_shared_entry.values[rhs]);
    }

    __syncthreads();


    // rs = rs - gamma*(rs - r)
    // xs = xs - gamma*(xs - x)
    for (int li = threadIdx.x;
         li < rs_shared_entry.num_rows * rs_shared_entry.num_rhs;
         li += blockDim.x) {
        const int row = li / rs_shared_entry.num_rhs;
        const int rhs = li % rs_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << rhs);

        if (conv) {
            continue;
        }

        rs_shared_entry.values[row * rs_shared_entry.stride + rhs] =
            (one<ValueType>() - gamma_shared_entry.values[rhs]) *
                rs_shared_entry.values[row * rs_shared_entry.stride + rhs] +
            gamma_shared_entry.values[rhs] *
                r_shared_entry.values[row * r_shared_entry.stride + rhs];

        xs_shared_entry.values[row * xs_shared_entry.stride + rhs] =
            (one<ValueType>() - gamma_shared_entry.values[rhs]) *
                xs_shared_entry.values[row * xs_shared_entry.stride + rhs] +
            gamma_shared_entry.values[rhs] *
                x_shared_entry.values[row * x_shared_entry.stride + rhs];
    }
}

template <typename ValueType>
__device__ __forceinline__ void compute_omega(
    const gko::batch_dense::BatchEntry<const ValueType> &t_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rho_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &t_r_dot_shared_entry,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>
        &norms_t_shared_entry,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>
        &norms_r_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_shared_entry,
    const gko::remove_complex<ValueType> kappa, const uint32 &converged)
{
    compute_dot_product(gko::batch::to_const(t_shared_entry),
                        gko::batch::to_const(r_shared_entry),
                        t_r_dot_shared_entry, converged);

    compute_norm2(gko::batch::to_const(t_shared_entry), norms_t_shared_entry,
                  converged);


    compute_norm2(gko::batch::to_const(r_shared_entry), norms_r_shared_entry,
                  converged);

    __syncthreads();


    // omega = ( t * r )/ (t * t)
    // rho = (t * r ) /(||t|| * || r||)
    for (int rhs = threadIdx.x; rhs < rho_shared_entry.num_rhs;
         rhs += blockDim.x) {
        const uint32 conv = converged & (1 << rhs);

        if (conv) {
            continue;
        }

        omega_shared_entry.values[rhs] =
            t_r_dot_shared_entry.values[rhs] /
            static_cast<ValueType>(norms_t_shared_entry.values[rhs] *
                                   norms_t_shared_entry.values[rhs]);

        rho_shared_entry.values[rhs] =
            t_r_dot_shared_entry.values[rhs] /
            static_cast<ValueType>(norms_t_shared_entry.values[rhs] *
                                   norms_r_shared_entry.values[rhs]);
    }

    __syncthreads();


    // if |rho| < kappa
    //      omega = omega * kappa / |rho|
    // end if
    for (int rhs = threadIdx.x; rhs < rho_shared_entry.num_rhs;
         rhs += blockDim.x) {
        const uint32 conv = converged & (1 << rhs);

        if (conv) {
            continue;
        }

        if (abs(rho_shared_entry.values[rhs]) < kappa) {
            omega_shared_entry.values[rhs] *=
                kappa / abs(rho_shared_entry.values[rhs]);
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_r_and_x_outer_loop(
    const gko::batch_dense::BatchEntry<const ValueType> &t_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < t_shared_entry.num_rows * t_shared_entry.num_rhs;
         li += blockDim.x) {
        const int row = li / t_shared_entry.num_rhs;
        const int rhs = li % t_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << rhs);

        if (conv) {
            continue;
        }

        const ValueType omega = omega_shared_entry.values[rhs];

        r_shared_entry.values[row * r_shared_entry.stride + rhs] -=
            omega * t_shared_entry.values[row * t_shared_entry.stride + rhs];

        x_shared_entry.values[row * x_shared_entry.stride + rhs] +=
            omega * v_shared_entry.values[row * v_shared_entry.stride + rhs];
    }
}


}  // unnamed namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(
    const int max_iter, const gko::remove_complex<ValueType> tol,
    const int subspace_dim, const gko::remove_complex<ValueType> kappa,
    const bool smoothing, const bool deterministic, LogType logger,
    PrecType prec_shared, const BatchMatrixType a,
    const gko::batch_dense::UniformBatch<const ValueType> left,
    const gko::batch_dense::UniformBatch<const ValueType> right,
    const gko::batch_dense::UniformBatch<ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> x,
    const gko::batch_dense::BatchEntry<const ValueType>
        Subspace_vectors_global_entry)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = a.num_batch;
    const int nrows = a.num_rows;
    const int nrhs = b.num_rhs;

    assert(batch_config<ValueType>::max_num_rhs >=
           nrhs);  // required for static allocation in stopping criterion

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
        extern __shared__ char local_mem_sh[];
        ValueType *const r_sh = reinterpret_cast<ValueType *>(local_mem_sh);
        ValueType *const t_sh = r_sh + nrows * nrhs;
        ValueType *const v_sh = t_sh + nrows * nrhs;
        ValueType *const x_sh = v_sh + nrows * nrhs;
        ValueType *const xs_sh = x_sh + nrows * nrhs;
        ValueType *const rs_sh = xs_sh + nrows * nrhs;
        ValueType *const helper_sh = rs_sh + nrows * nrhs;
        ValueType *const f_sh = helper_sh + nrows * nrhs;
        ValueType *const c_sh = f_sh + subspace_dim * nrhs;
        ValueType *const Subspace_vectors_sh = c_sh + subspace_dim * nrhs;
        ValueType *const G_sh = Subspace_vectors_sh + nrows * subspace_dim;
        ValueType *const U_sh = G_sh + nrows * subspace_dim * nrhs;
        ValueType *const M_sh = U_sh + nrows * subspace_dim * nrhs;
        ValueType *const prec_work_sh =
            M_sh + subspace_dim * subspace_dim * nrhs;
        ValueType *const temp_for_single_rhs_sh =
            prec_work_sh + PrecType::dynamic_work_size(nrows, a.num_nnz);
        ValueType *const omega_sh = temp_for_single_rhs_sh + nrows;
        ValueType *const temp1_sh = omega_sh + nrhs;
        ValueType *const temp2_sh = temp1_sh + nrhs;
        real_type *const norms_t_sh =
            reinterpret_cast<real_type *>(temp2_sh + nrhs);
        real_type *const norms_r_sh = norms_t_sh + nrhs;
        real_type *const norms_rhs_sh = norms_r_sh + nrhs;
        real_type *const norms_res_sh = norms_rhs_sh + nrhs;
        real_type *const norms_tmp_sh = norms_res_sh + nrhs;


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
            x_sh, static_cast<size_type>(nrhs), nrows,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> r_shared_entry{
            r_sh, static_cast<size_type>(nrhs), nrows,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> t_shared_entry{
            t_sh, static_cast<size_type>(nrhs), nrows,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> v_shared_entry{
            v_sh, static_cast<size_type>(nrhs), nrows,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> xs_shared_entry{
            xs_sh, static_cast<size_type>(nrhs), nrows,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> rs_shared_entry{
            rs_sh, static_cast<size_type>(nrhs), nrows,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> helper_shared_entry{
            helper_sh, static_cast<size_type>(nrhs), nrows,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> f_shared_entry{
            f_sh, static_cast<size_type>(nrhs), subspace_dim,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> c_shared_entry{
            c_sh, static_cast<size_type>(nrhs), subspace_dim,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix


        // P = [ p_0 , p_1 , ... , p_(subspace_dim - 1) ] , subspace S is the
        // left null space of matrix P to store subspace defining vectors: p_i ,
        // i = 0, ..., subspace_dim -1 , we use a matrix named Subspace_vectors
        const gko::batch_dense::BatchEntry<ValueType>
            Subspace_vectors_shared_entry{Subspace_vectors_sh, 1,
                                          nrows * subspace_dim, 1};
        // storage:row-major order , subspace vectors
        // are stored in a single col. one after the other-(matrix
        // Subspace_vectors on paper)(to have efficient memory accesses).  And
        // to get p_i : that is ith subspace vector : p_i_entry{
        // &Subspace_vectors[i* Subspace_vectors_entry.stride * nrows],
        // Subspace_vectors_entry.stride , nrows, 1 }; So, effectively the cols.
        // are stored contiguously in memory one after the other as
        // Subspace_vectors_entry.stride = 1


        // to store vectors: g_i , i = 0, ..., subspace_dim -1, we use matrix G
        const gko::batch_dense::BatchEntry<ValueType> G_shared_entry{
            G_sh, static_cast<size_type>(nrhs), nrows * subspace_dim, nrhs};
        // storage:row-major order , vectors corr. to each rhs
        // are stored in a single col. one after the other-(matrix G on
        // paper)(to have efficient memory accesses). And to get g_i : that is
        // ith  vector for each rhs: g_i_entry{  &G[i* G_entry.stride * nrows],
        // G_entry.stride , nrows, nrhs}; So if nrhs=1, effectively the cols.
        // are stored contiguously in memory one after the other.


        // to store vectors: u_i , i = 0, ..., subspace_dim -1 , we use matrix U
        const gko::batch_dense::BatchEntry<ValueType> U_shared_entry{
            U_sh, static_cast<size_type>(nrhs), nrows * subspace_dim, nrhs};
        // storage:row-major order , vectors corr. to each rhs
        // are stored in a single col. one after the other-(matrix U on
        // paper)(to have efficient memory accesses). And to get u_i : that is
        // ith  vector for each rhs: u_i_entry{  &U[i* U_entry.stride * nrows],
        // U_entry.stride , nrows, nrhs}; So if nrhs=1, effectively the cols.
        // are stored contiguously in memory one after the other.


        const gko::batch_dense::BatchEntry<ValueType> M_shared_entry{
            M_sh, static_cast<size_type>(nrhs * subspace_dim), subspace_dim,
            nrhs * subspace_dim};
        // storage:row-major ,  entry (i,j) for different RHSs are placed one
        // after the other in a row - when drawn on paper-(to have efficient
        // memory accesses), (and the same is true for actual storage as the
        // storage order is row-major) to get entry (i,j) for rhs: rhs_k ,
        // scalar_M_i_j_for_rhs_k =  M[M_entry.stride*i + j*nrhs  + rhs_k ]


        const gko::batch_dense::BatchEntry<ValueType>
            temp_for_single_rhs_shared_entry{
                temp_for_single_rhs_sh, static_cast<size_type>(1), nrows, 1};

        const gko::batch_dense::BatchEntry<ValueType> omega_shared_entry{
            omega_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> temp1_shared_entry{
            temp1_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> temp2_shared_entry{
            temp2_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> t_norms_shared_entry{
            norms_t_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<real_type> r_norms_shared_entry{
            norms_r_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> rhs_norms_shared_entry{
            norms_rhs_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> res_norms_shared_entry{
            norms_res_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> tmp_norms_shared_entry{
            norms_tmp_sh, static_cast<size_type>(subspace_dim), 1,
            static_cast<int>(subspace_dim)};


        // generate preconditioner
        prec_shared.generate(A_global_entry, prec_work_sh);


        // initialization
        // compute b norms
        // r = b - A*x
        // compute residual norms
        // initialize G, U with zeroes
        // M = Identity
        // xs = x and rs = r if smoothing is enabled
        // initialize (either random numbers or deterministically) and
        // orthonormalize Subspace_vectors omega = 1
        initialize(
            A_global_entry, b_global_entry,
            gko::batch::to_const(x_global_entry), x_shared_entry,
            r_shared_entry, G_shared_entry, U_shared_entry, M_shared_entry,
            Subspace_vectors_shared_entry, Subspace_vectors_global_entry,
            deterministic, xs_shared_entry, rs_shared_entry, smoothing,
            omega_shared_entry, rhs_norms_shared_entry, res_norms_shared_entry,
            temp_for_single_rhs_shared_entry, tmp_norms_shared_entry);


        __syncthreads();

        // stopping criterion object
        StopType stop(nrhs, max_iter, tol, rhs_norms_shared_entry.values,
                      converged);

        int outer_iter = -1;


        while (1) {
            outer_iter++;


            __syncthreads();

            bool all_converged =
                stop.check_converged(outer_iter, res_norms_shared_entry.values,
                                     {NULL, 0, 0, 0}, converged);


            logger.log_iteration(ibatch, outer_iter,
                                 res_norms_shared_entry.values, converged);


            __syncthreads();

            if (all_converged) {
                break;
            }

            // f = HermitianTranspose(P) * r
            update_f(gko::batch::to_const(Subspace_vectors_shared_entry),
                     gko::batch::to_const(r_shared_entry), subspace_dim,
                     f_shared_entry, converged);

            __syncthreads();


            for (int k = 0; k < subspace_dim; k++) {
                const gko::batch_dense::BatchEntry<ValueType> u_k_shared_entry{
                    &U_shared_entry.values[k * nrows * U_shared_entry.stride],
                    U_shared_entry.stride, nrows, U_shared_entry.num_rhs};

                const gko::batch_dense::BatchEntry<ValueType> g_k_shared_entry{
                    &G_shared_entry.values[k * nrows * G_shared_entry.stride],
                    G_shared_entry.stride, nrows, G_shared_entry.num_rhs};


                // solve c from Mc = f (Lower Triangular solve)
                update_c(gko::batch::to_const(M_shared_entry),
                         gko::batch::to_const(f_shared_entry), c_shared_entry,
                         converged);

                __syncthreads();

                // v = r - ( c(k) * g_k  +  c(k+1) * g_(k+1)  + ...  +
                // c(subspace_dim - 1) * g_(subspace_dim - 1))
                update_v(gko::batch::to_const(G_shared_entry),
                         gko::batch::to_const(c_shared_entry),
                         gko::batch::to_const(r_shared_entry), v_shared_entry,
                         k, converged);

                __syncthreads();

                // helper = v
                copy(gko::batch::to_const(v_shared_entry), helper_shared_entry,
                     converged);

                __syncthreads();

                // v = precond * helper
                prec_shared.apply(gko::batch::to_const(helper_shared_entry),
                                  v_shared_entry);

                __syncthreads();

                // u_k = omega * v + (c(k) * u_k  +  c(k+1) * u_(k+1) + ...  +
                // c(subspace_dim - 1) * u_(subspace_dim - 1) )
                update_u_k(gko::batch::to_const(omega_shared_entry),
                           gko::batch::to_const(c_shared_entry),
                           gko::batch::to_const(v_shared_entry),
                           gko::batch::to_const(U_shared_entry), k,
                           helper_shared_entry, u_k_shared_entry, converged);

                __syncthreads();


                // g_k = A * u_k
                spmv_kernel(A_global_entry,
                            gko::batch::to_const(u_k_shared_entry),
                            g_k_shared_entry);

                __syncthreads();


                // for i = 0 to k-1
                //     alpha = (p_i * g_k)/M(i,i)
                //     g_k = g_k - alpha * g_i
                //     u_k = u_k - alpha * u_i
                // end
                const gko::batch_dense::BatchEntry<ValueType>
                    &alpha_shared_entry = temp1_shared_entry;
                update_g_k_and_u_k(
                    k, gko::batch::to_const(G_shared_entry),
                    gko::batch::to_const(U_shared_entry),
                    gko::batch::to_const(Subspace_vectors_shared_entry),
                    gko::batch::to_const(M_shared_entry), alpha_shared_entry,
                    g_k_shared_entry, u_k_shared_entry, converged);

                __syncthreads();

                // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1
                update_M(gko::batch::to_const(g_k_shared_entry),
                         gko::batch::to_const(Subspace_vectors_shared_entry),
                         M_shared_entry, k, converged);

                __syncthreads();

                // beta = f(k)/M(k,k)
                const gko::batch_dense::BatchEntry<ValueType>
                    &beta_shared_entry = temp1_shared_entry;
                for (int rhs = threadIdx.x; rhs < nrhs; rhs += blockDim.x) {
                    const uint32 conv = converged & (1 << rhs);

                    if (conv) {
                        continue;
                    }

                    beta_shared_entry.values[rhs] =
                        f_shared_entry.values[k * f_shared_entry.stride + rhs] /
                        M_shared_entry
                            .values[k * M_shared_entry.stride + k * nrhs + rhs];
                }

                __syncthreads();


                // r = r - beta * g_k
                // x = x + beta * u_k
                update_r_and_x_inner_loop(
                    gko::batch::to_const(g_k_shared_entry),
                    gko::batch::to_const(u_k_shared_entry),
                    gko::batch::to_const(beta_shared_entry), r_shared_entry,
                    x_shared_entry, converged);


                __syncthreads();


                if (smoothing == true) {
                    const gko::batch_dense::BatchEntry<ValueType>
                        &gamma_shared_entry = temp2_shared_entry;
                    smoothing_operation(gko::batch::to_const(x_shared_entry),
                                        gko::batch::to_const(r_shared_entry),
                                        gamma_shared_entry, t_shared_entry,
                                        xs_shared_entry, rs_shared_entry,
                                        t_norms_shared_entry, converged);
                }


                // if k + 1 <= subspace_dim - 1
                //     f(i) = 0 , where i = 0,...,k
                //     f(i) = f(i) - beta * M(i,k) ,where i = k + 1, ... ,
                //     subspace_dim -1
                // end if
                if (k + 1 <= subspace_dim - 1) {
                    for (int li = threadIdx.x;
                         li < f_shared_entry.num_rows * f_shared_entry.num_rhs;
                         li += blockDim.x) {
                        int row = li / f_shared_entry.num_rhs;
                        int rhs = li % f_shared_entry.num_rhs;

                        if (row <= k) {
                            f_shared_entry
                                .values[row * f_shared_entry.stride + rhs] =
                                zero<ValueType>();
                        } else {
                            f_shared_entry
                                .values[row * f_shared_entry.stride + rhs] -=
                                beta_shared_entry.values[rhs] *
                                M_shared_entry
                                    .values[row * M_shared_entry.stride +
                                            k * nrhs + rhs];
                        }
                    }
                }

                __syncthreads();
            }


            __syncthreads();

            // v = precond * r
            prec_shared.apply(gko::batch::to_const(r_shared_entry),
                              v_shared_entry);

            __syncthreads();

            // t = A *v
            spmv_kernel(A_global_entry, gko::batch::to_const(v_shared_entry),
                        t_shared_entry);

            __syncthreads();

            // omega = ( t * r )/ (t * t)
            // rho = (t * r ) /(||t|| * || r||)
            // if |rho| < kappa
            //      omega = omega * kappa / |rho|
            // end if
            const gko::batch_dense::BatchEntry<ValueType>
                &t_r_dot_shared_entry = temp1_shared_entry;
            const gko::batch_dense::BatchEntry<ValueType> &rho_shared_entry =
                temp2_shared_entry;
            compute_omega(gko::batch::to_const(t_shared_entry),
                          gko::batch::to_const(r_shared_entry),
                          rho_shared_entry, t_r_dot_shared_entry,
                          t_norms_shared_entry, r_norms_shared_entry,
                          omega_shared_entry, kappa, converged);

            __syncthreads();


            // r = r - omega * t
            // x = x + omega * v
            update_r_and_x_outer_loop(gko::batch::to_const(t_shared_entry),
                                      gko::batch::to_const(v_shared_entry),
                                      gko::batch::to_const(omega_shared_entry),
                                      r_shared_entry, x_shared_entry,
                                      converged);

            __syncthreads();


            if (smoothing == true) {
                const gko::batch_dense::BatchEntry<ValueType>
                    &gamma_shared_entry = temp2_shared_entry;
                smoothing_operation(gko::batch::to_const(x_shared_entry),
                                    gko::batch::to_const(r_shared_entry),
                                    gamma_shared_entry, t_shared_entry,
                                    xs_shared_entry, rs_shared_entry,
                                    t_norms_shared_entry, converged);

                __syncthreads();

                compute_norm2<ValueType>(gko::batch::to_const(rs_shared_entry),
                                         res_norms_shared_entry,
                                         converged);  // store residual norms


            } else {
                compute_norm2<ValueType>(gko::batch::to_const(r_shared_entry),
                                         res_norms_shared_entry, converged);
            }

            __syncthreads();
        }

        if (smoothing == true) {
            copy(gko::batch::to_const(xs_shared_entry), x_shared_entry);
            copy(gko::batch::to_const(rs_shared_entry), r_shared_entry);
        }

        __syncthreads();

        if (left_global_entry.values) {
            batch_scale(right_global_entry, x_shared_entry);
        }

        __syncthreads();

        // copy x back to global memory
        copy(gko::batch::to_const(x_shared_entry), x_global_entry);
    }
}
