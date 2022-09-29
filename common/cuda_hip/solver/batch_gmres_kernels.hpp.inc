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


template <typename BatchMatrixType_entry, typename PrecType, typename ValueType>
__device__ __forceinline__ void initialize(
    group::thread_block_tile<config::warp_size>& warp_grp, const int restart,
    const BatchMatrixType_entry& a_gl_entry,
    const ValueType* const b_global_entry,
    const ValueType* const x_global_entry, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry, const PrecType& prec_shared,
    ValueType* const z_shared_entry, ValueType* const V_shared_entry,
    ValueType* const H_shared_entry, ValueType* const cs_shared_entry,
    ValueType* const sn_shared_entry,
    typename gko::remove_complex<ValueType>& rhs_norms,
    typename gko::remove_complex<ValueType>& res_norms)
{
    for (int i = threadIdx.x; i < a_gl_entry.num_rows; i += blockDim.x) {
        x_shared_entry[i] = x_global_entry[i];
        r_shared_entry[i] = b_global_entry[i];
    }
    __syncthreads();

    // r = b - A*x
    single_advanced_matvec_kernel(static_cast<ValueType>(-1.0), a_gl_entry,
                                  x_shared_entry, static_cast<ValueType>(1.0),
                                  r_shared_entry);
    __syncthreads();

    if (threadIdx.x / config::warp_size == 0) {
        single_compute_norm2(warp_grp, a_gl_entry.num_rows, b_global_entry,
                             rhs_norms);
    } else if (threadIdx.x / config::warp_size == 1) {
        single_compute_norm2(warp_grp, a_gl_entry.num_rows, r_shared_entry,
                             res_norms);
    }

    // z = precond * r
    prec_shared.apply(a_gl_entry.num_rows, r_shared_entry, z_shared_entry);

    for (int li = threadIdx.x; li < (restart + 1) * restart; li += blockDim.x) {
        H_shared_entry[li] = zero<ValueType>();
    }

    for (int i = threadIdx.x; i < a_gl_entry.num_rows * (restart + 1);
         i += blockDim.x) {
        V_shared_entry[i] = zero<ValueType>();
    }

    for (int li = threadIdx.x; li < restart; li += blockDim.x) {
        cs_shared_entry[li] = zero<ValueType>();
        sn_shared_entry[li] = zero<ValueType>();
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_v_naught_and_s(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const int restart, const ValueType* const z_shared_entry,
    ValueType* const V_shared_entry, ValueType* const s_shared_entry)
{
    using real_type = typename gko::remove_complex<ValueType>;

    __shared__ real_type tmp_norm;
    if (threadIdx.x / config::warp_size == 0) {
        single_compute_norm2(warp_grp, num_rows, z_shared_entry, tmp_norm);
    }
    __syncthreads();

    for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
        V_shared_entry[li] = z_shared_entry[li] / tmp_norm;
    }

    for (int li = threadIdx.x; li < restart + 1; li += blockDim.x) {
        s_shared_entry[li] = zero<ValueType>();
    }
    if (threadIdx.x == 0) {
        s_shared_entry[0] = static_cast<ValueType>(tmp_norm);
    }
}


template <typename BatchMatrixType_entry, typename ValueType, typename PrecType>
__device__ __forceinline__ void arnoldi(
    group::thread_block_tile<config::warp_size>& warp_grp, const int restart,
    const BatchMatrixType_entry& a_gl_entry, const int inner_iter,
    ValueType* const V_shared_entry, ValueType* const H_shared_entry,
    const size_type h_stride, ValueType* const w_shared_entry,
    ValueType* const helper_shared_entry, const PrecType& prec_shared)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const int num_rows = a_gl_entry.num_rows;

    const ValueType* const v_i_sh = V_shared_entry + inner_iter * num_rows;

    single_matvec_kernel(a_gl_entry, v_i_sh, helper_shared_entry);
    __syncthreads();

    prec_shared.apply(num_rows, helper_shared_entry, w_shared_entry);
    __syncthreads();

    for (int k = 0; k <= inner_iter; k++) {
        const ValueType* const v_k_sh = V_shared_entry + k * num_rows;
        ValueType* const h_k_i_sh = H_shared_entry + k * h_stride + inner_iter;

        if (threadIdx.x / config::warp_size == 0) {
            single_compute_dot_product(warp_grp, num_rows, w_shared_entry,
                                       v_k_sh, h_k_i_sh[0]);
        }
        __syncthreads();

        for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
            w_shared_entry[li] -= h_k_i_sh[0] * v_k_sh[li];
        }
        __syncthreads();
    }

    __shared__ remove_complex<ValueType> tmp_norm;
    if (threadIdx.x / config::warp_size == 0) {
        single_compute_norm2(warp_grp, num_rows, w_shared_entry, tmp_norm);
    }
    __syncthreads();

    if (threadIdx.x == config::warp_size) {
        H_shared_entry[(inner_iter + 1) * h_stride + inner_iter] = tmp_norm;
    }

    ValueType* const v_iplus1_sh = V_shared_entry + (inner_iter + 1) * num_rows;

    for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
        v_iplus1_sh[li] = w_shared_entry[li] / tmp_norm;
    }
}


template <typename ValueType>
__device__ __forceinline__ void apply_plane_rotation(const ValueType* const cs,
                                                     const ValueType* const sn,
                                                     ValueType* const x,
                                                     ValueType* const y)
{
    if (threadIdx.x == 0) {
        const ValueType temp = cs[0] * x[0] + sn[0] * y[0];
        y[0] = -one<ValueType>() * conj(sn[0]) * x[0] + conj(cs[0]) * y[0];
        x[0] = temp;
    }
}


template <typename ValueType>
__device__ __forceinline__ void generate_and_apply_plane_rot_fused(
    ValueType* const cs_i, ValueType* const sn_i, ValueType* const H_i_i,
    ValueType* const H_i_plus_1_i, ValueType* const s_i,
    ValueType* const s_i_plus_1)
{
    if (threadIdx.x == 0) {
        if (H_i_plus_1_i[0] == zero<ValueType>()) {
            cs_i[0] = one<ValueType>();
            sn_i[0] = zero<ValueType>();
        } else if (abs(H_i_plus_1_i[0]) > abs(H_i_i[0])) {
            const ValueType temp = H_i_i[0] / H_i_plus_1_i[0];
            sn_i[0] = (one<ValueType>() * conj(H_i_plus_1_i[0])) /
                      (sqrt(one<ValueType>() + temp * temp) * H_i_plus_1_i[0]);
            cs_i[0] = (conj(H_i_i[0]) / conj(H_i_plus_1_i[0])) * sn_i[0];

        } else {
            const ValueType temp = H_i_plus_1_i[0] / H_i_i[0];
            cs_i[0] = (one<ValueType>() * conj(H_i_i[0])) /
                      (sqrt(one<ValueType>() + temp * temp) * H_i_i[0]);
            sn_i[0] = (conj(H_i_plus_1_i[0]) / conj(H_i_i[0])) * cs_i[0];
        }

        const ValueType temp = cs_i[0] * s_i[0];
        s_i_plus_1[0] = -one<ValueType>() * conj(sn_i[0]) * s_i[0];
        s_i[0] = temp;
        H_i_i[0] = cs_i[0] * H_i_i[0] + sn_i[0] * H_i_plus_1_i[0];
        H_i_plus_1_i[0] = zero<ValueType>();
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const int m, const int restart, const ValueType* const H_shared_entry,
    const size_type h_stride, const ValueType* const s_shared_entry,
    const ValueType* const V_shared_entry, ValueType* const x_shared_entry,
    ValueType* const y_shared_entry)
{
    // triangular solve
    if (threadIdx.x == 0) {
        for (int r = m; r >= 0; r--) {
            ValueType temp_sum = zero<ValueType>();
            for (int t = m; t > r; t--) {
                temp_sum +=
                    H_shared_entry[r * h_stride + t] * y_shared_entry[t];
            }

            y_shared_entry[r] = (s_shared_entry[r] - temp_sum) /
                                H_shared_entry[r * h_stride + r];
        }
    }
    __syncthreads();

    // dense mat * vec multiplication
    for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
        for (int a = 0; a < m + 1; a++) {
            x_shared_entry[li] +=
                V_shared_entry[a * num_rows + li] * y_shared_entry[a];
        }
    }
}


}  // unnamed namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(const int global_gap, const int max_iter,
                             const gko::remove_complex<ValueType> tol,
                             const int restart, LogType logger,
                             PrecType prec_shared, const BatchMatrixType a,
                             const ValueType* const __restrict__ b,
                             ValueType* const __restrict__ x,
                             ValueType* const workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = a.num_batch;
    const auto nrows = a.num_rows;

    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto warp_grp = group::tiled_partition<tile_size>(thread_block);

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
#if GKO_CUDA_BATCH_GMRES_HAVE_NO_SHMEM
        assert(workspace != nullptr);
        ValueType* const r_sh = workspace + ibatch * (global_gap);
#else
        extern __shared__ char local_mem_sh[];
        ValueType* const r_sh = reinterpret_cast<ValueType*>(local_mem_sh);
#endif
        ValueType* const z_sh = r_sh + nrows;
        ValueType* const w_sh = z_sh + nrows;
        ValueType* const x_sh = w_sh + nrows;
        ValueType* const helper_sh = x_sh + nrows;
        ValueType* const cs_sh = helper_sh + nrows;
        ValueType* const sn_sh = cs_sh + restart;
        ValueType* const y_sh = sn_sh + restart;
        ValueType* const s_sh = y_sh + restart;

        // Hessenberg matrix
        // const gko::batch_dense::BatchEntry<ValueType> H_shared_entry{
        //     H_sh, static_cast<size_type>(nrhs * restart), restart + 1,
        //     nrhs * restart};
        // storage:row-major ,  entry (i,j) for different RHSs are placed after
        // the other in a row - when drawn on paper, (and the same is true for
        // actual storage as the storage order is row-major) to get entry (i,j)
        // for rhs: rhs_k , H_entry.stride*i + j*nrhs  + rhs_k
        ValueType* const H_sh = s_sh + (restart + 1);

        // Krylov subspace basis vectors
        // storage:row-major order , subspace basis vectors corr. to each rhs
        // are stored in a single col. one after the other-(on paper). This
        // allows for efficient memory access And to get vi : that is ith basis
        // vector for each rhs: vi_entry{  &V[i* V_entry.stride * nrows],
        // V_entry.stride , nrows, nrhs}; So if nrhs=1, effectively the cols.
        // are stored contiguously in memory one after the other.
        // const gko::batch_dense::BatchEntry<ValueType> V_shared_entry{
        //     V_sh, static_cast<size_type>(nrhs), nrows * (restart + 1), nrhs};
        ValueType* const V_sh = H_sh + restart * (restart + 1);

        ValueType* const prec_work_sh = V_sh + nrows * (restart + 1);

        // real_type *const norms_rhs_sh = reinterpret_cast<real_type *>(
        //     prec_work_sh + PrecType::dynamic_work_size(nrows, a.num_nnz));
        // real_type *const norms_res_sh = norms_rhs_sh + nrhs;
        // real_type *const norms_tmp_sh = norms_res_sh + nrhs;
        __shared__ real_type norms_rhs_sh[1];
        __shared__ real_type norms_res_sh[1];

        const auto a_gl_entry = gko::batch::batch_entry(a, ibatch);
        const ValueType* const b_entry_ptr =
            gko::batch::batch_entry_ptr(b, 1, nrows, ibatch);
        ValueType* const x_gl_entry_ptr =
            gko::batch::batch_entry_ptr(x, 1, nrows, ibatch);

        // Stride of Hessenberg matrix
        const auto h_stride = static_cast<size_type>(restart);

        prec_shared.generate(ibatch, a_gl_entry, prec_work_sh);

        // initialization
        // compute b norms
        // r = b - A*x
        // z = precond*r
        // compute residual norms
        // initialize V, H, cs, sn with zeroes
        initialize(warp_grp, restart, a_gl_entry, b_entry_ptr, x_gl_entry_ptr,
                   x_sh, r_sh, prec_shared, z_sh, V_sh, H_sh, cs_sh, sn_sh,
                   norms_rhs_sh[0], norms_res_sh[0]);
        __syncthreads();

        // stopping criterion object
        StopType stop(tol, norms_rhs_sh);

        int total_iter = 0;

        // Note: restart - inner iterations and the outer iteration:  are
        // counted as (restart + 1) number of iterations instead of one.
        while (1) {
            if (stop.check_converged(norms_res_sh) || total_iter >= max_iter) {
                break;
            }

            // KrylovBasis_0 = z/norm(z)
            // s -> fill with zeroes
            // s(0) = norm(z)
            update_v_naught_and_s(warp_grp, nrows, restart, z_sh, V_sh, s_sh);
            __syncthreads();

            bool inner_loop_break_flag = false;

            for (int inner_iter = 0; inner_iter < restart; inner_iter++) {
                // w_temp = A * v_i
                // w = precond * w_temp
                // i = inner_iter
                // for k = 0 to inner_iter
                //     Hessenburg(k,i) =  w' * v_k
                //     w = w - Hessenburg(k,i) * v_k
                // end
                // Hessenburg(i+1, i) = norm(w)
                // KrylovBasis_i+1 = w / Hessenburg(i+1,i)
                arnoldi(warp_grp, restart, a_gl_entry, inner_iter, V_sh, H_sh,
                        h_stride, w_sh, helper_sh, prec_shared);
                __syncthreads();

                for (int k = 0; k < inner_iter; k++) {
                    // temp = cs(k) * Hessenberg( k, inner_iter )  +   sn(k) *
                    // Hessenberg(k + 1, inner_iter)
                    // Hessenberg(k + 1, inner_iter) = -1 * conj(sn(k)) *
                    // Hessenberg( k , inner_iter) + conj(cs(k)) * Hessenberg(k
                    // + 1 , inner_iter) Hessenberg(k,inner_iter) = temp
                    apply_plane_rotation(
                        &cs_sh[k], &sn_sh[k], &H_sh[k * h_stride + inner_iter],
                        &H_sh[(k + 1) * h_stride + inner_iter]);
                    __syncthreads();
                }

                // compute sine and cos
                // temp = cs(inner_iter) * s(inner_iter)
                // s(inner_iter + 1) = -1 * conj(sn(inner_iter)) * s(inner_iter)
                // s(inner_iter) = temp
                // Hessenberg(inner_iter , inner_iter) = cs(inner_iter) *
                // Hessenberg(inner_iter , inner_iter) + sn(inner_iter) *
                // Hessenberg(inner_iter + 1, inner_iter)
                // Hessenberg(inner_iter + 1, inner_iter) = 0
                generate_and_apply_plane_rot_fused(
                    &cs_sh[inner_iter], &sn_sh[inner_iter],
                    &H_sh[inner_iter * h_stride + inner_iter],
                    &H_sh[(inner_iter + 1) * h_stride + inner_iter],
                    &s_sh[inner_iter], &s_sh[(inner_iter + 1)]);
                __syncthreads();

                // estimate of residual norms
                if (threadIdx.x == 0) {
                    norms_res_sh[0] = abs(s_sh[(inner_iter + 1)]);
                }
                __syncthreads();

                total_iter++;

                if (stop.check_converged(norms_res_sh) ||
                    total_iter >= max_iter) {
                    update_x(warp_grp, nrows, inner_iter, restart, H_sh,
                             h_stride, s_sh, V_sh, x_sh, y_sh);
                    inner_loop_break_flag = true;
                    break;
                }
            }

            if (inner_loop_break_flag == true) {
                break;
            }

            // y = Hessenburg(0:restart - 1,0:restart - 1) \ s(0:restart-1)
            // x = x + KrylovBasis(:,0 : restart - 1) * y
            update_x(warp_grp, nrows, restart - 1, restart, H_sh, h_stride,
                     s_sh, V_sh, x_sh, y_sh);
            __syncthreads();

            // r = b
            single_copy(nrows, b_entry_ptr, r_sh);
            __syncthreads();
            // r = r - A*x
            single_advanced_matvec_kernel(static_cast<ValueType>(-1.0),
                                          a_gl_entry, x_sh,
                                          static_cast<ValueType>(1.0), r_sh);
            __syncthreads();

            if (threadIdx.x / config::warp_size == 0) {
                single_compute_norm2(warp_grp, nrows, r_sh, norms_res_sh[0]);
            }

            prec_shared.apply(nrows, r_sh, z_sh);
            __syncthreads();
        }

        // copy x back to global memory
        single_copy(nrows, x_sh, x_gl_entry_ptr);

        logger.log_iteration(ibatch, total_iter, norms_res_sh[0]);
    }
}
