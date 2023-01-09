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


template <typename BatchMatrixType_entry, typename PrecType, typename ValueType>
__device__ __forceinline__ void initialize(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const BatchMatrixType_entry& A_global_entry,
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
    single_advanced_matvec_kernel(static_cast<ValueType>(-1.0), A_global_entry,
                                  x_shared_entry, static_cast<ValueType>(1.0),
                                  r_shared_entry);
    __syncthreads();

    // z = precond * r
    prec_shared.apply(num_rows, r_shared_entry, z_shared_entry);
    __syncthreads();

    if (threadIdx.x / config::warp_size == 0) {
        // Compute norms of rhs
        single_compute_norm2(warp_grp, num_rows, b_global_entry, rhs_norms_sh);
    } else if (threadIdx.x / config::warp_size == 1) {
        // rho_old = r' * z
        single_compute_dot_product(warp_grp, num_rows, r_shared_entry,
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


template <typename ValueType>
__device__ __forceinline__ void update_x_and_r(
    group::thread_block_tile<config::warp_size>& warp_grp, const int num_rows,
    const ValueType& rho_old_shared_entry,
    const ValueType* const p_shared_entry,
    const ValueType* const Ap_shared_entry, ValueType& alpha_shared_entry,
    ValueType* const x_shared_entry, ValueType* const r_shared_entry)
{
    if (threadIdx.x / config::warp_size == 0) {
        single_compute_dot_product(warp_grp, num_rows, p_shared_entry,
                                   Ap_shared_entry, alpha_shared_entry);
    }
    __syncthreads();

    for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
        const ValueType alpha = rho_old_shared_entry / alpha_shared_entry;
        x_shared_entry[li] += alpha * p_shared_entry[li];
        r_shared_entry[li] -= alpha * Ap_shared_entry[li];
    }
}

}  // namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(const gko::kernels::batch_cg::StorageConfig sconf,
                             const int max_iter,
                             const gko::remove_complex<ValueType> tol,
                             LogType logger, PrecType prec_shared,
                             const BatchMatrixType a,
                             const ValueType* const __restrict__ b,
                             ValueType* const __restrict__ x,
                             ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = a.num_batch;
    const auto nrows = a.num_rows;

    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto warp_grp = group::tiled_partition<tile_size>(thread_block);

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
        const int gmem_offset =
            ibatch * sconf.gmem_stride_bytes / sizeof(ValueType);
        extern __shared__ char local_mem_sh[];
        ValueType* r_sh;
        ValueType* z_sh;
        ValueType* p_sh;
        ValueType* Ap_sh;
        ValueType* x_sh;
        ValueType* prec_work_sh;
        if (sconf.n_shared >= 1) {
            r_sh = reinterpret_cast<ValueType*>(local_mem_sh);
        } else {
            r_sh = workspace + gmem_offset;
        }
        if (sconf.n_shared == 1) {
            z_sh = workspace + gmem_offset;
        } else {
            z_sh = r_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 2) {
            p_sh = workspace + gmem_offset;
        } else {
            p_sh = z_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 3) {
            Ap_sh = workspace + gmem_offset;
        } else {
            Ap_sh = p_sh + sconf.padded_vec_len;
        }
        if (!sconf.prec_shared && sconf.n_shared == 4) {
            prec_work_sh = workspace + gmem_offset;
        } else {
            prec_work_sh = Ap_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 4 && sconf.prec_shared) {
            x_sh = workspace + gmem_offset;
        } else {
            x_sh = prec_work_sh + PrecType::dynamic_work_size(nrows, a.num_nnz);
        }
        __shared__ uninitialized_array<ValueType, 1> rho_old_sh;
        __shared__ uninitialized_array<ValueType, 1> rho_new_sh;
        __shared__ uninitialized_array<ValueType, 1> alpha_sh;
        __shared__ real_type norms_rhs_sh[1];
        __shared__ real_type norms_res_sh[1];

        const auto A_global_entry = gko::batch::batch_entry(a, ibatch);
        const ValueType* const b_global_entry =
            gko::batch::batch_entry_ptr(b, 1, nrows, ibatch);
        ValueType* const x_global_entry =
            gko::batch::batch_entry_ptr(x, 1, nrows, ibatch);

        // generate preconditioner
        prec_shared.generate(ibatch, A_global_entry, prec_work_sh);

        // initialization
        // compute b norms
        // r = b - A*x
        // z = precond*r
        // rho_old = r' * z (' is for hermitian transpose)
        // p = z
        initialize(warp_grp, nrows, A_global_entry, b_global_entry,
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
                break;
            }

            // Ap = A * p
            single_matvec_kernel(A_global_entry, p_sh, Ap_sh);
            __syncthreads();

            // alpha = rho_old / (p' * Ap)
            // x = x + alpha * p
            // r = r - alpha * Ap
            update_x_and_r(warp_grp, nrows, rho_old_sh[0], p_sh, Ap_sh,
                           alpha_sh[0], x_sh, r_sh);
            __syncthreads();

            // z = precond * r
            prec_shared.apply(nrows, r_sh, z_sh);
            __syncthreads();

            if (threadIdx.x / config::warp_size == 0) {
                // rho_new =  (r)' * (z)
                single_compute_dot_product(warp_grp, nrows, r_sh, z_sh,
                                           rho_new_sh[0]);
            }
            __syncthreads();

            // beta = rho_new / rho_old
            // p = z + beta * p
            update_p(nrows, rho_new_sh[0], rho_old_sh[0], z_sh, p_sh);
            __syncthreads();

            // rho_old = rho_new
            if (threadIdx.x == 0) {
                rho_old_sh[0] = rho_new_sh[0];
            }
            __syncthreads();
        }

        logger.log_iteration(ibatch, iter, norms_res_sh[0]);

        // copy x back to global memory
        single_copy(nrows, x_sh, x_global_entry);
        __syncthreads();
    }
}
