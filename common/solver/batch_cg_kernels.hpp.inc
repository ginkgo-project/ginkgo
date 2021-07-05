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


template <typename BatchMatrixType_entry, typename PrecType, typename ValueType>
__device__ __forceinline__ void initialize(
    const BatchMatrixType_entry &A_global_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &b_global_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_global_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_shared_entry,
    const PrecType &prec_shared,
    const gko::batch_dense::BatchEntry<ValueType> &z_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &p_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &rhs_norms_shared_entry)
{
    // Compute norms of rhs
    compute_norm2<ValueType>(b_global_entry, rhs_norms_shared_entry);


    // copy x from global to shared memory
    copy(x_global_entry, x_shared_entry);


    // r = b
    copy(b_global_entry, r_shared_entry);

    __syncthreads();


    // r = b - A*x
    advanced_spmv_kernel(static_cast<ValueType>(-1.0), A_global_entry,
                         gko::batch::to_const(x_shared_entry),
                         static_cast<ValueType>(1.0), r_shared_entry);
    __syncthreads();


    // z = precond * r
    prec_shared.apply(gko::batch::to_const(r_shared_entry), z_shared_entry);

    __syncthreads();


    // p = z
    copy(gko::batch::to_const(z_shared_entry), p_shared_entry);


    // rho_old = r' * z
    compute_dot_product(gko::batch::to_const(r_shared_entry),
                        gko::batch::to_const(z_shared_entry),
                        rho_old_shared_entry);
}


template <typename ValueType>
__device__ __forceinline__ void update_p(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_new_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &z_shared_entry,
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
            rho_new_shared_entry.values[c] / rho_old_shared_entry.values[c];

        p_shared_entry.values[r * p_shared_entry.stride + c] =
            z_shared_entry.values[r * z_shared_entry.stride + c] +
            beta * p_shared_entry.values[r * p_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x_and_r(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &Ap_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_shared_entry,
    const uint32 &converged)
{
    compute_dot_product<ValueType>(p_shared_entry, Ap_shared_entry,
                                   alpha_shared_entry, converged);
    __syncthreads();


    for (int li = threadIdx.x;
         li < x_shared_entry.num_rows * x_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / x_shared_entry.num_rhs;
        const int c = li % x_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        const ValueType alpha =
            rho_old_shared_entry.values[c] / alpha_shared_entry.values[c];

        x_shared_entry.values[r * x_shared_entry.stride + c] +=

            alpha * p_shared_entry.values[r * p_shared_entry.stride + c];


        r_shared_entry.values[r * r_shared_entry.stride + c] -=
            alpha * Ap_shared_entry.values[r * Ap_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void use_implicit_norms(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &res_norms_shared_entry,
    const uint32 &converged)
{
    for (int c = threadIdx.x; c < res_norms_shared_entry.num_rhs;
         c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        res_norms_shared_entry.values[c] =
            sqrt(abs(rho_old_shared_entry.values[c]));
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

    // required for static allocation in stopping criterion
    assert(batch_config<ValueType>::max_num_rhs >= nrhs);

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
        extern __shared__ char local_mem_sh[];
        ValueType *const r_sh = reinterpret_cast<ValueType *>(local_mem_sh);
        ValueType *const z_sh = r_sh + nrows * nrhs;
        ValueType *const p_sh = z_sh + nrows * nrhs;
        ValueType *const Ap_sh = p_sh + nrows * nrhs;
        ValueType *const x_sh = Ap_sh + nrows * nrhs;
        ValueType *const prec_work_sh = x_sh + nrows * nrhs;
        ValueType *const rho_old_sh =
            prec_work_sh + PrecType::dynamic_work_size(nrows, a.num_nnz);
        ValueType *const rho_new_sh = rho_old_sh + nrhs;
        ValueType *const alpha_sh = rho_new_sh + nrhs;
        real_type *const norms_rhs_sh =
            reinterpret_cast<real_type *>(alpha_sh + nrhs);
        real_type *const norms_res_sh = norms_rhs_sh + nrhs;


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

        const gko::batch_dense::BatchEntry<ValueType> z_shared_entry{
            z_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> p_shared_entry{
            p_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> Ap_shared_entry{
            Ap_sh, static_cast<size_type>(nrhs), nrows, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> rho_old_shared_entry{
            rho_old_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> rho_new_shared_entry{
            rho_new_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> alpha_shared_entry{
            alpha_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> rhs_norms_shared_entry{
            norms_rhs_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> res_norms_shared_entry{
            norms_res_sh, static_cast<size_type>(nrhs), 1, nrhs};


        // generate preconditioner
        prec_shared.generate(A_global_entry, prec_work_sh);


        // initialization
        // compute b norms
        // r = b - A*x
        // z = precond*r
        // rho_old = r' * z (' is for hermitian transpose)
        // p = z
        initialize(A_global_entry, b_global_entry,
                   gko::batch::to_const(x_global_entry), x_shared_entry,
                   r_shared_entry, prec_shared, z_shared_entry,
                   rho_old_shared_entry, p_shared_entry,
                   rhs_norms_shared_entry);


        __syncthreads();


        // stopping criterion object
        StopType stop(nrhs, max_iter, tol, rhs_norms_shared_entry.values,
                      converged);

        int iter = -1;


        while (1) {
            ++iter;

            use_implicit_norms(gko::batch::to_const(rho_old_shared_entry),
                               res_norms_shared_entry,
                               converged);  // use implicit residual norms

            __syncthreads();

            bool all_converged =
                stop.check_converged(iter, res_norms_shared_entry.values,
                                     {NULL, 0, 0, 0}, converged);


            logger.log_iteration(ibatch, iter, res_norms_shared_entry.values,
                                 converged);
            if (all_converged) {
                break;
            }

            // Ap = A * p
            spmv_kernel(A_global_entry, gko::batch::to_const(p_shared_entry),
                        Ap_shared_entry);

            __syncthreads();

            // alpha = rho_old / (p' * Ap)
            // x = x + alpha * p
            // r = r - alpha * Ap
            update_x_and_r(gko::batch::to_const(rho_old_shared_entry),
                           gko::batch::to_const(p_shared_entry),
                           gko::batch::to_const(Ap_shared_entry),
                           alpha_shared_entry, x_shared_entry, r_shared_entry,
                           converged);

            __syncthreads();


            // z = precond * r
            prec_shared.apply(gko::batch::to_const(r_shared_entry),
                              z_shared_entry);

            __syncthreads();

            // rho_new =  (r)' * (z)
            compute_dot_product<ValueType>(gko::batch::to_const(r_shared_entry),
                                           gko::batch::to_const(z_shared_entry),
                                           rho_new_shared_entry, converged);

            __syncthreads();


            // beta = rho_new / rho_old
            // p = z + beta * p
            update_p(gko::batch::to_const(rho_new_shared_entry),
                     gko::batch::to_const(rho_old_shared_entry),
                     gko::batch::to_const(z_shared_entry), p_shared_entry,
                     converged);

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
