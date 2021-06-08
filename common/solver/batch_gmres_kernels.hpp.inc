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
    const gko::batch_dense::BatchEntry<ValueType> &V_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &H_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &cs_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &sn_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &rhs_norms_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &res_norms_shared_entry)
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

    compute_norm2<ValueType>(gko::batch::to_const(r_shared_entry),
                             res_norms_shared_entry);


    // z = precond * r
    prec_shared.apply(gko::batch::to_const(r_shared_entry), z_shared_entry);


    for (int li = threadIdx.x;
         li < H_shared_entry.num_rows * H_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / H_shared_entry.num_rhs;
        const int c = li % H_shared_entry.num_rhs;

        H_shared_entry.values[r * H_shared_entry.stride + c] =
            zero<ValueType>();
    }


    for (int tid = threadIdx.x;
         tid < V_shared_entry.num_rows * V_shared_entry.num_rhs;
         tid += blockDim.x) {
        const int row_index =
            (tid / V_shared_entry.num_rhs) % x_shared_entry.num_rows;

        const int basis_vec_index =
            (tid / V_shared_entry.num_rhs) / x_shared_entry.num_rows;

        const int rhs_index = tid % V_shared_entry.num_rhs;

        V_shared_entry
            .values[(basis_vec_index * x_shared_entry.num_rows + row_index) *
                        V_shared_entry.stride +
                    rhs_index] = zero<ValueType>();
    }


    for (int li = threadIdx.x;
         li < cs_shared_entry.num_rows * cs_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / cs_shared_entry.num_rhs;
        const int c = li % cs_shared_entry.num_rhs;

        cs_shared_entry.values[r * cs_shared_entry.stride + c] =
            zero<ValueType>();
    }


    for (int li = threadIdx.x;
         li < sn_shared_entry.num_rows * sn_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / sn_shared_entry.num_rhs;
        const int c = li % sn_shared_entry.num_rhs;

        sn_shared_entry.values[r * sn_shared_entry.stride + c] =
            zero<ValueType>();
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_v_naught_and_s(
    const gko::batch_dense::BatchEntry<const ValueType> &z_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &V_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &s_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &tmp_norms_shared_entry,
    const uint32 &converged)
{
    using real_type = typename gko::remove_complex<ValueType>;

    const gko::batch_dense::BatchEntry<ValueType> v_naught_shared_entry{
        V_shared_entry.values, V_shared_entry.stride, z_shared_entry.num_rows,
        V_shared_entry.num_rhs};


    const gko::batch_dense::BatchEntry<real_type> &z_norms_shared_entry =
        tmp_norms_shared_entry;

    compute_norm2(z_shared_entry, z_norms_shared_entry, converged);

    __syncthreads();

    for (int li = threadIdx.x;
         li < v_naught_shared_entry.num_rows * v_naught_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / v_naught_shared_entry.num_rhs;
        const int c = li % v_naught_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }


        v_naught_shared_entry.values[r * v_naught_shared_entry.stride + c] =
            z_shared_entry.values[r * z_shared_entry.stride + c] /
            static_cast<ValueType>(z_norms_shared_entry.values[c]);
    }


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
            zero<ValueType>();

        if (r == 0) {
            s_shared_entry.values[r * s_shared_entry.stride + c] =
                static_cast<ValueType>(z_norms_shared_entry.values[c]);
        }
    }
}


template <typename BatchMatrixType_entry, typename ValueType, typename PrecType>
__device__ __forceinline__ void arnoldi(
    const BatchMatrixType_entry &A_global_entry, const int inner_iter,
    const gko::batch_dense::BatchEntry<ValueType> &V_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &H_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &w_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &helper_shared_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &tmp_norms_shared_entry,
    const PrecType &prec_shared, const uint32 &converged)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const int &i = inner_iter;
    const int num_rows = w_shared_entry.num_rows;
    const int num_rhs = w_shared_entry.num_rhs;

    const gko::batch_dense::BatchEntry<const ValueType> v_i_shared_entry{
        &V_shared_entry.values[i * V_shared_entry.stride * num_rows],
        V_shared_entry.stride, num_rows, V_shared_entry.num_rhs};


    spmv_kernel(A_global_entry, v_i_shared_entry, helper_shared_entry);

    __syncthreads();

    prec_shared.apply(gko::batch::to_const(helper_shared_entry),
                      w_shared_entry);
    __syncthreads();


    for (int k = 0; k <= i; k++) {
        const gko::batch_dense::BatchEntry<const ValueType> v_k_shared_entry{
            &V_shared_entry.values[k * V_shared_entry.stride * num_rows],
            V_shared_entry.stride, num_rows, V_shared_entry.num_rhs};

        const gko::batch_dense::BatchEntry<ValueType> h_k_i_shared_entry{
            &H_shared_entry.values[k * H_shared_entry.stride + i * num_rhs],
            static_cast<size_type>(w_shared_entry.num_rhs), 1,
            w_shared_entry.num_rhs};

        compute_dot_product(gko::batch::to_const(w_shared_entry),
                            v_k_shared_entry, h_k_i_shared_entry, converged);

        __syncthreads();


        for (int li = threadIdx.x;
             li < w_shared_entry.num_rows * w_shared_entry.num_rhs;
             li += blockDim.x) {
            const int r = li / w_shared_entry.num_rhs;

            const int c = li % w_shared_entry.num_rhs;

            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            const ValueType h_k_i_scalar = h_k_i_shared_entry.values[c];

            w_shared_entry.values[r * w_shared_entry.stride + c] -=
                h_k_i_scalar *
                v_k_shared_entry.values[r * v_k_shared_entry.stride + c];
        }

        __syncthreads();
    }


    const gko::batch_dense::BatchEntry<real_type> &w_norms_shared_entry =
        tmp_norms_shared_entry;

    compute_norm2(gko::batch::to_const(w_shared_entry), w_norms_shared_entry,
                  converged);

    const gko::batch_dense::BatchEntry<ValueType> h_i_plus_1_i_shared_entry{
        &H_shared_entry.values[(i + 1) * H_shared_entry.stride + i * num_rhs],
        static_cast<size_type>(w_shared_entry.num_rhs), 1,
        w_shared_entry.num_rhs};

    __syncthreads();


    for (int c = threadIdx.x; c < w_norms_shared_entry.num_rhs;
         c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        h_i_plus_1_i_shared_entry.values[c] =
            static_cast<ValueType>(w_norms_shared_entry.values[c]);
    }


    __syncthreads();

    const gko::batch_dense::BatchEntry<ValueType> v_i_plus_1_shared_entry{
        &V_shared_entry.values[(i + 1) * V_shared_entry.stride * num_rows],
        V_shared_entry.stride, num_rows, V_shared_entry.num_rhs};


    for (int li = threadIdx.x;
         li < w_shared_entry.num_rows * w_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / w_shared_entry.num_rhs;
        const int c = li % w_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        v_i_plus_1_shared_entry.values[r * v_i_plus_1_shared_entry.stride + c] =
            w_shared_entry.values[r * w_shared_entry.stride + c] /
            h_i_plus_1_i_shared_entry.values[c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void generate_plane_rotation(
    const ValueType *const x, const ValueType *const y, const int nrhs,
    ValueType *const cs, ValueType *const sn, const uint32 &converged)
{
    for (int c = threadIdx.x; c < nrhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        if (y[c] == zero<ValueType>()) {
            cs[c] = one<ValueType>();
            sn[c] = zero<ValueType>();
        } else if (abs(y[c]) > abs(x[c])) {
            const ValueType temp = x[c] / y[c];
            sn[c] = (one<ValueType>() * conj(y[c])) /
                    (sqrt(one<ValueType>() + temp * temp) * y[c]);
            cs[c] = (conj(x[c]) / conj(y[c])) * sn[c];

        } else {
            const ValueType temp = y[c] / x[c];
            cs[c] = (one<ValueType>() * conj(x[c])) /
                    (sqrt(one<ValueType>() + temp * temp) * x[c]);
            sn[c] = (conj(y[c]) / conj(x[c])) * cs[c];
        }
    }
}

template <typename ValueType>
__device__ __forceinline__ void apply_plane_rotation(
    const ValueType *const cs, const ValueType *const sn, const int nrhs,
    ValueType *const x, ValueType *const y, const uint32 &converged)
{
    for (int c = threadIdx.x; c < nrhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        const ValueType temp = cs[c] * x[c] + sn[c] * y[c];
        y[c] = -one<ValueType>() * conj(sn[c]) * x[c] + conj(cs[c]) * y[c];
        x[c] = temp;
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x(
    const int m,
    const gko::batch_dense::BatchEntry<const ValueType> &H_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &V_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &y_shared_entry,
    const uint32 &converged)
{
    // triangular solve
    for (int rhs_index = threadIdx.x; rhs_index < y_shared_entry.num_rhs;
         rhs_index += blockDim.x) {
        for (int r = m; r >= 0; r--) {
            ValueType temp_sum = zero<ValueType>();

            for (int t = m; t > r; t--) {
                temp_sum +=
                    H_shared_entry
                        .values[r * H_shared_entry.stride +
                                t * y_shared_entry.num_rhs + rhs_index] *
                    y_shared_entry
                        .values[t * y_shared_entry.stride + rhs_index];
            }

            y_shared_entry.values[r * y_shared_entry.stride + rhs_index] =
                (s_shared_entry.values[r * s_shared_entry.stride + rhs_index] -
                 temp_sum) /
                H_shared_entry.values[r * H_shared_entry.stride +
                                      r * y_shared_entry.num_rhs + rhs_index];
        }
    }

    __syncthreads();


    // dense mat * vec multiplication

    for (int li = threadIdx.x;
         li < x_shared_entry.num_rows * x_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / x_shared_entry.num_rhs;
        const int rhs_index = li % x_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << rhs_index);

        if (conv) {
            continue;
        }

        for (int a = 0; a < m + 1; a++) {
            x_shared_entry.values[r * x_shared_entry.stride + rhs_index] +=
                V_shared_entry.values[a * V_shared_entry.stride *
                                          x_shared_entry.num_rows +
                                      r * V_shared_entry.stride + rhs_index] *
                y_shared_entry.values[a * y_shared_entry.stride + rhs_index];
        }
    }
}


}  // unnamed namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(
    const int max_iter, const gko::remove_complex<ValueType> tol,
    const int restart, LogType logger, PrecType prec_shared,
    const BatchMatrixType a,
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
        ValueType *const z_sh = r_sh + nrows * nrhs;
        ValueType *const w_sh = z_sh + nrows * nrhs;
        ValueType *const x_sh = w_sh + nrows * nrhs;
        ValueType *const helper_sh = x_sh + nrows * nrhs;
        ValueType *const cs_sh = helper_sh + nrows * nrhs;
        ValueType *const sn_sh = cs_sh + restart * nrhs;
        ValueType *const y_sh = sn_sh + restart * nrhs;
        ValueType *const s_sh = y_sh + restart * nrhs;
        ValueType *const H_sh = s_sh + (restart + 1) * nrhs;
        ValueType *const V_sh = H_sh + restart * (restart + 1) * nrhs;
        ValueType *const prec_work_sh = V_sh + nrows * (restart + 1) * nrhs;

        real_type *const norms_rhs_sh = reinterpret_cast<real_type *>(
            prec_work_sh + PrecType::dynamic_work_size(nrows, a.num_nnz));
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

        const gko::batch_dense::BatchEntry<ValueType> z_shared_entry{
            z_sh, static_cast<size_type>(nrhs), nrows,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> w_shared_entry{
            w_sh, static_cast<size_type>(nrhs), nrows,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> helper_shared_entry{
            helper_sh, static_cast<size_type>(nrhs), nrows,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> cs_shared_entry{
            cs_sh, static_cast<size_type>(nrhs), restart,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix


        const gko::batch_dense::BatchEntry<ValueType> sn_shared_entry{
            sn_sh, static_cast<size_type>(nrhs), restart,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> y_shared_entry{
            y_sh, static_cast<size_type>(nrhs), restart,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> s_shared_entry{
            s_sh, static_cast<size_type>(nrhs), restart + 1,
            nrhs};  // storage:row-major , residual vector corresponding to each
                    // rhs is stored as a col. of the matrix

        // Hessenberg matrix
        const gko::batch_dense::BatchEntry<ValueType> H_shared_entry{
            H_sh, static_cast<size_type>(nrhs * restart), restart + 1,
            nrhs * restart};
        // storage:row-major ,  entry (i,j) for different RHSs are placed after
        // the other in a row - when drawn on paper, (and the same is true for
        // actual storage as the storage order is row-major) to get entry (i,j)
        // for rhs: rhs_k , H_entry.stride*i + j*nrhs  + rhs_k

        // Krylov subspace basis vectors
        const gko::batch_dense::BatchEntry<ValueType> V_shared_entry{
            V_sh, static_cast<size_type>(nrhs), nrows * (restart + 1), nrhs};
        // storage:row-major order , subspace basis vectors corr. to each rhs
        // are stored in a single col. one after the other-(on paper). This
        // allows for efficient memory access And to get vi : that is ith basis
        // vector for each rhs: vi_entry{  &V[i* V_entry.stride * nrows],
        // V_entry.stride , nrows, nrhs}; So if nrhs=1, effectively the cols.
        // are stored contiguously in memory one after the other.


        const gko::batch_dense::BatchEntry<real_type> rhs_norms_shared_entry{
            norms_rhs_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> res_norms_shared_entry{
            norms_res_sh, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> tmp_norms_shared_entry{
            norms_tmp_sh, static_cast<size_type>(nrhs), 1, nrhs};

        // generate preconditioner
        prec_shared.generate(A_global_entry, prec_work_sh);


        // initialization
        // compute b norms
        // r = b - A*x
        // z = precond*r
        // compute residual norms
        // initialize V, H, cs, sn with zeroes
        initialize(A_global_entry, b_global_entry,
                   gko::batch::to_const(x_global_entry), x_shared_entry,
                   r_shared_entry, prec_shared, z_shared_entry, V_shared_entry,
                   H_shared_entry, cs_shared_entry, sn_shared_entry,
                   rhs_norms_shared_entry, res_norms_shared_entry);


        __syncthreads();


        // stopping criterion object
        StopType stop(nrhs, max_iter, tol, rhs_norms_shared_entry.values,
                      converged);

        int outer_iter = -1;
        bool inner_loop_break_flag = false;

        // Note: restart - inner iterations and the outer iteration:  are
        // counted as (restart + 1) number of iterations instead of one.
        while (1) {
            ++outer_iter;


            bool all_converged = stop.check_converged(
                outer_iter * (restart + 1), res_norms_shared_entry.values,
                {NULL, 0, 0, 0}, converged);


            logger.log_iteration(ibatch, outer_iter * (restart + 1),
                                 res_norms_shared_entry.values, converged);
            if (all_converged) {
                break;
            }


            // KrylovBasis_0 = z/norm(z)
            // s -> fill with zeroes
            // s(0) = norm(z)
            update_v_naught_and_s(gko::batch::to_const(z_shared_entry),
                                  V_shared_entry, s_shared_entry,
                                  tmp_norms_shared_entry, converged);

            __syncthreads();


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
                arnoldi(A_global_entry, inner_iter, V_shared_entry,
                        H_shared_entry, w_shared_entry, helper_shared_entry,
                        tmp_norms_shared_entry, prec_shared, converged);

                __syncthreads();


                for (int k = 0; k < inner_iter; k++) {
                    // temp = cs(k) * Hessenberg( k, inner_iter )  +   sn(k) *
                    // Hessenberg(k + 1, inner_iter)
                    // Hessenberg(k + 1, inner_iter) = -1 * conj(sn(k)) *
                    // Hessenberg( k , inner_iter) + conj(cs(k)) * Hessenberg(k
                    // + 1 , inner_iter) Hessenberg(k,inner_iter) = temp
                    apply_plane_rotation(
                        &cs_shared_entry.values[k * cs_shared_entry.stride],
                        &sn_shared_entry.values[k * sn_shared_entry.stride],
                        nrhs,
                        &H_shared_entry.values[k * H_shared_entry.stride +
                                               inner_iter * nrhs],
                        &H_shared_entry.values[(k + 1) * H_shared_entry.stride +
                                               inner_iter * nrhs],
                        converged);
                    __syncthreads();
                }


                // compute sine and cos
                generate_plane_rotation(
                    &H_shared_entry.values[inner_iter * H_shared_entry.stride +
                                           inner_iter * nrhs],
                    &H_shared_entry
                         .values[(inner_iter + 1) * H_shared_entry.stride +
                                 inner_iter * nrhs],
                    nrhs,
                    &cs_shared_entry
                         .values[inner_iter * cs_shared_entry.stride],
                    &sn_shared_entry
                         .values[inner_iter * cs_shared_entry.stride],
                    converged);
                __syncthreads();


                // temp = cs(inner_iter) * s(inner_iter)
                // s(inner_iter + 1) = -1 * conj(sn(inner_iter)) * s(inner_iter)
                // s(inner_iter) = temp
                // Hessenberg(inner_iter , inner_iter) = cs(inner_iter) *
                // Hessenberg(inner_iter , inner_iter) + sn(inner_iter) *
                // Hessenberg(inner_iter + 1, inner_iter)
                // Hessenberg(inner_iter + 1, inner_iter) = 0
                apply_plane_rotation(

                    &cs_shared_entry
                         .values[inner_iter * cs_shared_entry.stride],
                    &sn_shared_entry
                         .values[inner_iter * sn_shared_entry.stride],
                    nrhs,
                    &s_shared_entry.values[inner_iter * s_shared_entry.stride],
                    &s_shared_entry
                         .values[(inner_iter + 1) * s_shared_entry.stride],
                    converged);

                apply_plane_rotation(

                    &cs_shared_entry
                         .values[inner_iter * cs_shared_entry.stride],
                    &sn_shared_entry
                         .values[inner_iter * sn_shared_entry.stride],
                    nrhs,
                    &H_shared_entry.values[inner_iter * H_shared_entry.stride +
                                           inner_iter * nrhs],
                    &H_shared_entry
                         .values[(inner_iter + 1) * H_shared_entry.stride +
                                 inner_iter * nrhs],
                    converged);

                __syncthreads();

                for (int c = threadIdx.x; c < nrhs; c += blockDim.x) {
                    const uint32 conv = converged & (1 << c);

                    if (conv) {
                        continue;
                    }
                    H_shared_entry
                        .values[(inner_iter + 1) * H_shared_entry.stride +
                                inner_iter * nrhs + c] = zero<ValueType>();
                }
                __syncthreads();


                // estimate of residual norms
                // residual = abs(s(inner_iter + 1))
                for (int c = threadIdx.x; c < res_norms_shared_entry.num_rhs;
                     c += blockDim.x) {
                    const uint32 conv = converged & (1 << c);

                    if (conv) {
                        continue;
                    }

                    res_norms_shared_entry.values[c] = abs(
                        s_shared_entry
                            .values[(inner_iter + 1) * s_shared_entry.stride +
                                    c]);
                }

                __syncthreads();


                const uint32 converged_prev = converged;

                all_converged = stop.check_converged(
                    outer_iter * (restart + 1) + inner_iter + 1,
                    res_norms_shared_entry.values, {NULL, 0, 0, 0}, converged);


                const uint32 converged_recent = converged_prev ^ converged;


                // y = Hessenburg(0 : inner_iter,0 : inner_iter) \ s(0 :
                // inner_iter) x = x + KrylovBasis(:, 0 : inner_iter ) * y
                update_x(inner_iter, gko::batch::to_const(H_shared_entry),
                         gko::batch::to_const(s_shared_entry),
                         gko::batch::to_const(V_shared_entry), x_shared_entry,
                         y_shared_entry, ~converged_recent);


                logger.log_iteration(
                    ibatch, outer_iter * (restart + 1) + inner_iter + 1,
                    res_norms_shared_entry.values, converged);

                if (all_converged) {
                    __syncthreads();

                    inner_loop_break_flag = true;

                    break;
                }
            }

            if (inner_loop_break_flag == true) {
                break;
            }

            // y = Hessenburg(0:restart - 1,0:restart - 1) \ s(0:restart-1)
            // x = x + KrylovBasis(:,0 : restart - 1) * y
            update_x(restart - 1, gko::batch::to_const(H_shared_entry),
                     gko::batch::to_const(s_shared_entry),
                     gko::batch::to_const(V_shared_entry), x_shared_entry,
                     y_shared_entry, converged);

            // r = b
            copy(b_global_entry, r_shared_entry, converged);
            __syncthreads();
            // r = r - A*x
            advanced_spmv_kernel(static_cast<ValueType>(-1.0), A_global_entry,
                                 gko::batch::to_const(x_shared_entry),
                                 static_cast<ValueType>(1.0), r_shared_entry);
            __syncthreads();


            compute_norm2<ValueType>(gko::batch::to_const(r_shared_entry),
                                     res_norms_shared_entry, converged);


            prec_shared.apply(gko::batch::to_const(r_shared_entry),
                              z_shared_entry);

            __syncthreads();
        }


        if (left_global_entry.values) {
            batch_scale(right_global_entry, x_shared_entry);
        }

        __syncthreads();

        // copy x back to global memory
        copy(gko::batch::to_const(x_shared_entry), x_global_entry);
    }
}