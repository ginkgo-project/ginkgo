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

template <typename BatchMatrixType_entry, typename PrecType, typename ValueType>
inline void initialize(
    const BatchMatrixType_entry& A_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& b_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& x_entry,
    const gko::batch_dense::BatchEntry<ValueType>& r_entry,
    const PrecType& prec,
    const gko::batch_dense::BatchEntry<ValueType>& z_entry,
    const gko::batch_dense::BatchEntry<ValueType>& cs_entry,
    const gko::batch_dense::BatchEntry<ValueType>& sn_entry,
    const gko::batch_dense::BatchEntry<ValueType>& V_entry,
    const gko::batch_dense::BatchEntry<ValueType>& H_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        rhs_norms_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        res_norms_entry)
{
    // Compute norms of rhs
    compute_norm2<ValueType>(b_entry, rhs_norms_entry);


    // r = b
    copy(b_entry, r_entry);

    // r = b - A*x
    advanced_matvec_kernel(static_cast<ValueType>(-1.0), A_entry,
                           gko::batch::to_const(x_entry),
                           static_cast<ValueType>(1.0), r_entry);
    compute_norm2<ValueType>(gko::batch::to_const(r_entry), res_norms_entry);


    // z = precond * r
    prec.apply(gko::batch::to_const(r_entry), z_entry);

    // initialize cs with zeroes
    for (int r = 0; r < cs_entry.num_rows; r++) {
        for (int c = 0; c < cs_entry.num_rhs; c++) {
            cs_entry.values[r * cs_entry.stride + c] = zero<ValueType>();
        }
    }

    // initialize sn with zeroes
    for (int r = 0; r < sn_entry.num_rows; r++) {
        for (int c = 0; c < sn_entry.num_rhs; c++) {
            sn_entry.values[r * sn_entry.stride + c] = zero<ValueType>();
        }
    }

    // initialize V with zeroes
    for (int r = 0; r < V_entry.num_rows; r++) {
        for (int c = 0; c < V_entry.num_rhs; c++) {
            V_entry.values[r * V_entry.stride + c] = zero<ValueType>();
        }
    }


    // initialize H with zeroes
    for (int r = 0; r < H_entry.num_rows; r++) {
        for (int c = 0; c < H_entry.num_rhs; c++) {
            H_entry.values[r * H_entry.stride + c] = zero<ValueType>();
        }
    }
}


template <typename ValueType>
inline void apply_plane_rotation(const ValueType* const cs,
                                 const ValueType* const sn, ValueType* const x,
                                 ValueType* const y)
{
    const ValueType temp = cs[0] * x[0] + sn[0] * y[0];
    y[0] = -one<ValueType>() * conj(sn[0]) * x[0] + conj(cs[0]) * y[0];
    x[0] = temp;
}


template <typename ValueType>
inline void generate_and_apply_plane_rot_fused(ValueType* const cs_i,
                                               ValueType* const sn_i,
                                               ValueType* const H_i_i,
                                               ValueType* const H_i_plus_1_i,
                                               ValueType* const s_i,
                                               ValueType* const s_i_plus_1)
{
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


template <typename ValueType>
inline void update_v_naught_and_s(
    const gko::batch_dense::BatchEntry<const ValueType>& z_entry,
    const gko::batch_dense::BatchEntry<ValueType>& V_entry,
    const gko::batch_dense::BatchEntry<ValueType>& s_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        tmp_norms_entry)
{
    using real_type = typename gko::remove_complex<ValueType>;

    const gko::batch_dense::BatchEntry<ValueType> v_naught_entry{
        V_entry.values, V_entry.stride, z_entry.num_rows, V_entry.num_rhs};

    const gko::batch_dense::BatchEntry<real_type>& z_norms_entry =
        tmp_norms_entry;

    compute_norm2(z_entry, z_norms_entry);

    for (int r = 0; r < v_naught_entry.num_rows; r++) {
        v_naught_entry.values[r * v_naught_entry.stride] =
            z_entry.values[r * z_entry.stride] /
            static_cast<ValueType>(z_norms_entry.values[0]);
    }


    for (int r = 0; r < s_entry.num_rows; r++) {
        s_entry.values[r * s_entry.stride] = zero<ValueType>();
    }

    s_entry.values[0] = static_cast<ValueType>(z_norms_entry.values[0]);
}


template <typename BatchMatrixEntry, typename ValueType, typename PrecType>
inline void arnoldi_method(
    const BatchMatrixEntry& A_entry, const int inner_iter,
    const gko::batch_dense::BatchEntry<ValueType>& V_entry,
    const gko::batch_dense::BatchEntry<ValueType>& H_entry,
    const gko::batch_dense::BatchEntry<ValueType>& w_entry,
    const gko::batch_dense::BatchEntry<ValueType>& helper_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        tmp_norms_entry,
    const PrecType& prec)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const int& i = inner_iter;
    const int num_rows = w_entry.num_rows;
    const int num_rhs = w_entry.num_rhs;

    const gko::batch_dense::BatchEntry<const ValueType> v_i_entry{
        &V_entry.values[i * V_entry.stride * num_rows], V_entry.stride,
        num_rows, V_entry.num_rhs};

    matvec_kernel(A_entry, v_i_entry, helper_entry);

    prec.apply(gko::batch::to_const(helper_entry), w_entry);

    for (int k = 0; k <= i; k++) {
        const gko::batch_dense::BatchEntry<const ValueType> v_k_entry{
            &V_entry.values[k * V_entry.stride * num_rows], V_entry.stride,
            num_rows, V_entry.num_rhs};

        const gko::batch_dense::BatchEntry<ValueType> h_k_i_entry{
            &H_entry.values[k * H_entry.stride + i * num_rhs],
            static_cast<size_type>(w_entry.num_rhs), 1, w_entry.num_rhs};

        compute_dot_product(gko::batch::to_const(w_entry), v_k_entry,
                            h_k_i_entry);

        for (int r = 0; r < w_entry.num_rows; r++) {
            ValueType h_k_i_scalar = h_k_i_entry.values[0];

            w_entry.values[r * w_entry.stride] -=
                h_k_i_scalar * v_k_entry.values[r * v_k_entry.stride];
        }
    }

    const gko::batch_dense::BatchEntry<real_type>& w_norms_entry =
        tmp_norms_entry;

    compute_norm2(gko::batch::to_const(w_entry), w_norms_entry);

    const gko::batch_dense::BatchEntry<ValueType> h_i_plus_1_i_entry{
        &H_entry.values[(i + 1) * H_entry.stride + i * num_rhs],
        static_cast<size_type>(w_entry.num_rhs), 1, w_entry.num_rhs};

    h_i_plus_1_i_entry.values[0] =
        static_cast<ValueType>(w_norms_entry.values[0]);

    const gko::batch_dense::BatchEntry<ValueType> v_i_plus_1_entry{
        &V_entry.values[(i + 1) * V_entry.stride * num_rows], V_entry.stride,
        num_rows, V_entry.num_rhs};

    for (int r = 0; r < w_entry.num_rows; r++) {
        v_i_plus_1_entry.values[r * v_i_plus_1_entry.stride] =
            w_entry.values[r * w_entry.stride] / h_i_plus_1_i_entry.values[0];
    }
}

template <typename ValueType>
inline void update_x(
    const int m, const gko::batch_dense::BatchEntry<const ValueType>& H_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& s_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& V_entry,
    const gko::batch_dense::BatchEntry<ValueType>& x_entry,
    const gko::batch_dense::BatchEntry<ValueType>& y_entry)
{
    // upper triangular solve
    for (int r = m; r >= 0; r--) {
        ValueType temp_sum = zero<ValueType>();
        for (int t = m; t > r; t--) {
            temp_sum +=
                H_entry.values[r * H_entry.stride + t * y_entry.num_rhs] *
                y_entry.values[t * y_entry.stride];
        }
        y_entry.values[r * y_entry.stride] =
            (s_entry.values[r * s_entry.stride] - temp_sum) /
            H_entry.values[r * H_entry.stride + r * y_entry.num_rhs];
    }

    // dense mat * vec multiplication
    for (int a = 0; a < m + 1; a++) {
        for (int r = 0; r < x_entry.num_rows; r++) {
            x_entry.values[r * x_entry.stride] +=
                V_entry.values[a * V_entry.stride * x_entry.num_rows +
                               r * V_entry.stride] *
                y_entry.values[a * y_entry.stride];
        }
    }
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
inline void batch_entry_gmres_impl(
    const gko::kernels::batch_gmres::BatchGmresOptions<
        remove_complex<ValueType>>& opts,
    LogType logger, PrecType prec, const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x, const size_type ibatch,
    unsigned char* const local_space)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;
    const auto restart = opts.restart_num;

    ValueType* const r = reinterpret_cast<ValueType*>(local_space);
    ValueType* const z = r + nrows * nrhs;
    ValueType* const w = z + nrows * nrhs;
    ValueType* const helper = w + nrows * nrhs;
    ValueType* const cs = helper + nrows * nrhs;
    ValueType* const sn = cs + restart * nrhs;
    ValueType* const y = sn + restart * nrhs;
    ValueType* const s = y + restart * nrhs;
    ValueType* const H = s + (restart + 1) * nrhs;  // Hessenberg matrix
    ValueType* const V =
        H + restart * (restart + 1) * nrhs;  // Krylov subspace basis vectors
    ValueType* const prec_work = V + nrows * (restart + 1) * nrhs;

    real_type* const norms_rhs = reinterpret_cast<real_type*>(
        prec_work + PrecType::dynamic_work_size(nrows, a.num_nnz));
    real_type* const norms_res = norms_rhs + nrhs;
    real_type* const norms_tmp = norms_res + nrhs;

    const auto A_entry = gko::batch::batch_entry(a, ibatch);
    const gko::batch_dense::BatchEntry<const ValueType> b_entry =
        gko::batch::batch_entry(b, ibatch);
    const gko::batch_dense::BatchEntry<ValueType> x_entry =
        gko::batch::batch_entry(x, ibatch);

    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix
    const gko::batch_dense::BatchEntry<ValueType> r_entry{
        r, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> z_entry{
        z, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> w_entry{
        w, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> helper_entry{
        helper, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> cs_entry{
        cs, static_cast<size_type>(nrhs), restart, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> sn_entry{
        sn, static_cast<size_type>(nrhs), restart, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> y_entry{
        y, static_cast<size_type>(nrhs), restart, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> s_entry{
        s, static_cast<size_type>(nrhs), restart + 1, nrhs};

    // storage:row-major ,  entry (i,j) for different RHSs are placed after
    // the other in a row - when drawn on paper, (and the same is true for
    // actual storage as the storage order is row-major) to get entry (i,j)
    // for rhs: rhs_k , H_entry.stride*i + j*nrhs  + rhs_k
    const gko::batch_dense::BatchEntry<ValueType> H_entry{
        H, static_cast<size_type>(nrhs * restart), restart + 1, restart * nrhs};

    // storage:row-major order , subspace basis vectors corr. to each rhs
    // are stored in a single col. one after the other-(on paper). And to
    // get vi : that is ith basis vector for each rhs: vi_entry{  &V[i*
    // V_entry.stride * nrows], V_entry.stride , nrows, nrhs}; So if nrhs=1,
    // effectively the cols. are stored contiguously in memory one after the
    // other.
    const gko::batch_dense::BatchEntry<ValueType> V_entry{
        V, static_cast<size_type>(nrhs), nrows * (restart + 1), nrhs};

    const gko::batch_dense::BatchEntry<real_type> rhs_norms_entry{
        norms_rhs, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<real_type> res_norms_entry{
        norms_res, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<real_type> tmp_norms_entry{
        norms_tmp, static_cast<size_type>(nrhs), 1, nrhs};

    // generate preconditioner
    prec.generate(ibatch, A_entry, prec_work);

    // initialization
    // compute b norms
    // r = b - A*x
    // z = precond*r
    // compute residual norms
    // initialize V,H,cs,sn with zeroes
    initialize(A_entry, b_entry, gko::batch::to_const(x_entry), r_entry, prec,
               z_entry, cs_entry, sn_entry, V_entry, H_entry, rhs_norms_entry,
               res_norms_entry);

    // stopping criterion object
    StopType stop(opts.residual_tol, rhs_norms_entry.values);

    int total_iter = 0;

    // Note: restart - inner iterations and the outer iteration:  are
    // counted as (restart + 1) number of iterations instead of one.
    while (1) {
        if (stop.check_converged(res_norms_entry.values)) {
            break;
        }

        // KrylovBasis_0 = z/norm(z)
        // s -> fill with zeroes
        // s(0) = norm(z)
        update_v_naught_and_s(gko::batch::to_const(z_entry), V_entry, s_entry,
                              tmp_norms_entry);

        bool inner_loop_break_flag = false;
        int inner_iter = 0;
        for (; inner_iter < restart; inner_iter++) {
            // w_temp = A * v_i
            // w = precond * w_temp
            // i = inner_iter
            // for k = 0 to inner_iter
            //     Hessenburg(k,i) =  w' * v_k
            //     w = w - Hessenburg(k,i) * v_k
            // end
            // Hessenburg(i+1, i) = norm(w)
            // KrylovBasis_i+1 = w / Hessenburg(i+1,i)
            arnoldi_method(A_entry, inner_iter, V_entry, H_entry, w_entry,
                           helper_entry, tmp_norms_entry, prec);


            for (int k = 0; k < inner_iter; k++) {
                // temp = cs(k) * Hessenberg( k, inner_iter )  +   sn(k) *
                // Hessenberg(k + 1, inner_iter)
                // Hessenberg(k + 1, inner_iter) = -1 * conj(sn(k)) *
                // Hessenberg( k , inner_iter) + conj(cs(k)) * Hessenberg(k
                // + 1 , inner_iter) Hessenberg(k,inner_iter) = temp
                apply_plane_rotation(
                    &cs_entry.values[k * cs_entry.stride],
                    &sn_entry.values[k * sn_entry.stride],
                    &H_entry.values[k * H_entry.stride + inner_iter * nrhs],
                    &H_entry
                         .values[(k + 1) * H_entry.stride + inner_iter * nrhs]);
            }

            // compute sine and cos
            // temp = cs(inner_iter) * s(inner_iter)
            // s(inner_iter + 1) = -1 * conj(sn(inner_iter)) * s(inner_iter)
            // s(inner_iter) = temp
            // Hessenberg(inner_iter , inner_iter) = cs(inner_iter) *
            // Hessenberg(inner_iter , inner_iter) + sn(inner_iter) *
            // Hessenberg(inner_iter + 1, inner_iter) Hessenberg(inner_iter
            // + 1, inner_iter) = 0
            generate_and_apply_plane_rot_fused(
                &cs_entry.values[inner_iter * cs_entry.stride],
                &sn_entry.values[inner_iter * sn_entry.stride],
                &H_entry
                     .values[inner_iter * H_entry.stride + inner_iter * nrhs],
                &H_entry.values[(inner_iter + 1) * H_entry.stride +
                                inner_iter * nrhs],
                &s_entry.values[inner_iter * s_entry.stride],
                &s_entry.values[(inner_iter + 1) * s_entry.stride]);

            // estimate of residual norms
            // residual = abs(s(inner_iter + 1))
            res_norms_entry.values[0] =
                abs(s_entry.values[(inner_iter + 1) * s_entry.stride]);

            total_iter++;

            // y = Hessenburg(0 : inner_iter,0 : inner_iter) \ s(0 :
            // inner_iter) x = x + KrylovBasis(:, 0 : inner_iter ) * y
            if (stop.check_converged(res_norms_entry.values)) {
                update_x(inner_iter, gko::batch::to_const(H_entry),
                         gko::batch::to_const(s_entry),
                         gko::batch::to_const(V_entry), x_entry, y_entry);
                inner_loop_break_flag = true;
                break;
            }
        }

        if (inner_loop_break_flag == true) {
            break;
        }

        // y = Hessenburg(0:restart - 1,0:restart - 1) \ s(0:restart-1)
        // x = x + KrylovBasis(:,0 : restart - 1) * y
        update_x(restart - 1, gko::batch::to_const(H_entry),
                 gko::batch::to_const(s_entry), gko::batch::to_const(V_entry),
                 x_entry, y_entry);

        // r = b
        copy(b_entry, r_entry);

        // r = r - A*x
        advanced_matvec_kernel(static_cast<ValueType>(-1.0), A_entry,
                               gko::batch::to_const(x_entry),
                               static_cast<ValueType>(1.0), r_entry);
        compute_norm2<ValueType>(gko::batch::to_const(r_entry),
                                 res_norms_entry);

        // z = precond * r
        prec.apply(gko::batch::to_const(r_entry), z_entry);
    }

    logger.log_iteration(ibatch, total_iter, res_norms_entry.values[0]);
}
