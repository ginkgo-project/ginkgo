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

for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
    byte *const shared_space = local_space.get_data();
    ValueType *const r = reinterpret_cast<ValueType *>(shared_space);
    ValueType *const t = r + nrows * nrhs;
    ValueType *const v = t + nrows * nrhs;
    ValueType *const xs = v + nrows * nrhs;
    ValueType *const rs = xs + nrows * nrhs;
    ValueType *const helper = rs + nrows * nrhs;
    ValueType *const f = helper + nrows * nrhs;
    ValueType *const c = f + subspace_dim * nrhs;
    ValueType *const Subspace_vectors = c + subspace_dim * nrhs;
    ValueType *const G = Subspace_vectors + nrows * subspace_dim;
    ValueType *const U = G + nrows * subspace_dim * nrhs;
    ValueType *const M = U + nrows * subspace_dim * nrhs;
    ValueType *const prec_work = M + subspace_dim * subspace_dim * nrhs;
    ValueType *const temp_for_single_rhs =
        prec_work + PrecType::dynamic_work_size(nrows, a.num_nnz);
    ValueType *const omega = temp_for_single_rhs + nrows;
    ValueType *const temp1 = omega + nrhs;
    ValueType *const temp2 = temp1 + nrhs;
    real_type *const norms_t = reinterpret_cast<real_type *>(temp2 + nrhs);
    real_type *const norms_r = norms_t + nrhs;
    real_type *const norms_rhs = norms_r + nrhs;
    real_type *const norms_res = norms_rhs + nrhs;
    real_type *const norms_tmp = norms_res + nrhs;


    uint32 converged = 0;

    const gko::batch_dense::BatchEntry<const ValueType> left_entry =
        gko::batch::batch_entry(left, ibatch);

    const gko::batch_dense::BatchEntry<const ValueType> right_entry =
        gko::batch::batch_entry(right, ibatch);

    // scale the matrix and rhs
    if (left_entry.values) {
        const typename BatchMatrixType::entry_type A_entry =
            gko::batch::batch_entry(a, ibatch);
        const gko::batch_dense::BatchEntry<ValueType> b_entry =
            gko::batch::batch_entry(b, ibatch);
        batch_scale(left_entry, right_entry, A_entry);
        batch_dense::batch_scale(left_entry, b_entry);
    }

    // const typename BatchMatrixType::entry_type A_entry =
    const auto A_entry =
        gko::batch::batch_entry(gko::batch::to_const(a), ibatch);

    const gko::batch_dense::BatchEntry<const ValueType> b_entry =
        gko::batch::batch_entry(gko::batch::to_const(b), ibatch);

    const gko::batch_dense::BatchEntry<ValueType> x_entry =
        gko::batch::batch_entry(x, ibatch);


    const gko::batch_dense::BatchEntry<ValueType> r_entry{
        r, static_cast<size_type>(nrhs), nrows, nrhs};
    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix

    const gko::batch_dense::BatchEntry<ValueType> t_entry{
        t, static_cast<size_type>(nrhs), nrows, nrhs};
    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix

    const gko::batch_dense::BatchEntry<ValueType> v_entry{
        v, static_cast<size_type>(nrhs), nrows, nrhs};
    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix

    const gko::batch_dense::BatchEntry<ValueType> helper_entry{
        helper, static_cast<size_type>(nrhs), nrows, nrhs};
    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix

    const gko::batch_dense::BatchEntry<ValueType> xs_entry{
        xs, static_cast<size_type>(nrhs), nrows, nrhs};
    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix

    const gko::batch_dense::BatchEntry<ValueType> rs_entry{
        rs, static_cast<size_type>(nrhs), nrows, nrhs};
    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix

    const gko::batch_dense::BatchEntry<ValueType> f_entry{
        f, static_cast<size_type>(nrhs), static_cast<int>(subspace_dim), nrhs};
    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix

    const gko::batch_dense::BatchEntry<ValueType> c_entry{
        c, static_cast<size_type>(nrhs), static_cast<int>(subspace_dim), nrhs};
    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix

    // P = [ p_0 , p_1 , ... , p_(subspace_dim - 1) ] , subspace S is the
    // left null space of matrix P to store subspace defining vectors: p_i ,
    // i = 0, ..., subspace_dim -1 , we use a matrix named Subspace_vectors
    const gko::batch_dense::BatchEntry<ValueType> Subspace_vectors_entry{
        Subspace_vectors, 1, nrows * static_cast<int>(subspace_dim), 1};
    // storage:row-major order , subspace vectors
    // are stored in a single col. one after the other-(matrix
    // Subspace_vectors on paper). And to get p_i : that is ith subspace
    // vector : p_i_entry{  &Subspace_vectors[i*
    // Subspace_vectors_entry.stride * nrows], Subspace_vectors_entry.stride
    // , nrows, 1 }; So, effectively the cols. are stored contiguously in
    // memory one after the other as Subspace_vectors_entry.stride = 1

    // to store vectors: u_i , i = 0, ..., subspace_dim -1 , we use matrix U
    const gko::batch_dense::BatchEntry<ValueType> U_entry{
        U, static_cast<size_type>(nrhs), nrows * static_cast<int>(subspace_dim),
        nrhs};
    // storage:row-major order , vectors corr. to each rhs
    // are stored in a single col. one after the other-(matrix U on paper).
    // And to get u_i : that is ith  vector for each rhs: u_i_entry{  &U[i*
    // U_entry.stride * nrows], U_entry.stride , nrows, nrhs}; So if nrhs=1,
    // effectively the cols. are stored contiguously in memory one after the
    // other.

    // to store vectors: g_i , i = 0, ..., subspace_dim -1, we use matrix G
    const gko::batch_dense::BatchEntry<ValueType> G_entry{
        G, static_cast<size_type>(nrhs), nrows * static_cast<int>(subspace_dim),
        nrhs};
    // storage:row-major order , vectors corr. to each rhs
    // are stored in a single col. one after the other-(matrix G on paper).
    // And to get g_i : that is ith  vector for each rhs: g_i_entry{  &G[i*
    // G_entry.stride * nrows], G_entry.stride , nrows, nrhs}; So if nrhs=1,
    // effectively the cols. are stored contiguously in memory one after the
    // other.


    const gko::batch_dense::BatchEntry<ValueType> M_entry{
        M, subspace_dim * static_cast<size_type>(nrhs),
        static_cast<int>(subspace_dim), static_cast<int>(subspace_dim) * nrhs};
    // storage:row-major ,  entry (i,j) for different RHSs are placed one
    // after the other in a row - when drawn on paper, (and the same is true
    // for actual storage as the storage order is row-major) to get entry
    // (i,j) for rhs: rhs_k , scalar_M_i_j_for_rhs_k =  M[M_entry.stride*i +
    // j*nrhs  + rhs_k ]


    const gko::batch_dense::BatchEntry<ValueType> temp_for_single_rhs_entry{
        temp_for_single_rhs, static_cast<size_type>(1), nrows, 1};

    const gko::batch_dense::BatchEntry<ValueType> omega_entry{
        omega, static_cast<size_type>(nrhs), 1, nrhs};


    const gko::batch_dense::BatchEntry<ValueType> temp1_entry{
        temp1, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> temp2_entry{
        temp2, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<real_type> t_norms_entry{
        norms_t, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<real_type> r_norms_entry{
        norms_r, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<real_type> rhs_norms_entry{
        norms_rhs, static_cast<size_type>(nrhs), 1, nrhs};


    const gko::batch_dense::BatchEntry<real_type> res_norms_entry{
        norms_res, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<real_type> tmp_norms_entry{
        norms_tmp, static_cast<size_type>(subspace_dim), 1,
        static_cast<int>(subspace_dim)};

    // generate preconditioner
    prec.generate(A_entry, prec_work);

    // initialization
    // compute b norms
    // r = b - A*x
    // compute residual norms
    // initialize G, U with zeroes
    // M = Identity
    // xs = x and rs = r if smoothing is enabled
    // initialize (either random numbers or deterministically) and
    // orthonormalize Subspace_vectors omega = 1
    initialize(A_entry, b_entry, gko::batch::to_const(x_entry), r_entry,
               G_entry, U_entry, M_entry, Subspace_vectors_entry, deterministic,
               xs_entry, rs_entry, smoothing, omega_entry, rhs_norms_entry,
               res_norms_entry, temp_for_single_rhs_entry, tmp_norms_entry);


    // stopping criterion object
    StopType stop(nrhs, opts.max_its, opts.residual_tol, rhs_norms_entry.values,
                  converged);

    int outer_iter = -1;

    while (1) {
        outer_iter++;


        bool all_converged = stop.check_converged(
            outer_iter, res_norms_entry.values, {NULL, 0, 0, 0}, converged);

        logger.log_iteration(ibatch, outer_iter, res_norms_entry.values,
                             converged);

        if (all_converged) {
            break;
        }

        // f = HermitianTranspose(P) * r
        update_f(gko::batch::to_const(Subspace_vectors_entry),
                 gko::batch::to_const(r_entry), subspace_dim, f_entry,
                 converged);


        for (size_type k = 0; k < subspace_dim; k++) {
            const gko::batch_dense::BatchEntry<ValueType> u_k_entry{
                &U_entry.values[k * nrows * U_entry.stride], U_entry.stride,
                nrows, nrhs};

            const gko::batch_dense::BatchEntry<ValueType> g_k_entry{
                &G_entry.values[k * nrows * G_entry.stride], G_entry.stride,
                nrows, nrhs};


            // solve c from Mc = f (Lower Triangular solve)
            update_c(gko::batch::to_const(M_entry),
                     gko::batch::to_const(f_entry), c_entry, temp1_entry,
                     converged);

            // v = r - ( c(k) * g_k  +  c(k+1) * g_(k+1)  + ...  +
            // c(subspace_dim - 1) * g_(subspace_dim - 1))
            update_v(gko::batch::to_const(G_entry),
                     gko::batch::to_const(c_entry),
                     gko::batch::to_const(r_entry), v_entry, k, converged);


            // helper = v
            batch_dense::copy(gko::batch::to_const(v_entry), helper_entry,
                              converged);

            // v = precond * helper
            prec.apply(gko::batch::to_const(helper_entry), v_entry);


            // u_k = omega * v + (c(k) * u_k  +  c(k+1) * u_(k+1) + ...  +
            // c(subspace_dim - 1) * u_(subspace_dim - 1) )
            update_u_k(gko::batch::to_const(omega_entry),
                       gko::batch::to_const(c_entry),
                       gko::batch::to_const(v_entry),
                       gko::batch::to_const(U_entry), k, helper_entry,
                       u_k_entry, converged);


            // g_k = A * u_k
            spmv_kernel(A_entry, gko::batch::to_const(u_k_entry), g_k_entry);


            // for i = 0 to k-1
            //     alpha = (p_i * g_k)/M(i,i)
            //     g_k = g_k - alpha * g_i
            //     u_k = u_k - alpha * u_i
            // end
            const gko::batch_dense::BatchEntry<ValueType> &alpha_entry =
                temp1_entry;
            update_g_k_and_u_k(k, gko::batch::to_const(G_entry),
                               gko::batch::to_const(U_entry),
                               gko::batch::to_const(Subspace_vectors_entry),
                               gko::batch::to_const(M_entry), alpha_entry,
                               g_k_entry, u_k_entry, converged);


            // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1
            update_M(gko::batch::to_const(g_k_entry),
                     gko::batch::to_const(Subspace_vectors_entry), M_entry, k,
                     converged);


            // beta = f(k)/M(k,k)
            const gko::batch_dense::BatchEntry<ValueType> &beta_entry =
                temp1_entry;
            for (int rhs = 0; rhs < nrhs; rhs++) {
                const uint32 conv = converged & (1 << rhs);

                if (conv) {
                    continue;
                }

                beta_entry.values[rhs] =
                    f_entry.values[k * f_entry.stride + rhs] /
                    M_entry.values[k * M_entry.stride + k * nrhs + rhs];
            }


            // r = r - beta * g_k
            // x = x + beta * u_k
            update_r_and_x_inner_loop(gko::batch::to_const(g_k_entry),
                                      gko::batch::to_const(u_k_entry),
                                      gko::batch::to_const(beta_entry), r_entry,
                                      x_entry, converged);


            if (smoothing == true) {
                const gko::batch_dense::BatchEntry<ValueType> &gamma_entry =
                    temp2_entry;
                smoothing_operation(gko::batch::to_const(x_entry),
                                    gko::batch::to_const(r_entry), gamma_entry,
                                    t_entry, xs_entry, rs_entry, t_norms_entry,
                                    converged);
            }


            // if k + 1 <= subspace_dim - 1
            //     f(i) = 0 , where i = 0,...,k
            //     f(i) = f(i) - beta * M(i,k) ,where i = k + 1, ... ,
            //     subspace_dim -1
            // end if
            if (k + 1 <= subspace_dim - 1) {
                for (int row = 0; row <= k; row++) {
                    for (int rhs = 0; rhs < nrhs; rhs++) {
                        f_entry.values[row * f_entry.stride + rhs] =
                            zero<ValueType>();
                    }
                }

                for (int row = k + 1; row < subspace_dim; row++) {
                    for (int rhs = 0; rhs < nrhs; rhs++) {
                        f_entry.values[row * f_entry.stride + rhs] -=
                            beta_entry.values[rhs] *
                            M_entry
                                .values[row * M_entry.stride + k * nrhs + rhs];
                    }
                }
            }


        }  // end of inner loop


        // v = precond * r
        prec.apply(gko::batch::to_const(r_entry), v_entry);

        // t = A *v
        spmv_kernel(A_entry, gko::batch::to_const(v_entry), t_entry);


        // omega = ( t * r )/ (t * t)
        // rho = (t * r ) /(||t|| * || r||)
        // if |rho| < kappa
        //      omega = omega * kappa / |rho|
        // end if
        const gko::batch_dense::BatchEntry<ValueType> &t_r_dot_entry =
            temp1_entry;
        const gko::batch_dense::BatchEntry<ValueType> &rho_entry = temp2_entry;
        compute_omega(gko::batch::to_const(t_entry),
                      gko::batch::to_const(r_entry), rho_entry, t_r_dot_entry,
                      t_norms_entry, r_norms_entry, omega_entry, kappa,
                      converged);


        // r = r - omega * t
        // x = x + omega * v
        update_r_and_x_outer_loop(
            gko::batch::to_const(t_entry), gko::batch::to_const(v_entry),
            gko::batch::to_const(omega_entry), r_entry, x_entry, converged);


        if (smoothing == true) {
            const gko::batch_dense::BatchEntry<ValueType> &gamma_entry =
                temp2_entry;
            smoothing_operation(gko::batch::to_const(x_entry),
                                gko::batch::to_const(r_entry), gamma_entry,
                                t_entry, xs_entry, rs_entry, t_norms_entry,
                                converged);

            batch_dense::compute_norm2<ValueType>(
                gko::batch::to_const(rs_entry), res_norms_entry,
                converged);  // residual norms


        } else {
            batch_dense::compute_norm2<ValueType>(gko::batch::to_const(r_entry),
                                                  res_norms_entry,
                                                  converged);  // residual norms
        }
    }

    if (smoothing == true) {
        batch_dense::copy(gko::batch::to_const(xs_entry), x_entry);
        batch_dense::copy(gko::batch::to_const(rs_entry), r_entry);
    }

    if (left_entry.values) {
        batch_dense::batch_scale(right_entry, x_entry);
    }
}