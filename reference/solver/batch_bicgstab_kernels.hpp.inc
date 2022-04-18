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

template <typename BatchMatrixType_entry, typename ValueType>
inline void initialize(
    const BatchMatrixType_entry& A_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& b_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& x_entry,
    const gko::batch_dense::BatchEntry<ValueType>& rho_old_entry,
    const gko::batch_dense::BatchEntry<ValueType>& omega_entry,
    const gko::batch_dense::BatchEntry<ValueType>& alpha_entry,
    const gko::batch_dense::BatchEntry<ValueType>& r_entry,
    const gko::batch_dense::BatchEntry<ValueType>& r_hat_entry,
    const gko::batch_dense::BatchEntry<ValueType>& p_entry,
    const gko::batch_dense::BatchEntry<ValueType>& v_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        rhs_norms_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        res_norms_entry)
{
    for (int c = 0; c < rho_old_entry.num_rhs; c++) {
        rho_old_entry.values[c] = one<ValueType>();
        omega_entry.values[c] = one<ValueType>();
        alpha_entry.values[c] = one<ValueType>();
    }

    // Compute norms of rhs
    compute_norm2<ValueType>(b_entry, rhs_norms_entry);


    // r = b
    copy(b_entry, r_entry);

    // r = b - A*x
    advanced_matvec_kernel(static_cast<ValueType>(-1.0), A_entry,
                           gko::batch::to_const(x_entry),
                           static_cast<ValueType>(1.0), r_entry);
    compute_norm2<ValueType>(gko::batch::to_const(r_entry), res_norms_entry);

    copy(gko::batch::to_const(r_entry), r_hat_entry);

    for (int r = 0; r < p_entry.num_rows; r++) {
        for (int c = 0; c < p_entry.num_rhs; c++) {
            p_entry.values[r * p_entry.stride + c] = zero<ValueType>();
            v_entry.values[r * v_entry.stride + c] = zero<ValueType>();
        }
    }
}


template <typename ValueType>
inline void update_p(
    const gko::batch_dense::BatchEntry<const ValueType>& rho_new_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& rho_old_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& omega_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& r_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& v_entry,
    const gko::batch_dense::BatchEntry<ValueType>& p_entry)
{
    const ValueType beta = (rho_new_entry.values[0] / rho_old_entry.values[0]) *
                           (alpha_entry.values[0] / omega_entry.values[0]);
    for (int r = 0; r < p_entry.num_rows; r++) {
        p_entry.values[r * p_entry.stride] =
            r_entry.values[r * r_entry.stride] +
            beta * (p_entry.values[r * p_entry.stride] -
                    omega_entry.values[0] * v_entry.values[r * v_entry.stride]);
    }
}


template <typename ValueType>
inline void compute_alpha(
    const gko::batch_dense::BatchEntry<const ValueType>& rho_new_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& r_hat_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& v_entry,
    const gko::batch_dense::BatchEntry<ValueType>& alpha_entry)
{
    compute_dot_product<ValueType>(r_hat_entry, v_entry, alpha_entry);
    alpha_entry.values[0] = rho_new_entry.values[0] / alpha_entry.values[0];
}


template <typename ValueType>
inline void update_s(
    const gko::batch_dense::BatchEntry<const ValueType>& r_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& v_entry,
    const gko::batch_dense::BatchEntry<ValueType>& s_entry)
{
    for (int r = 0; r < s_entry.num_rows; r++) {
        s_entry.values[r * s_entry.stride] =
            r_entry.values[r * r_entry.stride] -
            alpha_entry.values[0] * v_entry.values[r * v_entry.stride];
    }
}


template <typename ValueType>
inline void compute_omega(
    const gko::batch_dense::BatchEntry<const ValueType>& t_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& s_entry,
    const gko::batch_dense::BatchEntry<ValueType>& temp_entry,
    const gko::batch_dense::BatchEntry<ValueType>& omega_entry)
{
    compute_dot_product<ValueType>(t_entry, s_entry, omega_entry);
    compute_dot_product<ValueType>(t_entry, t_entry, temp_entry);
    omega_entry.values[0] /= temp_entry.values[0];
}


template <typename ValueType>
inline void update_x_and_r(
    const gko::batch_dense::BatchEntry<const ValueType>& p_hat_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& s_hat_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& omega_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& s_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& t_entry,
    const gko::batch_dense::BatchEntry<ValueType>& x_entry,
    const gko::batch_dense::BatchEntry<ValueType>& r_entry)
{
    const ValueType omega = omega_entry.values[0];
    for (int r = 0; r < x_entry.num_rows; r++) {
        x_entry.values[r * x_entry.stride] =
            x_entry.values[r * x_entry.stride] +
            alpha_entry.values[0] * p_hat_entry.values[r * p_hat_entry.stride] +
            omega * s_hat_entry.values[r * s_hat_entry.stride];

        r_entry.values[r * r_entry.stride] =
            s_entry.values[r * s_entry.stride] -
            omega * t_entry.values[r * t_entry.stride];
    }
}

template <typename ValueType>
inline void update_x_middle(
    const gko::batch_dense::BatchEntry<const ValueType>& alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& p_hat_entry,
    const gko::batch_dense::BatchEntry<ValueType>& x_entry)
{
    for (int r = 0; r < x_entry.num_rows; r++) {
        x_entry.values[r * x_entry.stride] =
            x_entry.values[r * x_entry.stride] +
            alpha_entry.values[0] * p_hat_entry.values[r * p_hat_entry.stride];
    }
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
inline void batch_entry_bicgstab_impl(
    const gko::kernels::batch_bicgstab::BatchBicgstabOptions<
        remove_complex<ValueType>>& opts,
    LogType logger, PrecType prec, const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x, const size_type ibatch,
    unsigned char* const local_space)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;
    GKO_ASSERT((nrhs <= max_num_rhs));

    unsigned char* const shared_space = local_space;
    ValueType* const r = reinterpret_cast<ValueType*>(shared_space);
    ValueType* const r_hat = r + nrows * nrhs;
    ValueType* const p = r_hat + nrows * nrhs;
    ValueType* const p_hat = p + nrows * nrhs;
    ValueType* const v = p_hat + nrows * nrhs;
    ValueType* const s = v + nrows * nrhs;
    ValueType* const s_hat = s + nrows * nrhs;
    ValueType* const t = s_hat + nrows * nrhs;
    ValueType* const prec_work = t + nrows * nrhs;
    ValueType rho_old[max_num_rhs];
    ValueType rho_new[max_num_rhs];
    ValueType omega[max_num_rhs];
    ValueType alpha[max_num_rhs];
    ValueType temp[max_num_rhs];
    real_type norms_rhs[max_num_rhs];
    real_type norms_res[max_num_rhs];

    const auto A_entry =
        gko::batch::batch_entry(gko::batch::to_const(a), ibatch);

    const gko::batch_dense::BatchEntry<const ValueType> b_entry =
        gko::batch::batch_entry(gko::batch::to_const(b), ibatch);

    const gko::batch_dense::BatchEntry<ValueType> x_entry =
        gko::batch::batch_entry(x, ibatch);


    const gko::batch_dense::BatchEntry<ValueType> r_entry{
        r, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> r_hat_entry{
        r_hat, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> p_entry{
        p, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> p_hat_entry{
        p_hat, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> v_entry{
        v, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> s_entry{
        s, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> s_hat_entry{
        s_hat, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> t_entry{
        t, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> rho_old_entry{
        rho_old, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> rho_new_entry{
        rho_new, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> omega_entry{
        omega, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> alpha_entry{
        alpha, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> temp_entry{
        temp, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<real_type> rhs_norms_entry{
        norms_rhs, static_cast<size_type>(nrhs), 1, nrhs};

    const gko::batch_dense::BatchEntry<real_type> res_norms_entry{
        norms_res, static_cast<size_type>(nrhs), 1, nrhs};

    // generate preconditioner
    prec.generate(ibatch, A_entry, prec_work);

    // initialization
    // rho_old = 1, omega = 1, alpha = 1
    // compute b norms
    // r = b - A*x
    // compute residual norms
    // r_hat = r
    // p = 0
    // v = 0
    initialize(A_entry, b_entry, gko::batch::to_const(x_entry), rho_old_entry,
               omega_entry, alpha_entry, r_entry, r_hat_entry, p_entry, v_entry,
               rhs_norms_entry, res_norms_entry);

    // stopping criterion object
    StopType stop(opts.residual_tol, rhs_norms_entry.values);

    int iter{};

    for (iter = 0; iter < opts.max_its; iter++) {
        if (stop.check_converged(res_norms_entry.values)) {
            break;
        }

        // rho_new =  < r_hat , r > = (r_hat)' * (r)
        compute_dot_product<ValueType>(gko::batch::to_const(r_hat_entry),
                                       gko::batch::to_const(r_entry),
                                       rho_new_entry);


        // beta = (rho_new / rho_old)*(alpha / omega)
        // p = r + beta*(p - omega * v)
        update_p(gko::batch::to_const(rho_new_entry),
                 gko::batch::to_const(rho_old_entry),
                 gko::batch::to_const(alpha_entry),
                 gko::batch::to_const(omega_entry),
                 gko::batch::to_const(r_entry), gko::batch::to_const(v_entry),
                 p_entry);


        // p_hat = precond * p
        prec.apply(gko::batch::to_const(p_entry), p_hat_entry);

        // v = A * p_hat
        matvec_kernel(A_entry, gko::batch::to_const(p_hat_entry), v_entry);

        // alpha = rho_new / < r_hat , v>
        compute_alpha(gko::batch::to_const(rho_new_entry),
                      gko::batch::to_const(r_hat_entry),
                      gko::batch::to_const(v_entry), alpha_entry);


        // s = r - alpha*v
        update_s(gko::batch::to_const(r_entry),
                 gko::batch::to_const(alpha_entry),
                 gko::batch::to_const(v_entry), s_entry);
        // an estimate of residual norms
        compute_norm2<ValueType>(gko::batch::to_const(s_entry),
                                 res_norms_entry);


        if (stop.check_converged(res_norms_entry.values)) {
            // update x for the sytems (rhs) which converge at this point...  x
            // = x + alpha*p_hat
            // note bits could change from 0 to 1, not the other way round, so
            // we can use xor to get info about recent convergence...
            // const uint32 converged_recent = converged_prev ^ converged;
            update_x_middle(gko::batch::to_const(alpha_entry),
                            gko::batch::to_const(p_hat_entry), x_entry);
            break;
        }

        // s_hat = precond * s
        prec.apply(gko::batch::to_const(s_entry), s_hat_entry);

        // t = A * s_hat
        matvec_kernel(A_entry, gko::batch::to_const(s_hat_entry), t_entry);

        // omega = <t,s> / <t,t>
        compute_omega(gko::batch::to_const(t_entry),
                      gko::batch::to_const(s_entry), temp_entry, omega_entry);


        // x = x + alpha*p_hat + omega *s_hat
        // r = s - omega * t
        update_x_and_r(gko::batch::to_const(p_hat_entry),
                       gko::batch::to_const(s_hat_entry),
                       gko::batch::to_const(alpha_entry),
                       gko::batch::to_const(omega_entry),
                       gko::batch::to_const(s_entry),
                       gko::batch::to_const(t_entry), x_entry, r_entry);

        compute_norm2<ValueType>(gko::batch::to_const(r_entry),
                                 res_norms_entry);

        // rho_old = rho_new
        copy(gko::batch::to_const(rho_new_entry), rho_old_entry);
    }

    logger.log_iteration(ibatch, iter, res_norms_entry.values[0]);
}
