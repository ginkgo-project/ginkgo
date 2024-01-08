// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename BatchMatrixType_entry, typename ValueType>
inline void initialize(
    const BatchMatrixType_entry& A_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& b_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& x_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& rho_old_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& omega_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& alpha_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& r_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& r_hat_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& p_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& p_hat_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& v_entry,
    const gko::batch::multi_vector::batch_item<
        typename gko::remove_complex<ValueType>>& rhs_norms_entry,
    const gko::batch::multi_vector::batch_item<
        typename gko::remove_complex<ValueType>>& res_norms_entry)
{
    rho_old_entry.values[0] = one<ValueType>();
    omega_entry.values[0] = one<ValueType>();
    alpha_entry.values[0] = one<ValueType>();

    // Compute norms of rhs
    compute_norm2_kernel<ValueType>(b_entry, rhs_norms_entry);

    // r = b
    copy_kernel(b_entry, r_entry);

    // r = b - A*x
    advanced_apply_kernel(static_cast<ValueType>(-1.0), A_entry,
                          gko::batch::to_const(x_entry),
                          static_cast<ValueType>(1.0), r_entry);
    compute_norm2_kernel<ValueType>(gko::batch::to_const(r_entry),
                                    res_norms_entry);

    for (int r = 0; r < p_entry.num_rows; r++) {
        r_hat_entry.values[r * r_hat_entry.stride] =
            r_entry.values[r * r_entry.stride];
        p_entry.values[r * p_entry.stride] = zero<ValueType>();
        p_hat_entry.values[r * p_hat_entry.stride] = zero<ValueType>();
        v_entry.values[r * v_entry.stride] = zero<ValueType>();
    }
}


template <typename ValueType>
inline void update_p(
    const gko::batch::multi_vector::batch_item<const ValueType>& rho_new_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& rho_old_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& alpha_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& omega_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& r_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& v_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& p_entry)
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
    const gko::batch::multi_vector::batch_item<const ValueType>& rho_new_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& r_hat_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& v_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& alpha_entry)
{
    compute_dot_product_kernel<ValueType>(r_hat_entry, v_entry, alpha_entry);
    alpha_entry.values[0] = rho_new_entry.values[0] / alpha_entry.values[0];
}


template <typename ValueType>
inline void update_s(
    const gko::batch::multi_vector::batch_item<const ValueType>& r_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& alpha_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& v_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& s_entry)
{
    for (int r = 0; r < s_entry.num_rows; r++) {
        s_entry.values[r * s_entry.stride] =
            r_entry.values[r * r_entry.stride] -
            alpha_entry.values[0] * v_entry.values[r * v_entry.stride];
    }
}


template <typename ValueType>
inline void compute_omega(
    const gko::batch::multi_vector::batch_item<const ValueType>& t_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& s_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& temp_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& omega_entry)
{
    compute_dot_product_kernel<ValueType>(t_entry, s_entry, omega_entry);
    compute_dot_product_kernel<ValueType>(t_entry, t_entry, temp_entry);
    omega_entry.values[0] /= temp_entry.values[0];
}


template <typename ValueType>
inline void update_x_and_r(
    const gko::batch::multi_vector::batch_item<const ValueType>& p_hat_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& s_hat_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& alpha_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& omega_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& s_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& t_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& x_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& r_entry)
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
    const gko::batch::multi_vector::batch_item<const ValueType>& alpha_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& p_hat_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& x_entry)
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
    const gko::kernels::batch_bicgstab::settings<remove_complex<ValueType>>&
        settings,
    LogType logger, PrecType prec, const BatchMatrixType& a,
    const gko::batch::multi_vector::uniform_batch<const ValueType>& b,
    const gko::batch::multi_vector::uniform_batch<ValueType>& x,
    const size_type batch_item_id, unsigned char* const local_space)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto num_rows = a.num_rows;
    const auto num_rhs = b.num_rhs;
    GKO_ASSERT(num_rhs <= max_num_rhs);

    unsigned char* const shared_space = local_space;
    ValueType* const r = reinterpret_cast<ValueType*>(shared_space);
    ValueType* const r_hat = r + num_rows * num_rhs;
    ValueType* const p = r_hat + num_rows * num_rhs;
    ValueType* const p_hat = p + num_rows * num_rhs;
    ValueType* const v = p_hat + num_rows * num_rhs;
    ValueType* const s = v + num_rows * num_rhs;
    ValueType* const s_hat = s + num_rows * num_rhs;
    ValueType* const t = s_hat + num_rows * num_rhs;
    ValueType* const prec_work = t + num_rows * num_rhs;
    ValueType rho_old[max_num_rhs];
    ValueType rho_new[max_num_rhs];
    ValueType omega[max_num_rhs];
    ValueType alpha[max_num_rhs];
    ValueType temp[max_num_rhs];
    real_type norms_rhs[max_num_rhs];
    real_type norms_res[max_num_rhs];

    const auto A_entry = gko::batch::matrix::extract_batch_item(
        gko::batch::matrix::to_const(a), batch_item_id);
    const gko::batch::multi_vector::batch_item<const ValueType> b_entry =
        gko::batch::extract_batch_item(gko::batch::to_const(b), batch_item_id);
    const gko::batch::multi_vector::batch_item<ValueType> x_entry =
        gko::batch::extract_batch_item(x, batch_item_id);

    const gko::batch::multi_vector::batch_item<ValueType> r_entry{
        r, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> r_hat_entry{
        r_hat, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> p_entry{
        p, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> p_hat_entry{
        p_hat, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> v_entry{
        v, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> s_entry{
        s, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> s_hat_entry{
        s_hat, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> t_entry{
        t, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> rho_old_entry{
        rho_old, num_rhs, 1, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> rho_new_entry{
        rho_new, num_rhs, 1, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> omega_entry{
        omega, num_rhs, 1, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> alpha_entry{
        alpha, num_rhs, 1, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> temp_entry{
        temp, num_rhs, 1, num_rhs};
    const gko::batch::multi_vector::batch_item<real_type> rhs_norms_entry{
        norms_rhs, num_rhs, 1, num_rhs};
    const gko::batch::multi_vector::batch_item<real_type> res_norms_entry{
        norms_res, num_rhs, 1, num_rhs};

    // generate preconditioner
    prec.generate(batch_item_id, A_entry, prec_work);

    // initialization
    // rho_old = 1, omega = 1, alpha = 1
    // compute b norms
    // r = b - A*x
    // compute residual norms
    // r_hat = r
    // p = 0
    // p_hat = 0
    // v = 0
    initialize(A_entry, b_entry, gko::batch::to_const(x_entry), rho_old_entry,
               omega_entry, alpha_entry, r_entry, r_hat_entry, p_entry,
               p_hat_entry, v_entry, rhs_norms_entry, res_norms_entry);

    // stopping criterion object
    StopType stop(settings.residual_tol, rhs_norms_entry.values);

    int iter{};

    for (iter = 0; iter < settings.max_iterations; iter++) {
        if (stop.check_converged(res_norms_entry.values)) {
            logger.log_iteration(batch_item_id, iter,
                                 res_norms_entry.values[0]);
            break;
        }

        // rho_new =  < r_hat , r > = (r_hat)' * (r)
        compute_dot_product_kernel<ValueType>(gko::batch::to_const(r_hat_entry),
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
        simple_apply_kernel(A_entry, gko::batch::to_const(p_hat_entry),
                            v_entry);

        // alpha = rho_new / < r_hat , v>
        compute_alpha(gko::batch::to_const(rho_new_entry),
                      gko::batch::to_const(r_hat_entry),
                      gko::batch::to_const(v_entry), alpha_entry);

        // s = r - alpha*v
        update_s(gko::batch::to_const(r_entry),
                 gko::batch::to_const(alpha_entry),
                 gko::batch::to_const(v_entry), s_entry);

        // an estimate of residual norms
        compute_norm2_kernel<ValueType>(gko::batch::to_const(s_entry),
                                        res_norms_entry);

        if (stop.check_converged(res_norms_entry.values)) {
            // update x for the systems
            // x = x + alpha * p_hat
            update_x_middle(gko::batch::to_const(alpha_entry),
                            gko::batch::to_const(p_hat_entry), x_entry);
            logger.log_iteration(batch_item_id, iter,
                                 res_norms_entry.values[0]);
            break;
        }

        // s_hat = precond * s
        prec.apply(gko::batch::to_const(s_entry), s_hat_entry);

        // t = A * s_hat
        simple_apply_kernel(A_entry, gko::batch::to_const(s_hat_entry),
                            t_entry);
        // omega = <t,s> / <t,t>
        compute_omega(gko::batch::to_const(t_entry),
                      gko::batch::to_const(s_entry), temp_entry, omega_entry);


        // x = x + alpha * p_hat + omega * s_hat
        // r = s - omega * t
        update_x_and_r(gko::batch::to_const(p_hat_entry),
                       gko::batch::to_const(s_hat_entry),
                       gko::batch::to_const(alpha_entry),
                       gko::batch::to_const(omega_entry),
                       gko::batch::to_const(s_entry),
                       gko::batch::to_const(t_entry), x_entry, r_entry);

        compute_norm2_kernel<ValueType>(gko::batch::to_const(r_entry),
                                        res_norms_entry);

        // rho_old = rho_new
        copy_kernel(gko::batch::to_const(rho_new_entry), rho_old_entry);
    }

    logger.log_iteration(batch_item_id, iter, res_norms_entry.values[0]);
}
