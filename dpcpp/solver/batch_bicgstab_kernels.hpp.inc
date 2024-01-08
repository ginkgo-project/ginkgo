// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename BatchMatrixType_entry, typename ValueType>
__dpct_inline__ void initialize(
    const int num_rows, const BatchMatrixType_entry& mat_global_entry,
    const ValueType* const b_global_entry,
    const ValueType* const x_global_entry, ValueType& rho_old, ValueType& omega,
    ValueType& alpha, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry, ValueType* const r_hat_shared_entry,
    ValueType* const p_shared_entry, ValueType* const v_shared_entry,
    ValueType* const p_hat_shared_entry,
    typename gko::remove_complex<ValueType>& rhs_norm,
    typename gko::remove_complex<ValueType>& res_norm,
    sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group_size = item_ct1.get_local_range().size();
    const auto group = item_ct1.get_group();

    rho_old = one<ValueType>();
    omega = one<ValueType>();
    alpha = one<ValueType>();

    // copy x from global to shared memory
    // r = b
    for (int iz = tid; iz < num_rows; iz += group_size) {
        x_shared_entry[iz] = x_global_entry[iz];
        r_shared_entry[iz] = b_global_entry[iz];
    }
    item_ct1.barrier(sycl::access::fence_space::global_and_local);

    // r = b - A*x
    advanced_apply_kernel(static_cast<ValueType>(-1.0), mat_global_entry,
                          x_shared_entry, static_cast<ValueType>(1.0),
                          r_shared_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::global_and_local);

    if (sg_id == 0) {
        single_rhs_compute_norm2_sg(num_rows, r_shared_entry, res_norm,
                                    item_ct1);
    } else if (sg_id == 1) {
        single_rhs_compute_norm2_sg(num_rows, b_global_entry, rhs_norm,
                                    item_ct1);
    }
    item_ct1.barrier(sycl::access::fence_space::global_and_local);


    for (int iz = tid; iz < num_rows; iz += group_size) {
        r_hat_shared_entry[iz] = r_shared_entry[iz];
        p_shared_entry[iz] = zero<ValueType>();
        p_hat_shared_entry[iz] = zero<ValueType>();
        v_shared_entry[iz] = zero<ValueType>();
    }
}


template <typename ValueType>
__dpct_inline__ void update_p(const int num_rows, const ValueType& rho_new,
                              const ValueType& rho_old, const ValueType& alpha,
                              const ValueType& omega,
                              const ValueType* const r_shared_entry,
                              const ValueType* const v_shared_entry,
                              ValueType* const p_shared_entry,
                              sycl::nd_item<3> item_ct1)
{
    const ValueType beta = (rho_new / rho_old) * (alpha / omega);
    for (int r = item_ct1.get_local_linear_id(); r < num_rows;
         r += item_ct1.get_local_range().size()) {
        p_shared_entry[r] =
            r_shared_entry[r] +
            beta * (p_shared_entry[r] - omega * v_shared_entry[r]);
    }
}


template <typename ValueType>
__dpct_inline__ void compute_alpha(const int num_rows, const ValueType& rho_new,
                                   const ValueType* const r_hat_shared_entry,
                                   const ValueType* const v_shared_entry,
                                   ValueType& alpha, sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    if (sg_id == 0) {
        single_rhs_compute_conj_dot_sg(num_rows, r_hat_shared_entry,
                                       v_shared_entry, alpha, item_ct1);
    }
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
    if (tid == 0) {
        alpha = rho_new / alpha;
    }
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
}


template <typename ValueType>
__dpct_inline__ void update_s(const int num_rows,
                              const ValueType* const r_shared_entry,
                              const ValueType& alpha,
                              const ValueType* const v_shared_entry,
                              ValueType* const s_shared_entry,
                              sycl::nd_item<3> item_ct1)
{
    for (int r = item_ct1.get_local_linear_id(); r < num_rows;
         r += item_ct1.get_local_range().size()) {
        s_shared_entry[r] = r_shared_entry[r] - alpha * v_shared_entry[r];
    }
}


template <typename ValueType>
__dpct_inline__ void compute_omega(const int num_rows,
                                   const ValueType* const t_shared_entry,
                                   const ValueType* const s_shared_entry,
                                   ValueType& temp, ValueType& omega,
                                   sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    if (sg_id == 0) {
        single_rhs_compute_conj_dot_sg(num_rows, t_shared_entry, s_shared_entry,
                                       omega, item_ct1);
    } else if (sg_id == 1) {
        single_rhs_compute_conj_dot_sg(num_rows, t_shared_entry, t_shared_entry,
                                       temp, item_ct1);
    }
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
    if (tid == 0) {
        omega /= temp;
    }
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
}


template <typename ValueType>
__dpct_inline__ void update_x_and_r(
    const int num_rows, const ValueType* const p_hat_shared_entry,
    const ValueType* const s_hat_shared_entry, const ValueType& alpha,
    const ValueType& omega, const ValueType* const s_shared_entry,
    const ValueType* const t_shared_entry, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry, sycl::nd_item<3> item_ct1)
{
    for (int r = item_ct1.get_local_linear_id(); r < num_rows;
         r += item_ct1.get_local_range().size()) {
        x_shared_entry[r] = x_shared_entry[r] + alpha * p_hat_shared_entry[r] +
                            omega * s_hat_shared_entry[r];
        r_shared_entry[r] = s_shared_entry[r] - omega * t_shared_entry[r];
    }
}


template <typename ValueType>
__dpct_inline__ void update_x_middle(const int num_rows, const ValueType& alpha,
                                     const ValueType* const p_hat_shared_entry,
                                     ValueType* const x_shared_entry,
                                     sycl::nd_item<3> item_ct1)
{
    for (int r = item_ct1.get_local_linear_id(); r < num_rows;
         r += item_ct1.get_local_range().size()) {
        x_shared_entry[r] = x_shared_entry[r] + alpha * p_hat_shared_entry[r];
    }
}


template <typename StopType, const int n_shared_total, typename PrecType,
          typename LogType, typename BatchMatrixType, typename ValueType>
void apply_kernel(const gko::kernels::batch_bicgstab::storage_config sconf,
                  const int max_iter, const gko::remove_complex<ValueType> tol,
                  LogType logger, PrecType prec_shared,
                  const BatchMatrixType mat_global_entry,
                  const ValueType* const __restrict__ b_global_entry,
                  ValueType* const __restrict__ x_global_entry,
                  const size_type num_rows, const size_type nnz,
                  ValueType* const __restrict__ slm_values,
                  sycl::nd_item<3> item_ct1,
                  ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;

    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int tid = item_ct1.get_local_linear_id();
    auto group = item_ct1.get_group();
    const int group_size = item_ct1.get_local_range().size();

    const auto batch_id = item_ct1.get_group_linear_id();

    ValueType* rho_old_sh;
    ValueType* rho_new_sh;
    ValueType* alpha_sh;
    ValueType* omega_sh;
    ValueType* temp_sh;
    real_type* norms_rhs_sh;
    real_type* norms_res_sh;

    using tile_value_t = ValueType[5];
    tile_value_t& values =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_value_t>(
            group);
    using tile_real_t = real_type[2];
    tile_real_t& reals =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_real_t>(
            group);
    rho_old_sh = &values[0];
    rho_new_sh = &values[1];
    alpha_sh = &values[2];
    omega_sh = &values[3];
    temp_sh = &values[4];
    norms_rhs_sh = &reals[0];
    norms_res_sh = &reals[1];
    const int gmem_offset =
        batch_id * sconf.gmem_stride_bytes / sizeof(ValueType);
    ValueType* p_hat_sh;
    ValueType* s_hat_sh;
    ValueType* s_sh;
    ValueType* p_sh;
    ValueType* r_sh;
    ValueType* r_hat_sh;
    ValueType* v_sh;
    ValueType* t_sh;
    ValueType* x_sh;
    ValueType* prec_work_sh;

    if constexpr (n_shared_total >= 1) {
        p_hat_sh = slm_values;
    } else {
        p_hat_sh = workspace + gmem_offset;
    }
    if constexpr (n_shared_total == 1) {
        s_hat_sh = workspace + gmem_offset;
    } else {
        s_hat_sh = p_hat_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 2) {
        v_sh = workspace + gmem_offset;
    } else {
        v_sh = s_hat_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 3) {
        t_sh = workspace + gmem_offset;
    } else {
        t_sh = v_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 4) {
        p_sh = workspace + gmem_offset;
    } else {
        p_sh = t_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 5) {
        s_sh = workspace + gmem_offset;
    } else {
        s_sh = p_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 6) {
        r_sh = workspace + gmem_offset;
    } else {
        r_sh = s_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 7) {
        r_hat_sh = workspace + gmem_offset;
    } else {
        r_hat_sh = r_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 8) {
        x_sh = workspace + gmem_offset;
    } else {
        x_sh = r_hat_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 9) {
        prec_work_sh = workspace + gmem_offset;
    } else {
        prec_work_sh = x_sh + sconf.padded_vec_len;
    }

    // generate preconditioner
    prec_shared.generate(batch_id, mat_global_entry, prec_work_sh, item_ct1);

    // initialization
    // rho_old = 1, omega = 1, alpha = 1
    // compute b norms
    // copy x from global to shared memory
    // r = b - A*x
    // compute residual norms
    // r_hat = r
    // p = 0
    // p_hat = 0
    // v = 0
    initialize(num_rows, mat_global_entry, b_global_entry, x_global_entry,
               rho_old_sh[0], omega_sh[0], alpha_sh[0], x_sh, r_sh, r_hat_sh,
               p_sh, p_hat_sh, v_sh, norms_rhs_sh[0], norms_res_sh[0],
               item_ct1);
    item_ct1.barrier(sycl::access::fence_space::global_and_local);

    // stopping criterion object
    StopType stop(tol, norms_rhs_sh);

    int iter = 0;
    for (; iter < max_iter; iter++) {
        if (stop.check_converged(norms_res_sh)) {
            logger.log_iteration(batch_id, iter, norms_res_sh[0]);
            break;
        }

        // rho_new =  < r_hat , r > = (r_hat)' * (r)
        if (sg_id == 0) {
            single_rhs_compute_conj_dot_sg(num_rows, r_hat_sh, r_sh,
                                           rho_new_sh[0], item_ct1);
        }
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // beta = (rho_new / rho_old)*(alpha / omega)
        // p = r + beta*(p - omega * v)
        update_p(num_rows, rho_new_sh[0], rho_old_sh[0], alpha_sh[0],
                 omega_sh[0], r_sh, v_sh, p_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // p_hat = precond * p
        prec_shared.apply(num_rows, p_sh, p_hat_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // v = A * p_hat
        simple_apply_kernel(mat_global_entry, p_hat_sh, v_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // alpha = rho_new / < r_hat , v>
        compute_alpha(num_rows, rho_new_sh[0], r_hat_sh, v_sh, alpha_sh[0],
                      item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // s = r - alpha*v
        update_s(num_rows, r_sh, alpha_sh[0], v_sh, s_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // an estimate of residual norms
        if (sg_id == 0) {
            single_rhs_compute_norm2_sg(num_rows, s_sh, norms_res_sh[0],
                                        item_ct1);
        }
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        if (stop.check_converged(norms_res_sh)) {
            update_x_middle(num_rows, alpha_sh[0], p_hat_sh, x_sh, item_ct1);
            logger.log_iteration(batch_id, iter, norms_res_sh[0]);
            break;
        }

        // s_hat = precond * s
        prec_shared.apply(num_rows, s_sh, s_hat_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // t = A * s_hat
        simple_apply_kernel(mat_global_entry, s_hat_sh, t_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // omega = <t,s> / <t,t>
        compute_omega(num_rows, t_sh, s_sh, temp_sh[0], omega_sh[0], item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // x = x + alpha*p_hat + omega *s_hat
        // r = s - omega * t
        update_x_and_r(num_rows, p_hat_sh, s_hat_sh, alpha_sh[0], omega_sh[0],
                       s_sh, t_sh, x_sh, r_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        if (sg_id == 0)
            single_rhs_compute_norm2_sg(num_rows, r_sh, norms_res_sh[0],
                                        item_ct1);
        if (tid == group_size - 1) {
            rho_old_sh[0] = rho_new_sh[0];
        }
        item_ct1.barrier(sycl::access::fence_space::global_and_local);
    }

    logger.log_iteration(batch_id, iter, norms_res_sh[0]);

    // copy x back to global memory
    copy_kernel(num_rows, x_sh, x_global_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
}
