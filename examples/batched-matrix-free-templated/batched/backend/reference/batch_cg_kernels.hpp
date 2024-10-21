// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#if GINKGO_BUILD_REFERENCE

#include <ginkgo/core/log/batch_logger.hpp>

#include "../../batch_cg_settings.hpp"
#include "../../batch_criteria.hpp"
#include "../../batch_identity.hpp"
#include "../../batch_logger.hpp"
#include "batch_csr_kernels.hpp"
#include "batch_multi_vector_kernels.hpp"


namespace gko {
namespace kernels {

constexpr int max_num_rhs = 1;


namespace reference {
namespace batch_template {
namespace batch_single_kernels {
namespace batch_cg {
template <typename BatchMatrixType_entry, typename ValueType>
void initialize(
    const BatchMatrixType_entry& A_entry,
    const batch::multi_vector::batch_item<const ValueType>& b_entry,
    const batch::multi_vector::batch_item<const ValueType>& x_entry,
    const batch::multi_vector::batch_item<ValueType>& rho_old_entry,
    const batch::multi_vector::batch_item<ValueType>& rho_new_entry,
    const batch::multi_vector::batch_item<ValueType>& r_entry,
    const batch::multi_vector::batch_item<ValueType>& p_entry,
    const batch::multi_vector::batch_item<ValueType>& z_entry,
    const batch::multi_vector::batch_item<ValueType>& Ap_entry,
    const batch::multi_vector::batch_item<remove_complex<ValueType>>&
        rhs_norms_entry)
{
    rho_new_entry.values[0] = zero<ValueType>();
    rho_old_entry.values[0] = one<ValueType>();

    for (int r = 0; r < p_entry.num_rows; r++) {
        p_entry.values[r * p_entry.stride] = zero<ValueType>();
        z_entry.values[r * z_entry.stride] = zero<ValueType>();
        Ap_entry.values[r * Ap_entry.stride] = zero<ValueType>();
    }

    // Compute norms of rhs
    compute_norm2_kernel<ValueType>(b_entry, rhs_norms_entry);

    // r = b
    copy_kernel(b_entry, r_entry);

    // r = b - A*x
    advanced_apply(static_cast<ValueType>(-1.0), A_entry,
                   batch::to_const(x_entry), static_cast<ValueType>(1.0),
                   r_entry);
}

template <typename ValueType>
inline void update_p(
    const batch::multi_vector::batch_item<const ValueType>& rho_new_entry,
    const batch::multi_vector::batch_item<const ValueType>& rho_old_entry,
    const batch::multi_vector::batch_item<const ValueType>& z_entry,
    const batch::multi_vector::batch_item<ValueType>& p_entry)
{
    if (rho_old_entry.values[0] == zero<ValueType>()) {
        copy_kernel(z_entry, p_entry);
        return;
    }
    const ValueType beta = rho_new_entry.values[0] / rho_old_entry.values[0];
    for (int row = 0; row < p_entry.num_rows; row++) {
        p_entry.values[row * p_entry.stride] =
            z_entry.values[row * z_entry.stride] +
            beta * p_entry.values[row * p_entry.stride];
    }
}


template <typename ValueType>
inline void update_x_and_r(
    const batch::multi_vector::batch_item<const ValueType>& rho_old_entry,
    const batch::multi_vector::batch_item<const ValueType>& p_entry,
    const batch::multi_vector::batch_item<const ValueType>& Ap_entry,
    const batch::multi_vector::batch_item<ValueType>& alpha_entry,
    const batch::multi_vector::batch_item<ValueType>& x_entry,
    const batch::multi_vector::batch_item<ValueType>& r_entry)
{
    compute_conj_dot_product_kernel<ValueType>(p_entry, Ap_entry, alpha_entry);

    const ValueType temp = rho_old_entry.values[0] / alpha_entry.values[0];
    for (int row = 0; row < r_entry.num_rows; row++) {
        x_entry.values[row * x_entry.stride] +=
            temp * p_entry.values[row * p_entry.stride];
        r_entry.values[row * r_entry.stride] -=
            temp * Ap_entry.values[row * Ap_entry.stride];
    }
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
void batch_entry_cg_impl(
    const kernels::batch_cg::settings<remove_complex<ValueType>>& settings,
    LogType logger, PrecType prec, const BatchMatrixType& a,
    batch::multi_vector::batch_item<const ValueType> b,
    batch::multi_vector::batch_item<ValueType> x, const size_type batch_item_id,
    unsigned char* const local_space)
{
    using real_type = remove_complex<ValueType>;
    const auto num_rows = static_cast<int32>(a.num_rows);
    const auto num_rhs = static_cast<int32>(b.num_rhs);
    GKO_ASSERT(num_rhs <= max_num_rhs);

    unsigned char* const shared_space = local_space;
    ValueType* const r = reinterpret_cast<ValueType*>(shared_space);
    ValueType* const z = r + num_rows * num_rhs;
    ValueType* const p = z + num_rows * num_rhs;
    ValueType* const Ap = p + num_rows * num_rhs;
    ValueType* const prec_work = Ap + num_rows * num_rhs;
    ValueType rho_old[max_num_rhs];
    ValueType rho_new[max_num_rhs];
    ValueType alpha[max_num_rhs];
    ValueType temp[max_num_rhs];
    real_type norms_rhs[max_num_rhs];
    real_type norms_res[max_num_rhs];

    const auto A_entry = batch::matrix::extract_batch_item(a, batch_item_id);
    const auto b_entry = b;
    const auto x_entry = x;

    const batch::multi_vector::batch_item<ValueType> r_entry{r, num_rhs,
                                                             num_rows, num_rhs};
    const batch::multi_vector::batch_item<ValueType> z_entry{z, num_rhs,
                                                             num_rows, num_rhs};
    const batch::multi_vector::batch_item<ValueType> p_entry{p, num_rhs,
                                                             num_rows, num_rhs};
    const batch::multi_vector::batch_item<ValueType> Ap_entry{
        Ap, num_rhs, num_rows, num_rhs};
    const batch::multi_vector::batch_item<ValueType> rho_old_entry{
        rho_old, num_rhs, 1, num_rhs};
    const batch::multi_vector::batch_item<ValueType> rho_new_entry{
        rho_new, num_rhs, 1, num_rhs};
    const batch::multi_vector::batch_item<ValueType> alpha_entry{alpha, num_rhs,
                                                                 1, num_rhs};
    const batch::multi_vector::batch_item<ValueType> rhs_norms_entry{
        norms_rhs, num_rhs, 1, num_rhs};
    const batch::multi_vector::batch_item<ValueType> res_norms_entry{
        norms_res, num_rhs, 1, num_rhs};

    // generate preconditioner
    prec.generate(batch_item_id, A_entry, prec_work);

    // initialization
    // compute b norms
    // r = b - A*x
    // p = z = Ap = 0
    // rho_old = 1, rho_new = 0
    initialize(A_entry, batch::to_const(b_entry), batch::to_const(x_entry),
               rho_old_entry, rho_new_entry, r_entry, p_entry, z_entry,
               Ap_entry, rhs_norms_entry);

    // stopping criterion object
    StopType stop(settings.residual_tol, rhs_norms_entry.values);

    int iter = 0;

    while (true) {
        // z = precond * r
        prec.apply(batch::to_const(r_entry), z_entry);

        // rho_new =  < r , z > = (r)' * (z)
        compute_conj_dot_product_kernel<ValueType>(
            batch::to_const(r_entry), batch::to_const(z_entry), rho_new_entry);
        ++iter;
        // use implicit residual norms
        res_norms_entry.values[0] = sqrt(abs(rho_new_entry.values[0]));

        if (iter >= settings.max_iterations ||
            stop.check_converged(res_norms_entry.values)) {
            logger.log_iteration(batch_item_id, iter,
                                 res_norms_entry.values[0]);
            break;
        }

        // beta = (rho_new / rho_old)
        // p = z + beta * p
        update_p(batch::to_const(rho_new_entry), batch::to_const(rho_old_entry),
                 batch::to_const(z_entry), p_entry);

        // Ap = A * p
        simple_apply(A_entry, batch::to_const(p_entry), Ap_entry);

        // temp= rho_old / (p' * Ap)
        // x = x + temp * p
        // r = r - temp * Ap
        update_x_and_r(batch::to_const(rho_new_entry), batch::to_const(p_entry),
                       batch::to_const(Ap_entry), alpha_entry, x_entry,
                       r_entry);

        // rho_old = rho_new
        copy_kernel(batch::to_const(rho_new_entry), rho_old_entry);
    }

    logger.log_iteration(batch_item_id, iter, res_norms_entry.values[0]);
}
}  // namespace batch_cg
}  // namespace batch_single_kernels
namespace batch_cg {

template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op mat, batch::multi_vector::uniform_batch<const ValueType> b,
    batch::multi_vector::uniform_batch<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    using real_type = remove_complex<ValueType>;
    const size_type num_batch_items = mat.num_batch_items;
    const auto num_rows = mat.num_rows;
    const auto num_rhs = b.num_rhs;
    if (num_rhs > 1) {
        GKO_NOT_IMPLEMENTED;
    }

    const size_type local_size_bytes =
        kernels::batch_cg::local_memory_requirement<ValueType>(num_rows,
                                                               num_rhs);
    array<unsigned char> local_space(exec, local_size_bytes);

    batch_log::SimpleFinalLogger<real_type> logger(
        logdata.res_norms.get_data(), logdata.iter_counts.get_data());

    auto prec = batch_preconditioner::Identity<ValueType>();

    for (size_type batch_id = 0; batch_id < num_batch_items; batch_id++) {
        batch_single_kernels::batch_cg::batch_entry_cg_impl<
            batch_stop::SimpleRelResidual<ValueType>>(
            options, logger, prec, mat, batch::extract_batch_item(b, batch_id),
            batch::extract_batch_item(x, batch_id), batch_id,
            local_space.get_data());
    }
}
}  // namespace batch_cg
}  // namespace batch_template
}  // namespace reference
}  // namespace kernels
}  // namespace gko

#endif
