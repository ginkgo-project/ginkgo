// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_SOLVER_BATCH_BICGSTAB_KERNELS_HPP_
#define GKO_REFERENCE_SOLVER_BATCH_BICGSTAB_KERNELS_HPP_


#include "core/solver/batch_cg_kernels.hpp"

#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "reference/base/batch_multi_vector_kernels.hpp"
#include "reference/base/batch_struct.hpp"
#include "reference/matrix/batch_csr_kernels.hpp"
#include "reference/matrix/batch_dense_kernels.hpp"
#include "reference/matrix/batch_ell_kernels.hpp"
#include "reference/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_single_kernels {


constexpr int max_num_rhs = 1;


template <typename BatchMatrixType_entry, typename ValueType>
inline void initialize(
    const BatchMatrixType_entry& A_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& b_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& x_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& rho_old_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& rho_new_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& r_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& p_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& z_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& Ap_entry,
    const gko::batch::multi_vector::batch_item<
        typename gko::remove_complex<ValueType>>& rhs_norms_entry)
{
    rho_new_entry.values[0] = zero<ValueType>();
    rho_old_entry.values[0] = one<ValueType>();

    for (int r = 0; r < p_entry.num_rows; r++) {
        p_entry.values[r * p_entry.stride] = zero<ValueType>();
        z_entry.values[r * z_entry.stride] = zero<ValueType>();
        Ap_entry.values[r * Ap_entry.stride] = zero<ValueType>();
    }

    // Compute norms of rhs
    batch_single_kernels::compute_norm2_kernel<ValueType>(b_entry,
                                                          rhs_norms_entry);

    // r = b
    batch_single_kernels::copy_kernel(b_entry, r_entry);

    // r = b - A*x
    batch_single_kernels::advanced_apply(static_cast<ValueType>(-1.0), A_entry,
                                         gko::batch::to_const(x_entry),
                                         static_cast<ValueType>(1.0), r_entry);
}


template <typename ValueType>
inline void update_p(
    const gko::batch::multi_vector::batch_item<const ValueType>& rho_new_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& rho_old_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& z_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& p_entry)
{
    if (rho_old_entry.values[0] == zero<ValueType>()) {
        batch_single_kernels::copy_kernel(z_entry, p_entry);
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
    const gko::batch::multi_vector::batch_item<const ValueType>& rho_new_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& p_entry,
    const gko::batch::multi_vector::batch_item<const ValueType>& Ap_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& alpha_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& x_entry,
    const gko::batch::multi_vector::batch_item<ValueType>& r_entry)
{
    batch_single_kernels::compute_conj_dot_product_kernel<ValueType>(
        p_entry, Ap_entry, alpha_entry);

    const ValueType temp = rho_new_entry.values[0] / alpha_entry.values[0];
    for (int row = 0; row < r_entry.num_rows; row++) {
        x_entry.values[row * x_entry.stride] +=
            temp * p_entry.values[row * p_entry.stride];
        r_entry.values[row * r_entry.stride] -=
            temp * Ap_entry.values[row * Ap_entry.stride];
    }
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
inline void batch_entry_cg_impl(
    const gko::kernels::batch_cg::settings<remove_complex<ValueType>>& settings,
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

    const auto A_entry = gko::batch::extract_batch_item(
        gko::batch::matrix::to_const(a), batch_item_id);
    const gko::batch::multi_vector::batch_item<const ValueType> b_entry =
        gko::batch::extract_batch_item(gko::batch::to_const(b), batch_item_id);
    const gko::batch::multi_vector::batch_item<ValueType> x_entry =
        gko::batch::extract_batch_item(x, batch_item_id);

    const gko::batch::multi_vector::batch_item<ValueType> r_entry{
        r, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> z_entry{
        z, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> p_entry{
        p, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> Ap_entry{
        Ap, num_rhs, num_rows, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> rho_old_entry{
        rho_old, num_rhs, 1, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> rho_new_entry{
        rho_new, num_rhs, 1, num_rhs};
    const gko::batch::multi_vector::batch_item<ValueType> alpha_entry{
        alpha, num_rhs, 1, num_rhs};
    const gko::batch::multi_vector::batch_item<real_type> rhs_norms_entry{
        norms_rhs, num_rhs, 1, num_rhs};
    const gko::batch::multi_vector::batch_item<real_type> res_norms_entry{
        norms_res, num_rhs, 1, num_rhs};

    // generate preconditioner
    prec.generate(batch_item_id, A_entry, prec_work);

    // initialization
    // compute b norms
    // r = b - A*x
    // p = z = Ap = 0
    // rho_old = 1, rho_new = 0
    initialize(A_entry, b_entry, gko::batch::to_const(x_entry), rho_old_entry,
               rho_new_entry, r_entry, p_entry, z_entry, Ap_entry,
               rhs_norms_entry);

    // stopping criterion object
    StopType stop(settings.residual_tol, rhs_norms_entry.values);

    int iter = 0;

    while (true) {
        // z = precond * r
        prec.apply(gko::batch::to_const(r_entry), z_entry);

        // rho_new =  < r , z > = (r)' * (z)
        batch_single_kernels::compute_conj_dot_product_kernel<ValueType>(
            gko::batch::to_const(r_entry), gko::batch::to_const(z_entry),
            rho_new_entry);
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
        update_p(gko::batch::to_const(rho_new_entry),
                 gko::batch::to_const(rho_old_entry),
                 gko::batch::to_const(z_entry), p_entry);

        // Ap = A * p
        batch_single_kernels::simple_apply(
            A_entry, gko::batch::to_const(p_entry), Ap_entry);

        // temp= rho_old / (p' * Ap)
        // x = x + temp * p
        // r = r - temp * Ap
        update_x_and_r(
            gko::batch::to_const(rho_new_entry), gko::batch::to_const(p_entry),
            gko::batch::to_const(Ap_entry), alpha_entry, x_entry, r_entry);

        // rho_old = rho_new
        batch_single_kernels::copy_kernel(gko::batch::to_const(rho_new_entry),
                                          rho_old_entry);
    }

    logger.log_iteration(batch_item_id, iter, res_norms_entry.values[0]);
}


}  // namespace batch_single_kernels
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif
