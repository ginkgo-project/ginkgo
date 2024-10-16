// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/core/log/batch_logger.hpp>

#include "../../batch_cg_settings.hpp"
#include "../../batch_criteria.hpp"
#include "../../batch_identity.hpp"
#include "../../batch_logger.hpp"
#include "../../batch_multi_vector.hpp"


namespace gko {
namespace kernels {

constexpr int max_num_rhs = 1;


namespace reference {
namespace batch_tempalte {
namespace batch_cg {


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
void batch_entry_cg_impl(
    const kernels::batch_cg::settings<remove_complex<ValueType>>& settings,
    LogType logger, PrecType prec, const BatchMatrixType& a,
    multi_vector_view_item<const ValueType> b,
    multi_vector_view_item<ValueType> x, const size_type batch_item_id,
    unsigned char* const local_space)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto num_rows = static_cast<int32>(a.get_size().get_common_size()[0]);
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

    const auto A_entry = a.extract_batch_item(batch_item_id);
    const auto b_entry = b;
    const auto x_entry = x;

    const multi_vector_view_item<ValueType> r_entry{r, num_rhs, num_rows,
                                                    num_rhs};
    const multi_vector_view_item<ValueType> z_entry{z, num_rhs, num_rows,
                                                    num_rhs};
    const multi_vector_view_item<ValueType> p_entry{p, num_rhs, num_rows,
                                                    num_rhs};
    const multi_vector_view_item<ValueType> Ap_entry{Ap, num_rhs, num_rows,
                                                     num_rhs};
    const multi_vector_view_item<ValueType> rho_old_entry{rho_old, num_rhs, 1,
                                                          num_rhs};
    const multi_vector_view_item<ValueType> rho_new_entry{rho_new, num_rhs, 1,
                                                          num_rhs};
    const multi_vector_view_item<ValueType> alpha_entry{alpha, num_rhs, 1,
                                                        num_rhs};
    const multi_vector_view_item<ValueType> rhs_norms_entry{norms_rhs, num_rhs,
                                                            1, num_rhs};
    const multi_vector_view_item<ValueType> res_norms_entry{norms_res, num_rhs,
                                                            1, num_rhs};

    // generate preconditioner
    prec.generate(batch_item_id, A_entry, prec_work);

    // initialization
    // compute b norms
    // r = b - A*x
    // p = z = Ap = 0
    // rho_old = 1, rho_new = 0
    // initialize(A_entry, b_entry, batch::to_const(x_entry), rho_old_entry,
    // rho_new_entry, r_entry, p_entry, z_entry, Ap_entry,
    // rhs_norms_entry);

    // stopping criterion object
    StopType stop(settings.residual_tol, rhs_norms_entry.values);

    int iter = 0;

    while (true) {
        // z = precond * r
        prec.apply(r_entry, z_entry);

        // rho_new =  < r , z > = (r)' * (z)
        // compute_conj_dot_product_kernel<ValueType>(
        // batch::to_const(r_entry), batch::to_const(z_entry),
        // rho_new_entry);
        ++iter;
        // use implicit residual norms
        // res_norms_entry.values[0] = sqrt(abs(rho_new_entry.values[0]));

        // if (iter >= settings.max_iterations ||
        //     stop.check_converged(res_norms_entry.values)) {
        //     logger.log_iteration(batch_item_id, iter,
        //                          res_norms_entry.values[0]);
        //     break;
        // }

        // beta = (rho_new / rho_old)
        // p = z + beta * p
        // update_p(batch::to_const(rho_new_entry),
        // batch::to_const(rho_old_entry),
        // batch::to_const(z_entry), p_entry);

        // Ap = A * p
        apply(A_entry, p_entry, Ap_entry);

        // temp= rho_old / (p' * Ap)
        // x = x + temp * p
        // r = r - temp * Ap
        // update_x_and_r(
        // batch::to_const(rho_new_entry), batch::to_const(p_entry),
        // batch::to_const(Ap_entry), alpha_entry, x_entry, r_entry);

        // rho_old = rho_new
        // copy_kernel(batch::to_const(rho_new_entry), rho_old_entry);
    }

    logger.log_iteration(batch_item_id, iter, res_norms_entry.values[0]);
}


template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op* mat, multi_vector_view<const ValueType> b,
    multi_vector_view<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const size_type num_batch_items = mat->get_size().get_num_batch_items();
    const auto num_rows = mat->get_size().get_common_size()[0];
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
        batch_entry_cg_impl<batch_stop::SimpleRelResidual<ValueType>>(
            options, logger, prec, *mat, b.extract_batch_item(batch_id),
            x.extract_batch_item(batch_id), batch_id, local_space.get_data());
    }
}
}  // namespace batch_cg
}  // namespace batch_tempalte
}  // namespace reference
}  // namespace kernels
}  // namespace gko
