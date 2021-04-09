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

#include "core/solver/batch_bicgstab_kernels.hpp"


#include <ginkgo/core/preconditioner/batch_preconditioner_strings.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>

#include "reference/base/config.hpp"
// include device kernels for every matrix and preconditioner type
#include "reference/log/batch_logger.hpp"
#include "reference/matrix/batch_csr_kernels.hpp"
#include "reference/matrix/batch_dense_kernels.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_identity.hpp"
#include "reference/preconditioner/batch_jacobi.hpp"
#include "reference/stop/batch_criteria.hpp"


namespace gko {
namespace kernels {
namespace reference {


#include "core/matrix/batch_sparse_ops.hpp.inc"


/**
 * @brief The batch Bicgstab solver namespace.
 *
 * @ingroup batch_bicgstab
 */
namespace batch_bicgstab {

namespace {


template <typename BatchMatrixType_entry, typename ValueType>
inline void initialize(
    const gko::batch_dense::BatchEntry<ValueType> &rho_old_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_old_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_entry,
    const BatchMatrixType_entry &A_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &b_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_hat_entry,
    const gko::batch_dense::BatchEntry<ValueType> &p_entry,
    const gko::batch_dense::BatchEntry<ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &rhs_norms_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &res_norms_entry)
{
    for (int c = 0; c < rho_old_entry.num_rhs; c++) {
        rho_old_entry.values[0 * rho_old_entry.stride + c] = one<ValueType>();
        omega_old_entry.values[0 * omega_old_entry.stride + c] =
            one<ValueType>();
        alpha_entry.values[0 * alpha_entry.stride + c] = one<ValueType>();
    }

    // Compute norms of rhs
    batch_dense::compute_norm2<ValueType>(b_entry, rhs_norms_entry);


    // r = b
    for (int r = 0; r < r_entry.num_rows; r++) {
        for (int c = 0; c < r_entry.num_rhs; c++) {
            r_entry.values[r * r_entry.stride + c] =
                b_entry.values[r * b_entry.stride + c];
        }
    }
    // r = b - A*x
    batch_adv_spmv_single(static_cast<ValueType>(-1.0), A_entry,
                          gko::batch_dense::to_const(x_entry),
                          static_cast<ValueType>(1.0), r_entry);
    batch_dense::compute_norm2<ValueType>(gko::batch_dense::to_const(r_entry),
                                          res_norms_entry);


    for (int r = 0; r < r_entry.num_rows; r++) {
        for (int c = 0; c < r_entry.num_rhs; c++) {
            r_hat_entry.values[r * r_hat_entry.stride + c] =
                r_entry.values[r * r_entry.stride + c];
            p_entry.values[r * p_entry.stride + c] = zero<ValueType>();
            v_entry.values[r * v_entry.stride + c] = zero<ValueType>();
        }
    }
}


template <typename ValueType>
inline void compute_beta(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_new_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_old_entry,
    const gko::batch_dense::BatchEntry<ValueType> &beta_entry,
    const uint32 &converged)
{
    for (int c = 0; c < beta_entry.num_rhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        beta_entry.values[0 * beta_entry.stride + c] =
            (rho_new_entry.values[0 * rho_new_entry.stride + c] /
             rho_old_entry.values[0 * rho_old_entry.stride + c]) *
            (alpha_entry.values[0 * alpha_entry.stride + c] /
             omega_old_entry.values[0 * omega_old_entry.stride + c]);
    }
}


template <typename ValueType>
inline void update_p(
    const gko::batch_dense::BatchEntry<const ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &beta_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_old_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<ValueType> &p_entry,
    const uint32 &converged)
{
    for (int r = 0; r < p_entry.num_rows; r++) {
        for (int c = 0; c < p_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            p_entry.values[r * p_entry.stride + c] =
                r_entry.values[r * r_entry.stride + c] +
                beta_entry.values[0 * beta_entry.stride + c] *
                    (p_entry.values[r * p_entry.stride + c] -
                     omega_old_entry.values[0 * omega_old_entry.stride + c] *
                         v_entry.values[r * v_entry.stride + c]);
        }
    }
}


template <typename ValueType>
inline void compute_alpha(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_new_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_hat_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_entry,
    const uint32 &converged)
{
    const auto nrhs = rho_new_entry.num_rhs;

    ValueType temp[batch_config<ValueType>::max_num_rhs];
    const gko::batch_dense::BatchEntry<ValueType> temp_entry{
        temp, static_cast<size_type>(nrhs), 1, nrhs};

    batch_dense::compute_dot_product<ValueType>(r_hat_entry, v_entry,
                                                temp_entry);

    for (int c = 0; c < alpha_entry.num_rhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        alpha_entry.values[c] = rho_new_entry.values[c] / temp_entry.values[c];
    }
}


template <typename ValueType>
inline void update_s(
    const gko::batch_dense::BatchEntry<const ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<ValueType> &s_entry,
    const uint32 &converged)
{
    for (int r = 0; r < s_entry.num_rows; r++) {
        for (int c = 0; c < s_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }
            s_entry.values[r * s_entry.stride + c] =
                r_entry.values[r * r_entry.stride + c] -
                alpha_entry.values[0 * alpha_entry.stride + c] *
                    v_entry.values[r * v_entry.stride + c];
        }
    }
}

template <typename ValueType>
inline void compute_omega_new(
    const gko::batch_dense::BatchEntry<const ValueType> &t_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_new_entry,
    const uint32 &converged)
{
    const auto nrhs = omega_new_entry.num_rhs;

    ValueType t_s[batch_config<ValueType>::max_num_rhs];
    const gko::batch_dense::BatchEntry<ValueType> t_s_entry{
        t_s, static_cast<size_type>(nrhs), 1, nrhs};

    ValueType t_t[batch_config<ValueType>::max_num_rhs];
    const gko::batch_dense::BatchEntry<ValueType> t_t_entry{
        t_t, static_cast<size_type>(nrhs), 1, nrhs};

    batch_dense::compute_dot_product<ValueType>(t_entry, s_entry, t_s_entry);
    batch_dense::compute_dot_product<ValueType>(t_entry, t_entry, t_t_entry);

    for (int c = 0; c < omega_new_entry.num_rhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        omega_new_entry.values[c] = t_s_entry.values[c] / t_t_entry.values[c];
    }
}

template <typename ValueType>
inline void update_x(
    const gko::batch_dense::BatchEntry<ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_hat_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_hat_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_new_entry,
    const uint32 &converged)
{
    for (int r = 0; r < x_entry.num_rows; r++) {
        for (int c = 0; c < x_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            x_entry.values[r * x_entry.stride + c] =
                x_entry.values[r * x_entry.stride + c] +
                alpha_entry.values[c] *
                    p_hat_entry.values[r * p_hat_entry.stride + c] +
                omega_new_entry.values[c] *
                    s_hat_entry.values[r * s_hat_entry.stride + c];
        }
    }
}


template <typename ValueType>
inline void update_x_middle(
    const gko::batch_dense::BatchEntry<ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_hat_entry,
    const uint32 &converged_recent)
{
    for (int r = 0; r < x_entry.num_rows; r++) {
        for (int c = 0; c < x_entry.num_rhs; c++) {
            const uint32 conv = converged_recent & (1 << c);

            if (conv) {
                x_entry.values[r * x_entry.stride + c] =
                    x_entry.values[r * x_entry.stride + c] +
                    alpha_entry.values[c] *
                        p_hat_entry.values[r * p_hat_entry.stride + c];
            }
        }
    }
}

template <typename ValueType>
inline void update_r(
    const gko::batch_dense::BatchEntry<const ValueType> &s_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_new_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &t_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const uint32 &converged)
{
    for (int r = 0; r < r_entry.num_rows; r++) {
        for (int c = 0; c < r_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            r_entry.values[r * r_entry.stride + c] =
                s_entry.values[r * s_entry.stride + c] -
                omega_new_entry.values[c] *
                    t_entry.values[r * t_entry.stride + c];
        }
    }
}


}  // unnamed namespace


template <typename T>
using BatchBicgstabOptions =
    gko::kernels::batch_bicgstab::BatchBicgstabOptions<T>;

template <typename PrecType, typename StopType, typename LogType,
          typename BatchMatrixType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchBicgstabOptions<remove_complex<ValueType>> &opts, LogType logger,
    const BatchMatrixType *const a,
    const gko::batch_dense::UniformBatch<const ValueType> *const b,
    const gko::batch_dense::UniformBatch<ValueType> *const x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const size_type nbatch = a->num_batch;
    const auto nrows = a->num_rows;
    const auto nrhs = b->num_rhs;


    constexpr int max_nrhs = batch_config<ValueType>::max_num_rhs;

    GKO_ASSERT((batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs >=
                nrows * nrhs));
    GKO_ASSERT(batch_config<ValueType>::max_num_rows >= nrows);
    GKO_ASSERT(batch_config<ValueType>::max_num_rhs >= nrhs);

    for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
        ValueType r[batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs];
        ValueType r_hat[batch_config<ValueType>::max_num_rows *
                        batch_config<ValueType>::max_num_rhs];
        ValueType p[batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs];
        ValueType p_hat[batch_config<ValueType>::max_num_rows *
                        batch_config<ValueType>::max_num_rhs];
        ValueType v[batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs];
        ValueType s[batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs];
        ValueType s_hat[batch_config<ValueType>::max_num_rows *
                        batch_config<ValueType>::max_num_rhs];
        ValueType t[batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs];

        ValueType prec_work[PrecType::work_size];
        uint32 converged = 0;

        const typename BatchMatrixType::entry_type A_entry =
            gko::batch::batch_entry(*a, ibatch);

        const gko::batch_dense::BatchEntry<const ValueType> b_entry =
            gko::batch::batch_entry(*b, ibatch);

        const gko::batch_dense::BatchEntry<ValueType> x_entry =
            gko::batch::batch_entry(*x, ibatch);

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

        ValueType rho_old[max_nrhs];
        const gko::batch_dense::BatchEntry<ValueType> rho_old_entry{
            rho_old, static_cast<size_type>(nrhs), 1, nrhs};

        ValueType rho_new[max_nrhs];
        const gko::batch_dense::BatchEntry<ValueType> rho_new_entry{
            rho_new, static_cast<size_type>(nrhs), 1, nrhs};

        ValueType omega_old[max_nrhs];
        const gko::batch_dense::BatchEntry<ValueType> omega_old_entry{
            omega_old, static_cast<size_type>(nrhs), 1, nrhs};

        ValueType omega_new[max_nrhs];
        const gko::batch_dense::BatchEntry<ValueType> omega_new_entry{
            omega_new, static_cast<size_type>(nrhs), 1, nrhs};

        ValueType alpha[max_nrhs];
        const gko::batch_dense::BatchEntry<ValueType> alpha_entry{
            alpha, static_cast<size_type>(nrhs), 1, nrhs};

        ValueType beta[max_nrhs];
        const gko::batch_dense::BatchEntry<ValueType> beta_entry{
            beta, static_cast<size_type>(nrhs), 1, nrhs};

        real_type norms_rhs[max_nrhs];
        const gko::batch_dense::BatchEntry<real_type> rhs_norms_entry{
            norms_rhs, static_cast<size_type>(nrhs), 1, nrhs};

        real_type norms_res[max_nrhs];
        const gko::batch_dense::BatchEntry<real_type> res_norms_entry{
            norms_res, static_cast<size_type>(nrhs), 1, nrhs};

        // generate preconditioner
        PrecType prec(A_entry, prec_work);

        // initialization
        // rho_old = 1, omega_old = 1, alpha = 1
        // compute b norms
        // r = b - A*x
        // compute residual norms
        // r_hat = r
        // p = 0
        // v = 0
        initialize(rho_old_entry, omega_old_entry, alpha_entry, A_entry,
                   b_entry, gko::batch_dense::to_const(x_entry), r_entry,
                   r_hat_entry, p_entry, v_entry, rhs_norms_entry,
                   res_norms_entry);

        // stopping criterion object
        StopType stop(nrhs, opts.max_its, opts.abs_residual_tol,
                      opts.rel_residual_tol,
                      static_cast<stop::tolerance>(opts.tol_type), converged,
                      rhs_norms_entry.values);

        int iter = -1;

        while (1) {
            ++iter;


            bool all_converged = stop.check_converged(
                iter, res_norms_entry.values, {NULL, 0, 0, 0}, converged);

            logger.log_iteration(ibatch, iter, res_norms_entry.values,
                                 converged);

            if (all_converged) {
                break;
            }

            // rho_new =  < r_hat , r > = (r_hat)' * (r)
            batch_dense::compute_dot_product<ValueType>(
                gko::batch_dense::to_const(r_hat_entry),
                gko::batch_dense::to_const(r_entry), rho_new_entry);


            // beta = (rho_new / rho_old)*(alpha / omega_old)
            compute_beta(gko::batch_dense::to_const(rho_new_entry),
                         gko::batch_dense::to_const(rho_old_entry),
                         gko::batch_dense::to_const(alpha_entry),
                         gko::batch_dense::to_const(omega_old_entry),
                         beta_entry, converged);

            // p = r + beta*(p - omega_old * v)
            update_p(gko::batch_dense::to_const(r_entry),
                     gko::batch_dense::to_const(beta_entry),
                     gko::batch_dense::to_const(omega_old_entry),
                     gko::batch_dense::to_const(v_entry), p_entry, converged);

            // p_hat = precond * p
            prec.apply(gko::batch_dense::to_const(p_entry), p_hat_entry);

            // v = A * p_hat
            batch_spmv_single(A_entry, gko::batch_dense::to_const(p_hat_entry),
                              v_entry);

            // alpha = rho_new / < r_hat , v>
            compute_alpha(gko::batch_dense::to_const(rho_new_entry),
                          gko::batch_dense::to_const(r_hat_entry),
                          gko::batch_dense::to_const(v_entry), alpha_entry,
                          converged);


            // s = r - alpha*v
            update_s(gko::batch_dense::to_const(r_entry),
                     gko::batch_dense::to_const(alpha_entry),
                     gko::batch_dense::to_const(v_entry), s_entry, converged);
            batch_dense::compute_norm2<ValueType>(
                gko::batch_dense::to_const(s_entry),
                res_norms_entry);  // an estimate of residual norms

            const uint32 converged_prev = converged;

            all_converged = stop.check_converged(iter, res_norms_entry.values,
                                                 {NULL, 0, 0, 0}, converged);

            // update x for the sytems (rhs) which converge at this point...  x
            // = x + alpha*p_hat

            // note bits could change from 0 to 1, not the other way round, so
            // we can use xor to get info about recent convergence...
            const uint32 converged_recent = converged_prev ^ converged;
            update_x_middle(x_entry, gko::batch_dense::to_const(alpha_entry),
                            gko::batch_dense::to_const(p_hat_entry),
                            converged_recent);

            logger.log_iteration(ibatch, iter, res_norms_entry.values,
                                 converged);

            if (all_converged) {
                break;
            }

            // s_hat = precond * s
            prec.apply(gko::batch_dense::to_const(s_entry), s_hat_entry);

            // t = A * s_hat
            batch_spmv_single(A_entry, gko::batch_dense::to_const(s_hat_entry),
                              t_entry);

            // omega_new = <t,s> / <t,t>
            compute_omega_new(gko::batch_dense::to_const(t_entry),
                              gko::batch_dense::to_const(s_entry),
                              omega_new_entry, converged);

            // x = x + alpha*p_hat + omega_new*s_hat
            update_x(x_entry, gko::batch_dense::to_const(p_hat_entry),
                     gko::batch_dense::to_const(s_hat_entry),
                     gko::batch_dense::to_const(alpha_entry),
                     gko::batch_dense::to_const(omega_new_entry), converged);

            // r = s - omega_new*t
            update_r(gko::batch_dense::to_const(s_entry),
                     gko::batch_dense::to_const(omega_new_entry),
                     gko::batch_dense::to_const(t_entry), r_entry, converged);
            batch_dense::compute_norm2<ValueType>(
                gko::batch_dense::to_const(r_entry),
                res_norms_entry);  // residual norms

            // rho_old = rho_new
            // omega_old = omega_new
            for (int c = 0; c < rho_old_entry.num_rhs; c++) {
                const uint32 conv = converged & (1 << c);

                if (conv) {
                    continue;
                }

                rho_old_entry.values[c] = rho_new_entry.values[c];
                omega_old_entry.values[c] = omega_new_entry.values[c];
            }
        }
    }
}


template <typename BatchType, typename LoggerType, typename ValueType>
void apply_select_prec(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchBicgstabOptions<remove_complex<ValueType>> &opts,
    const LoggerType logger, const BatchType *const a,
    const gko::batch_dense::UniformBatch<const ValueType> *const b,
    const gko::batch_dense::UniformBatch<ValueType> *const x)
{
    if (opts.preconditioner == gko::preconditioner::batch::none_str) {
        apply_impl<BatchIdentity<ValueType>,
                   stop::AbsAndRelResidualMaxIter<ValueType>>(exec, opts,
                                                              logger, a, b, x);

    } else if (opts.preconditioner == gko::preconditioner::batch::jacobi_str) {
        apply_impl<BatchJacobi<ValueType>,
                   stop::AbsAndRelResidualMaxIter<ValueType>>(exec, opts,
                                                              logger, a, b, x);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const BatchBicgstabOptions<remove_complex<ValueType>> &opts,
           const BatchLinOp *const a,
           const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           gko::log::BatchLogData<ValueType> &logdata)
{
    batch_log::FinalLogger<remove_complex<ValueType>> logger(
        b->get_size().at(0)[1], opts.max_its, logdata.res_norms->get_values(),
        logdata.iter_counts.get_data());
    const gko::batch_dense::UniformBatch<const ValueType> b_b =
        get_batch_struct(b);
    const gko::batch_dense::UniformBatch<ValueType> x_b = get_batch_struct(x);
    if (auto a_mat = dynamic_cast<const matrix::BatchCsr<ValueType> *>(a)) {
        const gko::batch_csr::UniformBatch<const ValueType> a_b =
            get_batch_struct(a_mat);
        apply_select_prec(exec, opts, logger, &a_b, &b_b, &x_b);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL);


}  // namespace batch_bicgstab
}  // namespace reference
}  // namespace kernels
}  // namespace gko
