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

#include "core/solver/batch_cg_kernels.hpp"

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


/**
 * @brief The batch Cg solver namespace.
 *
 * @ingroup batch_cg
 */
namespace batch_cg {

namespace {


template <typename ValueType>
inline void copy(
    const gko::batch_dense::BatchEntry<const ValueType> &source_entry,
    const gko::batch_dense::BatchEntry<ValueType> &destination_entry,
    const uint32 &converged)
{
    for (int r = 0; r < source_entry.num_rows; r++) {
        for (int c = 0; c < source_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            destination_entry.values[r * destination_entry.stride + c] =
                source_entry.values[r * source_entry.stride + c];
        }
    }
}


template <typename BatchMatrixType_entry, typename PrecType, typename ValueType>
inline void initialize(
    const BatchMatrixType_entry &A_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &b_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const PrecType &prec,
    const gko::batch_dense::BatchEntry<ValueType> &z_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rho_old_entry,
    const gko::batch_dense::BatchEntry<ValueType> &p_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &rhs_norms_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &res_norms_entry)
{
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
    advanced_spmv_kernel(static_cast<ValueType>(-1.0), A_entry,
                         gko::batch::to_const(x_entry),
                         static_cast<ValueType>(1.0), r_entry);
    batch_dense::compute_norm2<ValueType>(gko::batch::to_const(r_entry),
                                          res_norms_entry);


    // z = precond * r
    prec.apply(gko::batch::to_const(r_entry), z_entry);


    // p = z
    for (int r = 0; r < p_entry.num_rows; r++) {
        for (int c = 0; c < p_entry.num_rhs; c++) {
            p_entry.values[r * p_entry.stride + c] =
                z_entry.values[r * z_entry.stride + c];
        }
    }

    // rho_old = r' * z
    batch_dense::compute_dot_product(gko::batch::to_const(r_entry),
                                     gko::batch::to_const(z_entry),
                                     rho_old_entry);
}


template <typename ValueType>
inline void compute_beta(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_new_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_entry,
    const gko::batch_dense::BatchEntry<ValueType> &beta_entry,
    const uint32 &converged)
{
    for (int c = 0; c < beta_entry.num_rhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        beta_entry.values[0 * beta_entry.stride + c] =
            rho_new_entry.values[0 * rho_new_entry.stride + c] /
            rho_old_entry.values[0 * rho_old_entry.stride + c];
    }
}


template <typename ValueType>
inline void update_p(
    const gko::batch_dense::BatchEntry<const ValueType> &z_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &beta_entry,
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
                z_entry.values[r * z_entry.stride + c] +
                beta_entry.values[0 * beta_entry.stride + c] *
                    p_entry.values[r * p_entry.stride + c];
        }
    }
}


template <typename ValueType>
inline void compute_alpha(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &Ap_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_entry,
    const uint32 &converged)
{
    const auto nrhs = rho_old_entry.num_rhs;

    ValueType temp[batch_config<ValueType>::max_num_rhs];
    const gko::batch_dense::BatchEntry<ValueType> temp_entry{
        temp, static_cast<size_type>(nrhs), 1, nrhs};

    batch_dense::compute_dot_product<ValueType>(p_entry, Ap_entry, temp_entry);

    for (int c = 0; c < alpha_entry.num_rhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        alpha_entry.values[c] = rho_old_entry.values[c] / temp_entry.values[c];
    }
}


template <typename ValueType>
inline void update_x(
    const gko::batch_dense::BatchEntry<ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_entry,
    const uint32 &converged)
{
    for (int r = 0; r < x_entry.num_rows; r++) {
        for (int c = 0; c < x_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            x_entry.values[r * x_entry.stride + c] +=

                alpha_entry.values[c] * p_entry.values[r * p_entry.stride + c];
        }
    }
}


template <typename ValueType>
inline void update_r(
    const gko::batch_dense::BatchEntry<const ValueType> &Ap_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const uint32 &converged)
{
    for (int r = 0; r < r_entry.num_rows; r++) {
        for (int c = 0; c < r_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            r_entry.values[r * r_entry.stride + c] -=

                alpha_entry.values[c] *
                Ap_entry.values[r * Ap_entry.stride + c];
        }
    }
}


}  // unnamed namespace


template <typename T>
using BatchCgOptions = gko::kernels::batch_cg::BatchCgOptions<T>;

template <typename PrecType, typename StopType, typename LogType,
          typename BatchMatrixType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchCgOptions<remove_complex<ValueType>> &opts, LogType logger,
    const BatchMatrixType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &left,
    const gko::batch_dense::UniformBatch<const ValueType> &right,
    const gko::batch_dense::UniformBatch<ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;


    constexpr int max_nrhs = batch_config<ValueType>::max_num_rhs;

    GKO_ASSERT((batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs >=
                nrows * nrhs));
    GKO_ASSERT(batch_config<ValueType>::max_num_rows >= nrows);
    GKO_ASSERT(batch_config<ValueType>::max_num_rhs >= nrhs);

    for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
        ValueType r[batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs];
        ValueType z[batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs];
        ValueType p[batch_config<ValueType>::max_num_rows *
                    batch_config<ValueType>::max_num_rhs];
        ValueType Ap[batch_config<ValueType>::max_num_rows *
                     batch_config<ValueType>::max_num_rhs];


        ValueType prec_work[PrecType::work_size];
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

        const gko::batch_dense::BatchEntry<ValueType> z_entry{
            z, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> p_entry{
            p, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> Ap_entry{
            Ap, static_cast<size_type>(nrhs), nrows, nrhs};


        ValueType rho_old[max_nrhs];
        const gko::batch_dense::BatchEntry<ValueType> rho_old_entry{
            rho_old, static_cast<size_type>(nrhs), 1, nrhs};

        ValueType rho_new[max_nrhs];
        const gko::batch_dense::BatchEntry<ValueType> rho_new_entry{
            rho_new, static_cast<size_type>(nrhs), 1, nrhs};

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

        real_type norms_res_temp[max_nrhs];
        const gko::batch_dense::BatchEntry<real_type> res_norms_temp_entry{
            norms_res_temp, static_cast<size_type>(nrhs), 1, nrhs};

        // generate preconditioner
        const PrecType prec(A_entry, prec_work);

        // initialization
        // compute b norms (precond b or what ?)
        // r = b - A*x
        // z = precond*r
        // compute residual norms (? precond res or what ?)
        // rho_old = r' * z (' is for hermitian transpose)
        // p = z
        initialize(A_entry, b_entry, gko::batch::to_const(x_entry), r_entry,
                   prec, z_entry, rho_old_entry, p_entry, rhs_norms_entry,
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

            // Ap = A * p
            spmv_kernel(A_entry, gko::batch::to_const(p_entry), Ap_entry);

            // alpha = rho_old / (p' * Ap)
            compute_alpha(gko::batch::to_const(rho_old_entry),
                          gko::batch::to_const(p_entry),
                          gko::batch::to_const(Ap_entry), alpha_entry,
                          converged);

            // x = x + alpha * p
            update_x(x_entry, gko::batch::to_const(p_entry),
                     gko::batch::to_const(alpha_entry), converged);

            // r = r - alpha * Ap
            update_r(gko::batch::to_const(Ap_entry),
                     gko::batch::to_const(alpha_entry), r_entry, converged);
            batch_dense::compute_norm2<ValueType>(
                gko::batch::to_const(r_entry),
                res_norms_temp_entry);  // store residual norms in temp entry
            copy(gko::batch::to_const(res_norms_temp_entry), res_norms_entry,
                 converged);  // copy into res_norms entry only for those RHSs
                              // which have not yet converged.

            // z = precond * r
            prec.apply(gko::batch::to_const(r_entry), z_entry);

            // rho_new =  (r)' * (z)
            batch_dense::compute_dot_product<ValueType>(
                gko::batch::to_const(r_entry), gko::batch::to_const(z_entry),
                rho_new_entry);


            // beta = rho_new / rho_old
            compute_beta(gko::batch::to_const(rho_new_entry),
                         gko::batch::to_const(rho_old_entry), beta_entry,
                         converged);

            // p = z + beta * p
            update_p(gko::batch::to_const(z_entry),
                     gko::batch::to_const(beta_entry), p_entry, converged);


            // rho_old = rho_new
            copy(gko::batch::to_const(rho_new_entry), rho_old_entry, converged);
        }

        if (left_entry.values) {
            batch_dense::batch_scale(right_entry, x_entry);
        }
    }
}


template <typename BatchType, typename LoggerType, typename ValueType>
void apply_select_prec(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchCgOptions<remove_complex<ValueType>> &opts,
    const LoggerType logger, const BatchType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &left,
    const gko::batch_dense::UniformBatch<const ValueType> &right,
    const gko::batch_dense::UniformBatch<ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x)
{
    if (opts.preconditioner == gko::preconditioner::batch::type::none) {
        apply_impl<BatchIdentity<ValueType>,
                   stop::AbsAndRelResidualMaxIter<ValueType>>(
            exec, opts, logger, a, left, right, b, x);

    }
    // else if (opts.preconditioner == gko::preconditioner::batch::jacobi_str) {
    //     apply_impl<BatchJacobi<ValueType>,
    //                stop::AbsAndRelResidualMaxIter<ValueType>>(
    //         exec, opts, logger, a, left, right, b, x);
    // }
    else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const BatchCgOptions<remove_complex<ValueType>> &opts,
           const BatchLinOp *const a,
           const matrix::BatchDense<ValueType> *const left_scale,
           const matrix::BatchDense<ValueType> *const right_scale,
           const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           gko::log::BatchLogData<ValueType> &logdata)
{
    batch_log::FinalLogger<remove_complex<ValueType>> logger(
        b->get_size().at(0)[1], opts.max_its, logdata.res_norms->get_values(),
        logdata.iter_counts.get_data());

    const gko::batch_dense::UniformBatch<const ValueType> b_b =
        get_batch_struct(b);

    const gko::batch_dense::UniformBatch<const ValueType> left_sb =
        maybe_null_batch_struct(left_scale);
    const gko::batch_dense::UniformBatch<const ValueType> right_sb =
        maybe_null_batch_struct(right_scale);
    const auto to_scale = left_sb.values || right_sb.values;
    if (to_scale) {
        if (!left_sb.values || !right_sb.values) {
            // one-sided scaling not implemented
            GKO_NOT_IMPLEMENTED;
        }
    }

    const gko::batch_dense::UniformBatch<ValueType> x_b = get_batch_struct(x);
    if (auto a_mat = dynamic_cast<const matrix::BatchCsr<ValueType> *>(a)) {
        // if(to_scale) {
        // We pinky-promise not to change the matrix and RHS if no scaling was
        // requested
        const gko::batch_csr::UniformBatch<ValueType> a_b =
            get_batch_struct(const_cast<matrix::BatchCsr<ValueType> *>(a_mat));
        const gko::batch_dense::UniformBatch<ValueType> b_b =
            get_batch_struct(const_cast<matrix::BatchDense<ValueType> *>(b));
        apply_select_prec(exec, opts, logger, a_b, left_sb, right_sb, b_b, x_b);
        // } else {
        // 	const gko::batch_csr::UniformBatch<const ValueType> a_b =
        // get_batch_struct(a_mat); 	apply_select_prec(exec, opts, logger,
        // a_b, left_sb, right_sb, &b_b, b_b, x_b);
        // }

    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG_APPLY_KERNEL);


}  // namespace batch_cg
}  // namespace reference
}  // namespace kernels
}  // namespace gko
