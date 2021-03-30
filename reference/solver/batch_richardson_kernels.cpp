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

#include "core/solver/batch_richardson_kernels.hpp"


#include <ginkgo/core/preconditioner/batch_preconditioner_strings.hpp>


#include "reference/base/config.hpp"
// include device kernels for every matrix and preconditioner type
#include "reference/log/batch_logger.hpp"
#include "reference/matrix/batch_csr_kernels.hpp"
#include "reference/matrix/batch_dense_kernels.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_jacobi.hpp"
#include "reference/stop/batch_criteria.hpp"


namespace gko {
namespace kernels {
namespace reference {


#include "core/matrix/batch_sparse_ops.hpp.inc"


/**
 * @brief The batch Richardson solver namespace.
 *
 * @ingroup batch_rich
 */
namespace batch_rich {


template <typename T>
using BatchRichardsonOptions =
    gko::kernels::batch_rich::BatchRichardsonOptions<T>;

template <typename PrecType, typename StopType, typename LogType,
          typename BatchMatrixType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchRichardsonOptions<remove_complex<ValueType>> &opts,
    LogType logger, const BatchMatrixType *const a,
    const gko::batch_dense::UniformBatch<const ValueType> *const b,
    const gko::batch_dense::UniformBatch<ValueType> *const x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const size_type nbatch = a->num_batch;
    const auto nrows = a->num_rows;
    const auto nrhs = b->num_rhs;
    constexpr int max_nrhs = batch_config<ValueType>::max_num_rhs;
    const auto stride = b->stride;

    GKO_ASSERT((nrhs == x->num_rhs));
    GKO_ASSERT((batch_config<ValueType>::max_num_rows >= nrows * nrhs));

    for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
        ValueType residual[batch_config<ValueType>::max_num_rows];
        ValueType delta_x[batch_config<ValueType>::max_num_rows];
        ValueType prec_work[PrecType::work_size];
        uint32 converged = 0;

        const typename BatchMatrixType::entry_type a_b =
            gko::batch::batch_entry(*a, ibatch);
        const gko::batch_dense::BatchEntry<const ValueType> b_b =
            gko::batch::batch_entry(*b, ibatch);
        const gko::batch_dense::BatchEntry<ValueType> x_b =
            gko::batch::batch_entry(*x, ibatch);
        const gko::batch_dense::BatchEntry<ValueType> r_b{
            residual, static_cast<size_type>(nrhs), nrows, nrhs};
        const gko::batch_dense::BatchEntry<ValueType> dx_b{
            delta_x, static_cast<size_type>(nrhs), nrows, nrhs};

        const ValueType one[] = {1.0};
        const gko::batch_dense::BatchEntry<const ValueType> one_b{one, 1, 1, 1};
        real_type norms[max_nrhs];
        for (int j = 0; j < max_nrhs; j++) {
            norms[j] = 0;
        }
        const gko::batch_dense::BatchEntry<real_type> norms_b{norms, max_nrhs,
                                                              1, nrhs};

        PrecType prec(a_b, prec_work);

        // initial residual
        for (int iz = 0; iz < nrows * nrhs; iz++) {
            const int i = iz / nrhs;
            const int j = iz % nrhs;
            r_b.values[i * r_b.stride + j] = b_b.values[i * b_b.stride + j];
        }
        batch_dense::compute_norm2<ValueType>(gko::batch_dense::to_const(r_b),
                                              norms_b);

        real_type init_rel_res_norm[max_nrhs];
        for (int j = 0; j < nrhs; j++) {
            init_rel_res_norm[j] = sqrt(norms_b.values[j]);
        }

        StopType stop(nrhs, opts.max_its, opts.rel_residual_tol, converged,
                      init_rel_res_norm);

        int iter = 0;
        while (1) {
            // r <- r - Adx  This causes instability!
            // batch_adv_spmv_single(static_cast<ValueType>(-1.0), a_b,
            // 					  gko::batch_dense::to_const(dx_b),
            // static_cast<ValueType>(1.0), r_b);

            // r <- b - Ax
            for (int iz = 0; iz < nrows * nrhs; iz++) {
                const int i = iz / nrhs;
                const int j = iz % nrhs;
                r_b.values[i * r_b.stride + j] = b_b.values[i * b_b.stride + j];
            }
            batch_adv_spmv_single(static_cast<ValueType>(-1.0), a_b,
                                  gko::batch_dense::to_const(x_b),
                                  static_cast<ValueType>(1.0), r_b);

            batch_dense::compute_norm2<ValueType>(
                gko::batch_dense::to_const(r_b), norms_b);
            for (int j = 0; j < nrhs; j++) {
                norms_b.values[j] = sqrt(norms_b.values[j]);
            }

            const bool all_converged =
                stop.check_converged(iter, norms, {NULL, 0, 0, 0}, converged);
            logger.log_iteration(ibatch, iter, norms, converged);
            if (all_converged) {
                break;
            }

            prec.apply(gko::batch_dense::to_const(r_b), dx_b);

            // zero out dx for rhs's which do not need to be updated,
            //  though this is unnecessary for this solver.
            for (int j = 0; j < nrhs; j++) {
                const uint32 conv = converged & (1 << j);
                if (conv) {
                    for (int i = 0; i < nrows; i++) {
                        dx_b.values[i * dx_b.stride + j] = 0.0;
                    }
                }
            }

            batch_dense::add_scaled(one_b, gko::batch_dense::to_const(dx_b),
                                    x_b);
            iter++;
        }
    }
}

template <typename BatchType, typename LoggerType, typename ValueType>
void apply_select_prec(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchRichardsonOptions<remove_complex<ValueType>> &opts,
    const LoggerType logger, const BatchType *const a,
    const gko::batch_dense::UniformBatch<const ValueType> *const b,
    const gko::batch_dense::UniformBatch<ValueType> *const x)
{
    if (opts.preconditioner == gko::preconditioner::batch::jacobi_str) {
        apply_impl<BatchJacobi<ValueType>, stop::RelResidualMaxIter<ValueType>>(
            exec, opts, logger, a, b, x);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const BatchRichardsonOptions<remove_complex<ValueType>> &opts,
           const LinOp *const a, const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           gko::log::BatchLogData<ValueType> &logdata)
{
    batch_log::FinalLogger<remove_complex<ValueType>> logger(
        b->get_batch_sizes()[0][1], opts.max_its,
        logdata.res_norms->get_values(), logdata.iter_counts.get_data());
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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_RICHARDSON_APPLY_KERNEL);


}  // namespace batch_rich
}  // namespace reference
}  // namespace kernels
}  // namespace gko
