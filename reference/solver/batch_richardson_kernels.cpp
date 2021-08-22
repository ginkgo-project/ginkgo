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


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"
#include "reference/log/batch_logger.hpp"
// include device kernels for every matrix and preconditioner type
#include "reference/matrix/batch_csr_kernels.hpp"
#include "reference/matrix/batch_dense_kernels.hpp"
//#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_identity.hpp"
#include "reference/preconditioner/batch_jacobi.hpp"
#include "reference/stop/batch_criteria.hpp"


namespace gko {
namespace kernels {
namespace reference {


/**
 * @brief The batch Richardson solver namespace.
 *
 * @ingroup batch_rich
 */
namespace batch_rich {


template <typename T>
using BatchRichardsonOptions =
    gko::kernels::batch_rich::BatchRichardsonOptions<T>;

template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchRichardsonOptions<remove_complex<ValueType>> &opts,
    LogType logger, PrecType prec, const BatchMatrixType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;

    GKO_ASSERT((nrhs == x.num_rhs));

    const int local_size_bytes =
        gko::kernels::batch_rich::local_memory_requirement<ValueType>(nrows,
                                                                      nrhs) +
        PrecType::dynamic_work_size(nrows, a.num_nnz) * sizeof(ValueType);
    using byte = unsigned char;
    Array<byte> local_space(exec, local_size_bytes);

    for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
        byte *const shared_space = local_space.get_data();
        const auto residual = reinterpret_cast<ValueType *>(shared_space);
        ValueType *const delta_x = residual + nrows * nrhs;
        ValueType *const prec_work = delta_x + nrows * nrhs;
        real_type norms[batch_config<ValueType>::max_num_rhs];
        real_type init_res_norm[batch_config<ValueType>::max_num_rhs];

        uint32 converged = 0;

        const auto a_b =
            gko::batch::batch_entry(gko::batch::to_const(a), ibatch);
        const auto b_b =
            gko::batch::batch_entry(gko::batch::to_const(b), ibatch);
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        const gko::batch_dense::BatchEntry<ValueType> r_b{
            residual, static_cast<size_type>(nrhs), nrows, nrhs};
        const gko::batch_dense::BatchEntry<ValueType> dx_b{
            delta_x, static_cast<size_type>(nrhs), nrows, nrhs};

        const auto relax = static_cast<ValueType>(opts.relax_factor);
        const gko::batch_dense::BatchEntry<const ValueType> relax_b{&relax, 1,
                                                                    1, 1};
        const gko::batch_dense::BatchEntry<real_type> norms_b{
            norms, static_cast<size_type>(nrhs), 1, nrhs};
        const gko::batch_dense::BatchEntry<real_type> init_res_norm_b{
            init_res_norm, static_cast<size_type>(nrhs), 1, nrhs};

        prec.generate(a_b, prec_work);

        // initial residual
        batch_dense::compute_norm2<ValueType>(gko::batch::to_const(b_b),
                                              norms_b);
        batch_dense::copy(gko::batch::to_const(norms_b), init_res_norm_b);

        StopType stop(nrhs, opts.max_its, opts.residual_tol,
                      init_res_norm_b.values, converged);

        int iter = 0;
        while (true) {
            // r <- r - Adx  This causes instability!
            // adv_spmv_ker(static_cast<ValueType>(-1.0), a_b,
            //              gko::batch::to_const(dx_b),
            // static_cast<ValueType>(1.0), r_b);

            // r <- b - Ax
            batch_dense::copy(b_b, r_b);
            batch_csr::advanced_spmv_kernel(static_cast<ValueType>(-1.0), a_b,
                                            gko::batch::to_const(x_b),
                                            static_cast<ValueType>(1.0), r_b);

            batch_dense::compute_norm2(gko::batch::to_const(r_b), norms_b);

            const bool all_converged = stop.check_converged(
                iter, norms, {nullptr, 0, 0, 0}, converged);
            logger.log_iteration(ibatch, iter, norms, converged);
            if (all_converged) {
                break;
            }

            prec.apply(gko::batch::to_const(r_b), dx_b);

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

            batch_dense::add_scaled(relax_b, gko::batch::to_const(dx_b), x_b);
            iter++;
        }
    }
}

template <typename BatchType, typename LoggerType, typename ValueType>
void apply_select_prec(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchRichardsonOptions<remove_complex<ValueType>> &opts,
    const LoggerType logger, const BatchType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x)
{
    if (opts.preconditioner == gko::preconditioner::batch::type::none) {
        BatchIdentity<ValueType> prec;
        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            apply_impl<stop::AbsResidualMaxIter<ValueType>>(exec, opts, logger,
                                                            prec, a, b, x);
        } else {
            apply_impl<stop::RelResidualMaxIter<ValueType>>(exec, opts, logger,
                                                            prec, a, b, x);
        }
    } else if (opts.preconditioner ==
               gko::preconditioner::batch::type::jacobi) {
        BatchJacobi<ValueType> prec;
        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            apply_impl<stop::AbsResidualMaxIter<ValueType>>(exec, opts, logger,
                                                            prec, a, b, x);
        } else {
            apply_impl<stop::RelResidualMaxIter<ValueType>>(exec, opts, logger,
                                                            prec, a, b, x);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const BatchRichardsonOptions<remove_complex<ValueType>> &opts,
           const BatchLinOp *const a,
           const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           gko::log::BatchLogData<ValueType> &logdata)
{
    batch_log::FinalLogger<remove_complex<ValueType>> logger(
        static_cast<int>(b->get_size().at(0)[1]), opts.max_its,
        logdata.res_norms->get_values(), logdata.iter_counts.get_data());

    const gko::batch_dense::UniformBatch<ValueType> x_b =
        host::get_batch_struct(x);
    if (auto a_mat = dynamic_cast<const matrix::BatchCsr<ValueType> *>(a)) {
        const auto a_b = host::get_batch_struct(a_mat);
        const auto b_b = host::get_batch_struct(b);
        apply_select_prec(exec, opts, logger, a_b, b_b, x_b);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_RICHARDSON_APPLY_KERNEL);


}  // namespace batch_rich
}  // namespace reference
}  // namespace kernels
}  // namespace gko
