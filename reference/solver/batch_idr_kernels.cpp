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

#include "core/solver/batch_idr_kernels.hpp"

#include <ctime>
#include <random>


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
 * @brief The batch Idr solver namespace.
 *
 * @ingroup batch_idr
 */
namespace batch_idr {

namespace {

#include "reference/solver/batch_idr_kernels.hpp"
}  // unnamed namespace

template <typename T>
using BatchIdrOptions = gko::kernels::batch_idr::BatchIdrOptions<T>;


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchIdrOptions<remove_complex<ValueType>> &opts, LogType logger,
    PrecType prec, const BatchMatrixType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &left,
    const gko::batch_dense::UniformBatch<const ValueType> &right,
    const gko::batch_dense::UniformBatch<ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;
    const auto subspace_dim = opts.subspace_dim_val;
    const auto kappa = opts.kappa_val;
    const auto smoothing = opts.to_use_smoothing;
    const auto deterministic = opts.deterministic_gen;

    GKO_ASSERT(batch_config<ValueType>::max_num_rhs >=
               nrhs);  // required for static allocation in stopping criterion

    const int local_size_bytes =
        gko::kernels::batch_idr::local_memory_requirement<ValueType>(
            nrows, nrhs, subspace_dim) +
        PrecType::dynamic_work_size(nrows, a.num_nnz) * sizeof(ValueType);
    using byte = unsigned char;
    Array<byte> local_space(exec, local_size_bytes);

#include "reference/solver/batch_idr_body.hpp"
}

template <typename BatchType, typename LoggerType, typename ValueType>
void apply_select_prec(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchIdrOptions<remove_complex<ValueType>> &opts,
    const LoggerType logger, const BatchType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &left,
    const gko::batch_dense::UniformBatch<const ValueType> &right,
    const gko::batch_dense::UniformBatch<ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x)
{
    if (opts.preconditioner == gko::preconditioner::batch::type::none) {
        BatchIdentity<ValueType> prec;

        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            apply_impl<stop::AbsResidualMaxIter<ValueType>>(
                exec, opts, logger, prec, a, left, right, b, x);
        } else {
            apply_impl<stop::RelResidualMaxIter<ValueType>>(
                exec, opts, logger, prec, a, left, right, b, x);
        }


    } else if (opts.preconditioner ==
               gko::preconditioner::batch::type::jacobi) {
        BatchJacobi<ValueType> prec;

        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            apply_impl<stop::AbsResidualMaxIter<ValueType>>(
                exec, opts, logger, prec, a, left, right, b, x);
        } else {
            apply_impl<stop::RelResidualMaxIter<ValueType>>(
                exec, opts, logger, prec, a, left, right, b, x);
        }

    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const BatchIdrOptions<remove_complex<ValueType>> &opts,
           const BatchLinOp *const a,
           const matrix::BatchDense<ValueType> *const left_scale,
           const matrix::BatchDense<ValueType> *const right_scale,
           const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           gko::log::BatchLogData<ValueType> &logdata)
{
    if (opts.is_complex_subspace == true && !is_complex<ValueType>()) {
        GKO_NOT_IMPLEMENTED;
    }

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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDR_APPLY_KERNEL);


}  // namespace batch_idr
}  // namespace reference
}  // namespace kernels
}  // namespace gko