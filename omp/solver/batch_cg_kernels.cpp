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


#include "omp/base/config.hpp"
#include "reference/matrix/batch_struct.hpp"
// include device kernels for every matrix and preconditioner type
#include "reference/log/batch_logger.hpp"
#include "reference/matrix/batch_dense_kernels.hpp"
#include "reference/preconditioner/batch_identity.hpp"
#include "reference/preconditioner/batch_jacobi.hpp"
#include "reference/stop/batch_criteria.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The batch Cg solver namespace.
 *
 * @ingroup batch_cg
 */
namespace batch_cg {


// namespace {


namespace batch_dense = gko::kernels::reference::batch_dense;
using gko::kernels::reference::BatchIdentity;
using gko::kernels::reference::BatchJacobi;
namespace batch_log = gko::kernels::reference::batch_log;
namespace stop = gko::kernels::reference::stop;

constexpr int max_num_rhs = 1;

#include "reference/matrix/batch_csr_kernels.hpp.inc"
#include "reference/matrix/batch_ell_kernels.hpp.inc"
#include "reference/solver/batch_cg_kernels.hpp.inc"


//}  // unnamed namespace


template <typename T>
using BatchCgOptions = gko::kernels::batch_cg::BatchCgOptions<T>;

template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
static void apply_impl(std::shared_ptr<const OmpExecutor> exec,
                       const BatchCgOptions<remove_complex<ValueType>>& opts,
                       LogType logger, PrecType prec, const BatchMatrixType& a,
                       const gko::batch_dense::UniformBatch<const ValueType>& b,
                       const gko::batch_dense::UniformBatch<ValueType>& x)
{
    const size_type nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;

    const int local_size_bytes =
        gko::kernels::batch_cg::local_memory_requirement<ValueType>(nrows,
                                                                    nrhs) +
        PrecType::dynamic_work_size(nrows, a.num_nnz) * sizeof(ValueType);
    Array<unsigned char> local_space(exec, local_size_bytes);

#pragma omp parallel for firstprivate(logger) firstprivate(local_space)
    for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
        batch_entry_cg_impl<StopType, PrecType, LogType, BatchMatrixType,
                            ValueType>(opts, logger, prec, a, b, x, ibatch,
                                       local_space);
    }
}


#define BATCH_CG_LAUNCH_KERNEL(_stoptype, _prectype)           \
    apply_impl<stop::_stoptype<ValueType>>(exec, opts, logger, \
                                           _prectype<ValueType>(), a, b, x)

template <typename BatchType, typename LoggerType, typename ValueType>
void apply_select_prec(std::shared_ptr<const OmpExecutor> exec,
                       const BatchCgOptions<remove_complex<ValueType>>& opts,
                       const LoggerType logger, const BatchType& a,
                       const gko::batch_dense::UniformBatch<const ValueType>& b,
                       const gko::batch_dense::UniformBatch<ValueType>& x)
{
    if (opts.preconditioner == gko::preconditioner::batch::type::none) {
        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            BATCH_CG_LAUNCH_KERNEL(SimpleAbsResidual, BatchIdentity);
        } else {
            BATCH_CG_LAUNCH_KERNEL(SimpleRelResidual, BatchIdentity);
        }
    } else if (opts.preconditioner ==
               gko::preconditioner::batch::type::jacobi) {
        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            BATCH_CG_LAUNCH_KERNEL(SimpleAbsResidual, BatchJacobi);
        } else {
            BATCH_CG_LAUNCH_KERNEL(SimpleRelResidual, BatchJacobi);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void apply(std::shared_ptr<const OmpExecutor> exec,
           const BatchCgOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           gko::log::BatchLogData<ValueType>& logdata)
{
    batch_log::SimpleFinalLogger<remove_complex<ValueType>> logger(
        logdata.res_norms->get_values(), logdata.iter_counts.get_data());

    const gko::batch_dense::UniformBatch<ValueType> x_b =
        host::get_batch_struct(x);
    if (auto a_mat = dynamic_cast<const matrix::BatchCsr<ValueType>*>(a)) {
        const auto a_b = host::get_batch_struct(a_mat);
        const auto b_b = host::get_batch_struct(b);
        apply_select_prec(exec, opts, logger, a_b, b_b, x_b);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

#if 0
template <typename ValueType>
class KernelCaller {
public:
    KernelCaller(std::shared_ptr<const OmpExecutor> exec,
                 const BatchCgOptions<remove_complex<ValueType>> opts)
        : exec_{exec}, opts_{opts}
    {}

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(LogType logger, const BatchMatrixType& a,
                     const gko::batch_dense::UniformBatch<const ValueType>& b,
                     const gko::batch_dense::UniformBatch<ValueType>& x) const
    {
        using real_type = typename gko::remove_complex<ValueType>;
        const size_type nbatch = a.num_batch;
        const auto nrows = a.num_rows;
        const auto nrhs = b.num_rhs;
        GKO_ASSERT(nrhs == 1);

        const int local_size_bytes =
            gko::kernels::batch_cg::local_memory_requirement<ValueType>(nrows,
                                                                        nrhs) +
            PrecType::dynamic_work_size(nrows, a.num_nnz) * sizeof(ValueType);
        Array<unsigned char> local_space(exec_, local_size_bytes);

#pragma omp parallel for firstprivate(logger) firstprivate(local_space)
        for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
            batch_entry_cg_impl<StopType, PrecType, LogType, BatchMatrixType,
                                ValueType>(opts_, logger, PrecType(), a, b, x,
                                           ibatch, local_space);
        }
    }

private:
    std::shared_ptr<const OmpExecutor> exec_;
    const BatchCgOptions<remove_complex<ValueType>> opts_;
};

namespace {

using namespace gko::kernels::host;

#include "core/solver/batch_dispatch.hpp.inc"

}  // namespace


template <typename ValueType>
void apply(std::shared_ptr<const OmpExecutor> exec,
           const BatchCgOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    auto dispatcher = create_dispatcher<ValueType, ValueType>(
        KernelCaller<ValueType>(exec, opts), exec, opts);
    dispatcher.apply(a, b, x, logdata);
}
#endif

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG_APPLY_KERNEL);


}  // namespace batch_cg
}  // namespace omp
}  // namespace kernels
}  // namespace gko
