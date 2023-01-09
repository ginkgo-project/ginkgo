/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/solver/batch_gmres_kernels.hpp"


#include "core/solver/batch_dispatch.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace omp {


/**
 * @brief The batch Gmres solver namespace.
 *
 * @ingroup batch_gmres
 */
namespace batch_gmres {
namespace {


#include "reference/matrix/batch_csr_kernels.hpp.inc"
#include "reference/matrix/batch_dense_kernels.hpp.inc"
#include "reference/matrix/batch_ell_kernels.hpp.inc"
#include "reference/solver/batch_gmres_kernels.hpp.inc"


}  // unnamed namespace


template <typename T>
using BatchGmresOptions = gko::kernels::batch_gmres::BatchGmresOptions<T>;


template <typename ValueType>
class KernelCaller {
public:
    KernelCaller(std::shared_ptr<const OmpExecutor> exec,
                 const BatchGmresOptions<remove_complex<ValueType>> opts)
        : exec_{exec}, opts_{opts}
    {}

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(LogType logger, const BatchMatrixType& a, PrecType prec,
                     const gko::batch_dense::UniformBatch<const ValueType>& b,
                     const gko::batch_dense::UniformBatch<ValueType>& x) const
    {
        using real_type = typename gko::remove_complex<ValueType>;
        const size_type nbatch = a.num_batch;
        const auto nrows = a.num_rows;
        const auto nrhs = b.num_rhs;
        GKO_ASSERT(nrhs == 1);

        const int local_size_bytes =
            gko::kernels::batch_gmres::local_memory_requirement<ValueType>(
                nrows, nrhs, opts_.restart_num) +
            PrecType::dynamic_work_size(nrows, a.num_nnz) * sizeof(ValueType);
        // For some reason, gko::Array allocation fails here
        // array<unsigned char> local_space(exec_, local_size_bytes);

#pragma omp parallel for firstprivate(logger)
        for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
            // TODO: Align to cache line boundary
            // TODO: Allocate and free once per thread rather than once per
            // work-item.
            const auto local_space =
                static_cast<unsigned char*>(malloc(local_size_bytes));
            batch_entry_gmres_impl<StopType, PrecType, LogType, BatchMatrixType,
                                   ValueType>(opts_, logger, prec, a, b, x,
                                              ibatch, local_space);
            free(local_space);
        }
    }

private:
    std::shared_ptr<const OmpExecutor> exec_;
    const BatchGmresOptions<remove_complex<ValueType>> opts_;
};


template <typename ValueType>
void apply(std::shared_ptr<const OmpExecutor> exec,
           const BatchGmresOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a, const BatchLinOp* const precon,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           gko::log::BatchLogData<ValueType>& logdata)
{
    auto dispatcher = batch_solver::create_dispatcher<ValueType>(
        KernelCaller<ValueType>(exec, opts), opts, a, precon);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_GMRES_APPLY_KERNEL);


}  // namespace batch_gmres
}  // namespace omp
}  // namespace kernels
}  // namespace gko
