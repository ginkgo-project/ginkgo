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

#include "core/solver/batch_richardson_kernels.hpp"


#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace reference {


/**
 * @brief The batch Richardson solver namespace.
 *
 * @ingroup batch_rich
 */
namespace batch_rich {


namespace {


constexpr int max_num_rhs = 1;

#include "reference/matrix/batch_csr_kernels.hpp.inc"
#include "reference/matrix/batch_dense_kernels.hpp.inc"
#include "reference/matrix/batch_ell_kernels.hpp.inc"
#include "reference/solver/batch_richardson_kernels.hpp.inc"


}  // unnamed namespace


template <typename T>
using BatchRichardsonOptions =
    gko::kernels::batch_rich::BatchRichardsonOptions<T>;


template <typename ValueType>
class KernelCaller {
public:
    KernelCaller(std::shared_ptr<const ReferenceExecutor> exec,
                 const BatchRichardsonOptions<remove_complex<ValueType>> opts)
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
            gko::kernels::batch_rich::local_memory_requirement<ValueType>(
                nrows, nrhs) +
            PrecType::dynamic_work_size(nrows, a.num_nnz) * sizeof(ValueType);
        std::vector<unsigned char> local_space(local_size_bytes);

        for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
            batch_entry_richardson_impl<StopType, PrecType, LogType,
                                        BatchMatrixType, ValueType>(
                opts_, logger, prec, a, b, x, ibatch, local_space.data());
        }
    }

private:
    std::shared_ptr<const ReferenceExecutor> exec_;
    const BatchRichardsonOptions<remove_complex<ValueType>> opts_;
};


template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const BatchRichardsonOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a, const BatchLinOp* const prec,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    auto dispatcher = batch_solver::create_dispatcher<ValueType>(
        KernelCaller<ValueType>(exec, opts), opts, a, prec);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_RICHARDSON_APPLY_KERNEL);


}  // namespace batch_rich
}  // namespace reference
}  // namespace kernels
}  // namespace gko
