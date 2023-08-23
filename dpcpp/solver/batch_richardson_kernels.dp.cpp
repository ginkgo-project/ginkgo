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


#include <ginkgo/core/base/math.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/matrix/batch_struct.hpp"

namespace gko {
namespace kernels {
namespace dpcpp {

/**
 * @brief The batch Richardson solver namespace.
 *
 * @ingroup batch_rich
 */
namespace batch_rich {


// include all depedencies (note: do not remove this comment)
#include "dpcpp/matrix/batch_csr_kernels.hpp.inc"
#include "dpcpp/matrix/batch_dense_kernels.hpp.inc"
#include "dpcpp/matrix/batch_ell_kernels.hpp.inc"
#include "dpcpp/matrix/batch_vector_kernels.hpp.inc"
#include "dpcpp/solver/batch_richardson_kernels.hpp.inc"


template <typename T>
using BatchRichardsonOptions =
    gko::kernels::batch_rich::BatchRichardsonOptions<T>;

template <typename ValueType>
class KernelCaller {
public:
    KernelCaller(std::shared_ptr<const DpcppExecutor> exec,
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
        const size_type num_batches = a.num_batch;
        const auto nrows = a.num_rows;
        const auto nrhs = b.num_rhs;
        GKO_ASSERT(nrhs == 1);

        auto device = exec_->get_queue()->get_device();
        auto group_size =
            device.get_info<sycl::info::device::max_work_group_size>();
        constexpr int subgroup_size = config::warp_size;
        GKO_ASSERT(group_size >= 2 * subgroup_size);

        const dim3 block(group_size);
        const dim3 grid(num_batches);

        size_type slm_size =
            device.get_info<sycl::info::device::local_mem_size>();

        const int shared_size =
            gko::kernels::batch_rich::local_memory_requirement<ValueType>(
                a.num_rows, b.num_rhs) +
            PrecType::dynamic_work_size(a.num_rows, a.num_nnz) *
                sizeof(ValueType);

        GKO_ASSERT(shared_size <= slm_size - 2 * sizeof(real_type));

        const auto max_iters = opts_.max_its;
        const auto res_tol = opts_.residual_tol;
        const auto relax_factor = opts_.relax_factor;
        auto x_values = x.values;
        auto b_values = b.values;

        (exec_->get_queue())->submit([&](sycl::handler& cgh) {
            sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                slm_values(sycl::range<1>(shared_size), cgh);
            sycl::accessor<real_type, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                slm_reals(sycl::range<1>(2), cgh);

            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    auto group = item_ct1.get_group();
                    auto batch_id = group.get_group_linear_id();
                    const auto a_b = gko::batch::batch_entry(a, batch_id);
                    const ValueType* const b_b = gko::batch::batch_entry_ptr(
                        b_values, 1, nrows, batch_id);
                    ValueType* const x_b = gko::batch::batch_entry_ptr(
                        x_values, 1, nrows, batch_id);
                    apply_kernel<StopType>(
                        max_iters, res_tol, relax_factor, logger, prec, a_b,
                        b_b, x_b, nrows, nrhs,
                        static_cast<ValueType*>(slm_values.get_pointer()),
                        static_cast<real_type*>(slm_reals.get_pointer()),
                        item_ct1);
                });
        });
    }

private:
    std::shared_ptr<const DpcppExecutor> exec_;
    const BatchRichardsonOptions<remove_complex<ValueType>> opts_;
};


template <typename ValueType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
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
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
