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

#include "core/solver/batch_cg_kernels.hpp"


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
 * @brief The batch Cg solver namespace.
 *
 * @ingroup batch_cg
 */
namespace batch_cg {


#include "dpcpp/matrix/batch_csr_kernels.hpp.inc"
#include "dpcpp/matrix/batch_dense_kernels.hpp.inc"
#include "dpcpp/matrix/batch_ell_kernels.hpp.inc"
#include "dpcpp/matrix/batch_vector_kernels.hpp.inc"
#include "dpcpp/solver/batch_cg_kernels.hpp.inc"


template <typename T>
using BatchCgOptions = gko::kernels::batch_cg::BatchCgOptions<T>;

int get_larger_power(int value, int guess = 32)
{
    return guess >= value ? guess : get_larger_power(value, guess << 1);
}

template <typename ValueType>
class KernelCaller {
public:
    KernelCaller(std::shared_ptr<const DpcppExecutor> exec,
                 const BatchCgOptions<remove_complex<ValueType>> opts)
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
        int group_size =
            device.get_info<sycl::info::device::max_work_group_size>();
        if (group_size > 2 * nrows) group_size = get_larger_power(nrows);
        constexpr int subgroup_size = config::warp_size;

        const dim3 block(group_size);
        const dim3 grid(num_batches);

        size_type shmem_per_blk =
            device.get_info<sycl::info::device::local_mem_size>();
        const auto matrix_size = a.get_entry_storage();
        const int shared_gap =
            nrows;  // TODO: check if it is neccessary to align
        const size_type prec_size =
            PrecType::dynamic_work_size(shared_gap, a.num_nnz) *
            sizeof(ValueType);
        const auto sconf =
            gko::kernels::batch_cg::compute_shared_storage<PrecType, ValueType>(
                shmem_per_blk, shared_gap, a.num_nnz, b.num_rhs);
        const size_t shared_size =
            sconf.n_shared * shared_gap +
            (sconf.prec_shared ? prec_size : 0) / sizeof(ValueType);
        auto workspace = gko::Array<ValueType>(
            exec_, sconf.gmem_stride_bytes * num_batches / sizeof(ValueType));
        assert(sconf.gmem_stride_bytes % sizeof(ValueType) == 0);

        ValueType* const workspace_data = workspace.get_data();
        auto b_values = b.values;
        auto x_values = x.values;
        auto max_iters = opts_.max_its;
        auto res_tol = opts_.residual_tol;

        (exec_->get_queue())->submit([&](sycl::handler& cgh) {
            sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                slm_values(sycl::range<1>(shared_size), cgh);
            cgh.parallel_for(
                sycl_nd_range(grid, block),
                [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(subgroup_size)]] {
                        auto group = item_ct1.get_group();
                        auto batch_id = group.get_group_linear_id();
                        const auto a_global_entry =
                            gko::batch::batch_entry(a, batch_id);
                        const ValueType* const b_global_entry =
                            gko::batch::batch_entry_ptr(b_values, 1, nrows,
                                                        batch_id);
                        ValueType* const x_global_entry =
                            gko::batch::batch_entry_ptr(x_values, 1, nrows,
                                                        batch_id);

                        ValueType* const slm_values_ptr =
                            slm_values.get_pointer();

                        if (sconf.n_global == 0) {
                            small_apply_kernel<StopType>(
                                sconf, max_iters, res_tol, logger, prec,
                                a_global_entry, b_global_entry, x_global_entry,
                                nrows, a.num_nnz, slm_values_ptr, item_ct1,
                                workspace_data);
                        } else {
                            apply_kernel<StopType>(
                                sconf, max_iters, res_tol, logger, prec,
                                a_global_entry, b_global_entry, x_global_entry,
                                nrows, a.num_nnz, slm_values_ptr, item_ct1,
                                workspace_data);
                        }
                    });
        });
    }

private:
    std::shared_ptr<const DpcppExecutor> exec_;
    const BatchCgOptions<remove_complex<ValueType>> opts_;
};

template <typename ValueType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
           const BatchCgOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a, const BatchLinOp* const prec,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           gko::log::BatchLogData<ValueType>& logdata)
{
    auto dispatcher = batch_solver::create_dispatcher<ValueType>(
        KernelCaller<ValueType>(exec, opts), opts, a, prec);
    dispatcher.apply(b, x, logdata);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG_APPLY_KERNEL);


}  // namespace batch_cg
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
