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

#include "core/solver/batch_lower_trs_kernels.hpp"


#include "core/matrix/batch_struct.hpp"
// #include "core/solver/batch_dispatch.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/matrix/batch_struct.hpp"

namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_lower_trs {

#include "dpcpp/matrix/batch_csr_kernels.hpp.inc"
#include "dpcpp/matrix/batch_dense_kernels.hpp.inc"
#include "dpcpp/matrix/batch_ell_kernels.hpp.inc"
#include "dpcpp/solver/batch_lower_trs_kernels.hpp.inc"

template <typename BatchMatrixType, typename ValueType>
void call_apply_kernel(std::shared_ptr<const DefaultExecutor> exec,
                       const BatchMatrixType& a,
                       const gko::batch_dense::UniformBatch<const ValueType>& b,
                       const gko::batch_dense::UniformBatch<ValueType>& x)
{
    const auto nbatch = a.num_batch;
    const auto num_rows = a.num_rows;
    const auto num_rhs = b.num_rhs;
    assert(num_rhs == 1);

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(nbatch);

    size_type slm_size = device.get_info<sycl::info::device::local_mem_size>();

    const int shared_size =
        gko::kernels::batch_lower_trs::local_memory_requirement<ValueType>(
            num_rows,
            num_rhs);  // TODO: make it works with SLM, atm shared_size = 0

    GKO_ASSERT(shared_size <= slm_size);

    auto x_values = x.values;
    auto b_values = b.values;

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            slm_values(sycl::range<1>(shared_size), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_group_linear_id();
                const auto a_b = gko::batch::batch_entry(a, batch_id);
                const ValueType* const b_b = gko::batch::batch_entry_ptr(
                    b_values, 1, num_rows, batch_id);
                ValueType* const x_b = gko::batch::batch_entry_ptr(
                    x_values, 1, num_rows, batch_id);
                apply_kernel(a_b, b_b, x_b, num_rows,
                             static_cast<ValueType*>(slm_values.get_pointer()),
                             item_ct1);
            });
    });
}


template <typename ValueType>
void dispatch_on_matrix_type(std::shared_ptr<const DefaultExecutor> exec,
                             const BatchLinOp* const sys_mat,
                             const matrix::BatchDense<ValueType>* const b,
                             matrix::BatchDense<ValueType>* const x)
{
    namespace device = gko::kernels::dpcpp;
    const auto b_b = device::get_batch_struct(b);
    const auto x_b = device::get_batch_struct(x);

    if (auto amat = dynamic_cast<const matrix::BatchCsr<ValueType>*>(sys_mat)) {
        auto m_b = device::get_batch_struct(amat);
        call_apply_kernel(exec, m_b, b_b, x_b);

    } else if (auto amat =
                   dynamic_cast<const matrix::BatchEll<ValueType>*>(sys_mat)) {
        auto m_b = device::get_batch_struct(amat);
        call_apply_kernel(exec, m_b, b_b, x_b);

    } else if (auto amat = dynamic_cast<const matrix::BatchDense<ValueType>*>(
                   sys_mat)) {
        auto m_b = device::get_batch_struct(amat);
        call_apply_kernel(exec, m_b, b_b, x_b);
    } else {
        GKO_NOT_SUPPORTED(sys_mat);
    }
}


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const BatchLinOp* const sys_mat,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x)
{
    dispatch_on_matrix_type(exec, sys_mat, b, x);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_LOWER_TRS_APPLY_KERNEL);


}  // namespace batch_lower_trs
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
