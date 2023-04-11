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

#include "core/preconditioner/batch_ilu_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_ilu {

#include "dpcpp/matrix/batch_vector_kernels.hpp.inc"
#include "dpcpp/preconditioner/batch_ilu.hpp.inc"
#include "dpcpp/preconditioner/batch_ilu_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void compute_ilu0_factorization(
    std::shared_ptr<const DefaultExecutor> exec,
    const IndexType* const diag_locs,
    matrix::BatchCsr<ValueType, IndexType>* const mat_fact)
{
    const auto num_rows = static_cast<int>(mat_fact->get_size().at(0)[0]);
    const auto nbatch = mat_fact->get_num_batch_entries();
    const auto nnz =
        static_cast<int>(mat_fact->get_num_stored_elements() / nbatch);

    const int shared_size = 2 * num_rows;

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    size_type slm_size = device.get_info<sycl::info::device::local_mem_size>();
    GKO_ASSERT(shared_size * sizeof(ValueType) <= slm_size);

    const dim3 block(group_size);
    const dim3 grid(nbatch);

    const auto row_ptrs = mat_fact->get_const_row_ptrs();
    const auto col_idxs = mat_fact->get_const_col_idxs();
    auto mat_values = mat_fact->get_values();
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            slm_storage(sycl::range<1>(shared_size), cgh);
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                generate_exact_ilu0_kernel(
                    num_rows, nnz, diag_locs, row_ptrs, col_idxs, mat_values,
                    static_cast<ValueType*>(slm_storage.get_pointer()),
                    item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_EXACT_ILU_COMPUTE_FACTORIZATION_KERNEL);


template <typename ValueType, typename IndexType>
void compute_parilu0_factorization(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    matrix::BatchCsr<ValueType, IndexType>* const mat_fact,
    const int parilu_num_sweeps, const IndexType* const dependencies,
    const IndexType* const nz_ptrs)
{
    const auto num_rows = static_cast<int>(sys_mat->get_size().at(0)[0]);
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto nnz =
        static_cast<int>(sys_mat->get_num_stored_elements() / nbatch);

    const int shared_size = nnz;

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    size_type slm_size = device.get_info<sycl::info::device::local_mem_size>();
    GKO_ASSERT(shared_size * sizeof(ValueType) <= slm_size);

    const dim3 block(group_size);
    const dim3 grid(nbatch);

    const auto sys_values = sys_mat->get_const_values();
    auto mat_values = mat_fact->get_values();
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            slm_storage(sycl::range<1>(shared_size), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                generate_parilu0_kernel(
                    num_rows, nnz, dependencies, nz_ptrs, parilu_num_sweeps,
                    sys_values, mat_values,
                    static_cast<ValueType*>(slm_storage.get_pointer()),
                    item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PARILU_COMPUTE_FACTORIZATION_KERNEL);


// Only for testing purpose
template <typename ValueType, typename IndexType>
void apply_ilu(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_matrix,
    const matrix::BatchCsr<ValueType, IndexType>* const factored_matrix,
    const IndexType* const diag_locs,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto num_rows =
        static_cast<int>(factored_matrix->get_size().at(0)[0]);
    const auto nbatch = factored_matrix->get_num_batch_entries();
    const auto factored_matrix_batch = get_batch_struct(factored_matrix);
    using prec_type = batch_ilu<ValueType>;
    prec_type prec(factored_matrix_batch, diag_locs);
    const auto shared_size = prec_type::dynamic_work_size(
        num_rows,
        static_cast<int>(sys_matrix->get_num_stored_elements() / nbatch));

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    size_type slm_size = device.get_info<sycl::info::device::local_mem_size>();
    GKO_ASSERT(shared_size * sizeof(ValueType) <= slm_size);

    const dim3 block(group_size);
    const dim3 grid(nbatch);

    const auto r_values = r->get_const_values();
    auto z_values = z->get_values();
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            slm_storage(sycl::range<1>(shared_size), cgh);
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto batch_id = item_ct1.get_group_linear_id();
                const auto r_b = r_values + batch_id * num_rows;
                auto z_b = z_values + batch_id * num_rows;
                batch_ilu_apply(
                    prec, num_rows, r_b, z_b,
                    static_cast<ValueType*>(slm_storage.get_pointer()),
                    item_ct1);
            });
    });
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void generate_common_pattern_to_fill_l_and_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_mat,
    const IndexType* const l_row_ptrs, const IndexType* const u_row_ptrs,
    IndexType* const l_col_holders, IndexType* const u_col_holders)
{
    const size_type num_rows = first_sys_mat->get_size()[0];

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    constexpr int subgroup_size = config::warp_size;

    const dim3 block(group_size);
    const dim3 grid(ceildiv(num_rows * subgroup_size, group_size));

    const auto row_ptrs = first_sys_mat->get_const_row_ptrs();
    const auto col_idxs = first_sys_mat->get_const_col_idxs();
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1)
                             [[intel::reqd_sub_group_size(subgroup_size)]] {
                                 generate_common_pattern_to_fill_L_and_U(
                                     num_rows, row_ptrs, col_idxs, l_row_ptrs,
                                     u_row_ptrs, l_col_holders, u_col_holders,
                                     item_ct1);
                             });
    });
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_GENERATE_COMMON_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_batch_l_and_batch_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    matrix::BatchCsr<ValueType, IndexType>* const l_factor,
    matrix::BatchCsr<ValueType, IndexType>* const u_factor,
    const IndexType* const l_col_holders, const IndexType* const u_col_holders)
{
    const auto num_rows = static_cast<int>(sys_mat->get_size().at(0)[0]);
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto nnz =
        static_cast<int>(sys_mat->get_num_stored_elements() / nbatch);
    const auto l_nnz =
        static_cast<int>(l_factor->get_num_stored_elements() / nbatch);
    const auto u_nnz =
        static_cast<int>(u_factor->get_num_stored_elements() / nbatch);
    const int greater_nnz = l_nnz > u_nnz ? l_nnz : u_nnz;

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    //    const auto row_ptrs = sys_mat->get_const_row_ptrs();
    const auto sys_col_idxs = sys_mat->get_const_col_idxs();
    auto sys_values = sys_mat->get_const_values();
    auto l_factor_col_idxs = l_factor->get_col_idxs();
    auto l_factor_values = l_factor->get_values();
    auto u_factor_col_idxs = u_factor->get_col_idxs();
    auto u_factor_values = u_factor->get_values();

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nbatch * greater_nnz, [=](auto id) {
            fill_L_and_U(nbatch, num_rows, nnz, sys_col_idxs, sys_values, l_nnz,
                         l_factor_col_idxs, l_factor_values, l_col_holders,
                         u_nnz, u_factor_col_idxs, u_factor_values,
                         u_col_holders, id);
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_INITIALIZE_BATCH_L_AND_BATCH_U);

}  // namespace batch_ilu
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
