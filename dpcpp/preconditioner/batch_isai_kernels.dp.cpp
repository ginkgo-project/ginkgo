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

#include "core/preconditioner/batch_isai_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/merging.dp.hpp"
#include "dpcpp/components/segment_scan.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/matrix/batch_struct.hpp"

namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_isai {

namespace {
#include "dpcpp/preconditioner/batch_isai.hpp.inc"
#include "dpcpp/preconditioner/batch_isai_kernels.hpp.inc"
}  // namespace


template <typename ValueType, typename IndexType>
void extract_dense_linear_sys_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const matrix::Csr<ValueType, IndexType>* const first_approx_inv,
    IndexType* const dense_mat_pattern, IndexType* const rhs_one_idxs,
    IndexType* const sizes, IndexType* num_matches_per_row_for_each_csr_sys)
{
    const auto nrows = first_approx_inv->get_size()[0];
    const auto nnz_aiA = first_approx_inv->get_num_stored_elements();

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    constexpr int subgroup_size = config::warp_size;

    const dim3 block(group_size);
    const dim3 grid(ceildiv(nnz_aiA * subgroup_size,
                            group_size));  // TODO: Phuong - rewrite this kernel
                                           // with  nnz_aiA * nrows subgroups
    const auto sys_row_ptrs = first_sys_csr->get_const_row_ptrs();
    const auto sys_col_idxs = first_sys_csr->get_const_col_idxs();
    const auto approx_row_ptrs = first_approx_inv->get_const_row_ptrs();
    const auto approx_col_idxs = first_approx_inv->get_const_col_idxs();
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[intel::reqd_sub_group_size(subgroup_size)]] {
                    extract_dense_linear_sys_pattern_kernel<subgroup_size>(
                        nrows, sys_row_ptrs, sys_col_idxs, approx_row_ptrs,
                        approx_col_idxs, dense_mat_pattern, rhs_one_idxs, sizes,
                        num_matches_per_row_for_each_csr_sys, item_ct1);
                });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_EXTRACT_DENSE_LINEAR_SYSTEM_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_values_dense_mat_and_solve(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const inv,
    const IndexType* const dense_mat_pattern,
    const IndexType* const rhs_one_idxs, const IndexType* const sizes,
    const gko::preconditioner::batch_isai_input_matrix_type&
        input_matrix_type_isai)
{
    const auto nbatch = inv->get_num_batch_entries();
    const auto nrows = static_cast<int>(inv->get_size().at(0)[0]);
    const auto A_nnz = sys_csr->get_num_stored_elements() / nbatch;
    const auto aiA_nnz = inv->get_num_stored_elements() / nbatch;

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    constexpr int subgroup_size = config::warp_size;

    const dim3 block(group_size);
    const dim3 grid(ceildiv(nbatch * nrows * subgroup_size, group_size));

    const auto sys_csr_values = sys_csr->get_const_values();
    auto inv_values = inv->get_values();
    const auto inv_row_ptrs = inv->get_const_row_ptrs();
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[intel::reqd_sub_group_size(subgroup_size)]] {
                    fill_values_dense_mat_and_solve_kernel<ValueType>(
                        nbatch, nrows, A_nnz, sys_csr_values, aiA_nnz,
                        inv_row_ptrs, inv_values, dense_mat_pattern,
                        rhs_one_idxs, sizes, input_matrix_type_isai, item_ct1);
                });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_FILL_VALUES_DENSE_MATRIX_AND_SOLVE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_isai(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
                const matrix::BatchCsr<ValueType, IndexType>* const approx_inv,
                const matrix::BatchDense<ValueType>* const r,
                matrix::BatchDense<ValueType>* const z)
{
    const auto num_rows = static_cast<int>(sys_mat->get_size().at(0)[0]);
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto approx_inv_batch = get_batch_struct(approx_inv);
    using prec_type = batch_isai<ValueType>;
    prec_type prec(approx_inv_batch);

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    constexpr int subgroup_size = config::warp_size;
    int shared_size = prec_type::dynamic_work_size(
        num_rows,
        static_cast<int>(sys_mat->get_num_stored_elements() / nbatch));
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
                auto r_b = r_values + batch_id * num_rows;
                auto z_b = z_values + batch_id * num_rows;
                batch_isai_apply(
                    prec, num_rows, r_b, z_b,
                    static_cast<ValueType*>(slm_storage.get_pointer()),
                    item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void extract_csr_sys_pattern(
    std::shared_ptr<const DefaultExecutor> exec, const int lin_sys_row,
    const int size,
    const matrix::Csr<ValueType, IndexType>* const first_approx_inv,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    matrix::Csr<gko::remove_complex<ValueType>, IndexType>* const csr_pattern)
{
    const auto nrows = first_approx_inv->get_size()[0];

    const auto approx_row_ptrs = first_approx_inv->get_const_row_ptrs();
    const auto approx_col_idxs = first_approx_inv->get_const_col_idxs();
    const auto sys_row_ptrs = first_sys_csr->get_const_row_ptrs();
    const auto sys_col_idxs = first_sys_csr->get_const_col_idxs();
    const auto csr_row_ptrs = csr_pattern->get_const_row_ptrs();
    auto csr_col_idxs = csr_pattern->get_col_idxs();
    auto csr_values = csr_pattern->get_values();

    auto n_items = approx_row_ptrs[lin_sys_row + 1];

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(n_items, [=](auto gid) {
            extract_csr_sys_pattern_kernel<ValueType>(
                lin_sys_row, approx_row_ptrs, approx_col_idxs, sys_row_ptrs,
                sys_col_idxs, csr_row_ptrs, csr_col_idxs, csr_values, gid);
        });
    });
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_EXTRACT_CSR_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_batch_csr_sys_with_values(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<gko::remove_complex<ValueType>, IndexType>* const
        csr_pattern,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const batch_csr_mats)
{
    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto csr_nnz = csr_pattern->get_num_stored_elements();
    const auto sys_nnz = sys_csr->get_num_stored_elements() / nbatch;

    auto n_items = nbatch * csr_nnz;

    const auto csr_pattern_values = csr_pattern->get_const_values();
    const auto sys_csr_values = sys_csr->get_const_values();
    auto batch_csr_mats_values = batch_csr_mats->get_values();
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(n_items, [=](auto gid) {
            fill_batch_csr_system_kernel(nbatch, csr_nnz, csr_pattern_values,
                                         sys_nnz, sys_csr_values,
                                         batch_csr_mats_values, gid);
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_FILL_BATCH_CSR_SYSTEM_USING_PATTERN);


template <typename ValueType, typename IndexType>
void initialize_b_and_x_vectors(std::shared_ptr<const DefaultExecutor> exec,
                                const IndexType rhs_one_idx,
                                matrix::BatchDense<ValueType>* const b,
                                matrix::BatchDense<ValueType>* const x)
{
    const auto nbatch = b->get_num_batch_entries();
    const auto size = b->get_size().at(0)[0];
    auto n_items = nbatch * size;

    auto b_values = b->get_values();
    auto x_values = x->get_values();
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(n_items, [=](auto gid) {
            initialize_b_and_x_vectors_kernel(nbatch, size, rhs_one_idx,
                                              b_values, x_values, gid);
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_INITIALIZE_B_AND_X);


template <typename ValueType, typename IndexType>
void write_large_sys_solution_to_inverse(
    std::shared_ptr<const DefaultExecutor> exec, const int lin_sys_row,
    const matrix::BatchDense<ValueType>* const x,
    matrix::BatchCsr<ValueType, IndexType>* const approx_inv)
{
    const auto nbatch = x->get_num_batch_entries();
    const auto size = x->get_size().at(0)[0];
    auto n_items = nbatch * size;

    const auto x_values = x->get_const_values();
    const auto approx_inv_n_elems =
        approx_inv->get_num_stored_elements() / nbatch;
    const auto approx_inv_row_ptrs = approx_inv->get_const_row_ptrs();
    auto approx_inv_values = approx_inv->get_values();

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(n_items, [=](auto gid) {
            write_large_sys_solution_to_inverse_kernel(
                nbatch, lin_sys_row, size, x_values, approx_inv_n_elems,
                approx_inv_row_ptrs, approx_inv_values, gid);
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_WRITE_SOLUTION_TO_INVERSE);

}  // namespace batch_isai
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
