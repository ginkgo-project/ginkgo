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

#include "core/matrix/batch_csr_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <utility>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>

#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_csr
 */
namespace batch_csr {


#include "dpcpp/matrix/batch_csr_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::BatchCsr<ValueType, IndexType>* a,
          const matrix::BatchDense<ValueType>* b,
          matrix::BatchDense<ValueType>* c)
{
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);

    // From data types of a -> find number of batches
    const auto num_batches = a_ub.num_batch;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    // Launch a kernel that has nbatches blocks, each block has max group size
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto a_b = batch::batch_entry(a_ub, group_id);
                const auto b_b = batch::batch_entry(b_ub, group_id);
                const auto c_b = batch::batch_entry(c_ub, group_id);
                matvec_kernel(a_b, b_b, c_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::BatchDense<ValueType>* alpha,
                   const matrix::BatchCsr<ValueType, IndexType>* a,
                   const matrix::BatchDense<ValueType>* b,
                   const matrix::BatchDense<ValueType>* beta,
                   matrix::BatchDense<ValueType>* c)
{
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);

    const auto num_batches = a_ub.num_batch;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    // Launch a kernel that has nbatches blocks, each block has max_group_size
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto a_b = batch::batch_entry(a_ub, group_id);
                const auto b_b = batch::batch_entry(b_ub, group_id);
                const auto c_b = batch::batch_entry(c_ub, group_id);
                const auto alpha_b = batch::batch_entry(alpha_ub, group_id);
                const auto beta_b = batch::batch_entry(beta_ub, group_id);
                advanced_matvec_kernel(alpha_b.values[0], a_b, b_b,
                                       beta_b.values[0], c_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const DpcppExecutor> exec,
                              const IndexType* ptrs, size_type num_rows,
                              IndexType* idxs) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::BatchCsr<ValueType, IndexType>* source,
                      matrix::BatchDense<ValueType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(std::shared_ptr<const DpcppExecutor> exec,
                             matrix::BatchCsr<ValueType, IndexType>* trans,
                             const matrix::BatchCsr<ValueType, IndexType>* orig,
                             UnaryOperator op) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::BatchCsr<ValueType, IndexType>* orig,
               matrix::BatchCsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::BatchCsr<ValueType, IndexType>* orig,
                    matrix::BatchCsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const DpcppExecutor> exec,
                          const matrix::BatchCsr<ValueType, IndexType>* source,
                          size_type* result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* source,
    size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* source,
    array<size_type>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const DpcppExecutor> exec,
                          matrix::BatchCsr<ValueType, IndexType>* to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* to_check,
    bool* is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void batch_scale(std::shared_ptr<const DpcppExecutor> exec,
                 const matrix::BatchDiagonal<ValueType>* const left_scale,
                 const matrix::BatchDiagonal<ValueType>* const right_scale,
                 matrix::BatchCsr<ValueType, IndexType>* const mat)
{
    if (!left_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    if (!right_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;

    const auto m_ub = get_batch_struct(mat);
    const auto ncols = mat->get_size().at()[1];
    const auto left_values = left_scale->get_const_values();
    const auto right_values = right_scale->get_const_values();

    const auto num_batches = m_ub.num_batch;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    constexpr auto subgroup_size = config::warp_size;

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=
        ](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                                            subgroup_size)]] {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto m_b = batch::batch_entry(m_ub, group_id);
                const auto left_b = batch::batch_entry_ptr(
                    left_values, 1, m_ub.num_rows, group_id);
                const auto right_b =
                    batch::batch_entry_ptr(right_values, 1, ncols, group_id);
                batch_scale_kernel(left_b, right_b, m_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SCALE);


template <typename ValueType, typename IndexType>
void pre_diag_transform_system(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchDiagonal<ValueType>* const left_op,
    const matrix::BatchDiagonal<ValueType>* const right_op,
    matrix::BatchCsr<ValueType, IndexType>* const a,
    matrix::BatchDense<ValueType>* const b)
{
    const int num_batches = a->get_num_batch_entries();
    const int num_rows = a->get_size().at()[0];
    const int num_cols = a->get_size().at()[1];
    const size_type a_batch_stride = a->get_num_stored_elements() / num_batches;
    const int num_rhs = b->get_size().at()[1];
    const size_type b_stride = b->get_stride().at();

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(num_batches);

    // Extracting values to avoid doing so inside the lambda kernel
    const auto left_values = left_op->get_const_values();
    const auto right_values = right_op->get_const_values();
    auto a_values = a->get_values();
    auto b_values = b->get_values();
    const auto col_idxs = a->get_const_col_idxs();
    const auto row_ptrs = a->get_const_row_ptrs();

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_group_linear_id();
                auto ab = a_values + a_batch_stride * batch_id;
                auto bb = batch::batch_entry_ptr(b_values, b_stride, num_rows,
                                                 batch_id);
                auto left_scaleb =
                    batch::batch_entry_ptr(left_values, 1, num_rows, batch_id);
                auto right_scaleb =
                    batch::batch_entry_ptr(right_values, 1, num_cols, batch_id);
                pre_diag_scale_kernel(num_rows, ab, col_idxs, row_ptrs, num_rhs,
                                      b_stride, bb, left_scaleb, right_scaleb,
                                      item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_PRE_DIAG_TRANSFORM_SYSTEM);


template <typename ValueType, typename IndexType>
void convert_to_batch_dense(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const src,
    matrix::BatchDense<ValueType>* const dest)
{
    const int num_batches = src->get_num_batch_entries();
    const auto num_rows = src->get_size().at()[0];
    const auto num_cols = src->get_size().at()[1];
    const auto nnz = src->get_num_stored_elements() / num_batches;
    const auto dense_stride = dest->get_stride().at();

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(num_batches);

    // Extracting values to avoid doing so inside the lambda kernel
    const auto src_values = src->get_const_values();
    auto dest_values = dest->get_values();
    const auto col_idxs = src->get_const_col_idxs();
    const auto row_ptrs = src->get_const_row_ptrs();

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_group_linear_id();
                const auto bvalues = src_values + batch_id * nnz;
                auto bdense = dest_values + batch_id * dense_stride * num_rows;
                convert_to_batch_dense_kernel(num_rows, num_cols, row_ptrs,
                                              col_idxs, bvalues, dense_stride,
                                              bdense, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_BATCH_DENSE);


template <typename ValueType, typename IndexType>
void check_diagonal_entries_exist(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const mtx,
    bool& has_all_diags)
{
    if (!mtx->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const auto nmin = static_cast<int>(
        std::min(mtx->get_size().at(0)[0], mtx->get_size().at(0)[1]));
    array<bool> d_result(exec, 1);

    // Here num_sg should be compile-time known
    constexpr auto group_size = config::max_block_size;
    constexpr auto sg_size = config::warp_size;
    constexpr auto num_sg = group_size / sg_size;
    const dim3 block(group_size);
    const dim3 grid(1);

    // Extracting values to avoid doing so inside the lambda kernel
    auto d_result_data = d_result.get_data();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto row_ptrs = mtx->get_const_row_ptrs();

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            tile_has_diags(sycl::range<1>(num_sg), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=
        ](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(sg_size)]] {
                check_all_diagonal_kernel(
                    nmin, row_ptrs, col_idxs, d_result_data,
                    tile_has_diags.get_pointer(), item_ct1);
            });
    });
    has_all_diags = exec->copy_val_to_host(d_result.get_const_data());
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CHECK_DIAGONAL_ENTRIES_EXIST);


template <typename ValueType, typename IndexType>
void add_scaled_identity(std::shared_ptr<const DpcppExecutor> exec,
                         const matrix::BatchDense<ValueType>* const a,
                         const matrix::BatchDense<ValueType>* const b,
                         matrix::BatchCsr<ValueType, IndexType>* const mtx)
{
    if (!mtx->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;

    const auto num_batches = mtx->get_num_batch_entries();
    const int nnz = mtx->get_num_stored_elements() / num_batches;
    const int num_rows = mtx->get_size().at()[0];
    const auto a_stride = a->get_stride().at();
    const auto b_stride = b->get_stride().at();

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(num_batches);

    // Extracting values to avoid doing so inside the lambda kernel
    const auto a_values = a->get_const_values();
    const auto b_values = b->get_const_values();
    auto mtx_values = mtx->get_values();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto row_ptrs = mtx->get_const_row_ptrs();

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_group_linear_id();

                ValueType* const values_b = mtx_values + batch_id * nnz;
                const ValueType* const alpha_b =
                    batch::batch_entry_ptr(a_values, a_stride, 1, batch_id);
                const ValueType* const beta_b =
                    batch::batch_entry_ptr(b_values, b_stride, 1, batch_id);
                add_scaled_identity_kernel(num_rows, row_ptrs, col_idxs,
                                           values_b, alpha_b[0], beta_b[0],
                                           item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_csr
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
