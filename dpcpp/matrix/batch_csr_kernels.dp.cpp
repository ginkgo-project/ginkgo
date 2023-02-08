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

// #include "core/matrix/batch_struct.hpp"
// #include "core/synthesizer/implementation_selection.hpp"
#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/segment_scan.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"
#include "dpcpp/matrix/batch_struct.hpp"

// #include "dpcpp/base/config.hpp"
// #include "dpcpp/base/dim3.dp.hpp"
// #include "dpcpp/base/onemkl_bindings.hpp"
// #include "dpcpp/components/atomic.dp.hpp"
// #include "dpcpp/components/cooperative_groups.dp.hpp"
// #include "dpcpp/components/reduction.dp.hpp"
// #include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_csr
 */
namespace batch_csr {

template <typename ValueType>
void matvec_kernel(sycl::nd_item<3>& item_ct1,
                   const gko::batch_csr::BatchEntry<const ValueType>& a,
                   const batch_dense::BatchEntry<const ValueType>& b,
                   const batch_dense::BatchEntry<ValueType>& c)
{
    auto sg = item_ct1.get_sub_group();

    for (int row_and_rhs_combination = sg.get_group_id();
         row_and_rhs_combination < a.num_rows * b.num_rhs;
         row_and_rhs_combination += sg.get_group_range().size()) {
        const int row = row_and_rhs_combination / b.num_rhs;
        const int rhs = row_and_rhs_combination % b.num_rhs;

        const int row_start = a.row_ptrs[row];
        const int row_end = a.row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = sg.get_local_id() + row_start; i < row_end;
             i += sg.get_local_range().size()) {
            const int col = a.col_idxs[i];
            const ValueType val = a.values[i];
            temp += val * b.values[col * b.stride + rhs];
        }

        temp = sycl::reduce_over_group(sg, temp, sycl::plus<>());

        if (sg.get_local_id() == 0) {
            c.values[row * c.stride + rhs] = temp;
        }
    }
}


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

    const dim3 block(group_size);  // Is there any perf diff between 1D vs 3D gr
    const dim3 grid(num_batches);

    // Launch a kernel that has nbatches blocks, each block has max group size
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) {  // TODO: enforce subgroup_size?
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto a_b = batch::batch_entry(a_ub, group_id);
                const auto b_b = batch::batch_entry(b_ub, group_id);
                const auto c_b = batch::batch_entry(c_ub, group_id);
                matvec_kernel(item_ct1, a_b, b_b, c_b);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SPMV_KERNEL);

template <typename ValueType>
void advanced_matvec_kernel(
    sycl::nd_item<3>& item_ct1, const ValueType alpha,
    const gko::batch_csr::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType>& c)
{
    auto sg = item_ct1.get_sub_group();

    for (int row_and_rhs_combination = sg.get_group_id();
         row_and_rhs_combination < a.num_rows * b.num_rhs;
         row_and_rhs_combination += sg.get_group_range().size()) {
        const int row = row_and_rhs_combination / b.num_rhs;
        const int rhs = row_and_rhs_combination % b.num_rhs;

        const int row_start = a.row_ptrs[row];
        const int row_end = a.row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = sg.get_local_id() + row_start; i < row_end;
             i += sg.get_local_range().size()) {
            const int col = a.col_idxs[i];
            const ValueType val = a.values[i];
            temp += alpha * val * b.values[col * b.stride + rhs];
        }

        temp = sycl::reduce_over_group(sg, temp, sycl::plus<>());

        if (sg.get_local_id() == 0) {
            c.values[row * c.stride + rhs] =
                temp + beta * c.values[row * c.stride + rhs];
        }
    }
}

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

    // From data types of a -> find number of batches
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
                advanced_matvec_kernel(item_ct1, alpha_b.values[0], a_b, b_b,
                                       beta_b.values[0], c_b);
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
inline void convert_batch_csr_to_csc(
    size_type num_rows, const IndexType* row_ptrs, const IndexType* col_idxs,
    const ValueType* batch_csr_vals, IndexType* row_idxs, IndexType* col_ptrs,
    ValueType* csc_vals, UnaryOperator op) GKO_NOT_IMPLEMENTED;


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

template <typename ValueType>
void batch_scale_kernel(  // TODO: consider to find a new kernel name
    sycl::nd_item<3>& item_ct1, const ValueType* const left_scale,
    const ValueType* const right_scale,
    const gko::batch_csr::BatchEntry<ValueType>& a)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int i_row = sg_id; i_row < a.num_rows; i_row += num_sg) {
        const ValueType rowscale = left_scale[i_row];
        for (int iz = a.row_ptrs[i_row] + sg.get_local_id();
             iz < a.row_ptrs[i_row + 1]; iz += sg_size) {
            a.values[iz] *= rowscale * right_scale[a.col_idxs[iz]];
        }
    }
}


template <typename ValueType, typename IndexType>
void batch_scale(std::shared_ptr<const DpcppExecutor> exec,
                 const matrix::BatchDiagonal<ValueType>* const left_scale,
                 const matrix::BatchDiagonal<ValueType>* const right_scale,
                 matrix::BatchCsr<ValueType, IndexType>* const mat)
{
    if (!left_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    if (!right_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;

    const auto ncols = mat->get_size().at()[1];
    const auto m_ub = get_batch_struct(mat);

    const auto num_batches = mat->get_num_batch_entries();
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto m_b = batch::batch_entry(m_ub, group_id);
                const auto left_b = batch::batch_entry_ptr(
                    left_scale->get_const_values(), 1, m_b.num_rows, group_id);
                const auto right_b = batch::batch_entry_ptr(
                    right_scale->get_const_values(), 1, ncols, group_id);
                batch_scale_kernel(item_ct1, left_b, right_b, m_b);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BATCH_CSR_SCALE);

#if 1
template <typename ValueType, typename IndexType>
void pre_diag_transform_system(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchDiagonal<ValueType>* const left_op,
    const matrix::BatchDiagonal<ValueType>* const right_op,
    matrix::BatchCsr<ValueType, IndexType>* const a,
    matrix::BatchDense<ValueType>* const b) GKO_NOT_IMPLEMENTED;
#else

template <typename ValueType>
__forceinline__ void pre_diag_scale_kernel(
    sycl::nd_item<3>& item_ct1, const int num_rows,
    ValueType* const __restrict__ a_values,
    const int* const __restrict__ col_idxs,
    const int* const __restrict__ row_ptrs, const int num_rhs,
    const size_type b_stride, ValueType* const __restrict__ b,
    const ValueType* const __restrict__ left_scale,
    const ValueType* const __restrict__ right_scale)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range();
    const int num_sg = sg.get_group_range();

    for (int i_row = sg_id; i_row < a.num_rows; i_row += num_sg) {
        const ValueType rowscale = left_scale[i_row];
        for (int iz = a.row_ptrs[i_row] + sg.get_local_id();
             iz < a.row_ptrs[i_row + 1]; iz += sg_size) {
            a.values[iz] *= rowscale * right_scale[a.col_idxs[iz]];
        }
    }
    for (int iz = item_ct1.get_local_linear_id(); iz < num_rows * num_rhs;
         iz += item_ct1.get_local_range(2)) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        b[row * b_stride + col] *= left_scale[row];
    }
}

template <typename ValueType, typename IndexType>
void pre_diag_transform_system(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchDiagonal<ValueType>* const left_op,
    const matrix::BatchDiagonal<ValueType>* const right_op,
    matrix::BatchCsr<ValueType, IndexType>* const a,
    matrix::BatchDense<ValueType>* const b)
{
    const int num_batches = mat->get_num_batch_entries();
    const auto num_rows = a->get_size().at()[0];
    const auto num_cols = a->get_size().at()[1];
    const auto a_batch_stride = a->get_num_stored_elements() / num_batches;
    const int num_rhs = b->get_size().at()[1];
    const auto b_stride = b->get_stride().at();


    const auto group_size = exec->get_exec_info().max_workgroup_size;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_linear_id();
                auto ab = a->get_values() + a_batch_stride * batch_id;
                auto bb = batch::batch_entry_ptr(b->get_values(), b_stride,
                                                 num_rows, batch_id);
                auto left_scaleb = batch::batch_entry_ptr(
                    left_scale->get_const_values(), 1, num_rows, batch_id);
                auto right_scaleb = batch::batch_entry_ptr(
                    right_scale->get_const_values(), 1, num_cols, batch_id);
                pre_diag_scale_kernel(item_ct1, num_rows, ab,
                                      a->get_const_col_idxs(),
                                      a->get_const_row_ptrs(), num_rhs,
                                      b_stride, bb, left_scaleb, right_scaleb);
            });
    });
}
#endif
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_PRE_DIAG_TRANSFORM_SYSTEM);

#if 1
template <typename ValueType, typename IndexType>
void convert_to_batch_dense(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const src,
    matrix::BatchDense<ValueType>* const dest) GKO_NOT_IMPLEMENTED;
#else

template <typename ValueType>
__forceinline__ void convert_to_batch_dense_kernel(
    sycl::nd_item<3>& item_ct1, const int num_rows, const int num_cols,
    const int* const row_ptrs, const int* const col_idxs,
    const ValueType* const values, const size_type dense_stride,
    ValueType* const dense)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range();
    const int num_sg = sg.get_group_range();

    for (int i_row = sg_id; i_row < num_rows; i_row += num_sg) {
        for (int j = sg.get_local_id(); j < num_cols; j += sg_size) {
            dense[i_row * dense_stride + j] = zero<ValueType>();
        }
        for (int iz = row_ptrs[i_row] + sg.get_local_id();
             iz < row_ptrs[i_row + 1]; iz += sg_size) {
            dense[i_row * dense_stride + col_idxs[iz]] = values[iz];
        }
    }
}
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

    const auto group_size = exec->get_exec_info().max_workgroup_size;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_linear_id();

                const auto bvalues = src->get_const_values() + batch_id * nnz;
                const auto bdense = dest->get_const_values() +
                                    batch_id * dense_stride * num_rows;

                convert_to_batch_dense_kernel(
                    item_ct1, num_rows, num_cols, src->get_const_row_ptrs(),
                    src->get_const_col_idxs(), bvalues, dense_stride, bdense);
            });
    });
}
#endif
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_BATCH_DENSE);

#if 1
template <typename ValueType, typename IndexType>
void check_diagonal_entries_exist(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const mtx,
    bool& has_all_diags) GKO_NOT_IMPLEMENTED;
#else

__forceinline__ void check_all_diagonal_kernel(
    sycl::nd_item<3>& item_ct1, const int min_rows_cols,
    const int* const __restrict__ row_ptrs,
    const int* const __restrict__ col_idxs, bool* const __restrict__ all_diags,
    int* tile_has_diags)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range();
    const int num_sg = sg.get_group_range();

    int this_tile_has_diags = 1;
    for (int row = sg_id; row < min_rows_cols; row += num_sg) {
        const int row_sz = row_ptrs[row + 1] - row_ptrs[row];
        int has_diag = 0;
        for (int iz = sg.get_local_id(); iz < row_sz; iz += sg_size) {
            has_diag = static_cast<int>(col_idxs[iz + row_ptrs[row]] == row);
            if (has_diag) {
                break;
            }
        }
        const config::lane_mask_type row_has_diag = tile.ballot(has_diag);
        this_tile_has_diags = this_tile_has_diags && row_has_diag;
    }
    if (sg.get_local_id() == 0) {
        tile_has_diags[sg_id] = this_tile_has_diags;
    }
    sg.barrier();  // TODO: sync with sg or group?

    // reduce array to one warp
    if (sg_id == 0) {
        for (int i = sg_size + sg.get_local_id(); i < num_sg; i += sg_size) {
            tile_has_diags[i % sg_size] =
                tile_has_diags[i % sg_size] && tile_has_diags[i];
        }
        // warp-reduce
        int var =
            sg.get_local_id() < num_sg ? tile_has_diags[sg.get_local_id()] : 1;
        /*
        for (int k = sg_size / 2; k >= 1; k >>= 1) {
          var = var && tile.shfl_down(var, k);
        }
        */
        var = sycl::reduce_over_group(sg, var, sycl::and<>());
        if (tile.thread_rank() == 0) {
            all_diags[0] = static_cast<bool>(var);
        }
    }
}

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

    const auto group_size = exec->get_exec_info().max_workgroup_size;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(1);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            tile_has_diags(sycl::range<1>(num_sg), cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
          check_all_diagonal_kernel(
              item_ct1,
              nmin,
              mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
              d_result.get_data(),
              tile_has_diags.get_pointer();
                         });
    });
    has_all_diags = exec->copy_val_to_host(d_result.get_const_data());
}
#endif
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CHECK_DIAGONAL_ENTRIES_EXIST);

#if 1
template <typename ValueType, typename IndexType>
void add_scaled_identity(std::shared_ptr<const DpcppExecutor> exec,
                         const matrix::BatchDense<ValueType>* const a,
                         const matrix::BatchDense<ValueType>* const b,
                         matrix::BatchCsr<ValueType, IndexType>* const mtx)
    GKO_NOT_IMPLEMENTED;
#else

template <typename ValueType>
__forceinline__ void add_scaled_identity_kernel(
    sycl::nd_item<3>& item_ct1, const int num_rows, const int* const row_ptrs,
    const int* const col_idxs, ValueType* const __restrict__ values,
    const ValueType& alpha, const ValueType& beta)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range();
    const int num_sg = sg.get_group_range();

    for (int row = sg_id; row < num_rows; row += num_sg) {
        for (int iz = row_ptrs[row] + sg.get_local_id(); iz < row_ptrs[row + 1];
             iz += sg_size) {
            values[iz] *= beta;
            if (row == col_idxs[iz]) {
                values[iz] += alpha;
            }
        }
    }
}

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

    add_scaled_identity<<<nbatch, default_block_size>>>(
        nbatch, nrows, nnz, mtx->get_const_row_ptrs(),
        mtx->get_const_col_idxs(), as_cuda_type(mtx->get_values()), astride,
        as_cuda_type(a->get_const_values()), bstride,
        as_cuda_type(b->get_const_values()));

    const auto group_size = exec->get_exec_info().max_workgroup_size;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_linear_id();

                ValueType* const values_b = mtx->get_values() + batch_id * nnz;
                const ValueType* const alpha_b = batch::batch_entry_ptr(
                    a->get_const_values(), a_stride, 1, batch_id);
                const ValueType* const beta_b = batch::batch_entry_ptr(
                    b->get_const_values(), b_stride, 1, batch_id);
                add_scaled_identity_kernel(
                    item_ct1, num_rows, mtx->get_const_row_ptrs,
                    mtx->get_const_col_idxs, values_b, alpha_b[0], beta_b[0]);
            });
    });
}

#endif

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_csr
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
