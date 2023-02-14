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

#include "core/matrix/batch_ell_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <utility>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


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


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_ell
 */
namespace batch_ell {

template <typename ValueType>
inline void matvec_kernel(
    sycl::nd_item<3> item_ct1,
    const gko::batch_ell::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const gko::batch_dense::BatchEntry<ValueType>& c)
{
    for (int tidx = item_ct1.get_local_linear_id(); tidx < a.num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < a.num_stored_elems_per_row; idx++) {
            const auto col_idx = a.col_idxs[tidx + idx * a.stride];
            if (col_idx < idx)
                break;
            else
                temp += a.values[tidx + idx * a.stride] *
                        b.values[col_idx * b.stride];
        }
        c.values[tidx * c.stride] = temp;
    }
}

template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::BatchEll<ValueType, IndexType>* const a,
          const matrix::BatchDense<ValueType>* const b,
          matrix::BatchDense<ValueType>* const c)
{
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);

    auto const num_batches = a->get_num_batch_entries();
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_group_linear_id();
                const auto a_b = gko::batch::batch_entry(a_ub, batch_id);
                const auto b_b = gko::batch::batch_entry(b_ub, batch_id);
                const auto c_b = gko::batch::batch_entry(c_ub, batch_id);
                matvec_kernel(item_ct1, a_b, b_b, c_b);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_SPMV_KERNEL);

template <typename ValueType>
inline void advanced_matvec_kernel(
    sycl::nd_item<3> item_ct1, const ValueType alpha,
    const gko::batch_ell::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType>& c)
{
    for (int tidx = item_ct1.get_local_linear_id(); tidx < a.num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < a.num_stored_elems_per_row; idx++) {
            const auto col_idx = a.col_idxs[tidx + idx * a.stride];
            if (col_idx < idx)
                break;
            else
                temp += alpha * a.values[tidx + idx * a.stride] *
                        b.values[col_idx * b.stride];
        }
        c.values[tidx * c.stride] = temp + beta * c.values[tidx * c.stride];
    }
}

template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::BatchDense<ValueType>* const alpha,
                   const matrix::BatchEll<ValueType, IndexType>* const a,
                   const matrix::BatchDense<ValueType>* const b,
                   const matrix::BatchDense<ValueType>* const beta,
                   matrix::BatchDense<ValueType>* const c)
{
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);

    auto const num_batches = a->get_num_batch_entries();
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(num_batches);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_group_linear_id();
                const auto a_b = gko::batch::batch_entry(a_ub, batch_id);
                const auto b_b = gko::batch::batch_entry(b_ub, batch_id);
                const auto c_b = gko::batch::batch_entry(c_ub, batch_id);
                const auto alpha_b =
                    gko::batch::batch_entry(alpha_ub, batch_id);
                const auto beta_b = gko::batch::batch_entry(beta_ub, batch_id);
                const ValueType alphav = alpha_b.values[0];
                const ValueType betav = beta_b.values[0];
                advanced_matvec_kernel(item_ct1, alphav, a_b, b_b, betav, c_b);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const DpcppExecutor> exec,
                              const IndexType* ptrs, size_type num_rows,
                              IndexType* idxs) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::BatchEll<ValueType, IndexType>* source,
                      matrix::BatchDense<ValueType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const DpcppExecutor> exec,
                          const matrix::BatchEll<ValueType, IndexType>* source,
                          size_type* result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::BatchEll<ValueType, IndexType>* orig,
               matrix::BatchEll<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::BatchEll<ValueType, IndexType>* orig,
                    matrix::BatchEll<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* source,
    size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* source,
    array<size_type>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const DpcppExecutor> exec,
                          matrix::BatchEll<ValueType, IndexType>* to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* to_check,
    bool* is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void batch_scale(std::shared_ptr<const DpcppExecutor> exec,
                 const matrix::BatchDiagonal<ValueType>* const left_scale,
                 const matrix::BatchDiagonal<ValueType>* const right_scale,
                 matrix::BatchEll<ValueType, IndexType>* const mat)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_SCALE);


template <typename ValueType, typename IndexType>
void pre_diag_scale_system(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchDense<ValueType>* const left_scale,
    const matrix::BatchDense<ValueType>* const right_scale,
    matrix::BatchEll<ValueType, IndexType>* const a,
    matrix::BatchDense<ValueType>* const b) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_PRE_DIAG_SCALE_SYSTEM);


template <typename ValueType, typename IndexType>
void convert_to_batch_dense(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* const src,
    matrix::BatchDense<ValueType>* const dest) GKO_NOT_IMPLEMENTED;
// TODO
// {
//     const size_type nbatches = src->get_num_batch_entries();
//     const int num_rows = src->get_size().at(0)[0];
//     const int num_cols = src->get_size().at(0)[1];
//     const int nnz = static_cast<int>(src->get_num_stored_elements() /
//     nbatches); const size_type dstride = dest->get_stride().at(0); const
//     size_type estride = src->get_stride().at(0); const auto col_idxs =
//     src->get_const_col_idxs(); const auto vals = src->get_const_values();

//     const dim3 block_size(config::warp_size,
//                           config::max_block_size / config::warp_size, 1);
//     const dim3 init_grid_dim(ceildiv(num_cols * nbatches, block_size.x),
//                              ceildiv(num_rows * nbatches, block_size.y), 1);
//     initialize_zero_dense<<<init_grid_dim, block_size>>>(
//         nbatches, num_rows, num_cols, dstride,
//         as_cuda_type(dest->get_values()));

//     const auto grid_dim = ceildiv(num_rows * nbatches, default_block_size);
//     fill_in_dense<<<grid_dim, default_block_size>>>(
//         nbatches, num_rows, src->get_num_stored_elements_per_row().at(0),
//         estride, as_cuda_type(col_idxs), as_cuda_type(vals), dstride,
//         as_cuda_type(dest->get_values()));
// }

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONVERT_TO_BATCH_DENSE);


template <typename ValueType, typename IndexType>
void convert_from_batch_csc(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::BatchEll<ValueType, IndexType>* ell, const array<ValueType>& values,
    const array<IndexType>& row_idxs,
    const array<IndexType>& col_ptrs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONVERT_FROM_BATCH_CSC);

template <typename IndexType>
inline void check_diagonal_entries_kernel(
    sycl::nd_item<3> item_ct1, const IndexType num_min_rows_cols,
    const size_type row_stride, const size_type max_nnz_per_row,
    const IndexType* const __restrict__ col_idxs,
    bool* const __restrict__ has_all_diags)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    if (item_ct1.get_local_linear_id() == 0) {
        *has_all_diags = true;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (sg_id == 0 && num_min_rows_cols > 0) {
        bool row_has_diag_local{false};
        if (sg.get_local_id() == 0) {
            if (col_idxs[0] == 0) {
                row_has_diag_local = true;
            }
        }
        auto row_has_diag =
            sycl::ext::oneapi::group_ballot(sg, row_has_diag_local).any();
        if (!row_has_diag) {
            if (sg.get_local_id() == 0) {
                *has_all_diags = false;
            }
            return;
        }
    } else if (sg_id < num_min_rows_cols) {
        bool row_has_diag_local{false};
        for (IndexType iz = sg.get_local_id(); iz < max_nnz_per_row;
             iz += sg_size) {
            if (col_idxs[iz * row_stride + sg_id] == sg_id) {  // or = sg_id
                row_has_diag_local = true;
                break;
            }
        }
        auto row_has_diag =
            sycl::ext::oneapi::group_ballot(sg, row_has_diag_local).any();
        if (!row_has_diag) {
            if (sg.get_local_id() == 0) {
                *has_all_diags = false;
            }
            return;
        }
    }
}

template <typename ValueType, typename IndexType>
void check_diagonal_entries_exist(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* const mtx,
    bool& has_all_diags)
{
    if (!mtx->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const auto nmin = static_cast<int>(
        std::min(mtx->get_size().at(0)[0], mtx->get_size().at(0)[1]));
    const auto row_stride = mtx->get_stride().at(0);
    const auto max_nnz_per_row =
        static_cast<int>(mtx->get_num_stored_elements_per_row().at(0));
    array<bool> d_result(exec, 1);
    const auto col_idxs = mtx->get_const_col_idxs();
    auto d_result_data = d_result.get_data();

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(1);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto batch_id = group.get_group_linear_id();
                check_diagonal_entries_kernel(item_ct1, nmin, row_stride,
                                              max_nnz_per_row, col_idxs,
                                              d_result_data);
            });
    });
    has_all_diags = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CHECK_DIAGONAL_ENTRIES_EXIST);

template <typename ValueType>
inline void add_scaled_identity_kernel(
    sycl::nd_item<3> item_ct1, const int nrows, const size_type row_stride,
    const int max_nnz_per_row, const int* const col_idxs,
    ValueType* const __restrict__ values, const ValueType& alpha,
    const ValueType& beta)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int row = sg_id; row < nrows; row += num_sg) {
        if (row == 0) {
            for (int iz = sg.get_local_id(); iz < max_nnz_per_row;
                 iz += sg_size) {
                values[iz * row_stride] *= beta;
            }
            if (sg.get_local_id() == 0 && col_idxs[0] == 0) {
                values[0] += alpha;
            }
        } else {
            for (int iz = sg.get_local_id(); iz < max_nnz_per_row;
                 iz += sg_size) {
                values[iz * row_stride + row] *= beta;
                if (row == col_idxs[iz * row_stride + row]) {
                    values[iz * row_stride + row] += alpha;
                }
            }
        }
    }
}

template <typename ValueType, typename IndexType>
void add_scaled_identity(std::shared_ptr<const DpcppExecutor> exec,
                         const matrix::BatchDense<ValueType>* const a,
                         const matrix::BatchDense<ValueType>* const b,
                         matrix::BatchEll<ValueType, IndexType>* const mtx)
{
    if (!mtx->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const size_type num_batches = mtx->get_num_batch_entries();
    const int nnz =
        static_cast<int>(mtx->get_num_stored_elements() / num_batches);
    const int nrows = mtx->get_size().at()[0];
    const auto row_stride = mtx->get_stride().at(0);
    const auto max_nnz_per_row =
        static_cast<int>(mtx->get_num_stored_elements_per_row().at(0));
    const size_type a_stride = a->get_stride().at();
    const size_type b_stride = b->get_stride().at();

    const auto col_idxs = mtx->get_const_col_idxs();
    const auto a_values = a->get_const_values();
    const auto b_values = b->get_const_values();
    auto mtx_values = mtx->get_values();

    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const dim3 block(group_size);
    const dim3 grid(num_batches);

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
                add_scaled_identity_kernel(item_ct1, nrows, row_stride,
                                           max_nnz_per_row, col_idxs, values_b,
                                           alpha_b[0], beta_b[0]);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_ell
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
