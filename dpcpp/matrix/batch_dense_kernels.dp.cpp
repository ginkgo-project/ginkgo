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

#include "core/matrix/batch_dense_kernels.hpp"


#include <algorithm>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>

#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/matrix/batch_dense_kernels.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The BatchDense matrix format namespace.
 *
 * @ingroup batch_dense
 */
namespace batch_dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::BatchDense<ValueType>* a,
                  const matrix::BatchDense<ValueType>* b,
                  matrix::BatchDense<ValueType>* c)
{
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);

    if (b_ub.num_rhs > 1) {
        GKO_NOT_IMPLEMENTED;
    }

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
                matvec_kernel(a_b, b_b.values, c_b.values, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::BatchDense<ValueType>* alpha,
           const matrix::BatchDense<ValueType>* a,
           const matrix::BatchDense<ValueType>* b,
           const matrix::BatchDense<ValueType>* beta,
           matrix::BatchDense<ValueType>* c)
{
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);

    if (b_ub.num_rhs > 1) {
        GKO_NOT_IMPLEMENTED;
    }

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
                const auto alpha_b = batch::batch_entry(alpha_ub, group_id);
                const auto beta_b = batch::batch_entry(beta_ub, group_id);
                advanced_matvec_kernel(alpha_b.values[0], a_b, b_b.values,
                                       beta_b.values[0], c_b.values, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::BatchDense<ValueType>* alpha,
           matrix::BatchDense<ValueType>* x)
{
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);

    const auto num_batches = x_ub.num_batch;
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
                const auto alpha_b = batch::batch_entry(alpha_ub, group_id);
                const auto x_b = batch::batch_entry(x_ub, group_id);
                single_scale_kernel(alpha_b, x_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const DpcppExecutor> exec,
                const matrix::BatchDense<ValueType>* const alpha,
                const matrix::BatchDense<ValueType>* const x,
                matrix::BatchDense<ValueType>* const y)
{
    const size_type num_rows = x->get_size().at(0)[0];
    const size_type num_cols = x->get_size().at(0)[1];

    const auto num_batches = x->get_num_batch_entries();
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_batches);
    /*
        if (num_cols == 1) {
            const auto alpha_values = alpha->get_const_values();
            const auto x_values = x->get_const_values();
            auto y_values = y->get_values();
            (exec->get_queue())->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                        auto group = item_ct1.get_group();
                        auto group_id = group.get_group_linear_id();
                        const auto x_b =
                            batch::batch_entry_ptr(x_values, 1, num_rows,
       group_id); const auto y_b = batch::batch_entry_ptr(y_values, 1, num_rows,
       group_id); single_add_scaled_kernel(item_ct1, num_rows, alpha_values[0],
       x_b, y_b);
                    });
            });
        } else {
        */
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto group = item_ct1.get_group();
                auto group_id = group.get_group_linear_id();
                const auto alpha_b = batch::batch_entry(alpha_ub, group_id);
                const auto x_b = batch::batch_entry(x_ub, group_id);
                const auto y_b = batch::batch_entry(y_ub, group_id);
                add_scaled_kernel(alpha_b, x_b, y_b, item_ct1);
            });
    });
    //}
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scale(std::shared_ptr<const DpcppExecutor> exec,  // Not a good name!!!
               const matrix::BatchDense<ValueType>* const alpha,
               const matrix::BatchDense<ValueType>* const x,
               const matrix::BatchDense<ValueType>* const beta,
               matrix::BatchDense<ValueType>* const y)
{
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);

    const auto num_batches = x->get_num_batch_entries();
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
                const auto alpha_b = batch::batch_entry(alpha_ub, group_id);
                const auto beta_b = batch::batch_entry(beta_ub, group_id);
                const auto x_b = batch::batch_entry(x_ub, group_id);
                const auto y_b = batch::batch_entry(y_ub, group_id);
                add_scaled_advanced_kernel(alpha_b, x_b, beta_b, y_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALE_KERNEL);


template <typename ValueType>
void convergence_add_scaled(std::shared_ptr<const DpcppExecutor> exec,
                            const matrix::BatchDense<ValueType>* alpha,
                            const matrix::BatchDense<ValueType>* x,
                            matrix::BatchDense<ValueType>* y,
                            const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const DpcppExecutor> exec,
                     const matrix::BatchDense<ValueType>* alpha,
                     const matrix::Diagonal<ValueType>* x,
                     matrix::BatchDense<ValueType>* y) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const DpcppExecutor> exec,
                 const matrix::BatchDense<ValueType>* x,
                 const matrix::BatchDense<ValueType>* y,
                 matrix::BatchDense<ValueType>* result)
{
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    const auto res_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch;
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
                const auto x_b = batch::batch_entry(x_ub, group_id);
                const auto y_b = batch::batch_entry(y_ub, group_id);
                const auto res_b = batch::batch_entry(res_ub, group_id);
                compute_dot_product_kernel(x_b, y_b, res_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void convergence_compute_dot(std::shared_ptr<const DpcppExecutor> exec,
                             const matrix::BatchDense<ValueType>* x,
                             const matrix::BatchDense<ValueType>* y,
                             matrix::BatchDense<ValueType>* result,
                             const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::BatchDense<ValueType>* x,
                   matrix::BatchDense<remove_complex<ValueType>>* result)
{
    const auto x_ub = get_batch_struct(x);
    const auto res_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch;
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
                const auto x_b = batch::batch_entry(x_ub, group_id);
                const auto res_b = batch::batch_entry(res_ub, group_id);
                compute_norm2_kernel(x_b, res_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void convergence_compute_norm2(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchDense<ValueType>* x,
    matrix::BatchDense<remove_complex<ValueType>>* result,
    const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_batch_csr(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::BatchDense<ValueType>* source,
                          matrix::BatchCsr<ValueType, IndexType>* other)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_DENSE_CONVERT_TO_BATCH_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::BatchDense<ValueType>* source,
                    size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const DpcppExecutor> exec,
                               const matrix::BatchDense<ValueType>* source,
                               size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const DpcppExecutor> exec,
                                const matrix::BatchDense<ValueType>* source,
                                array<size_type>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const DpcppExecutor> exec,
                          const matrix::BatchDense<ValueType>* source,
                          size_type* result, const size_type* stride_factor,
                          const size_type* slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::BatchDense<ValueType>* orig,
               matrix::BatchDense<ValueType>* trans)
{
    const auto orig_values = orig->get_const_values();
    auto trans_values = trans->get_values();

    const size_type orig_stride = orig->get_stride().at();
    const size_type trans_stride = trans->get_stride().at();
    const int nrows = orig->get_size().at()[0];
    const int ncols = orig->get_size().at()[1];


    const auto num_batches = orig->get_num_batch_entries();
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
                const ValueType* const orig_b =
                    orig_values + group_id * orig_stride * nrows;
                ValueType* const trans_b =
                    trans_values + group_id * trans_stride * ncols;
                transpose_kernel(
                    nrows, ncols, orig_stride, orig_b, trans_stride, trans_b,
                    [](ValueType x) { return x; }, item_ct1);
            });
    });
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::BatchDense<ValueType>* orig,
                    matrix::BatchDense<ValueType>* trans)
{
    const auto orig_values = orig->get_const_values();
    auto trans_values = trans->get_values();

    const size_type orig_stride = orig->get_stride().at();
    const size_type trans_stride = trans->get_stride().at();
    const int nrows = orig->get_size().at()[0];
    const int ncols = orig->get_size().at()[1];

    const auto num_batches = orig->get_num_batch_entries();
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
                const ValueType* const orig_b =
                    orig_values + group_id * orig_stride * nrows;
                ValueType* const trans_b =
                    trans_values + group_id * trans_stride * ncols;
                transpose_kernel(
                    nrows, ncols, orig_stride, orig_b, trans_stride, trans_b,
                    [](ValueType x) { return conj(x); }, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::BatchDense<ValueType>* x,
          matrix::BatchDense<ValueType>* result)
{
    const auto x_ub = get_batch_struct(x);
    const auto result_ub = get_batch_struct(result);

    const auto num_batches = x_ub.num_batch;
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
                const auto x_b = batch::batch_entry(x_ub, group_id);
                const auto result_b = batch::batch_entry(result_ub, group_id);
                copy_kernel(x_b, result_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COPY_KERNEL);

template <typename ValueType>
void convergence_copy(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::BatchDense<ValueType>* x,
                      matrix::BatchDense<ValueType>* result,
                      const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COPY_KERNEL);


template <typename ValueType>
void batch_scale(std::shared_ptr<const DpcppExecutor> exec,
                 const matrix::BatchDiagonal<ValueType>* const left_scale,
                 const matrix::BatchDiagonal<ValueType>* const right_scale,
                 matrix::BatchDense<ValueType>* const x)
{
    if (!left_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    if (!right_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    if (!x->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;

    const auto x_stride = x->get_stride().at();
    const auto num_rows = static_cast<int>(x->get_size().at()[0]);
    const auto num_rhs = static_cast<int>(x->get_size().at()[1]);
    const auto num_batches = x->get_num_batch_entries();

    const auto left_values = left_scale->get_const_values();
    const auto right_values = right_scale->get_const_values();
    auto x_values = x->get_values();

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
                const auto x_ptr = batch::batch_entry_ptr(x_values, x_stride,
                                                          num_rows, group_id);
                const auto left_ptr =
                    batch::batch_entry_ptr(left_values, 1, num_rows, group_id);
                const auto right_ptr =
                    batch::batch_entry_ptr(right_values, 1, num_rhs, group_id);

                batch_scale_kernel(num_rows, x_stride, num_rhs, left_ptr,
                                   right_ptr, x_ptr, item_ct1);
            });
    });
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_BATCH_SCALE_KERNEL);


template <typename ValueType>
void add_scaled_identity(std::shared_ptr<const DpcppExecutor> exec,
                         const matrix::BatchDense<ValueType>* const a,
                         const matrix::BatchDense<ValueType>* const b,
                         matrix::BatchDense<ValueType>* const mtx)
{
    if (!mtx->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;

    const auto num_batches = mtx->get_num_batch_entries();
    const auto num_rows = static_cast<int>(mtx->get_size().at(0)[0]);
    const auto num_cols = static_cast<int>(mtx->get_size().at(0)[1]);
    const auto mtx_stride = mtx->get_stride().at(0);
    const auto mtx_values = mtx->get_values();
    const auto alpha_values = a->get_const_values();
    const auto a_stride = a->get_stride().at(0);
    const auto b_stride = b->get_stride().at(0);
    const auto beta_values = b->get_const_values();

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
                const auto mtx_b =
                    mtx_values + group_id * mtx_stride * num_rows;
                const auto alpha_b = alpha_values[group_id * a_stride];
                const auto beta_b = beta_values[group_id * b_stride];
                add_scaled_identity_kernel(num_rows, num_cols, mtx_stride,
                                           mtx_b, alpha_b, beta_b, item_ct1);
            });
    });
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
