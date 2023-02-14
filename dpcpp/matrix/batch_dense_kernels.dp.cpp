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
 * @brief The BatchDense matrix format namespace.
 *
 * @ingroup batch_dense
 */
namespace batch_dense {

template <typename ValueType>
inline void single_matvec_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int row = sg_id; row < a.num_rows; row += num_sg) {
        ValueType temp = zero<ValueType>();
        for (int j = sg.get_local_id(); j < a.num_rhs; j += sg_size) {
            const ValueType val = a.values[row * a.stride + j];
            temp += val * b[j];
        }
        temp = sycl::reduce_over_group(sg, temp, sycl::plus<>());
        if (sg.get_local_id() == 0) {
            c[row] = temp;
        }
    }
}

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
                single_matvec_kernel(item_ct1, a_b, b_b.values, c_b.values);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);

template <typename ValueType>
inline void single_advanced_matvec_kernel(
    sycl::nd_item<3>& item_ct1, const ValueType alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ c)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int row = sg_id; row < a.num_rows; row += num_sg) {
        ValueType temp = zero<ValueType>();
        for (int j = sg.get_local_id(); j < a.num_rhs; j += sg_size) {
            const ValueType val = a.values[row * a.stride + j];
            temp += alpha * val * b[j];
        }
        temp = sycl::reduce_over_group(sg, temp, sycl::plus<>());
        if (sg.get_local_id() == 0) {
            c[row] = temp + beta * c[row];
        }
    }
}

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
                single_advanced_matvec_kernel(item_ct1, alpha_b.values[0], a_b,
                                              b_b.values, beta_b.values[0],
                                              c_b.values);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL);

template <typename ValueType>
inline void single_scale_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<ValueType>& x)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_linear_id(); li < max_li;
         li += item_ct1.get_local_range().size()) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            x.values[row * x.stride + col] =
                alpha.values[0] * x.values[row * x.stride + col];
        } else {
            x.values[row * x.stride + col] =
                alpha.values[col] * x.values[row * x.stride + col];
        }
    }
}

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
                single_scale_kernel(item_ct1, alpha_b, x_b);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);

template <typename ValueType>
inline void single_add_scaled_kernel(sycl::nd_item<3>& item_ct1,
                                     const int num_rows, const ValueType alpha,
                                     const ValueType* const x,
                                     ValueType* const y)
{
    for (int li = item_ct1.get_local_id(2); li < num_rows;
         li += item_ct1.get_local_range(2)) {
        y[li] += alpha * x[li];
    }
}

template <typename ValueType>
inline void add_scaled_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<ValueType>& y)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_id(2); li < max_li;
         li += item_ct1.get_local_range(2)) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            y.values[row * y.stride + col] +=
                alpha.values[0] * x.values[row * x.stride + col];
        } else {
            y.values[row * y.stride + col] +=
                alpha.values[col] * x.values[row * x.stride + col];
        }
    }
}

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
                add_scaled_kernel(item_ct1, alpha_b, x_b, y_b);
            });
    });
    //}
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
inline void add_scaled_advanced_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<const ValueType>& beta,
    const gko::batch_dense::BatchEntry<ValueType>& y)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_id(2); li < max_li;
         li += item_ct1.get_local_range(2)) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            y.values[row * y.stride + col] *= beta.values[0];
            y.values[row * y.stride + col] +=
                alpha.values[0] * x.values[row * x.stride + col];
        } else {
            y.values[row * y.stride + col] *= beta.values[col];
            y.values[row * y.stride + col] +=
                alpha.values[col] * x.values[row * x.stride + col];
        }
    }
}

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
                add_scaled_advanced_kernel(item_ct1, alpha_b, x_b, beta_b, y_b);
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
inline void compute_dot_product_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<const ValueType>& y,
    const gko::batch_dense::BatchEntry<ValueType>& result)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int rhs_index = sg_id; rhs_index < x.num_rhs; rhs_index += num_sg) {
        ValueType val = zero<ValueType>();

        for (int r = sg.get_local_id(); r < x.num_rows; r += sg_size) {
            val += conj(x.values[r * x.stride + rhs_index]) *
                   y.values[r * y.stride + rhs_index];
        }

        val = sycl::reduce_over_group(sg, val, sycl::plus<>());

        if (sg.get_local_id() == 0) {
            result.values[rhs_index] = val;
        }
    }
}

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
                compute_dot_product_kernel(item_ct1, x_b, y_b, res_b);
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
inline void compute_norm2_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<remove_complex<ValueType>>& result)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    using real_type = typename gko::remove_complex<ValueType>;
    for (int rhs_index = sg_id; rhs_index < x.num_rhs; rhs_index += num_sg) {
        real_type val = zero<real_type>();

        for (int r = sg.get_local_id(); r < x.num_rows; r += sg_size)
            val += squared_norm(x.values[r * x.stride + rhs_index]);

        val = sycl::reduce_over_group(sg, val, sycl::plus<>());

        if (sg.get_local_id() == 0) result.values[rhs_index] = sqrt(val);
    }
}

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
                compute_norm2_kernel(item_ct1, x_b, res_b);
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

template <typename Op, typename ValueType>
inline void transpose_kernel(sycl::nd_item<3>& item_ct1, const int src_nrows,
                             const int src_ncols, const size_type src_stride,
                             const ValueType* const src,
                             const size_type dest_stride, ValueType* const dest,
                             Op op)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int i_row = sg_id; i_row < src_nrows; i_row += num_sg) {
        for (int j = sg.get_local_id(); j < src_ncols; j += sg_size) {
            dest[j * dest_stride + i_row] = op(src[i_row * src_stride + j]);
        }
    }
}

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
                transpose_kernel(item_ct1, nrows, ncols, orig_stride, orig_b,
                                 trans_stride, trans_b,
                                 [](ValueType x) { return x; });
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
                transpose_kernel(item_ct1, nrows, ncols, orig_stride, orig_b,
                                 trans_stride, trans_b,
                                 [](ValueType x) { return conj(x); });
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL);

template <typename ValueType>
inline void copy_kernel(sycl::nd_item<3>& item_ct1,
                        const gko::batch_dense::BatchEntry<const ValueType>& in,
                        const gko::batch_dense::BatchEntry<ValueType>& out)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < in.num_rows * in.num_rhs;
         iz += item_ct1.get_local_range().size()) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}

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
                copy_kernel(item_ct1, x_b, result_b);
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
inline void batch_scale_kernel(sycl::nd_item<3>& item_ct1, const int num_rows,
                               const size_type stride, const int num_rhs,
                               const ValueType* const left_scale_vec,
                               const ValueType* const right_scale_vec,
                               ValueType* const a)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < num_rows * num_rhs;
         iz += item_ct1.get_local_range().size()) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        a[row * stride + col] *= left_scale_vec[row] * right_scale_vec[col];
    }
}


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

                batch_scale_kernel(item_ct1, num_rows, x_stride, num_rhs,
                                   left_ptr, right_ptr, x_ptr);
            });
    });
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_BATCH_SCALE_KERNEL);


template <typename ValueType>
inline void add_scaled_identity_kernel(sycl::nd_item<3>& item_ct1,
                                       const int nrows, const int ncols,
                                       const size_type stride,
                                       ValueType* const __restrict__ values,
                                       const ValueType alpha,
                                       const ValueType beta)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int row = sg_id; row < nrows; row += num_sg) {
        for (int col = sg.get_local_id(); col < ncols; col += sg_size) {
            values[row * stride + col] *= beta;
            if (col == row) {
                values[row * stride + row] += alpha;
            }
        }
    }
}

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
                add_scaled_identity_kernel(item_ct1, num_rows, num_cols,
                                           mtx_stride, mtx_b, alpha_b, beta_b);
            });
    });
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
