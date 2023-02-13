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
inline void add_scaled_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<ValueType>& y)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_linear_id(); li < max_li;
         li += item_ct1.get_local_range().size()) {
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
                const matrix::BatchDense<ValueType>* alpha,
                const matrix::BatchDense<ValueType>* x,
                matrix::BatchDense<ValueType>* y)
{
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);

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
                const auto alpha_b = batch::batch_entry(alpha_ub, group_id);
                const auto x_b = batch::batch_entry(x_ub, group_id);
                const auto y_b = batch::batch_entry(y_ub, group_id);
                add_scaled_kernel(item_ct1, alpha_b, x_b, y_b);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scale(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::BatchDense<ValueType>* const alpha,
               const matrix::BatchDense<ValueType>* const x,
               const matrix::BatchDense<ValueType>* const beta,
               matrix::BatchDense<ValueType>* const y) GKO_NOT_IMPLEMENTED;

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
                 matrix::BatchDense<ValueType>* result) GKO_NOT_IMPLEMENTED;

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
    GKO_NOT_IMPLEMENTED;

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
               matrix::BatchDense<ValueType>* trans) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::BatchDense<ValueType>* orig,
                    matrix::BatchDense<ValueType>* trans) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::BatchDense<ValueType>* x,
          matrix::BatchDense<ValueType>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COPY_KERNEL);


template <typename ValueType>
void convergence_copy(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::BatchDense<ValueType>* x,
                      matrix::BatchDense<ValueType>* result,
                      const uint32& converged) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COPY_KERNEL);


template <typename ValueType>
void batch_scale(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::BatchDiagonal<ValueType>* left_diag,
                 const matrix::BatchDiagonal<ValueType>* right_diag,
                 matrix::BatchDense<ValueType>* x) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_BATCH_SCALE_KERNEL);


template <typename ValueType>
void add_scaled_identity(std::shared_ptr<const DpcppExecutor> exec,
                         const matrix::BatchDense<ValueType>* const a,
                         const matrix::BatchDense<ValueType>* const b,
                         matrix::BatchDense<ValueType>* const mtx)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
