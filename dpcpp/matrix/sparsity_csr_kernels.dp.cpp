/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/matrix/sparsity_csr_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup sparsity
 */
namespace sparsity_csr {


using classical_kernels = syn::value_list<int, 1>;


constexpr int spmv_block_size = 128;
constexpr int classical_overweight = 32;


namespace kernel {
template <size_type subgroup_size, typename ValueType, typename IndexType,
          typename Closure>
void device_classical_spmv(const size_type num_rows,
                           const ValueType* __restrict__ val,
                           const IndexType* __restrict__ col_idxs,
                           const IndexType* __restrict__ row_ptrs,
                           const ValueType* __restrict__ b,
                           const size_type b_stride, ValueType* __restrict__ c,
                           const size_type c_stride, Closure scale,
                           sycl::nd_item<3> item_ct1)
{
    auto subgroup_tile = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    const auto subrow = thread::get_subwarp_num_flat<subgroup_size>(item_ct1);
    const auto subid = subgroup_tile.thread_rank();
    const auto column_id = item_ct1.get_group(1);
    const auto value = val[0];
    auto row = thread::get_subwarp_id_flat<subgroup_size>(item_ct1);
    for (; row < num_rows; row += subrow) {
        const auto ind_end = row_ptrs[row + 1];
        ValueType temp_val = zero<ValueType>();
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subgroup_size) {
            temp_val += value * b[col_idxs[ind] * b_stride + column_id];
        }
        auto subgroup_result = ::gko::kernels::dpcpp::reduce(
            subgroup_tile, temp_val,
            [](const ValueType& a, const ValueType& b) { return a + b; });
        // TODO: check the barrier
        subgroup_tile.sync();
        if (subid == 0) {
            c[row * c_stride + column_id] =
                scale(subgroup_result, c[row * c_stride + column_id]);
        }
    }
}


template <size_type subgroup_size, typename ValueType, typename IndexType>
void abstract_classical_spmv(
    const size_type num_rows, const ValueType* __restrict__ val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ b,
    const size_type b_stride, ValueType* __restrict__ c,
    const size_type c_stride, sycl::nd_item<3> item_ct1)
{
    device_classical_spmv<subgroup_size>(
        num_rows, val, col_idxs, row_ptrs, b, b_stride, c, c_stride,
        [](const ValueType& x, const ValueType& y) { return x; }, item_ct1);
}

template <size_type subgroup_size, typename ValueType, typename IndexType>
void abstract_classical_spmv(dim3 grid, dim3 block,
                             size_type dynamic_shared_memory,
                             sycl::queue* queue, const size_type num_rows,
                             const ValueType* val, const IndexType* col_idxs,
                             const IndexType* row_ptrs, const ValueType* b,
                             const size_type b_stride, ValueType* c,
                             const size_type c_stride)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_classical_spmv<subgroup_size>(num_rows, val, col_idxs,
                                                       row_ptrs, b, b_stride, c,
                                                       c_stride, item_ct1);
            });
    });
}


template <size_type subgroup_size, typename ValueType, typename IndexType>
void abstract_classical_spmv(
    const size_type num_rows, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ b,
    const size_type b_stride, const ValueType* __restrict__ beta,
    ValueType* __restrict__ c, const size_type c_stride,
    sycl::nd_item<3> item_ct1)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    device_classical_spmv<subgroup_size>(
        num_rows, val, col_idxs, row_ptrs, b, b_stride, c, c_stride,
        [&alpha_val, &beta_val](const ValueType& x, const ValueType& y) {
            return alpha_val * x + beta_val * y;
        },
        item_ct1);
}

template <size_type subgroup_size, typename ValueType, typename IndexType>
void abstract_classical_spmv(dim3 grid, dim3 block,
                             size_type dynamic_shared_memory,
                             sycl::queue* queue, const size_type num_rows,
                             const ValueType* alpha, const ValueType* val,
                             const IndexType* col_idxs,
                             const IndexType* row_ptrs, const ValueType* b,
                             const size_type b_stride, const ValueType* beta,
                             ValueType* c, const size_type c_stride)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             abstract_classical_spmv<subgroup_size>(
                                 num_rows, alpha, val, col_idxs, row_ptrs, b,
                                 b_stride, beta, c, c_stride, item_ct1);
                         });
    });
}


}  // namespace kernel


namespace host_kernel {


template <int subgroup_size, typename ValueType, typename IndexType>
void classical_spmv(syn::value_list<int, subgroup_size>,
                    std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::SparsityCsr<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c,
                    const matrix::Dense<ValueType>* alpha = nullptr,
                    const matrix::Dense<ValueType>* beta = nullptr)
{
    constexpr int threads_per_cu = 7;
    const auto num_subgroup =
        exec->get_num_computing_units() * threads_per_cu * classical_overweight;
    const auto nsg_in_group = spmv_block_size / subgroup_size;
    const auto gridx =
        std::min(ceildiv(a->get_size()[0], spmv_block_size / subgroup_size),
                 int64(num_subgroup / nsg_in_group));
    const dim3 grid(gridx, b->get_size()[1]);
    const dim3 block(spmv_block_size);

    if (alpha == nullptr && beta == nullptr) {
        kernel::abstract_classical_spmv<subgroup_size>(
            grid, block, 0, exec->get_queue(), a->get_size()[0],
            a->get_const_value(), a->get_const_col_idxs(),
            a->get_const_row_ptrs(), b->get_const_values(), b->get_stride(),
            c->get_values(), c->get_stride());
    } else if (alpha != nullptr && beta != nullptr) {
        kernel::abstract_classical_spmv<subgroup_size>(
            grid, block, 0, exec->get_queue(), a->get_size()[0],
            alpha->get_const_values(), a->get_const_value(),
            a->get_const_col_idxs(), a->get_const_row_ptrs(),
            b->get_const_values(), b->get_stride(), beta->get_const_values(),
            c->get_values(), c->get_stride());
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_classical_spmv, classical_spmv);


}  // namespace host_kernel


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::SparsityCsr<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    host_kernel::select_classical_spmv(
        classical_kernels(), [](int compiled_info) { return true; },
        syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::SparsityCsr<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    host_kernel::select_classical_spmv(
        classical_kernels(), [](int compiled_info) { return true; },
        syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha, beta);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::SparsityCsr<ValueType, IndexType>* input,
                   matrix::Dense<ValueType>* output) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void count_num_diagonal_elements(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* matrix,
    size_type* num_diagonal_elements) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_COUNT_NUM_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void remove_diagonal_elements(
    std::shared_ptr<const DpcppExecutor> exec, const IndexType* row_ptrs,
    const IndexType* col_idxs,
    matrix::SparsityCsr<ValueType, IndexType>* matrix) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_REMOVE_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::SparsityCsr<ValueType, IndexType>* orig,
               matrix::SparsityCsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const DpcppExecutor> exec,
                          matrix::SparsityCsr<ValueType, IndexType>* to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* to_check,
    bool* is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_IS_SORTED_BY_COLUMN_INDEX);


}  // namespace sparsity_csr
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
