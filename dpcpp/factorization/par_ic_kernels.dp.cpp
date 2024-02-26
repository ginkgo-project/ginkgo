// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ic_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The parallel ic factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ic_factorization {


constexpr int default_block_size = 256;


namespace kernel {


template <typename ValueType, typename IndexType>
void ic_init(const IndexType* __restrict__ l_row_ptrs,
             ValueType* __restrict__ l_vals, size_type num_rows,
             sycl::nd_item<3> item_ct1)
{
    auto row = thread::get_thread_id_flat(item_ct1);
    if (row >= num_rows) {
        return;
    }
    auto l_nz = l_row_ptrs[row + 1] - 1;
    auto diag = std::sqrt(l_vals[l_nz]);
    if (is_finite(diag)) {
        l_vals[l_nz] = diag;
    } else {
        l_vals[l_nz] = one<ValueType>();
    }
}

template <typename ValueType, typename IndexType>
void ic_init(dim3 grid, dim3 block, size_type dynamic_shared_memory,
             sycl::queue* queue, const IndexType* l_row_ptrs, ValueType* l_vals,
             size_type num_rows)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            ic_init(l_row_ptrs, l_vals, num_rows, item_ct1);
                        });
}


template <typename ValueType, typename IndexType>
void ic_sweep(const IndexType* __restrict__ a_row_idxs,
              const IndexType* __restrict__ a_col_idxs,
              const ValueType* __restrict__ a_vals,
              const IndexType* __restrict__ l_row_ptrs,
              const IndexType* __restrict__ l_col_idxs,
              ValueType* __restrict__ l_vals, IndexType l_nnz,
              sycl::nd_item<3> item_ct1)
{
    const auto l_nz = thread::get_thread_id_flat<IndexType>(item_ct1);
    if (l_nz >= l_nnz) {
        return;
    }
    const auto row = a_row_idxs[l_nz];
    const auto col = l_col_idxs[l_nz];
    const auto a_val = a_vals[l_nz];
    auto l_row_begin = l_row_ptrs[row];
    const auto l_row_end = l_row_ptrs[row + 1];
    auto lh_col_begin = l_row_ptrs[col];
    const auto lh_col_end = l_row_ptrs[col + 1];
    ValueType sum{};
    auto last_entry = col;
    while (l_row_begin < l_row_end && lh_col_begin < lh_col_end) {
        auto l_col = l_col_idxs[l_row_begin];
        auto lh_row = l_col_idxs[lh_col_begin];
        if (l_col == lh_row && l_col < last_entry) {
            sum += l_vals[l_row_begin] * conj(l_vals[lh_col_begin]);
        }
        l_row_begin += l_col <= lh_row;
        lh_col_begin += l_col >= lh_row;
    }
    auto to_write = row == col
                        ? std::sqrt(a_val - sum)
                        : (a_val - sum) / l_vals[l_row_ptrs[col + 1] - 1];
    if (is_finite(to_write)) {
        l_vals[l_nz] = to_write;
    }
}

template <typename ValueType, typename IndexType>
void ic_sweep(dim3 grid, dim3 block, size_type dynamic_shared_memory,
              sycl::queue* queue, const IndexType* a_row_idxs,
              const IndexType* a_col_idxs, const ValueType* a_vals,
              const IndexType* l_row_ptrs, const IndexType* l_col_idxs,
              ValueType* l_vals, IndexType l_nnz)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            ic_sweep(a_row_idxs, a_col_idxs, a_vals, l_row_ptrs,
                                     l_col_idxs, l_vals, l_nnz, item_ct1);
                        });
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void init_factor(std::shared_ptr<const DefaultExecutor> exec,
                 matrix::Csr<ValueType, IndexType>* l)
{
    auto num_rows = l->get_size()[0];
    auto num_blocks = ceildiv(num_rows, default_block_size);
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_vals = l->get_values();
    kernel::ic_init(num_blocks, default_block_size, 0, exec->get_queue(),
                    l_row_ptrs, l_vals, num_rows);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_INIT_FACTOR_KERNEL);


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    size_type iterations,
                    const matrix::Coo<ValueType, IndexType>* a_lower,
                    matrix::Csr<ValueType, IndexType>* l)
{
    auto nnz = l->get_num_stored_elements();
    auto num_blocks = ceildiv(nnz, default_block_size);
    for (size_type i = 0; i < iterations; ++i) {
        kernel::ic_sweep(num_blocks, default_block_size, 0, exec->get_queue(),
                         a_lower->get_const_row_idxs(),
                         a_lower->get_const_col_idxs(),
                         a_lower->get_const_values(), l->get_const_row_ptrs(),
                         l->get_const_col_idxs(), l->get_values(),
                         static_cast<IndexType>(l->get_num_stored_elements()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_COMPUTE_FACTOR_KERNEL);


}  // namespace par_ic_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
