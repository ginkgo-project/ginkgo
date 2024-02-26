// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilut_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"
#include "dpcpp/components/merging.dp.hpp"
#include "dpcpp/components/prefix_sum.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/searching.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


constexpr int default_block_size = 256;


// subwarp sizes for all warp-parallel kernels (filter, add_candidates)
using compiled_kernels = syn::value_list<int, 1, 16, 32>;

namespace kernel {


template <int subgroup_size, typename ValueType, typename IndexType>
void sweep(const IndexType* __restrict__ a_row_ptrs,
           const IndexType* __restrict__ a_col_idxs,
           const ValueType* __restrict__ a_vals,
           const IndexType* __restrict__ l_row_ptrs,
           const IndexType* __restrict__ l_row_idxs,
           const IndexType* __restrict__ l_col_idxs,
           ValueType* __restrict__ l_vals, IndexType l_nnz,
           const IndexType* __restrict__ u_row_idxs,
           const IndexType* __restrict__ u_col_idxs,
           ValueType* __restrict__ u_vals,
           const IndexType* __restrict__ ut_col_ptrs,
           const IndexType* __restrict__ ut_row_idxs,
           ValueType* __restrict__ ut_vals, IndexType u_nnz,
           sycl::nd_item<3> item_ct1)
{
    auto tidx = thread::get_subwarp_id_flat<subgroup_size, IndexType>(item_ct1);
    if (tidx >= l_nnz + u_nnz) {
        return;
    }
    // split the subwarps into two halves for lower and upper triangle
    auto l_nz = tidx;
    auto u_nz = l_nz - l_nnz;
    auto lower = u_nz < 0;
    auto row = lower ? l_row_idxs[l_nz] : u_row_idxs[u_nz];
    auto col = lower ? l_col_idxs[l_nz] : u_col_idxs[u_nz];
    if (lower && row == col) {
        // don't update the diagonal twice
        return;
    }
    auto subwarp = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    // find entry of A at (row, col)
    auto a_row_begin = a_row_ptrs[row];
    auto a_row_end = a_row_ptrs[row + 1];
    auto a_row_size = a_row_end - a_row_begin;
    auto a_idx =
        group_wide_search(a_row_begin, a_row_size, subwarp,
                          [&](IndexType i) { return a_col_idxs[i] >= col; });
    bool has_a = a_idx < a_row_end && a_col_idxs[a_idx] == col;
    auto a_val = has_a ? a_vals[a_idx] : zero<ValueType>();
    auto l_row_begin = l_row_ptrs[row];
    auto l_row_size = l_row_ptrs[row + 1] - l_row_begin;
    auto ut_col_begin = ut_col_ptrs[col];
    auto ut_col_size = ut_col_ptrs[col + 1] - ut_col_begin;
    ValueType sum{};
    IndexType ut_nz{};
    auto last_entry = min(row, col);
    group_merge<subgroup_size>(
        l_col_idxs + l_row_begin, l_row_size, ut_row_idxs + ut_col_begin,
        ut_col_size, subwarp,
        [&](IndexType l_idx, IndexType l_col, IndexType ut_idx,
            IndexType ut_row, IndexType, bool) {
            // we don't need to use the `bool valid` because last_entry is
            // already a smaller sentinel value than the one used in group_merge
            if (l_col == ut_row && l_col < last_entry) {
                sum += l_vals[l_idx + l_row_begin] *
                       ut_vals[ut_idx + ut_col_begin];
            }
            // remember the transposed element
            auto found_transp = subwarp.ballot(ut_row == row);
            if (found_transp) {
                ut_nz =
                    subwarp.shfl(ut_idx + ut_col_begin, ffs(found_transp) - 1);
            }
            return true;
        });
    // accumulate result from all threads
    sum = ::gko::kernels::dpcpp::reduce(
        subwarp, sum, [](ValueType a, ValueType b) { return a + b; });

    if (subwarp.thread_rank() == 0) {
        if (lower) {
            auto to_write = (a_val - sum) / ut_vals[ut_col_ptrs[col + 1] - 1];
            if (is_finite(to_write)) {
                l_vals[l_nz] = to_write;
            }
        } else {
            auto to_write = a_val - sum;
            if (is_finite(to_write)) {
                u_vals[u_nz] = to_write;
                ut_vals[ut_nz] = to_write;
            }
        }
    }
}

template <int subgroup_size, typename ValueType, typename IndexType>
void sweep(dim3 grid, dim3 block, size_type dynamic_shared_memory,
           sycl::queue* queue, const IndexType* a_row_ptrs,
           const IndexType* a_col_idxs, const ValueType* a_vals,
           const IndexType* l_row_ptrs, const IndexType* l_row_idxs,
           const IndexType* l_col_idxs, ValueType* l_vals, IndexType l_nnz,
           const IndexType* u_row_idxs, const IndexType* u_col_idxs,
           ValueType* u_vals, const IndexType* ut_col_ptrs,
           const IndexType* ut_row_idxs, ValueType* ut_vals, IndexType u_nnz)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                sweep<subgroup_size>(
                                    a_row_ptrs, a_col_idxs, a_vals, l_row_ptrs,
                                    l_row_idxs, l_col_idxs, l_vals, l_nnz,
                                    u_row_idxs, u_col_idxs, u_vals, ut_col_ptrs,
                                    ut_row_idxs, ut_vals, u_nnz, item_ct1);
                            });
}


}  // namespace kernel

namespace {


template <int subgroup_size, typename ValueType, typename IndexType>
void compute_l_u_factors(syn::value_list<int, subgroup_size>,
                         std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Csr<ValueType, IndexType>* a,
                         matrix::Csr<ValueType, IndexType>* l,
                         const matrix::Coo<ValueType, IndexType>* l_coo,
                         matrix::Csr<ValueType, IndexType>* u,
                         const matrix::Coo<ValueType, IndexType>* u_coo,
                         matrix::Csr<ValueType, IndexType>* u_csc)
{
    auto total_nnz = static_cast<IndexType>(l->get_num_stored_elements() +
                                            u->get_num_stored_elements());
    auto block_size = default_block_size / subgroup_size;
    auto num_blocks = ceildiv(total_nnz, block_size);
    kernel::sweep<subgroup_size>(
        num_blocks, default_block_size, 0, exec->get_queue(),
        a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(),
        l->get_const_row_ptrs(), l_coo->get_const_row_idxs(),
        l->get_const_col_idxs(), l->get_values(),
        static_cast<IndexType>(l->get_num_stored_elements()),
        u_coo->get_const_row_idxs(), u_coo->get_const_col_idxs(),
        u->get_values(), u_csc->get_const_row_ptrs(),
        u_csc->get_const_col_idxs(), u_csc->get_values(),
        static_cast<IndexType>(u->get_num_stored_elements()));
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_l_u_factors,
                                    compute_l_u_factors);


}  // namespace


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Csr<ValueType, IndexType>* a,
                         matrix::Csr<ValueType, IndexType>* l,
                         const matrix::Coo<ValueType, IndexType>* l_coo,
                         matrix::Csr<ValueType, IndexType>* u,
                         const matrix::Coo<ValueType, IndexType>* u_coo,
                         matrix::Csr<ValueType, IndexType>* u_csc)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz =
        l->get_num_stored_elements() + u->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_compute_l_u_factors(
        compiled_kernels(),
        [&](int compiled_subgroup_size) {
            return total_nnz_per_row <= compiled_subgroup_size ||
                   compiled_subgroup_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, a, l, l_coo, u, u_coo,
        u_csc);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_COMPUTE_LU_FACTORS_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
