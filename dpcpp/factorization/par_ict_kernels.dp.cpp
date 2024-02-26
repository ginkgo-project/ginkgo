// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ict_kernels.hpp"


#include <limits>


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
 * @brief The parallel ICT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ict_factorization {


constexpr int default_block_size = 256;


// subwarp sizes for all warp-parallel kernels (filter, add_candidates)
using compiled_kernels = syn::value_list<int, 1, 16, 32>;


namespace kernel {


template <int subgroup_size, typename IndexType>
void ict_tri_spgeam_nnz(const IndexType* __restrict__ llh_row_ptrs,
                        const IndexType* __restrict__ llh_col_idxs,
                        const IndexType* __restrict__ a_row_ptrs,
                        const IndexType* __restrict__ a_col_idxs,
                        IndexType* __restrict__ l_new_row_ptrs,
                        IndexType num_rows, sycl::nd_item<3> item_ct1)
{
    auto subwarp = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    auto row = thread::get_subwarp_id_flat<subgroup_size, IndexType>(item_ct1);
    if (row >= num_rows) {
        return;
    }

    auto llh_begin = llh_row_ptrs[row];
    auto llh_size = llh_row_ptrs[row + 1] - llh_begin;
    auto a_begin = a_row_ptrs[row];
    auto a_size = a_row_ptrs[row + 1] - a_begin;
    IndexType count{};
    group_merge<subgroup_size>(
        a_col_idxs + a_begin, a_size, llh_col_idxs + llh_begin, llh_size,
        subwarp,
        [&](IndexType a_nz, IndexType a_col, IndexType llh_nz,
            IndexType llh_col, IndexType out_nz, bool valid) {
            auto col = min(a_col, llh_col);
            // count the number of unique elements being merged
            count +=
                popcnt(subwarp.ballot(col <= row && a_col != llh_col && valid));
            return true;
        });
    if (subwarp.thread_rank() == 0) {
        l_new_row_ptrs[row] = count;
    }
}

template <int subgroup_size, typename IndexType>
void ict_tri_spgeam_nnz(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                        sycl::queue* queue, const IndexType* llh_row_ptrs,
                        const IndexType* llh_col_idxs,
                        const IndexType* a_row_ptrs,
                        const IndexType* a_col_idxs, IndexType* l_new_row_ptrs,
                        IndexType num_rows)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                ict_tri_spgeam_nnz<subgroup_size>(
                                    llh_row_ptrs, llh_col_idxs, a_row_ptrs,
                                    a_col_idxs, l_new_row_ptrs, num_rows,
                                    item_ct1);
                            });
}


template <int subgroup_size, typename ValueType, typename IndexType>
void ict_tri_spgeam_init(const IndexType* __restrict__ llh_row_ptrs,
                         const IndexType* __restrict__ llh_col_idxs,
                         const ValueType* __restrict__ llh_vals,
                         const IndexType* __restrict__ a_row_ptrs,
                         const IndexType* __restrict__ a_col_idxs,
                         const ValueType* __restrict__ a_vals,
                         const IndexType* __restrict__ l_row_ptrs,
                         const IndexType* __restrict__ l_col_idxs,
                         const ValueType* __restrict__ l_vals,
                         const IndexType* __restrict__ l_new_row_ptrs,
                         IndexType* __restrict__ l_new_col_idxs,
                         ValueType* __restrict__ l_new_vals, IndexType num_rows,
                         sycl::nd_item<3> item_ct1)
{
    auto subwarp = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    auto row = thread::get_subwarp_id_flat<subgroup_size, IndexType>(item_ct1);
    if (row >= num_rows) {
        return;
    }

    auto lane = static_cast<IndexType>(subwarp.thread_rank());
    auto lanemask_eq = config::lane_mask_type{1} << lane;
    auto lanemask_lt = lanemask_eq - 1;

    // merge lower triangle of A, L*L^T (and L)
    auto l_begin = l_row_ptrs[row];
    auto l_end = l_row_ptrs[row + 1];

    auto llh_begin = llh_row_ptrs[row];
    auto llh_end = llh_row_ptrs[row + 1];
    auto llh_size = llh_end - llh_begin;

    auto a_begin = a_row_ptrs[row];
    auto a_end = a_row_ptrs[row + 1];
    auto a_size = a_end - a_begin;

    IndexType out_begin{};
    auto out_size = llh_size + a_size;

    IndexType l_new_begin = l_new_row_ptrs[row];

    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    // load column indices and values for the first merge step
    auto a_col = checked_load(a_col_idxs, a_begin + lane, a_end, sentinel);
    auto a_val = checked_load(a_vals, a_begin + lane, a_end, zero<ValueType>());
    auto llh_col =
        checked_load(llh_col_idxs, llh_begin + lane, llh_end, sentinel);
    auto llh_val =
        checked_load(llh_vals, llh_begin + lane, llh_end, zero<ValueType>());
    auto l_col = checked_load(l_col_idxs, l_begin + lane, l_end, sentinel);
    auto l_val = checked_load(l_vals, l_begin + lane, l_end, zero<ValueType>());
    bool skip_first{};
    while (out_begin < out_size) {
        // merge subwarp.size() elements from A and L*L^H
        auto merge_result =
            group_merge_step<subgroup_size>(a_col, llh_col, subwarp);
        auto a_cur_col = merge_result.a_val;
        auto llh_cur_col = merge_result.b_val;
        auto a_cur_val = subwarp.shfl(a_val, merge_result.a_idx);
        auto llh_cur_val = subwarp.shfl(llh_val, merge_result.b_idx);
        auto valid = out_begin + lane < out_size;
        // check if the previous thread has matching columns
        auto equal_mask = subwarp.ballot(a_cur_col == llh_cur_col && valid);
        auto prev_equal_mask = equal_mask << 1 | skip_first;
        skip_first = bool(equal_mask >> (subgroup_size - 1));
        auto prev_equal = bool(prev_equal_mask & lanemask_eq);

        auto r_col = min(a_cur_col, llh_cur_col);
        // find matching entry of L
        // S(L) is a subset of S(A - L * L^H) since L has a diagonal
        auto l_source = synchronous_fixed_binary_search<subgroup_size>(
            [&](int i) { return subwarp.shfl(l_col, i) >= r_col; });
        auto l_cur_col = subwarp.shfl(l_col, l_source);
        auto l_cur_val = subwarp.shfl(l_val, l_source);

        // determine actual values of A and L*L^H at r_col
        if (r_col != a_cur_col) {
            a_cur_val = zero<ValueType>();
        }
        if (r_col != llh_cur_col) {
            llh_cur_val = zero<ValueType>();
        }
        auto r_val = a_cur_val - llh_cur_val;

        // early return when reaching the upper diagonal
        if (subwarp.all(r_col > row)) {
            break;
        }

        // determine which threads will write output to L
        auto use_l = l_cur_col == r_col;
        auto do_write = !prev_equal && valid && r_col <= row;
        auto l_new_advance_mask = subwarp.ballot(do_write);
        // store values
        if (do_write) {
            auto diag = l_vals[l_row_ptrs[r_col + 1] - 1];
            auto out_val = use_l ? l_cur_val : r_val / diag;
            auto ofs = popcnt(l_new_advance_mask & lanemask_lt);
            l_new_col_idxs[l_new_begin + ofs] = r_col;
            l_new_vals[l_new_begin + ofs] = out_val;
        }

        // advance *_begin offsets
        auto a_advance = merge_result.a_advance;
        auto llh_advance = merge_result.b_advance;
        auto l_advance = popcnt(subwarp.ballot(do_write && use_l));
        auto l_new_advance = popcnt(l_new_advance_mask);
        a_begin += a_advance;
        llh_begin += llh_advance;
        l_begin += l_advance;
        l_new_begin += l_new_advance;
        out_begin += subgroup_size;

        // shuffle the unmerged elements to the front
        a_col = subwarp.shfl_down(a_col, a_advance);
        a_val = subwarp.shfl_down(a_val, a_advance);
        llh_col = subwarp.shfl_down(llh_col, llh_advance);
        llh_val = subwarp.shfl_down(llh_val, llh_advance);
        l_col = subwarp.shfl_down(l_col, l_advance);
        l_val = subwarp.shfl_down(l_val, l_advance);
        /*
         * To optimize memory access, we load the new elements for `a` and `llh`
         * with a single load instruction:
         * the lower part of the group loads new elements for `a`
         * the upper part of the group loads new elements for `llh`
         * `load_lane` is the part-local lane idx
         * The elements for `a` have to be shuffled up afterwards.
         */
        auto load_a = lane < a_advance;
        auto load_lane = load_a ? lane : lane - a_advance;
        auto load_source_col = load_a ? a_col_idxs : llh_col_idxs;
        auto load_source_val = load_a ? a_vals : llh_vals;
        auto load_begin =
            load_a ? a_begin + llh_advance : llh_begin + a_advance;
        auto load_end = load_a ? a_end : llh_end;

        auto load_idx = load_begin + load_lane;
        auto loaded_col =
            checked_load(load_source_col, load_idx, load_end, sentinel);
        auto loaded_val = checked_load(load_source_val, load_idx, load_end,
                                       zero<ValueType>());
        // shuffle the `a` values to the end of the warp
        auto lower_loaded_col = subwarp.shfl_up(loaded_col, llh_advance);
        auto lower_loaded_val = subwarp.shfl_up(loaded_val, llh_advance);
        if (lane >= llh_advance) {
            a_col = lower_loaded_col;
            a_val = lower_loaded_val;
        }
        if (lane >= a_advance) {
            llh_col = loaded_col;
            llh_val = loaded_val;
        }
        // load the new values for l
        if (lane >= subgroup_size - l_advance) {
            auto l_idx = l_begin + lane;
            l_col = checked_load(l_col_idxs, l_idx, l_end, sentinel);
            l_val = checked_load(l_vals, l_idx, l_end, zero<ValueType>());
        }
    }
}

template <int subgroup_size, typename ValueType, typename IndexType>
void ict_tri_spgeam_init(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                         sycl::queue* queue, const IndexType* llh_row_ptrs,
                         const IndexType* llh_col_idxs,
                         const ValueType* llh_vals, const IndexType* a_row_ptrs,
                         const IndexType* a_col_idxs, const ValueType* a_vals,
                         const IndexType* l_row_ptrs,
                         const IndexType* l_col_idxs, const ValueType* l_vals,
                         const IndexType* l_new_row_ptrs,
                         IndexType* l_new_col_idxs, ValueType* l_new_vals,
                         IndexType num_rows)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                ict_tri_spgeam_init<subgroup_size>(
                                    llh_row_ptrs, llh_col_idxs, llh_vals,
                                    a_row_ptrs, a_col_idxs, a_vals, l_row_ptrs,
                                    l_col_idxs, l_vals, l_new_row_ptrs,
                                    l_new_col_idxs, l_new_vals, num_rows,
                                    item_ct1);
                            });
}


}  // namespace kernel
namespace kernel {


template <int subgroup_size, typename ValueType, typename IndexType>
void ict_sweep(const IndexType* __restrict__ a_row_ptrs,
               const IndexType* __restrict__ a_col_idxs,
               const ValueType* __restrict__ a_vals,
               const IndexType* __restrict__ l_row_ptrs,
               const IndexType* __restrict__ l_row_idxs,
               const IndexType* __restrict__ l_col_idxs,
               ValueType* __restrict__ l_vals, IndexType l_nnz,
               sycl::nd_item<3> item_ct1)
{
    auto l_nz = thread::get_subwarp_id_flat<subgroup_size, IndexType>(item_ct1);
    if (l_nz >= l_nnz) {
        return;
    }
    auto row = l_row_idxs[l_nz];
    auto col = l_col_idxs[l_nz];
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
    auto lh_col_begin = l_row_ptrs[col];
    auto lh_col_size = l_row_ptrs[col + 1] - lh_col_begin;
    ValueType sum{};
    IndexType lh_nz{};
    auto last_entry = col;
    group_merge<subgroup_size>(
        l_col_idxs + l_row_begin, l_row_size, l_col_idxs + lh_col_begin,
        lh_col_size, subwarp,
        [&](IndexType l_idx, IndexType l_col, IndexType lh_idx,
            IndexType lh_row, IndexType, bool) {
            // we don't need to use the `bool valid` because last_entry is
            // already a smaller sentinel value than the one used in group_merge
            if (l_col == lh_row && l_col < last_entry) {
                sum += l_vals[l_idx + l_row_begin] *
                       conj(l_vals[lh_idx + lh_col_begin]);
            }
            // remember the transposed element
            auto found_transp = subwarp.ballot(lh_row == row);
            if (found_transp) {
                lh_nz =
                    subwarp.shfl(lh_idx + lh_col_begin, ffs(found_transp) - 1);
            }
            return true;
        });
    // accumulate result from all threads
    sum = ::gko::kernels::dpcpp::reduce(
        subwarp, sum, [](ValueType a, ValueType b) { return a + b; });

    if (subwarp.thread_rank() == 0) {
        auto to_write = row == col
                            ? std::sqrt(a_val - sum)
                            : (a_val - sum) / l_vals[l_row_ptrs[col + 1] - 1];
        if (is_finite(to_write)) {
            l_vals[l_nz] = to_write;
        }
    }
}

template <int subgroup_size, typename ValueType, typename IndexType>
void ict_sweep(dim3 grid, dim3 block, size_type dynamic_shared_memory,
               sycl::queue* queue, const IndexType* a_row_ptrs,
               const IndexType* a_col_idxs, const ValueType* a_vals,
               const IndexType* l_row_ptrs, const IndexType* l_row_idxs,
               const IndexType* l_col_idxs, ValueType* l_vals, IndexType l_nnz)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                ict_sweep<subgroup_size>(
                                    a_row_ptrs, a_col_idxs, a_vals, l_row_ptrs,
                                    l_row_idxs, l_col_idxs, l_vals, l_nnz,
                                    item_ct1);
                            });
}


}  // namespace kernel


namespace {


template <int subgroup_size, typename ValueType, typename IndexType>
void add_candidates(syn::value_list<int, subgroup_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* llh,
                    const matrix::Csr<ValueType, IndexType>* a,
                    const matrix::Csr<ValueType, IndexType>* l,
                    matrix::Csr<ValueType, IndexType>* l_new)
{
    auto num_rows = static_cast<IndexType>(llh->get_size()[0]);
    auto subwarps_per_block = default_block_size / subgroup_size;
    auto num_blocks = ceildiv(num_rows, subwarps_per_block);
    matrix::CsrBuilder<ValueType, IndexType> l_new_builder(l_new);
    auto llh_row_ptrs = llh->get_const_row_ptrs();
    auto llh_col_idxs = llh->get_const_col_idxs();
    auto llh_vals = llh->get_const_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_const_values();
    auto l_new_row_ptrs = l_new->get_row_ptrs();
    // count non-zeros per row
    kernel::ict_tri_spgeam_nnz<subgroup_size>(
        num_blocks, default_block_size, 0, exec->get_queue(), llh_row_ptrs,
        llh_col_idxs, a_row_ptrs, a_col_idxs, l_new_row_ptrs, num_rows);

    // build row ptrs
    components::prefix_sum_nonnegative(exec, l_new_row_ptrs, num_rows + 1);

    // resize output arrays
    auto l_new_nnz = exec->copy_val_to_host(l_new_row_ptrs + num_rows);
    l_new_builder.get_col_idx_array().resize_and_reset(l_new_nnz);
    l_new_builder.get_value_array().resize_and_reset(l_new_nnz);

    auto l_new_col_idxs = l_new->get_col_idxs();
    auto l_new_vals = l_new->get_values();

    // fill columns and values
    kernel::ict_tri_spgeam_init<subgroup_size>(
        num_blocks, default_block_size, 0, exec->get_queue(), llh_row_ptrs,
        llh_col_idxs, llh_vals, a_row_ptrs, a_col_idxs, a_vals, l_row_ptrs,
        l_col_idxs, l_vals, l_new_row_ptrs, l_new_col_idxs, l_new_vals,
        num_rows);
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_add_candidates, add_candidates);


template <int subgroup_size, typename ValueType, typename IndexType>
void compute_factor(syn::value_list<int, subgroup_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* a,
                    matrix::Csr<ValueType, IndexType>* l,
                    const matrix::Coo<ValueType, IndexType>* l_coo)
{
    auto total_nnz = static_cast<IndexType>(l->get_num_stored_elements());
    auto block_size = default_block_size / subgroup_size;
    auto num_blocks = ceildiv(total_nnz, block_size);
    kernel::ict_sweep<subgroup_size>(
        num_blocks, default_block_size, 0, exec->get_queue(),
        a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(),
        l->get_const_row_ptrs(), l_coo->get_const_row_idxs(),
        l->get_const_col_idxs(), l->get_values(),
        static_cast<IndexType>(l->get_num_stored_elements()));
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_factor, compute_factor);


}  // namespace


template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* llh,
                    const matrix::Csr<ValueType, IndexType>* a,
                    const matrix::Csr<ValueType, IndexType>* l,
                    matrix::Csr<ValueType, IndexType>* l_new)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz =
        llh->get_num_stored_elements() + a->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_add_candidates(
        compiled_kernels(),
        [&](int compiled_subgroup_size) {
            return total_nnz_per_row <= compiled_subgroup_size ||
                   compiled_subgroup_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, llh, a, l, l_new);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ICT_ADD_CANDIDATES_KERNEL);


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* a,
                    matrix::Csr<ValueType, IndexType>* l,
                    const matrix::Coo<ValueType, IndexType>* l_coo)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz = 2 * l->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_compute_factor(
        compiled_kernels(),
        [&](int compiled_subgroup_size) {
            return total_nnz_per_row <= compiled_subgroup_size ||
                   compiled_subgroup_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, a, l, l_coo);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ICT_COMPUTE_FACTOR_KERNEL);


}  // namespace par_ict_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
