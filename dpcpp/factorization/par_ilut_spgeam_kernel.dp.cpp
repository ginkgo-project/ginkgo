// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilut_kernels.hpp"


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


// subwarp sizes for add_candidates kernels
using compiled_kernels = syn::value_list<int, 1, 16, 32>;


namespace kernel {


template <int subgroup_size, typename IndexType>
void tri_spgeam_nnz(const IndexType* __restrict__ lu_row_ptrs,
                    const IndexType* __restrict__ lu_col_idxs,
                    const IndexType* __restrict__ a_row_ptrs,
                    const IndexType* __restrict__ a_col_idxs,
                    IndexType* __restrict__ l_new_row_ptrs,
                    IndexType* __restrict__ u_new_row_ptrs, IndexType num_rows,
                    sycl::nd_item<3> item_ct1)
{
    auto subwarp = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    auto row = thread::get_subwarp_id_flat<subgroup_size, IndexType>(item_ct1);
    if (row >= num_rows) {
        return;
    }

    auto lu_begin = lu_row_ptrs[row];
    auto lu_size = lu_row_ptrs[row + 1] - lu_begin;
    auto a_begin = a_row_ptrs[row];
    auto a_size = a_row_ptrs[row + 1] - a_begin;
    IndexType l_count{};
    IndexType u_count{};
    group_merge<subgroup_size>(
        a_col_idxs + a_begin, a_size, lu_col_idxs + lu_begin, lu_size, subwarp,
        [&](IndexType a_nz, IndexType a_col, IndexType lu_nz, IndexType lu_col,
            IndexType out_nz, bool valid) {
            auto col = min(a_col, lu_col);
            // count the number of unique elements being merged
            l_count +=
                popcnt(subwarp.ballot(col <= row && a_col != lu_col && valid));
            u_count +=
                popcnt(subwarp.ballot(col >= row && a_col != lu_col && valid));
            return true;
        });
    if (subwarp.thread_rank() == 0) {
        l_new_row_ptrs[row] = l_count;
        u_new_row_ptrs[row] = u_count;
    }
}

template <int subgroup_size, typename IndexType>
void tri_spgeam_nnz(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                    sycl::queue* queue, const IndexType* lu_row_ptrs,
                    const IndexType* lu_col_idxs, const IndexType* a_row_ptrs,
                    const IndexType* a_col_idxs, IndexType* l_new_row_ptrs,
                    IndexType* u_new_row_ptrs, IndexType num_rows)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                tri_spgeam_nnz<subgroup_size>(
                                    lu_row_ptrs, lu_col_idxs, a_row_ptrs,
                                    a_col_idxs, l_new_row_ptrs, u_new_row_ptrs,
                                    num_rows, item_ct1);
                            });
}


// TODO: it may get similar/better performance from a thread-per-row
// parallelization
template <int subgroup_size, typename ValueType, typename IndexType>
void tri_spgeam_init(const IndexType* __restrict__ lu_row_ptrs,
                     const IndexType* __restrict__ lu_col_idxs,
                     const ValueType* __restrict__ lu_vals,
                     const IndexType* __restrict__ a_row_ptrs,
                     const IndexType* __restrict__ a_col_idxs,
                     const ValueType* __restrict__ a_vals,
                     const IndexType* __restrict__ l_row_ptrs,
                     const IndexType* __restrict__ l_col_idxs,
                     const ValueType* __restrict__ l_vals,
                     const IndexType* __restrict__ u_row_ptrs,
                     const IndexType* __restrict__ u_col_idxs,
                     const ValueType* __restrict__ u_vals,
                     const IndexType* __restrict__ l_new_row_ptrs,
                     IndexType* __restrict__ l_new_col_idxs,
                     ValueType* __restrict__ l_new_vals,
                     const IndexType* __restrict__ u_new_row_ptrs,
                     IndexType* __restrict__ u_new_col_idxs,
                     ValueType* __restrict__ u_new_vals, IndexType num_rows,
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

    // merge A, L*U (and L+U)
    auto l_begin = l_row_ptrs[row];
    auto l_end = l_row_ptrs[row + 1] - 1;  // ignore diagonal
    auto l_size = l_end - l_begin;

    auto u_begin = u_row_ptrs[row];
    auto u_end = u_row_ptrs[row + 1];
    auto u_size = u_end - u_begin;

    // lpu_* stores the entries of L + U with the diagonal from U
    // this allows us to act as if L and U were a single matrix
    auto lpu_begin = l_begin;
    auto lpu_end = lpu_begin + l_size + u_size;
    auto lpu_col_idxs =
        lpu_begin + lane < l_end ? l_col_idxs : u_col_idxs + u_begin - l_end;
    auto lpu_vals =
        lpu_begin + lane < l_end ? l_vals : u_vals + u_begin - l_end;

    auto lu_begin = lu_row_ptrs[row];
    auto lu_end = lu_row_ptrs[row + 1];
    auto lu_size = lu_end - lu_begin;

    auto a_begin = a_row_ptrs[row];
    auto a_end = a_row_ptrs[row + 1];
    auto a_size = a_end - a_begin;

    IndexType out_begin{};
    auto out_size = lu_size + a_size;

    IndexType l_new_begin = l_new_row_ptrs[row];
    IndexType u_new_begin = u_new_row_ptrs[row];

    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    // load column indices and values for the first merge step
    auto a_col = checked_load(a_col_idxs, a_begin + lane, a_end, sentinel);
    auto a_val = checked_load(a_vals, a_begin + lane, a_end, zero<ValueType>());
    auto lu_col = checked_load(lu_col_idxs, lu_begin + lane, lu_end, sentinel);
    auto lu_val =
        checked_load(lu_vals, lu_begin + lane, lu_end, zero<ValueType>());
    auto lpu_col =
        checked_load(lpu_col_idxs, lpu_begin + lane, lpu_end, sentinel);
    auto lpu_val =
        checked_load(lpu_vals, lpu_begin + lane, lpu_end, zero<ValueType>());
    bool skip_first{};
    while (out_begin < out_size) {
        // merge subwarp.size() elements from A and L*U
        auto merge_result =
            group_merge_step<subgroup_size>(a_col, lu_col, subwarp);
        auto a_cur_col = merge_result.a_val;
        auto lu_cur_col = merge_result.b_val;
        auto a_cur_val = subwarp.shfl(a_val, merge_result.a_idx);
        auto lu_cur_val = subwarp.shfl(lu_val, merge_result.b_idx);
        auto valid = out_begin + lane < out_size;
        // check if the previous thread has matching columns
        auto equal_mask = subwarp.ballot(a_cur_col == lu_cur_col && valid);
        auto prev_equal_mask = equal_mask << 1 | skip_first;
        skip_first = bool(equal_mask >> (subgroup_size - 1));
        auto prev_equal = bool(prev_equal_mask & lanemask_eq);

        auto r_col = min(a_cur_col, lu_cur_col);
        // find matching entry of L+U
        // S(L + U) is a subset of S(A - L * U) since L and U have a diagonal
        auto lpu_source = synchronous_fixed_binary_search<subgroup_size>(
            [&](int i) { return subwarp.shfl(lpu_col, i) >= r_col; });
        auto lpu_cur_col = subwarp.shfl(lpu_col, lpu_source);
        auto lpu_cur_val = subwarp.shfl(lpu_val, lpu_source);

        // determine actual values of A and L*U at r_col
        if (r_col != a_cur_col) {
            a_cur_val = zero<ValueType>();
        }
        if (r_col != lu_cur_col) {
            lu_cur_val = zero<ValueType>();
        }
        auto r_val = a_cur_val - lu_cur_val;

        // determine which threads will write output to L or U
        auto use_lpu = lpu_cur_col == r_col;
        auto l_new_advance_mask =
            subwarp.ballot(r_col <= row && !prev_equal && valid);
        auto u_new_advance_mask =
            subwarp.ballot(r_col >= row && !prev_equal && valid);
        // store values
        if (!prev_equal && valid) {
            auto diag =
                r_col < row ? u_vals[u_row_ptrs[r_col]] : one<ValueType>();
            auto out_val = use_lpu ? lpu_cur_val : r_val / diag;
            if (r_col <= row) {
                auto ofs = popcnt(l_new_advance_mask & lanemask_lt);
                l_new_col_idxs[l_new_begin + ofs] = r_col;
                l_new_vals[l_new_begin + ofs] =
                    r_col == row ? one<ValueType>() : out_val;
            }
            if (r_col >= row) {
                auto ofs = popcnt(u_new_advance_mask & lanemask_lt);
                u_new_col_idxs[u_new_begin + ofs] = r_col;
                u_new_vals[u_new_begin + ofs] = out_val;
            }
        }

        // advance *_begin offsets
        auto a_advance = merge_result.a_advance;
        auto lu_advance = merge_result.b_advance;
        auto lpu_advance =
            popcnt(subwarp.ballot(use_lpu && !prev_equal && valid));
        auto l_new_advance = popcnt(l_new_advance_mask);
        auto u_new_advance = popcnt(u_new_advance_mask);
        a_begin += a_advance;
        lu_begin += lu_advance;
        lpu_begin += lpu_advance;
        l_new_begin += l_new_advance;
        u_new_begin += u_new_advance;
        out_begin += subgroup_size;

        // shuffle the unmerged elements to the front
        a_col = subwarp.shfl_down(a_col, a_advance);
        a_val = subwarp.shfl_down(a_val, a_advance);
        lu_col = subwarp.shfl_down(lu_col, lu_advance);
        lu_val = subwarp.shfl_down(lu_val, lu_advance);
        lpu_col = subwarp.shfl_down(lpu_col, lpu_advance);
        lpu_val = subwarp.shfl_down(lpu_val, lpu_advance);
        /*
         * To optimize memory access, we load the new elements for `a` and `lu`
         * with a single load instruction:
         * the lower part of the group loads new elements for `a`
         * the upper part of the group loads new elements for `lu`
         * `load_lane` is the part-local lane idx
         * The elements for `a` have to be shuffled up afterwards.
         */
        auto load_a = lane < a_advance;
        auto load_lane = load_a ? lane : lane - a_advance;
        auto load_source_col = load_a ? a_col_idxs : lu_col_idxs;
        auto load_source_val = load_a ? a_vals : lu_vals;
        auto load_begin = load_a ? a_begin + lu_advance : lu_begin + a_advance;
        auto load_end = load_a ? a_end : lu_end;

        auto load_idx = load_begin + load_lane;
        auto loaded_col =
            checked_load(load_source_col, load_idx, load_end, sentinel);
        auto loaded_val = checked_load(load_source_val, load_idx, load_end,
                                       zero<ValueType>());
        // shuffle the `a` values to the end of the warp
        auto lower_loaded_col = subwarp.shfl_up(loaded_col, lu_advance);
        auto lower_loaded_val = subwarp.shfl_up(loaded_val, lu_advance);
        if (lane >= lu_advance) {
            a_col = lower_loaded_col;
            a_val = lower_loaded_val;
        }
        if (lane >= a_advance) {
            lu_col = loaded_col;
            lu_val = loaded_val;
        }
        // load the new values for lpu
        if (lane >= subgroup_size - lpu_advance) {
            auto lpu_idx = lpu_begin + lane;
            // update lpu pointer if we move from l to u
            if (lpu_idx >= l_end) {
                lpu_col_idxs = u_col_idxs + u_begin - l_end;
                lpu_vals = u_vals + u_begin - l_end;
            }
            lpu_col = checked_load(lpu_col_idxs, lpu_idx, lpu_end, sentinel);
            lpu_val =
                checked_load(lpu_vals, lpu_idx, lpu_end, zero<ValueType>());
        }
    }
}

template <int subgroup_size, typename ValueType, typename IndexType>
void tri_spgeam_init(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                     sycl::queue* queue, const IndexType* lu_row_ptrs,
                     const IndexType* lu_col_idxs, const ValueType* lu_vals,
                     const IndexType* a_row_ptrs, const IndexType* a_col_idxs,
                     const ValueType* a_vals, const IndexType* l_row_ptrs,
                     const IndexType* l_col_idxs, const ValueType* l_vals,
                     const IndexType* u_row_ptrs, const IndexType* u_col_idxs,
                     const ValueType* u_vals, const IndexType* l_new_row_ptrs,
                     IndexType* l_new_col_idxs, ValueType* l_new_vals,
                     const IndexType* u_new_row_ptrs, IndexType* u_new_col_idxs,
                     ValueType* u_new_vals, IndexType num_rows)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                tri_spgeam_init<subgroup_size>(
                                    lu_row_ptrs, lu_col_idxs, lu_vals,
                                    a_row_ptrs, a_col_idxs, a_vals, l_row_ptrs,
                                    l_col_idxs, l_vals, u_row_ptrs, u_col_idxs,
                                    u_vals, l_new_row_ptrs, l_new_col_idxs,
                                    l_new_vals, u_new_row_ptrs, u_new_col_idxs,
                                    u_new_vals, num_rows, item_ct1);
                            });
}


}  // namespace kernel

namespace {


template <int subgroup_size, typename ValueType, typename IndexType>
void add_candidates(syn::value_list<int, subgroup_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* lu,
                    const matrix::Csr<ValueType, IndexType>* a,
                    const matrix::Csr<ValueType, IndexType>* l,
                    const matrix::Csr<ValueType, IndexType>* u,
                    matrix::Csr<ValueType, IndexType>* l_new,
                    matrix::Csr<ValueType, IndexType>* u_new)
{
    auto num_rows = static_cast<IndexType>(lu->get_size()[0]);
    auto subwarps_per_block = default_block_size / subgroup_size;
    auto num_blocks = ceildiv(num_rows, subwarps_per_block);
    matrix::CsrBuilder<ValueType, IndexType> l_new_builder(l_new);
    matrix::CsrBuilder<ValueType, IndexType> u_new_builder(u_new);
    auto lu_row_ptrs = lu->get_const_row_ptrs();
    auto lu_col_idxs = lu->get_const_col_idxs();
    auto lu_vals = lu->get_const_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_const_values();
    auto u_row_ptrs = u->get_const_row_ptrs();
    auto u_col_idxs = u->get_const_col_idxs();
    auto u_vals = u->get_const_values();
    auto l_new_row_ptrs = l_new->get_row_ptrs();
    auto u_new_row_ptrs = u_new->get_row_ptrs();
    // count non-zeros per row
    kernel::tri_spgeam_nnz<subgroup_size>(
        num_blocks, default_block_size, 0, exec->get_queue(), lu_row_ptrs,
        lu_col_idxs, a_row_ptrs, a_col_idxs, l_new_row_ptrs, u_new_row_ptrs,
        num_rows);

    // build row ptrs
    components::prefix_sum_nonnegative(exec, l_new_row_ptrs, num_rows + 1);
    components::prefix_sum_nonnegative(exec, u_new_row_ptrs, num_rows + 1);

    // resize output arrays
    auto l_new_nnz = exec->copy_val_to_host(l_new_row_ptrs + num_rows);
    auto u_new_nnz = exec->copy_val_to_host(u_new_row_ptrs + num_rows);
    l_new_builder.get_col_idx_array().resize_and_reset(l_new_nnz);
    l_new_builder.get_value_array().resize_and_reset(l_new_nnz);
    u_new_builder.get_col_idx_array().resize_and_reset(u_new_nnz);
    u_new_builder.get_value_array().resize_and_reset(u_new_nnz);

    auto l_new_col_idxs = l_new->get_col_idxs();
    auto l_new_vals = l_new->get_values();
    auto u_new_col_idxs = u_new->get_col_idxs();
    auto u_new_vals = u_new->get_values();

    // fill columns and values
    kernel::tri_spgeam_init<subgroup_size>(
        num_blocks, default_block_size, 0, exec->get_queue(), lu_row_ptrs,
        lu_col_idxs, lu_vals, a_row_ptrs, a_col_idxs, a_vals, l_row_ptrs,
        l_col_idxs, l_vals, u_row_ptrs, u_col_idxs, u_vals, l_new_row_ptrs,
        l_new_col_idxs, l_new_vals, u_new_row_ptrs, u_new_col_idxs, u_new_vals,
        num_rows);
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_add_candidates, add_candidates);


}  // namespace


template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* lu,
                    const matrix::Csr<ValueType, IndexType>* a,
                    const matrix::Csr<ValueType, IndexType>* l,
                    const matrix::Csr<ValueType, IndexType>* u,
                    matrix::Csr<ValueType, IndexType>* l_new,
                    matrix::Csr<ValueType, IndexType>* u_new)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz =
        lu->get_num_stored_elements() + a->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_add_candidates(
        compiled_kernels(),
        [&](int compiled_subgroup_size) {
            return total_nnz_per_row <= compiled_subgroup_size ||
                   compiled_subgroup_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, lu, a, l, u, l_new,
        u_new);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
