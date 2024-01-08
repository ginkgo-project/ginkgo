// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace kernel {


template <int subwarp_size, typename IndexType>
__global__ __launch_bounds__(default_block_size) void ict_tri_spgeam_nnz(
    const IndexType* __restrict__ llh_row_ptrs,
    const IndexType* __restrict__ llh_col_idxs,
    const IndexType* __restrict__ a_row_ptrs,
    const IndexType* __restrict__ a_col_idxs,
    IndexType* __restrict__ l_new_row_ptrs, IndexType num_rows)
{
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    if (row >= num_rows) {
        return;
    }

    auto llh_begin = llh_row_ptrs[row];
    auto llh_size = llh_row_ptrs[row + 1] - llh_begin;
    auto a_begin = a_row_ptrs[row];
    auto a_size = a_row_ptrs[row + 1] - a_begin;
    IndexType count{};
    group_merge<subwarp_size>(
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


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void ict_tri_spgeam_init(
    const IndexType* __restrict__ llh_row_ptrs,
    const IndexType* __restrict__ llh_col_idxs,
    const ValueType* __restrict__ llh_vals,
    const IndexType* __restrict__ a_row_ptrs,
    const IndexType* __restrict__ a_col_idxs,
    const ValueType* __restrict__ a_vals,
    const IndexType* __restrict__ l_row_ptrs,
    const IndexType* __restrict__ l_col_idxs,
    const ValueType* __restrict__ l_vals,
    const IndexType* __restrict__ l_new_row_ptrs,
    IndexType* __restrict__ l_new_col_idxs, ValueType* __restrict__ l_new_vals,
    IndexType num_rows)
{
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
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

    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
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
            group_merge_step<subwarp_size>(a_col, llh_col, subwarp);
        auto a_cur_col = merge_result.a_val;
        auto llh_cur_col = merge_result.b_val;
        auto a_cur_val = subwarp.shfl(a_val, merge_result.a_idx);
        auto llh_cur_val = subwarp.shfl(llh_val, merge_result.b_idx);
        auto valid = out_begin + lane < out_size;
        // check if the previous thread has matching columns
        auto equal_mask = subwarp.ballot(a_cur_col == llh_cur_col && valid);
        auto prev_equal_mask = equal_mask << 1 | skip_first;
        skip_first = bool(equal_mask >> (subwarp_size - 1));
        auto prev_equal = bool(prev_equal_mask & lanemask_eq);

        auto r_col = min(a_cur_col, llh_cur_col);
        // find matching entry of L
        // S(L) is a subset of S(A - L * L^H) since L has a diagonal
        auto l_source = synchronous_fixed_binary_search<subwarp_size>(
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
        out_begin += subwarp_size;

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
        if (lane >= subwarp_size - l_advance) {
            auto l_idx = l_begin + lane;
            l_col = checked_load(l_col_idxs, l_idx, l_end, sentinel);
            l_val = checked_load(l_vals, l_idx, l_end, zero<ValueType>());
        }
    }
}


}  // namespace kernel
