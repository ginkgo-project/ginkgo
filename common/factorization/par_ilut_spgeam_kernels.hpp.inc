/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

namespace kernel {


template <int subwarp_size, typename IndexType>
__global__ __launch_bounds__(default_block_size) void tri_spgeam_nnz(
    const IndexType *__restrict__ lu_row_ptrs,
    const IndexType *__restrict__ lu_col_idxs,
    const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_col_idxs,
    IndexType *__restrict__ l_new_row_ptrs,
    IndexType *__restrict__ u_new_row_ptrs, IndexType num_rows)
{
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    if (row >= num_rows) {
        return;
    }

    auto lu_begin = lu_row_ptrs[row];
    auto lu_size = lu_row_ptrs[row + 1] - lu_begin;
    auto a_begin = a_row_ptrs[row];
    auto a_size = a_row_ptrs[row + 1] - a_begin;
    IndexType l_count{};
    IndexType u_count{};
    group_merge<subwarp_size>(
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


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void tri_spgeam_init(
    const IndexType *__restrict__ lu_row_ptrs,
    const IndexType *__restrict__ lu_col_idxs,
    const ValueType *__restrict__ lu_vals,
    const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_col_idxs,
    const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ l_row_ptrs,
    const IndexType *__restrict__ l_col_idxs,
    const ValueType *__restrict__ l_vals,
    const IndexType *__restrict__ u_row_ptrs,
    const IndexType *__restrict__ u_col_idxs,
    const ValueType *__restrict__ u_vals,
    const IndexType *__restrict__ l_new_row_ptrs,
    IndexType *__restrict__ l_new_col_idxs, ValueType *__restrict__ l_new_vals,
    const IndexType *__restrict__ u_new_row_ptrs,
    IndexType *__restrict__ u_new_col_idxs, ValueType *__restrict__ u_new_vals,
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

    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
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
            group_merge_step<subwarp_size>(a_col, lu_col, subwarp);
        auto a_cur_col = merge_result.a_val;
        auto lu_cur_col = merge_result.b_val;
        auto a_cur_val = subwarp.shfl(a_val, merge_result.a_idx);
        auto lu_cur_val = subwarp.shfl(lu_val, merge_result.b_idx);
        auto valid = out_begin + lane < out_size;
        // check if the previous thread has matching columns
        auto equal_mask = subwarp.ballot(a_cur_col == lu_cur_col && valid);
        auto prev_equal_mask = equal_mask << 1 | skip_first;
        skip_first = bool(equal_mask >> (subwarp_size - 1));
        auto prev_equal = bool(prev_equal_mask & lanemask_eq);

        auto r_col = min(a_cur_col, lu_cur_col);
        // find matching entry of L+U
        // S(L + U) is a subset of S(A - L * U) since L and U have a diagonal
        auto lpu_source = synchronous_fixed_binary_search<subwarp_size>(
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
        out_begin += subwarp_size;

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
        if (lane >= subwarp_size - lpu_advance) {
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


}  // namespace kernel