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


namespace detail {


template <bool IsSorted = false>
struct find_helper {
    template <typename Group, typename IndexType>
    static __device__ bool find(Group subwarp_grp, const IndexType *first,
                                const IndexType *last, const IndexType &value)
    {
        auto subwarp_idx = subwarp_grp.thread_rank();
        bool found{false};
        for (auto curr = first + subwarp_idx; curr < last;
             curr += subwarp_grp.size()) {
            found = (*curr == value);
            found = subwarp_grp.any(found);
            if (found) {
                break;
            }
        }
        return found;
    }
};


/* TODO write specialization for the sorted case.
template <>
struct find_helper<true> {
    template <int SubwarpSize, typename IndexType>
    static bool find(IndexType subwarp_idx, const IndexType *first, const
IndexType *last, const IndexType &value)
    {
        return std::binary_search(first, last, value);
    }
};
*/


}  // namespace detail

// SubwarpSize needs to be a power of 2
// Each subwarp works on one row
template <bool IsSorted, int SubwarpSize, typename IndexType>
__global__
    __launch_bounds__(default_block_size) void find_missing_diagonal_elements(
        IndexType num_rows, IndexType num_cols,
        const IndexType *__restrict__ col_idxs,
        const IndexType *__restrict__ row_ptrs,
        IndexType *__restrict__ elements_to_add_per_row,
        bool *__restrict__ changes_required)
{
    const auto rows_covered_by_threads =
        static_cast<size_type>(blockDim.x) * gridDim.x / SubwarpSize;
    const auto tidx =
        threadIdx.x + static_cast<size_type>(blockIdx.x) * blockDim.x;
    const auto begin_row = static_cast<IndexType>(tidx / SubwarpSize);

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<SubwarpSize>(thread_block);
    bool local_change{false};
    for (IndexType row = begin_row; row < num_rows;
         row += rows_covered_by_threads) {
        if (row >= num_cols) {
            elements_to_add_per_row[row] = 0;
            continue;
        }
        const auto *start_cols = col_idxs + row_ptrs[row];
        const auto *end_cols = col_idxs + row_ptrs[row + 1];
        if (detail::find_helper<IsSorted>::find(subwarp_grp, start_cols,
                                                end_cols, row)) {
            elements_to_add_per_row[row] = 0;
        } else {
            elements_to_add_per_row[row] = 1;
            local_change = true;
        }
    }
    if (local_change) {
        *changes_required = true;
    }
    if (tidx == 0) {
        elements_to_add_per_row[num_rows] = 0;
    }
}


// SubwarpSize needs to be a power of 2
// Each subwarp works on one row
template <int SubwarpSize, typename ValueType, typename IndexType>
__global__
    __launch_bounds__(default_block_size) void add_missing_diagonal_elements(
        IndexType num_rows, const ValueType *__restrict__ old_values,
        const IndexType *__restrict__ old_col_idxs,
        const IndexType *__restrict__ old_row_ptrs,
        ValueType *__restrict__ new_values,
        IndexType *__restrict__ new_col_idxs,
        const IndexType *__restrict__ row_ptrs_addition)
{
    const auto rows_covered_by_threads =
        static_cast<size_type>(blockDim.x) * gridDim.x / SubwarpSize;
    const auto tidx =
        threadIdx.x + static_cast<size_type>(blockIdx.x) * blockDim.x;
    const auto begin_row = static_cast<IndexType>(tidx / SubwarpSize);

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<SubwarpSize>(thread_block);
    const auto subwarp_idx = subwarp_grp.thread_rank();

    for (IndexType row = begin_row; row < num_rows;
         row += rows_covered_by_threads) {
        const IndexType old_row_start{old_row_ptrs[row]};
        const IndexType old_row_end{old_row_ptrs[row + 1]};
        const IndexType new_row_start{old_row_start + row_ptrs_addition[row]};
        const IndexType new_row_end{old_row_end + row_ptrs_addition[row + 1]};

        // if no element needs to be added, do a simple copy
        if (new_row_end - new_row_start == old_row_end - old_row_start) {
            for (IndexType i = subwarp_idx; i < new_row_end - new_row_start;
                 i += SubwarpSize) {
                const IndexType new_idx = new_row_start + i;
                const IndexType old_idx = old_row_start + i;
                new_values[new_idx] = old_values[old_idx];
                new_col_idxs[new_idx] = old_col_idxs[old_idx];
            }
        } else {
            IndexType new_idx = new_row_start + subwarp_idx;
            bool diagonal_added{false};
            for (IndexType old_idx = old_row_start + subwarp_idx;
                 old_idx < old_row_end;
                 old_idx += SubwarpSize, new_idx += SubwarpSize) {
                const auto col_idx = old_col_idxs[old_idx];
                bool diagonal_add_required = !diagonal_added && row < col_idx;
                auto ballot = subwarp_grp.ballot(diagonal_add_required);
                if (ballot) {
                    auto first_subwarp_idx = ffs(ballot) - 1;
                    if (first_subwarp_idx == subwarp_idx) {
                        new_values[new_idx] = zero<ValueType>();
                        new_col_idxs[new_idx] = row;
                    }
                    // if diagonal was inserted in a thread below this one,
                    // add it to the new_idx.
                    new_idx += (first_subwarp_idx <= subwarp_idx) ? 1 : 0;
                    new_values[new_idx] = old_values[old_idx];
                    new_col_idxs[new_idx] = col_idx;
                    // if diagonal is inserted in a thread after this one,
                    // it needs to be considered after writing the values
                    new_idx += (subwarp_idx < first_subwarp_idx) ? 1 : 0;
                    diagonal_added = true;
                } else {
                    new_values[new_idx] = old_values[old_idx];
                    new_col_idxs[new_idx] = col_idx;
                }
            }
            if (!diagonal_added) {
                new_idx = new_row_end - 1;
                new_values[new_idx] = zero<ValueType>();
                new_col_idxs[new_idx] = row;
            }
        }
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void update_row_ptrs(
    IndexType num_rows, IndexType *__restrict__ row_ptrs,
    IndexType *__restrict__ row_ptr_addition)
{
    const auto rows_covered_by_threads =
        static_cast<size_type>(blockDim.x) * gridDim.x;
    const auto begin_row =
        threadIdx.x + static_cast<size_type>(blockIdx.x) * blockDim.x;

    for (IndexType row = begin_row; row < num_rows;
         row += rows_covered_by_threads) {
        row_ptrs[row] += row_ptr_addition[row];
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void count_nnz_per_l_u_row(
    size_type num_rows, const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, IndexType *__restrict__ l_nnz_row,
    IndexType *__restrict__ u_nnz_row)
{
    const auto row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        IndexType l_row_nnz{};
        IndexType u_row_nnz{};
        bool has_diagonal{};
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            auto col = col_idxs[idx];
            l_row_nnz += (col <= row);
            u_row_nnz += (row <= col);
            has_diagonal |= col == row;
        }
        // if we didn't find it, add the diagonal entry
        l_nnz_row[row] = l_row_nnz + !has_diagonal;
        u_nnz_row[row] = u_row_nnz + !has_diagonal;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void initialize_l_u(
    size_type num_rows, const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values,
    const IndexType *__restrict__ l_row_ptrs,
    IndexType *__restrict__ l_col_idxs, ValueType *__restrict__ l_values,
    const IndexType *__restrict__ u_row_ptrs,
    IndexType *__restrict__ u_col_idxs, ValueType *__restrict__ u_values)
{
    const auto row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        auto l_idx = l_row_ptrs[row];
        auto u_idx = u_row_ptrs[row] + 1;  // we treat the diagonal separately
        bool has_diagonal{};
        ValueType diag_val{};
        for (size_type i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
            const auto col = col_idxs[i];
            const auto val = values[i];
            // save diagonal entry for later
            if (col == row) {
                has_diagonal = true;
                diag_val = val;
            }
            if (col < row) {
                l_col_idxs[l_idx] = col;
                l_values[l_idx] = val;
                ++l_idx;
            }
            if (row < col) {
                u_col_idxs[u_idx] = col;
                u_values[u_idx] = val;
                ++u_idx;
            }
        }
        // if there was no diagonal entry, set it to one
        if (!has_diagonal) {
            diag_val = one<ValueType>();
        }
        // store diagonal entries
        auto l_diag_idx = l_row_ptrs[row + 1] - 1;
        auto u_diag_idx = u_row_ptrs[row];
        l_col_idxs[l_diag_idx] = row;
        u_col_idxs[u_diag_idx] = row;
        l_values[l_diag_idx] = one<ValueType>();
        u_values[u_diag_idx] = diag_val;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void compute_l_u_factors(
    size_type num_elements, const IndexType *__restrict__ row_idxs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values,
    const IndexType *__restrict__ l_row_ptrs,
    const IndexType *__restrict__ l_col_idxs, ValueType *__restrict__ l_values,
    const IndexType *__restrict__ u_row_ptrs,
    const IndexType *__restrict__ u_col_idxs, ValueType *__restrict__ u_values)
{
    const auto elem_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (elem_id < num_elements) {
        const auto row = row_idxs[elem_id];
        const auto col = col_idxs[elem_id];
        const auto val = values[elem_id];
        auto l_idx = l_row_ptrs[row];
        auto u_idx = u_row_ptrs[col];
        ValueType sum{val};
        ValueType last_operation{};
        while (l_idx < l_row_ptrs[row + 1] && u_idx < u_row_ptrs[col + 1]) {
            const auto l_col = l_col_idxs[l_idx];
            const auto u_col = u_col_idxs[u_idx];
            last_operation = zero<ValueType>();
            if (l_col == u_col) {
                last_operation = l_values[l_idx] * u_values[u_idx];
                sum -= last_operation;
            }
            l_idx += (l_col <= u_col);
            u_idx += (u_col <= l_col);
        }
        sum += last_operation;  // undo the last operation
        if (row > col) {
            auto to_write = sum / u_values[u_row_ptrs[col + 1] - 1];
            if (::gko::isfinite(to_write)) {
                l_values[l_idx - 1] = to_write;
            }
        } else {
            auto to_write = sum;
            if (::gko::isfinite(to_write)) {
                u_values[u_idx - 1] = to_write;
            }
        }
    }
}


}  // namespace kernel
