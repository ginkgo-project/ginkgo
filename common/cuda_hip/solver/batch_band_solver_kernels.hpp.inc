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


namespace unblocked_LU_band_factorization {

template <typename Group, typename ValueType>
__device__ __forceinline__ void set_possible_fill_in_zero_in_U_part(
    Group& subwarp_grp, const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry)
{
    const int KV = KL + KU;
    const int AB_nrows = 2 * KL + KU + 1;

    // set all fill-in elements: zero
    // col: 0,1,2,..., KU : no fill-in possible in U part

    const int local_subwarp_id = threadIdx.x / subwarp_grp.size();
    const int total_num_subwarp_grps_in_block =
        (blockDim.x) / subwarp_grp.size();

    // a subwarp per column
    for (int AB_col_idx = KU + 1 + local_subwarp_id; AB_col_idx <= N - 1;
         AB_col_idx += total_num_subwarp_grps_in_block) {
        for (int AB_row_idx = KL - 1 - subwarp_grp.thread_rank();
             AB_row_idx >= max(0, KV - AB_col_idx);
             AB_row_idx -= subwarp_grp.size()) {
            AB_array_entry[AB_row_idx + AB_nrows * AB_col_idx] =
                zero<ValueType>();  // coalesced accesses
        }
    }
}

template <typename ValueType>
__device__ __forceinline__ int naive_find_pivot_row_index(
    const int N, const int KL, const int KU,
    const ValueType* const __restrict__ AB_array_entry, const int AB_col_idx)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int KV = KU + KL;
    int AB_piv_row_idx = KV;
    const int AB_row_idx_limit = KL + KU + (N - AB_col_idx - 1);
    ValueType piv_val = AB_array_entry[AB_piv_row_idx + AB_nrows * AB_col_idx];

    // uncoalesced accesses but atleast data locality is there
    for (int AB_row_idx = KV; AB_row_idx <= min(AB_row_idx_limit, KV + KL);
         AB_row_idx++) {
        const auto val = AB_array_entry[AB_row_idx + AB_nrows * AB_col_idx];
        if (abs(piv_val) < abs(val)) {
            piv_val = val;
            AB_piv_row_idx = AB_row_idx;
        }
    }

    return AB_piv_row_idx;
}


template <typename Group, typename ValueType>
__device__ __forceinline__ int find_pivot_row_index(
    Group& subwarp_grp, const int N, const int KL, const int KU,
    const ValueType* const __restrict__ AB_array_entry, const int AB_col_idx)
{
    // Each subwarp of the threadblock finds the pivot row for the given column.
    const int AB_nrows = 2 * KL + KU + 1;
    const int KV = KU + KL;
    const int AB_row_idx_limit = KL + KU + (N - AB_col_idx - 1);
    const int final_AB_row_idx_limit = min(AB_row_idx_limit, KV + KL);

    int my_AB_row_idx = KV + subwarp_grp.thread_rank();
    ValueType my_val;
    if (my_AB_row_idx <= final_AB_row_idx_limit) {
        my_val = AB_array_entry[my_AB_row_idx + AB_nrows * AB_col_idx];
        // coalesced accesses
    }

    const ValueType diag_val = subwarp_grp.shfl(my_val, 0);

    if (my_AB_row_idx > final_AB_row_idx_limit) {
        my_AB_row_idx = KV;
        my_val = diag_val;
    }

    for (int AB_row_idx = KV + subwarp_grp.size() + subwarp_grp.thread_rank();
         AB_row_idx <= final_AB_row_idx_limit;
         AB_row_idx += subwarp_grp.size()) {
        const ValueType val =
            AB_array_entry[AB_row_idx +
                           AB_nrows * AB_col_idx];  // coalesced accesses
        if (abs(val) > abs(my_val)) {
            my_val = val;
            my_AB_row_idx = AB_row_idx;
        }
    }
    subwarp_grp.sync();

    for (int offset = subwarp_grp.size() / 2; offset > 0; offset /= 2) {
        const auto other_val = subwarp_grp.shfl_down(my_val, offset);
        const auto other_AB_row_idx =
            subwarp_grp.shfl_down(my_AB_row_idx, offset);
        if (abs(other_val) > abs(my_val)) {
            my_val = other_val;
            my_AB_row_idx = other_AB_row_idx;
        }
        subwarp_grp.sync();
    }

    // 0th thread of the subwarp will have the correct pivot row index and value
    const int AB_piv_row_idx = subwarp_grp.shfl(my_AB_row_idx, 0);
    return AB_piv_row_idx;
}


template <typename ValueType>
__device__ __forceinline__ void swap_rows(
    const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry,
    const int A_affected_col_idx_last, const int A_diag_row_idx,
    const int A_pivot_row_idx)
{
    const int AB_nrows = 2 * KL + KU + 1;

    // uncoalesced access, also no data locality
    for (int A_col_idx = A_diag_row_idx + threadIdx.x;
         A_col_idx <= A_affected_col_idx_last; A_col_idx += blockDim.x) {
        const int AB_row_idx_1 = KL + KU + A_diag_row_idx - A_col_idx;
        const int AB_row_idx_2 = KL + KU + A_pivot_row_idx - A_col_idx;
        const ValueType temp =
            AB_array_entry[AB_row_idx_1 + AB_nrows * A_col_idx];
        AB_array_entry[AB_row_idx_1 + AB_nrows * A_col_idx] =
            AB_array_entry[AB_row_idx_2 + AB_nrows * A_col_idx];
        AB_array_entry[AB_row_idx_2 + AB_nrows * A_col_idx] = temp;
    }
}


template <typename ValueType>
__device__ __forceinline__ void modify_rows_below(
    const int N, const int KL, const int KU,
    ValueType* const __restrict__ ele_above_sh_arr_entry,
    const ValueType diag_val, ValueType* const __restrict__ AB_array_entry,
    const int AB_col_idx, const int A_diag_row_idx,
    const int A_affected_col_idx_last)
{
    const int AB_nrows = 2 * KL + KU + 1;

    // fill the ele_above_sh_arr_entry
    // this is to reduce uncoalesced global memory reads while updating the
    // submatrix
    for (int A_col_idx = A_diag_row_idx + 1 + threadIdx.x;
         A_col_idx <= A_affected_col_idx_last; A_col_idx += blockDim.x) {
        const int AB_main_row = KL + KU + A_diag_row_idx - A_col_idx;
        const auto ele_above =
            AB_array_entry[AB_main_row +
                           AB_nrows * A_col_idx];  // uncoalesced accesses but
                                                   // can't help
        ele_above_sh_arr_entry[A_col_idx] = ele_above;
    }

    __syncthreads();

    // each thread handles a row, and the acesses are coalesced AB is in col
    // major order
    for (int A_row_idx = A_diag_row_idx + 1 + threadIdx.x;
         A_row_idx <= min(N - 1, A_diag_row_idx + KL);
         A_row_idx += blockDim.x) {
        const int AB_row_idx_for_side_ele =
            KL + KU + A_row_idx - A_diag_row_idx;
        // scale the ele in col of diag_ele
        AB_array_entry[AB_row_idx_for_side_ele + AB_nrows * A_diag_row_idx] /=
            diag_val;
        const ValueType ele_side =
            AB_array_entry[AB_row_idx_for_side_ele +
                           AB_nrows * A_diag_row_idx];  // coalesced accesses

        for (int A_col_idx = A_diag_row_idx + 1;
             A_col_idx <= A_affected_col_idx_last; A_col_idx++) {
            const ValueType ele_above = ele_above_sh_arr_entry[A_col_idx];

            const int AB_row_idx_for_this_col_ele =
                KL + KU + A_row_idx - A_col_idx;
            AB_array_entry[AB_row_idx_for_this_col_ele +
                           AB_nrows * A_col_idx] -=
                ele_above * ele_side;  // coalesced accesses
        }
    }
}

template <typename Group, typename ValueType>
__device__ __forceinline__ void factorize_band(
    Group& subwarp_grp, const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry,
    int* const __restrict__ ipiv_sh_entry,
    ValueType* const __restrict__ temp_sh_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int KV = KU + KL;

    // write zeroes into the possible fill-in U part
    set_possible_fill_in_zero_in_U_part(subwarp_grp, N, KL, KU, AB_array_entry);
    __syncthreads();

    int A_affected_col_idx_last = 0;

    for (int AB_col_idx = 0; AB_col_idx <= N - 1; AB_col_idx++) {
        const int AB_diag_row_idx = KV;

        __syncthreads();
        const int AB_pivot_row_idx = find_pivot_row_index(
            subwarp_grp, N, KL, KU, AB_array_entry, AB_col_idx);

        const int A_pivot_row_idx = AB_pivot_row_idx - KV + AB_col_idx;
        const int A_diag_row_idx = AB_col_idx;

        // assert piv not equal to zero
        const ValueType piv_val =
            AB_array_entry[AB_pivot_row_idx + AB_nrows * AB_col_idx];
        assert(piv_val != zero<ValueType>());

        A_affected_col_idx_last =
            max(A_affected_col_idx_last, min(A_pivot_row_idx + KU, N - 1));

        // swap rows
        if (AB_diag_row_idx != AB_pivot_row_idx) {
            swap_rows(N, KL, KU, AB_array_entry, A_affected_col_idx_last,
                      A_diag_row_idx, A_pivot_row_idx);
        }

        // record the swap row info...
        if (threadIdx.x == 0) {
            ipiv_sh_entry[A_diag_row_idx] = A_pivot_row_idx;
        }
        __syncthreads();

        // scale A's column and update trailing matrix
        modify_rows_below(N, KL, KU, temp_sh_entry, piv_val, AB_array_entry,
                          AB_col_idx, A_diag_row_idx, A_affected_col_idx_last);

        __syncthreads();
    }
}

}  // namespace unblocked_LU_band_factorization

namespace unblocked_and_blocked_band_solver_common {

template <typename ValueType>
__device__ __forceinline__ void accommodate_partial_row_swaps_lower_trsv(
    const int N, const int KL, const int KU,
    const int* const __restrict__ ipiv_sh_entry,
    const ValueType* const __restrict__ AB_array_entry,
    const ValueType* const __restrict__ b_global_entry,
    ValueType* const __restrict__ temp_sh_entry)
{
    single_copy(N, b_global_entry, temp_sh_entry);
    __syncthreads();

    const int AB_nrows = 2 * KL + KU + 1;

    for (int j = 0; j <= N - 2; j++) {
        // swap temp[j] and temp[ipiv[j]]
        if (threadIdx.x == 0) {
            const ValueType tmp = temp_sh_entry[j];
            temp_sh_entry[j] = temp_sh_entry[ipiv_sh_entry[j]];
            temp_sh_entry[ipiv_sh_entry[j]] = tmp;
        }

        __syncthreads();

        // Now temp[j] holds the correct value.
        const int A_col_idx = j;
        const auto temp_at_row_j = temp_sh_entry[A_col_idx];

        for (int A_row_idx = j + 1 + threadIdx.x;
             A_row_idx <= min(N - 1, j + KL); A_row_idx += blockDim.x) {
            // temp[A_row_idx] += -1 * A(A_row_idx, A_col_idx) * temp[A_col_idx]

            const int AB_row_idx = KL + KU + A_row_idx - A_col_idx;

            temp_sh_entry[A_row_idx] +=
                -one<ValueType>() *
                AB_array_entry[AB_row_idx + A_col_idx * AB_nrows] *
                temp_at_row_j;  // band matrix accesses are coalseced
        }

        __syncthreads();
    }
}

template <typename ValueType>
__device__ __forceinline__ void naive_upper_trsv(
    const int N, const int KL, const int KU,
    const ValueType* const __restrict__ AB_array_entry,
    const ValueType* const __restrict__ y_sh_entry,
    ValueType* const __restrict__ x_sh_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    if (threadIdx.x == 0) {
        for (int A_row_idx = N - 1; A_row_idx >= 0; A_row_idx--) {
            const int A_col_idx_last = min(A_row_idx + KL + KU, N - 1);

            ValueType sum = zero<ValueType>();

            for (int A_col_idx = A_col_idx_last; A_col_idx > A_row_idx;
                 A_col_idx--) {
                // sum += A(A_row_idx, A_col_idx) * x(A_col_idx)
                const int AB_corr_row_idx = KL + KU + A_row_idx - A_col_idx;

                sum += AB_array_entry[AB_corr_row_idx + A_col_idx * AB_nrows] *
                       x_sh_entry[A_col_idx];
            }

            // x(A_row_idx) = (y(A_row_idx) - sum)/A(A_row_idx, A_row_idx);
            const int AB_row_idx_diag_ele = KL + KU;

            const ValueType diag_ele =
                AB_array_entry[AB_row_idx_diag_ele + A_row_idx * AB_nrows];

            x_sh_entry[A_row_idx] = (y_sh_entry[A_row_idx] - sum) / diag_ele;
        }
    }
    // No parallelism
    // No data locality since AB arr is in col major order....
}


template <typename ValueType>
__device__ __forceinline__ void upper_trsv(
    const int N, const int KL, const int KU,
    const ValueType* const __restrict__ AB_array_entry,
    const ValueType* const __restrict__ y_sh_entry,
    ValueType* const __restrict__ x_sh_entry)
{
    // naive_upper_trsv(N, KL, KU, AB_array_entry, y_sh_entry, x_sh_entry);
    const int AB_nrows = 2 * KL + KU + 1;
    const int KV = KL + KU;

    single_copy(N, y_sh_entry, x_sh_entry);
    __syncthreads();

    if (threadIdx.x == 0) {
        x_sh_entry[N - 1] /= AB_array_entry[KV + (N - 1) * AB_nrows];
    }
    __syncthreads();


    for (int j = N - 1; j > 0; j--) {
        const int A_col_idx = j;
        const auto x_at_row_j = x_sh_entry[j];

        for (int A_row_idx = j - 1 - threadIdx.x;
             A_row_idx >= max(0, j - KL - KU); A_row_idx -= blockDim.x) {
            const int AB_row_idx = KL + KU + A_row_idx - A_col_idx;
            // coalesced accesses
            if (A_row_idx == j - 1) {
                x_sh_entry[A_row_idx] =
                    (-1 * AB_array_entry[AB_row_idx + A_col_idx * AB_nrows] *
                         x_at_row_j +
                     x_sh_entry[A_row_idx]) /
                    AB_array_entry[KV + (A_col_idx - 1) * AB_nrows];
            } else {
                x_sh_entry[A_row_idx] +=
                    -1 * AB_array_entry[AB_row_idx + A_col_idx * AB_nrows] *
                    x_at_row_j;
            }
        }

        // case: A is a diagonal matrix
        if (KL == 0 & KU == 0)  // could be done - more efficiently of course -
                                // but the focus here is not this -
        // as this path is never used; there's another lightweight
        // implementation for the case of diagonal matrix solver.
        {
            if (threadIdx.x == 0) {
                x_sh_entry[j - 1] /= AB_array_entry[KV + (j - 1) * AB_nrows];
            }
        }

        __syncthreads();
    }
}

}  // namespace unblocked_and_blocked_band_solver_common


template <int subwarp_size, typename ValueType>
__global__ void band_solver_unblocked_kernel(const size_type nbatch,
                                             const int N, const int KL,
                                             const int KU,
                                             ValueType* const batch_band_array,
                                             const ValueType* const b,
                                             ValueType* const x)
{
    const int AB_nrows = 2 * KL + KU + 1;

    constexpr auto tile_size = subwarp_size;
    auto subwarp_grp =
        group::tiled_partition<tile_size>(group::this_thread_block());

    for (int batch_idx = blockIdx.x; batch_idx < nbatch;
         batch_idx += blockDim.x) {
        extern __shared__ char local_mem_sh[];
        ValueType* const __restrict__ x_sh_entry =
            reinterpret_cast<ValueType*>(local_mem_sh);
        ValueType* const __restrict__ temp_sh_entry =
            reinterpret_cast<ValueType*>(
                x_sh_entry + N);  // Will reuse this for vector y in trsv
        int* const __restrict__ ipiv_sh_entry =
            reinterpret_cast<int*>(temp_sh_entry + N);

        const ValueType* const __restrict__ b_global_entry =
            gko::batch::batch_entry_ptr(b, 1, N, batch_idx);
        ValueType* const __restrict__ x_global_entry =
            gko::batch::batch_entry_ptr(x, 1, N, batch_idx);
        ValueType* const __restrict__ AB_array_entry =
            batch_band_array + batch_idx * (N * AB_nrows);

        unblocked_LU_band_factorization::factorize_band(
            subwarp_grp, N, KL, KU, AB_array_entry, ipiv_sh_entry,
            temp_sh_entry);

        unblocked_and_blocked_band_solver_common::
            accommodate_partial_row_swaps_lower_trsv(
                N, KL, KU, ipiv_sh_entry, AB_array_entry, b_global_entry,
                temp_sh_entry);

        __syncthreads();

        unblocked_and_blocked_band_solver_common::upper_trsv(
            N, KL, KU, AB_array_entry, temp_sh_entry, x_sh_entry);

        __syncthreads();

        // copy x back to global memory
        single_copy(N, x_sh_entry, x_global_entry);
    }
}
