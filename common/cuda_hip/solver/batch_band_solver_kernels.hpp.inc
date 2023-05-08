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

namespace common_band_factorization {

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
    const int A_col_idx_start_inclusive, const int A_col_idx_end_inclusive,
    const int A_row_idx_1, const int A_row_idx_2)
{
    const int AB_nrows = 2 * KL + KU + 1;

    // uncoalesced access, also no data locality
    for (int A_col_idx = A_col_idx_start_inclusive + threadIdx.x;
         A_col_idx <= A_col_idx_end_inclusive; A_col_idx += blockDim.x) {
        const int AB_row_idx_1 = KL + KU + A_row_idx_1 - A_col_idx;
        const int AB_row_idx_2 = KL + KU + A_row_idx_2 - A_col_idx;
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
    const int AB_col_idx, const int A_col_idx_end_inclusive)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int A_diag_row_idx = AB_col_idx;

    // fill the ele_above_sh_arr_entry
    // this is to reduce uncoalesced global memory reads while updating the
    // submatrix
    for (int A_col_idx = A_diag_row_idx + 1 + threadIdx.x;
         A_col_idx <= A_col_idx_end_inclusive; A_col_idx += blockDim.x) {
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
             A_col_idx <= A_col_idx_end_inclusive; A_col_idx++) {
            const ValueType ele_above = ele_above_sh_arr_entry[A_col_idx];

            const int AB_row_idx_for_this_col_ele =
                KL + KU + A_row_idx - A_col_idx;
            AB_array_entry[AB_row_idx_for_this_col_ele +
                           AB_nrows * A_col_idx] -=
                ele_above * ele_side;  // coalesced accesses
        }
    }
}

}  // namespace common_band_factorization

namespace unblocked_LU_band_factorization {

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
    common_band_factorization::set_possible_fill_in_zero_in_U_part(
        subwarp_grp, N, KL, KU, AB_array_entry);
    __syncthreads();

    int A_affected_col_idx_last = 0;

    for (int AB_col_idx = 0; AB_col_idx <= N - 1; AB_col_idx++) {
        const int AB_diag_row_idx = KV;

        __syncthreads();
        const int AB_pivot_row_idx =
            common_band_factorization::find_pivot_row_index(
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
            common_band_factorization::swap_rows(
                N, KL, KU, AB_array_entry, AB_col_idx, A_affected_col_idx_last,
                A_diag_row_idx, A_pivot_row_idx);
        }

        // record the swap row info...
        if (threadIdx.x == 0) {
            ipiv_sh_entry[A_diag_row_idx] = A_pivot_row_idx;
        }
        __syncthreads();

        // scale A's column and update trailing matrix
        common_band_factorization::modify_rows_below(
            N, KL, KU, temp_sh_entry, piv_val, AB_array_entry, AB_col_idx,
            A_affected_col_idx_last);

        __syncthreads();
    }
}

}  // namespace unblocked_LU_band_factorization


namespace blocked_LU_band_factorization {

template <typename ValueType>
__device__ __forceinline__ void apply_row_interchanges_A01_A11_A21(
    const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry, const int j, const int JB,
    const int J1, const int* const __restrict__ ipiv_sh_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    for (int a = j; a < j + JB; a++) {
        const int A_row_1 = a;
        const int A_row_2 = ipiv_sh_entry[a];

        if (A_row_1 != A_row_2) {
            // uncoalesced accesses and no data locality
            for (int col_idx = j + JB + threadIdx.x; col_idx < j + JB + J1;
                 col_idx += blockDim.x) {
                const int AB_row_1 = KL + KU + A_row_1 - col_idx;
                const int AB_row_2 = KL + KU + A_row_2 - col_idx;
                const ValueType temp =
                    AB_array_entry[AB_row_1 + AB_nrows * col_idx];
                AB_array_entry[AB_row_1 + AB_nrows * col_idx] =
                    AB_array_entry[AB_row_2 + AB_nrows * col_idx];
                AB_array_entry[AB_row_2 + AB_nrows * col_idx] = temp;
            }

            __syncthreads();
        }
    }
}

template <typename ValueType>
__device__ __forceinline__ void apply_row_interchanges_A02_A12_A22(
    const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry, const int j, const int JB,
    const int J1, const int J2, const int* const __restrict__ ipiv_sh_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    // apply the row interchanges to A02, A12, and A22 columnwise

    // uncoalesced accesses, no data locality
    for (int A_col = j + JB + J1 + threadIdx.x; A_col < j + JB + J1 + J2;
         A_col += blockDim.x) {
        for (int a = max(j, A_col - KL - KU); a < j + JB; a++) {
            const int A_row_1 = a;
            const int A_row_2 = ipiv_sh_entry[a];

            if (A_row_1 != A_row_2) {
                const int AB_row_1 = KL + KU + A_row_1 - A_col;
                const int AB_row_2 = KL + KU + A_row_2 - A_col;
                const ValueType temp =
                    AB_array_entry[AB_row_1 + AB_nrows * A_col];
                AB_array_entry[AB_row_1 + AB_nrows * A_col] =
                    AB_array_entry[AB_row_2 + AB_nrows * A_col];
                AB_array_entry[AB_row_2 + AB_nrows * A_col] = temp;
            }
        }
    }
}


template <int subwarpgrp_size, typename ValueType>
__device__ __forceinline__ void lower_trsv_update_A01(
    const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry, const int j, const int JB,
    const int J1)
{
    const int AB_nrows = 2 * KL + KU + 1;
    /*
    //dense lower trsv-> sequential
    for(int A_col = j + JB + threadIdx.x; A_col < j + JB + J1; A_col +=
    blockDim.x)
    {
        for(int A_row = j; A_row < j + JB; A_row++)
        {
            ValueType sum = zero<ValueType>();

            for(int i = 0; i < A_row - j; i++)
            {
                // In first-> no data locality, but in second: there is data
    locality sum += AB_array_entry[ (KL + KU +   i + j   - A_col)  +  AB_nrows *
    A_col ] * AB_array_entry[ (KL + KU +  A_row - (i + j) ) + AB_nrows * (i +
    j)];
            }

            AB_array_entry[(KL + KU + A_row - A_col)  +  AB_nrows * A_col] -=
    sum; //data locality
        }
    }*/


    auto subwarpgrp =
        group::tiled_partition<subwarpgrp_size>(group::this_thread_block());

    const int subwarp_id = threadIdx.x / subwarpgrp_size;
    const int total_num_subwarp_grps = (blockDim.x) / subwarpgrp_size;
    const int local_id = subwarpgrp.thread_rank();

    for (int A_col = j + JB + subwarp_id; A_col < j + JB + J1;
         A_col += total_num_subwarp_grps) {
        for (int A_row = j; A_row < j + JB; A_row++) {
            ValueType sum = zero<ValueType>();

            subwarpgrp.sync();

            for (int i = 0 + local_id; i < A_row - j; i += subwarpgrp_size) {
                sum += AB_array_entry[(KL + KU + i + j - A_col) +
                                      AB_nrows * A_col] *
                       AB_array_entry[(KL + KU + A_row - (i + j)) +
                                      AB_nrows * (i + j)];
            }

            // subwarp level reduction
            for (int offset = subwarpgrp_size / 2; offset > 0;
                 offset = offset / 2) {
                sum += subwarpgrp.shfl_down(sum, offset);
            }

            if (local_id == 0) {
                AB_array_entry[(KL + KU + A_row - A_col) + AB_nrows * A_col] -=
                    sum;
            }
        }
    }
}

template <int subwarpgrp_size, typename ValueType>
__device__ __forceinline__ void lower_trsv_update_work02(
    const int N, const int KL, const int KU,
    const ValueType* const __restrict__ AB_array_entry, const int j,
    const int J2, const int JB, const int NB,
    ValueType* const __restrict__ work02_sh_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    /*
    for(int c = 0 + threadIdx.x; c < J2; c += blockDim.x)
    {
        for(int r = 0; r < JB ; r++)
        {
            ValueType sum = zero<ValueType>();

            for(int i = 0; i < r; i++)
            {
                const int A_col = i + j;
                sum += AB_array_entry[ (KL + KU + (r + j) - A_col ) + AB_nrows *
    A_col  ] * work02_sh_entry[i + c * NB];
            }

            work02_sh_entry[r + c * NB] -= sum;
        }
    }
    */
    auto subwarpgrp =
        group::tiled_partition<subwarpgrp_size>(group::this_thread_block());

    const int subwarp_id = threadIdx.x / subwarpgrp_size;
    const int total_num_subwarp_grps = (blockDim.x) / subwarpgrp_size;
    const int local_id = subwarpgrp.thread_rank();

    for (int c = 0 + subwarp_id; c < J2; c += total_num_subwarp_grps) {
        for (int r = 0; r < JB; r++) {
            ValueType sum = zero<ValueType>();

            subwarpgrp.sync();

            for (int i = local_id; i < r; i += subwarpgrp_size) {
                const int A_col = i + j;
                sum += AB_array_entry[(KL + KU + (r + j) - A_col) +
                                      AB_nrows * A_col] *
                       work02_sh_entry[i + c * NB];
            }

            // subwarp level reduction
            for (int offset = subwarpgrp_size / 2; offset > 0;
                 offset = offset / 2) {
                sum += subwarpgrp.shfl_down(sum, offset);
            }

            if (local_id == 0) {
                work02_sh_entry[r + c * NB] -= sum;
            }
        }
    }
}

template <int subwarpgrp_size, typename ValueType, typename Callable>
__device__ __forceinline__ void update_part_of_trailing_matrix(
    const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry, const int trail_row_start,
    const int trail_row_end_exclusive, const int trail_col_start,
    const int trail_col_end_exclusive, Callable compute_sum)
{
    const int AB_nrows = 2 * KL + KU + 1;

    auto subwarpgrp =
        group::tiled_partition<subwarpgrp_size>(group::this_thread_block());

    const int subwarp_id = threadIdx.x / subwarpgrp_size;
    const int total_num_subwarp_grps = (blockDim.x) / subwarpgrp_size;
    const int local_id = subwarpgrp.thread_rank();

    const int num_limit = (trail_row_end_exclusive - trail_row_start) *
                          (trail_col_end_exclusive - trail_col_start);

    for (int i = subwarp_id; i < num_limit; i += total_num_subwarp_grps) {
        const int trail_row =
            trail_row_start + (i % (trail_row_end_exclusive - trail_row_start));
        const int trail_col =
            trail_col_start + (i / (trail_row_end_exclusive - trail_row_start));

        ValueType temp = compute_sum(trail_row, trail_col, subwarpgrp);
        subwarpgrp.sync();
        // subwarp level reduction
        for (int offset = subwarpgrp_size / 2; offset > 0;
             offset = offset / 2) {
            temp += subwarpgrp.shfl_down(temp, offset);
        }
        if (local_id == 0) {
            AB_array_entry[(KL + KU + trail_row - trail_col) +
                           trail_col * AB_nrows] -= temp;
        }
    }
}


template <int subwarpgrp_size, typename ValueType>
__device__ __forceinline__ void update_A11(
    const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry, const int j, const int JB,
    const int I1, const int J1)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int trail_row_start = j + JB;
    const int trail_row_end_exclusive = j + JB + I1;
    const int trail_col_start = j + JB;
    const int trail_col_end_exclusive = j + JB + J1;

    using Group = group::thread_block_tile<subwarpgrp_size>;

    auto compute_sum = [=](const int trail_row, const int trail_col,
                           Group& subwarp_grp) {
        ValueType temp = zero<ValueType>();

        // coalesced accesses: second AB, but first AB --> uncoalesced accesses
        for (int k = subwarp_grp.thread_rank(); k < JB;
             k += subwarp_grp.size()) {
            const int c = k + j;
            const int r = k + j;
            temp += AB_array_entry[(KL + KU + trail_row - c) + c * AB_nrows] *
                    AB_array_entry[(KL + KU + r - trail_col) +
                                   trail_col * AB_nrows];
        }

        return temp;
    };

    update_part_of_trailing_matrix<subwarpgrp_size>(
        N, KL, KU, AB_array_entry, trail_row_start, trail_row_end_exclusive,
        trail_col_start, trail_col_end_exclusive, compute_sum);
}

template <int subwarpgrp_size, typename ValueType>
__device__ __forceinline__ void update_A21(
    const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry, const int j, const int JB,
    const int I1, const int I2, const int J1, const int NB,
    const ValueType* const __restrict__ work20_sh_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int trail_row_start = j + JB + I1;
    const int trail_row_end_exclusive = j + JB + I1 + I2;
    const int trail_col_start = j + JB;
    const int trail_col_end_exclusive = j + JB + J1;

    using Group = group::thread_block_tile<subwarpgrp_size>;

    auto compute_sum = [=](const int trail_row, const int trail_col,
                           Group& subwarp_grp) {
        ValueType temp = zero<ValueType>();
        // coalesced accesses: AB, but work20 -> coalesced accesses
        for (int k = subwarp_grp.thread_rank(); k < JB;
             k += subwarp_grp.size()) {
            const int r = k + j;
            temp += work20_sh_entry[(trail_row - (trail_row_start)) * NB + k] *
                    AB_array_entry[(KL + KU + r - trail_col) +
                                   trail_col * AB_nrows];
        }

        return temp;
    };

    update_part_of_trailing_matrix<subwarpgrp_size>(
        N, KL, KU, AB_array_entry, trail_row_start, trail_row_end_exclusive,
        trail_col_start, trail_col_end_exclusive, compute_sum);
}


template <int subwarpgrp_size, typename ValueType>
__device__ __forceinline__ void update_A12(
    const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry, const int j, const int JB,
    const int I1, const int J1, const int J2, const int NB,
    const ValueType* const __restrict__ work02_sh_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    const int trail_row_start = j + JB;
    const int trail_row_end_exclusive = j + JB + I1;
    const int trail_col_start = j + JB + J1;
    const int trail_col_end_exclusive = j + JB + J1 + J2;

    using Group = group::thread_block_tile<subwarpgrp_size>;

    auto compute_sum = [=](const int trail_row, const int trail_col,
                           Group& subwarp_grp) {
        ValueType temp = zero<ValueType>();

        // uncoalesced accesses: AB, but work02 --> coalesced accesses
        for (int k = subwarp_grp.thread_rank(); k < JB;
             k += subwarp_grp.size()) {
            const int c = k + j;
            const int offset_col = j + JB + J1;
            temp += AB_array_entry[(KL + KU + trail_row - c) + c * AB_nrows] *
                    work02_sh_entry[k + (trail_col - offset_col) * NB];
        }

        return temp;
    };

    update_part_of_trailing_matrix<subwarpgrp_size>(
        N, KL, KU, AB_array_entry, trail_row_start, trail_row_end_exclusive,
        trail_col_start, trail_col_end_exclusive, compute_sum);
}

template <int subwarpgrp_size, typename ValueType>
__device__ __forceinline__ void update_A22(
    const int N, const int KL, const int KU,
    ValueType* const __restrict__ AB_array_entry, const int j, const int JB,
    const int I1, const int I2, const int J1, const int J2, const int NB,
    const ValueType* const __restrict__ work20_sh_entry,
    const ValueType* const __restrict__ work02_sh_entry)
{
    const int trail_row_start = j + JB + I1;
    const int trail_row_end_exclusive = j + JB + I1 + I2;
    const int trail_col_start = j + JB + J1;
    const int trail_col_end_exclusive = j + JB + J1 + J2;
    using Group = group::thread_block_tile<subwarpgrp_size>;
    auto compute_sum = [=](const int trail_row, const int trail_col,
                           Group& subwarp_grp) {
        ValueType temp = zero<ValueType>();

        // work20 -> coalesced accesses,  work02 --> coalesced accesses
        for (int k = subwarp_grp.thread_rank(); k < JB;
             k += subwarp_grp.size()) {
            const int offset_col = j + JB + J1;
            const int offset_row = j + JB + I1;
            temp += work20_sh_entry[(trail_row - offset_row) * NB + k] *
                    work02_sh_entry[k + (trail_col - offset_col) * NB];
        }

        return temp;
    };

    update_part_of_trailing_matrix<subwarpgrp_size>(
        N, KL, KU, AB_array_entry, trail_row_start, trail_row_end_exclusive,
        trail_col_start, trail_col_end_exclusive, compute_sum);
}

template <int subwarp_size, typename ValueType>
__device__ __forceinline__ void factorize_band(
    const int N, const int KL, const int KU, const int NB,
    ValueType* const __restrict__ AB_array_entry,
    int* const __restrict__ ipiv_sh_entry,
    ValueType* const __restrict__ work02_sh_entry,
    ValueType* const __restrict__ work20_sh_entry,
    ValueType* const __restrict__ temp_sh_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int KV = KU + KL;

    auto subwarp_grp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int subwarp_id = threadIdx.x / subwarp_size;
    const int total_num_subwarp_grps = (blockDim.x) / subwarp_size;

    // write zeroes into the possible fill-in U part
    common_band_factorization::set_possible_fill_in_zero_in_U_part(
        subwarp_grp, N, KL, KU, AB_array_entry);

    for (int i = threadIdx.x; i < NB * NB; i += blockDim.x) {
        work02_sh_entry[i] = zero<ValueType>();
        work20_sh_entry[i] = zero<ValueType>();
    }

    __syncthreads();

    int A_affected_col_idx_last = 0;

    for (int j = 0; j < N; j += NB) {
        const int JB = min(NB, N - j);

        /*
         *      The active part of the matrix is partitioned
         *
         *              A00   A01   A02
         *              A10   A11   A12
         *              A20   A21   A22
         *
         *           Here A00, A10 and A20 denote the current block of JB
         * columns which is about to be factorized. The number of rows in the
         *           partitioning are JB, I1, I2 respectively, and the numbers
         *           of columns are JB, J1, J1. The superdiagonal elements of
         * A02 and the subdiagonal elements of A20 lie outside the band.
         *
         */

        const int I1 = min(KL - JB, N - j - JB);
        const int I2 = min(JB, N - j - KL);

        // J1 and J1 are computed after A_affected_col_idx_last has been
        // updated.

        // Factorize the current block of JB columns
        for (int jj = j; jj < j + JB; jj++) {
            // find pivot row
            __syncthreads();
            const int AB_pivot_row_idx =
                common_band_factorization::find_pivot_row_index(
                    subwarp_grp, N, KL, KU, AB_array_entry, jj);
            const int A_pivot_row_idx = AB_pivot_row_idx - KV + jj;

            // record the swap info...
            if (threadIdx.x == 0) {
                ipiv_sh_entry[jj] = A_pivot_row_idx;
            }

            const ValueType piv_val =
                AB_array_entry[(KL + KU + A_pivot_row_idx - jj) +
                               AB_nrows * jj];
            assert(piv_val != zero<ValueType>());

            A_affected_col_idx_last =
                max(A_affected_col_idx_last, min(A_pivot_row_idx + KU, N - 1));

            if (A_pivot_row_idx != jj) {
                __syncthreads();
                // swap: row swaps for columns j to j + JB - 1(inclusive)
                if (A_pivot_row_idx <= j + JB - 1 + I1) {
                    // swap in band array
                    common_band_factorization::swap_rows(
                        N, KL, KU, AB_array_entry, j, j + JB - 1, jj,
                        A_pivot_row_idx);
                } else {
                    // for cols. j to jj - 1 --> do in work20
                    // Uncoalesced accesses, and also no data locality
                    for (int A_col_idx = j + threadIdx.x; A_col_idx <= jj - 1;
                         A_col_idx += blockDim.x) {
                        const int AB_row_idx1 = KL + KU + jj - A_col_idx;
                        ValueType temp =
                            AB_array_entry[AB_row_idx1 + AB_nrows * A_col_idx];
                        const int r = A_pivot_row_idx - (j + JB + I1);
                        AB_array_entry[AB_row_idx1 + AB_nrows * A_col_idx] =
                            work20_sh_entry[r * NB + (A_col_idx - j)];
                        work20_sh_entry[r * NB + (A_col_idx - j)] = temp;
                    }

                    // for cols. jj to j + JB -1 --> can do in band array
                    common_band_factorization::swap_rows(
                        N, KL, KU, AB_array_entry, jj, j + JB - 1, jj,
                        A_pivot_row_idx);
                }
            }

            __syncthreads();

            // Modify the rows below in current col. panel (within the band)
            common_band_factorization::modify_rows_below(
                N, KL, KU, temp_sh_entry, piv_val, AB_array_entry, jj,
                j + JB - 1);
            __syncthreads();

            // Copy current column of A20 into the work array work20
            for (int A_row = j + JB + I1 + threadIdx.x;
                 A_row <= min(jj + KL, N - 1); A_row += blockDim.x) {
                const int AB_row = KL + KU + A_row - jj;
                const ValueType val =
                    AB_array_entry[AB_row +
                                   AB_nrows * jj];  // coalesced accesses
                const int r = A_row - (j + JB + I1);
                work20_sh_entry[r * NB + (jj - j)] =
                    val;  // uncoalesced accesses
            }

            __syncthreads();
        }
        // completed factorizing the current column panel

        // Apply row interchanges to other blocks
        const int J1 = min(A_affected_col_idx_last - j + 1, KV) - JB;
        const int J2 = max(0, A_affected_col_idx_last - j - KV + 1);

        // apply the row interchanges to A01, A11, and A21
        apply_row_interchanges_A01_A11_A21(N, KL, KU, AB_array_entry, j, JB, J1,
                                           ipiv_sh_entry);

        // apply the row interchanges to A02, A12, and A22 columnwise
        apply_row_interchanges_A02_A12_A22(N, KL, KU, AB_array_entry, j, JB, J1,
                                           J2, ipiv_sh_entry);

        __syncthreads();

        if (J1 > 0) {
            // lower trsv - update A01
            lower_trsv_update_A01<subwarp_size>(N, KL, KU, AB_array_entry, j,
                                                JB, J1);
            __syncthreads();

            if (I1 > 0) {
                update_A11<subwarp_size>(N, KL, KU, AB_array_entry, j, JB, I1,
                                         J1);
            }

            if (I2 > 0) {
                update_A21<subwarp_size>(N, KL, KU, AB_array_entry, j, JB, I1,
                                         I2, J1, NB, work20_sh_entry);
            }
        }

        if (J2 > 0) {
            // Copy part of A02 into the work array WORK02 (guess its lower
            // triangular)
            for (int A_col = j + JB + J1 + subwarp_id; A_col < j + JB + J1 + J2;
                 A_col += total_num_subwarp_grps) {
                const int A_row_start = max(0, A_col - KL - KU);
                for (int A_row = A_row_start + subwarp_grp.thread_rank();
                     A_row < j + JB; A_row += subwarp_grp.size()) {
                    const int AB_row = KL + KU + A_row - A_col;
                    const ValueType val =
                        AB_array_entry[AB_row +
                                       AB_nrows * A_col];  // coalesced accesses
                    const int r = A_row - j;
                    const int c = A_col - (j + JB + J1);
                    work02_sh_entry[r + c * NB] = val;  // coalesced accesses
                }
            }
            __syncthreads();

            lower_trsv_update_work02<subwarp_size>(N, KL, KU, AB_array_entry, j,
                                                   J2, JB, NB, work02_sh_entry);
            __syncthreads();

            if (I1 > 0) {
                update_A12<subwarp_size>(N, KL, KU, AB_array_entry, j, JB, I1,
                                         J1, J2, NB, work02_sh_entry);
            }

            if (I2 > 0) {
                update_A22<subwarp_size>(N, KL, KU, AB_array_entry, j, JB, I1,
                                         I2, J1, J2, NB, work20_sh_entry,
                                         work02_sh_entry);
            }

            __syncthreads();

            // Copy part of A02 back into place (guess, its lower triangular)
            for (int A_col = j + JB + J1 + subwarp_id; A_col < j + JB + J1 + J2;
                 A_col += total_num_subwarp_grps) {
                const int A_row_start = max(0, A_col - KL - KU);
                for (int A_row = A_row_start + subwarp_grp.thread_rank();
                     A_row < j + JB; A_row += subwarp_grp.size()) {
                    const int AB_row = KL + KU + A_row - A_col;
                    const int r = A_row - j;
                    const int c = A_col - (j + JB + J1);
                    const ValueType val = work02_sh_entry[r + c * NB];
                    AB_array_entry[AB_row + AB_nrows * A_col] = val;
                }
            }
        }
        __syncthreads();

        // undo partial row interchanges in the column panel - A00, A10, A20
        for (int a = j + JB - 1; a >= j; a--) {
            const int A_row_1 = a;
            const int A_row_2 = ipiv_sh_entry[a];

            // swap rows for cols. j to a - 1 (inclusive)

            // uncoalesced accesses
            if (A_row_2 <= j + JB + I1 - 1) {
                common_band_factorization::swap_rows(
                    N, KL, KU, AB_array_entry, j, a - 1, A_row_1, A_row_2);
            } else {
                // swap from col. j to a-1 (inclusive)
                for (int A_col = j + threadIdx.x; A_col <= a - 1;
                     A_col += blockDim.x) {
                    const int AB_row_idx1 = KL + KU + A_row_1 - A_col;
                    ValueType temp =
                        AB_array_entry[AB_row_idx1 + AB_nrows * A_col];
                    const int r = A_row_2 - (j + JB + I1);
                    AB_array_entry[AB_row_idx1 + AB_nrows * A_col] =
                        work20_sh_entry[r * NB + (A_col - j)];
                    work20_sh_entry[r * NB + (A_col - j)] = temp;
                }
            }

            __syncthreads();
        }

        // copy relevant part of work20 to A20
        for (int A_col = j + subwarp_id; A_col < j + JB;
             A_col += total_num_subwarp_grps) {
            for (int A_row = j + JB + I1 + subwarp_grp.thread_rank();
                 A_row < min(A_col + KL + 1, N); A_row += subwarp_grp.size()) {
                const int AB_row = KL + KU + A_row - A_col;
                const int r = A_row - (j + JB + I1);
                const int c = A_col - j;
                const ValueType val =
                    work20_sh_entry[r * NB + c];  // uncoalesced accesses
                AB_array_entry[AB_row + A_col * AB_nrows] =
                    val;  // coalesced accesses
            }
        }
        __syncthreads();
    }
}

}  // namespace blocked_LU_band_factorization


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
        if (KL == 0 && KU == 0)  // could be done - more efficiently of course -
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

        ValueType* __restrict__ AB_array_entry;
        if (is_matrix_in_shared_mem(N, KL, KU)) {
            AB_array_entry = reinterpret_cast<ValueType*>(ipiv_sh_entry + N);
            single_copy(N * AB_nrows,
                        batch_band_array + batch_idx * (N * AB_nrows),
                        AB_array_entry);
        } else {
            AB_array_entry = batch_band_array + batch_idx * (N * AB_nrows);
        }
        __syncthreads();

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


template <int subwarp_size, typename ValueType>
__global__ void band_solver_blocked_kernel(const size_type nbatch, const int N,
                                           const int KL, const int KU,
                                           const int NB,
                                           ValueType* const batch_band_array,
                                           const ValueType* const b,
                                           ValueType* const x)
{
    const int AB_nrows = 2 * KL + KU + 1;

    for (int batch_idx = blockIdx.x; batch_idx < nbatch;
         batch_idx += blockDim.x) {
        extern __shared__ char local_mem_sh[];
        ValueType* const __restrict__ x_sh_entry =
            reinterpret_cast<ValueType*>(local_mem_sh);
        ValueType* const __restrict__ temp_sh_entry =
            x_sh_entry + N;  // Will reuse this for vector y in trsv

        ValueType* const __restrict__ work02_sh_entry = temp_sh_entry + N;
        ValueType* const __restrict__ work20_sh_entry =
            work02_sh_entry + NB * NB;
        int* const __restrict__ ipiv_sh_entry =
            reinterpret_cast<int*>(work20_sh_entry + NB * NB);

        const ValueType* const __restrict__ b_global_entry =
            gko::batch::batch_entry_ptr(b, 1, N, batch_idx);
        ValueType* const __restrict__ x_global_entry =
            gko::batch::batch_entry_ptr(x, 1, N, batch_idx);

        // ValueType* const __restrict__ AB_array_entry =
        //     batch_band_array + batch_idx * (N * AB_nrows);

        ValueType* __restrict__ AB_array_entry;
        if (is_matrix_in_shared_mem(N, KL, KU)) {
            AB_array_entry = reinterpret_cast<ValueType*>(ipiv_sh_entry + N);
            single_copy(N * AB_nrows,
                        batch_band_array + batch_idx * (N * AB_nrows),
                        AB_array_entry);
        } else {
            AB_array_entry = batch_band_array + batch_idx * (N * AB_nrows);
        }
        __syncthreads();

        blocked_LU_band_factorization::factorize_band<subwarp_size>(
            N, KL, KU, NB, AB_array_entry, ipiv_sh_entry, work02_sh_entry,
            work20_sh_entry, temp_sh_entry);

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
