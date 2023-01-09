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

template <int subwarp_size>
__global__
    __launch_bounds__(default_block_size) void extract_dense_linear_sys_pattern_kernel(
        const int nrows, const int* const __restrict__ A_row_ptrs,
        const int* const __restrict__ A_col_idxs,
        const int* const __restrict__ aiA_row_ptrs,
        const int* const __restrict__ aiA_col_idxs,
        int* const dense_mat_pattern, int* const rhs_one_idxs, int* const sizes)
{
    using gko::preconditioner::batch_isai::row_size_limit;
    // assert(subwarp_size >= row_size_limit); //Not required here
    auto subwarpgrp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int subgrpwarp_id_in_grid =
        thread::get_subwarp_id_flat<subwarp_size, int>();
    const int total_num_subwarp_grps_in_grid =
        thread::get_subwarp_num_flat<subwarp_size, int>();
    const int id_within_warp = subwarpgrp.thread_rank();

    int ele_id_assigned_to_subwarpgrp = subgrpwarp_id_in_grid;

    for (int aiA_row = 0; aiA_row < nrows; aiA_row++) {
        int* const dense_mat_ptr =
            dense_mat_pattern + row_size_limit * row_size_limit * aiA_row;
        const int aiA_row_st = aiA_row_ptrs[aiA_row];
        const int aiA_row_end = aiA_row_ptrs[aiA_row + 1];

        for (int aiA_nz_id = aiA_row_st; aiA_nz_id < aiA_row_end; aiA_nz_id++) {
            if (ele_id_assigned_to_subwarpgrp != aiA_nz_id) {
                continue;
            }

            const int i_size = aiA_row_end - aiA_row_st;

            if (aiA_nz_id == aiA_row_st && subwarpgrp.thread_rank() == 0) {
                sizes[aiA_row] = i_size;
            }

            const int aiA_col = aiA_col_idxs[aiA_nz_id];

            if (aiA_col == aiA_row && subwarpgrp.thread_rank() == 0) {
                rhs_one_idxs[aiA_row] = aiA_nz_id - aiA_row_st;
            }

            const int A_row = aiA_col;
            const int A_row_st = A_row_ptrs[A_row];
            const int A_row_end = A_row_ptrs[A_row + 1];
            const int dense_sys_row = aiA_nz_id - aiA_row_st;

            assert(i_size <= row_size_limit);

            int* const dense_ptr_for_this_row =
                dense_mat_ptr + dense_sys_row * row_size_limit;

            auto do_this_on_match = [dense_ptr_for_this_row, aiA_row_st,
                                     A_row_st](
                                        int, int aiA_idx_match, int A_idx_match,
                                        config::lane_mask_type, bool valid) {
                if (valid) {
                    dense_ptr_for_this_row[aiA_idx_match] =
                        A_idx_match + A_row_st;
                }
            };

            // warp-parallel match algorithm
            group_match<subwarp_size>(
                aiA_col_idxs + aiA_row_st, aiA_row_end - aiA_row_st,
                A_col_idxs + A_row_st, A_row_end - A_row_st, subwarpgrp,
                do_this_on_match);

            subwarpgrp.sync();
            ele_id_assigned_to_subwarpgrp += total_num_subwarp_grps_in_grid;
        }
    }
}

template <typename Group, typename ValueType>
__device__ __forceinline__ ValueType
solve_lower_tri_dense_system(Group subwarpgrp, const int size,
                             ValueType* const local_row, const int rhs_one_idx)
{
    const int local_id = subwarpgrp.thread_rank();
    const ValueType rhs =
        local_id == rhs_one_idx ? one<ValueType>() : zero<ValueType>();
    ValueType sol = rhs;

    for (int dense_col_idx = 0; dense_col_idx < size; dense_col_idx++) {
        const ValueType ele = local_row[dense_col_idx];

        if (dense_col_idx == local_id) {
            sol = sol / ele;
        }

        subwarpgrp.sync();
        const ValueType top = subwarpgrp.shfl(sol, dense_col_idx);

        if (local_id > dense_col_idx) {
            sol = sol - top * ele;
        }
    }

    return sol;
}


template <typename Group, typename ValueType>
__device__ __forceinline__ ValueType
solve_upper_tri_dense_system(Group subwarpgrp, const int size,
                             ValueType* const local_row, const int rhs_one_idx)
{
    const int local_id = subwarpgrp.thread_rank();
    const ValueType rhs =
        local_id == rhs_one_idx ? one<ValueType>() : zero<ValueType>();
    ValueType sol = rhs;

    for (int dense_col_idx = size - 1; dense_col_idx >= 0; dense_col_idx--) {
        const ValueType ele = local_row[dense_col_idx];

        if (dense_col_idx == local_id) {
            sol = sol / ele;
        }

        subwarpgrp.sync();
        const ValueType bot = subwarpgrp.shfl(sol, dense_col_idx);

        if (local_id < dense_col_idx) {
            sol = sol - bot * ele;
        }
    }

    return sol;
}


template <typename Group, typename ValueType>
__device__ __forceinline__ int choose_pivot_row(
    Group subwarpgrp, const int diag_pos, const int size,
    const ValueType* const local_row)
{
    const int local_id = subwarpgrp.thread_rank();
    ValueType my_ele = local_row[diag_pos];
    int my_idx = local_id;
    const ValueType my_ele_1 = subwarpgrp.shfl(local_row[diag_pos], diag_pos);

    if (local_id < diag_pos || local_id >= size) {
        my_ele = my_ele_1;
        my_idx = diag_pos;
    }

    subwarpgrp.sync();

    for (int offset = subwarpgrp.size() / 2; offset > 0; offset /= 2) {
        ValueType other_ele = subwarpgrp.shfl_down(my_ele, offset);
        int other_idx = subwarpgrp.shfl_down(my_idx, offset);
        if (abs(other_ele) > abs(my_ele)) {
            my_ele = other_ele;
            my_idx = other_idx;
        }
        subwarpgrp.sync();
    }

    // thread 0 would have the correct piv_row_idx and piv_ele
    const int piv_row_idx = subwarpgrp.shfl(my_idx, 0);
    return piv_row_idx;
}

template <typename Group, typename ValueType>
__device__ __forceinline__ void swap_rows_and_rhs(
    Group subwarpgrp, const int diag_pos, const int piv_row_idx, const int size,
    ValueType* const local_row, ValueType& rhs)
{
    const int local_id = subwarpgrp.thread_rank();

    for (int col = 0; col < size; col++) {
        ValueType diag_tid_col_val = subwarpgrp.shfl(local_row[col], diag_pos);
        ValueType piv_row_tid_col_val =
            subwarpgrp.shfl(local_row[col], piv_row_idx);

        if (local_id == diag_pos) {
            local_row[col] = piv_row_tid_col_val;
        }

        if (local_id == piv_row_idx) {
            local_row[col] = diag_tid_col_val;
        }
    }

    ValueType diag_tid_rhs = subwarpgrp.shfl(rhs, diag_pos);
    ValueType piv_row_tid_rhs = subwarpgrp.shfl(rhs, piv_row_idx);

    if (local_id == diag_pos) {
        rhs = piv_row_tid_rhs;
    }

    if (local_id == piv_row_idx) {
        rhs = diag_tid_rhs;
    }
}


template <typename Group, typename ValueType>
__device__ __forceinline__ void row_transformation(Group subwarpgrp,
                                                   const int diag_pos,
                                                   const int size,
                                                   ValueType* const local_row,
                                                   ValueType& rhs)
{
    const int local_id = subwarpgrp.thread_rank();
    const ValueType diag_ele = subwarpgrp.shfl(local_row[diag_pos], diag_pos);
    assert(abs(diag_ele) != abs(zero<ValueType>()));
    const ValueType multiplier = local_row[diag_pos] / diag_ele;

    for (int col = 0; col < size; col++) {
        const ValueType col_key_val = subwarpgrp.shfl(local_row[col], diag_pos);

        if (local_id != diag_pos) {
            local_row[col] -= multiplier * col_key_val;
        }
    }

    const ValueType rhs_key_val = subwarpgrp.shfl(rhs, diag_pos);
    if (local_id != diag_pos) {
        rhs -= multiplier * rhs_key_val;
    }
}

template <typename Group, typename ValueType>
__device__ __forceinline__ ValueType
solve_general_dense_system(Group subwarpgrp, const int size,
                           ValueType* const local_row, const int rhs_one_idx)
{
    const int local_id = subwarpgrp.thread_rank();
    ValueType rhs =
        local_id == rhs_one_idx ? one<ValueType>() : zero<ValueType>();

    for (int diag_pos = 0; diag_pos < size; diag_pos++) {
        const int piv_row_idx =
            choose_pivot_row(subwarpgrp, diag_pos, size, local_row);
        if (piv_row_idx != diag_pos) {
            swap_rows_and_rhs(subwarpgrp, diag_pos, piv_row_idx, size,
                              local_row, rhs);
        }

        subwarpgrp.sync();
        row_transformation(subwarpgrp, diag_pos, size, local_row, rhs);
        subwarpgrp.sync();
    }

    rhs = rhs / local_row[local_id];
    return rhs;
}


template <int subwarp_size, typename ValueType>
__global__
    __launch_bounds__(default_block_size) void fill_values_dense_mat_and_solve_kernel(
        const int nbatch, const int nrows, const int A_nnz,
        const ValueType* const A_values, const int aiA_nnz,
        const int* const __restrict__ aiA_row_ptrs,
        ValueType* const __restrict__ aiA_values,
        const int* const __restrict__ dense_mat_pattern,
        const int* const __restrict__ rhs_one_idxs,
        const int* const __restrict__ sizes,
        const enum gko::preconditioner::batch_isai_input_matrix_type
            matrix_type)
{
    using gko::preconditioner::batch_isai::row_size_limit;
    static_assert(row_size_limit <= subwarp_size, "incompatible subwarp size");

    auto subwarpgrp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int subwarp_id_in_grid =
        thread::get_subwarp_id_flat<subwarp_size, int>();
    const int total_num_subwarp_grps_in_grid =
        thread::get_subwarp_num_flat<subwarp_size, int>();
    const int local_id = subwarpgrp.thread_rank();

    for (int i = subwarp_id_in_grid; i < nrows * nbatch;
         i += total_num_subwarp_grps_in_grid) {
        const int row = i % nrows;
        const int batch_id = i / nrows;

        const int* dense_ptr =
            dense_mat_pattern + row_size_limit * row_size_limit * row;
        const int size = sizes[row];
        const int rhs_one_idx = rhs_one_idxs[row];

        assert(size <= row_size_limit);

        ValueType local_row[row_size_limit];
        // Note: The Pattern shouldn't be  a transpose.

        // implicit transpose
        for (int k = 0; k < size; k++)  // coalesced access by subwarp_grp
        {
            if (local_id >= size) {
                break;
            }

            const int val_idx = dense_ptr[k * row_size_limit + local_id];

            if (val_idx != -1) {
                local_row[k] = A_values[val_idx + batch_id * A_nnz];
            } else {
                local_row[k] = 0;
            }
        }
        // Now the 0th thread of the subwarp contains the 0th row of the dense
        // system, the 1st thread, the 1st row and so on...

        subwarpgrp.sync();
        ValueType sol;

        if (matrix_type == gko::preconditioner::batch_isai_input_matrix_type::
                               lower_tri)  // input matrix: lower_tri =>
                                           // tranposed system: uppper_tri
        {
            sol = solve_upper_tri_dense_system(subwarpgrp, size, local_row,
                                               rhs_one_idx);
        } else if (matrix_type ==
                   gko::preconditioner::batch_isai_input_matrix_type::
                       upper_tri)  // input matrix: upper_tri => tranposed
                                   // system: lower_tri
        {
            sol = solve_lower_tri_dense_system(subwarpgrp, size, local_row,
                                               rhs_one_idx);
        } else if (matrix_type ==
                   gko::preconditioner::batch_isai_input_matrix_type::
                       general)  // general input matrix
        {
            sol = solve_general_dense_system(subwarpgrp, size, local_row,
                                             rhs_one_idx);
        } else {
            printf("\n No such case: line: %d and file: %s ", __LINE__,
                   __FILE__);
            assert(false);
        }

        if (local_id < size)  // subwarp - coalesced accesses
        {
            aiA_values[aiA_row_ptrs[row] + local_id + batch_id * aiA_nnz] = sol;
        }
    }
}
