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
__dpct_inline__ void extract_dense_linear_sys_pattern_kernel(
    const int nrows, const int* const __restrict__ A_row_ptrs,
    const int* const __restrict__ A_col_idxs,
    const int* const __restrict__ aiA_row_ptrs,
    const int* const __restrict__ aiA_col_idxs, int* const dense_mat_pattern,
    int* const rhs_one_idxs, int* const sizes,
    int* num_matches_per_row_for_each_csr_sys, sycl::nd_item<3> item_ct1)
{
    using gko::preconditioner::batch_isai::row_size_limit;
    // assert(subwarp_size >= row_size_limit); //Not required here
    auto subwarpgrp = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
    const int subgrpwarp_id_in_grid =
        thread::get_subwarp_id_flat<subwarp_size, int>(item_ct1);
    const int total_num_subwarp_grps_in_grid =
        thread::get_subwarp_num_flat<subwarp_size, int>(item_ct1);
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

            int* count_matches =
                &num_matches_per_row_for_each_csr_sys[aiA_nz_id];


            if (i_size <= row_size_limit) {
                // assert(*count_matches == -1);

                const int dense_sys_row = aiA_nz_id - aiA_row_st;

                int* const dense_ptr_for_this_row =
                    dense_mat_ptr + dense_sys_row * row_size_limit;

                auto do_this_on_match =
                    [dense_ptr_for_this_row, aiA_row_st, A_row_st](
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

            } else {
                if (subwarpgrp.thread_rank() == 0) {
                    *count_matches = 0;
                }

                auto do_this_on_match =
                    [count_matches](int, int aiA_idx_match, int A_idx_match,
                                    config::lane_mask_type, bool valid) {
                        if (valid) {
                            atomic_add(count_matches, 1);
                        }
                    };

                // warp-parallel match algorithm
                group_match<subwarp_size>(
                    aiA_col_idxs + aiA_row_st, aiA_row_end - aiA_row_st,
                    A_col_idxs + A_row_st, A_row_end - A_row_st, subwarpgrp,
                    do_this_on_match);
            }

            subwarpgrp.sync();
            ele_id_assigned_to_subwarpgrp += total_num_subwarp_grps_in_grid;
        }
    }
}

template <typename ValueType>
__dpct_inline__ ValueType
solve_lower_tri_dense_system(const int size, ValueType* const local_row,
                             const int rhs_one_idx, sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    auto sg_tid = sg.get_local_id();
    const ValueType rhs =
        sg_tid == rhs_one_idx ? one<ValueType>() : zero<ValueType>();
    ValueType sol = rhs;

    for (int dense_col_idx = 0; dense_col_idx < size; dense_col_idx++) {
        const ValueType ele = local_row[dense_col_idx];

        if (dense_col_idx == sg_tid) {
            sol = sol / ele;
        }

        sg.barrier();
        const ValueType top = sg.shuffle(sol, dense_col_idx);

        if (sg_tid > dense_col_idx) {
            sol = sol - top * ele;
        }
    }

    return sol;
}


template <typename ValueType>
__dpct_inline__ ValueType
solve_upper_tri_dense_system(const int size, ValueType* const local_row,
                             const int rhs_one_idx, sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_tid = sg.get_local_id();

    const ValueType rhs =
        sg_tid == rhs_one_idx ? one<ValueType>() : zero<ValueType>();
    ValueType sol = rhs;

    for (int dense_col_idx = size - 1; dense_col_idx >= 0; dense_col_idx--) {
        const ValueType ele = local_row[dense_col_idx];

        if (dense_col_idx == sg_tid) {
            sol = sol / ele;
        }

        sg.barrier();
        const ValueType bot = sg.shuffle(sol, dense_col_idx);

        if (sg_tid < dense_col_idx) {
            sol = sol - bot * ele;
        }
    }

    return sol;
}


template <typename ValueType>
__dpct_inline__ int choose_pivot_row(const int diag_pos, const int size,
                                     const ValueType* const local_row,
                                     sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_tid = sg.get_local_id();
    const int sg_size = sg.get_local_range().size();

    ValueType my_ele = local_row[diag_pos];
    int my_idx = sg_tid;
    const ValueType my_ele_1 = sg.shuffle(local_row[diag_pos], diag_pos);

    if (sg_tid < diag_pos || sg_tid >= size) {
        my_ele = my_ele_1;
        my_idx = diag_pos;
    }

    sg.barrier();

    for (int offset = sg_size / 2; offset > 0; offset /= 2) {
        ValueType other_ele = sg.shuffle_down(my_ele, offset);
        int other_idx = sg.shuffle_down(my_idx, offset);
        if (abs(other_ele) > abs(my_ele)) {
            my_ele = other_ele;
            my_idx = other_idx;
        }
        sg.barrier();
    }

    // thread 0 would have the correct piv_row_idx and piv_ele
    const int piv_row_idx = sg.shuffle(my_idx, 0);
    return piv_row_idx;
}

template <typename ValueType>
__dpct_inline__ void swap_rows_and_rhs(const int diag_pos,
                                       const int piv_row_idx, const int size,
                                       ValueType* const local_row,
                                       ValueType& rhs,
                                       sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_tid = sg.get_local_id();

    for (int col = 0; col < size; col++) {
        ValueType diag_tid_col_val = sg.shuffle(local_row[col], diag_pos);
        ValueType piv_row_tid_col_val = sg.shuffle(local_row[col], piv_row_idx);

        if (sg_tid == diag_pos) {
            local_row[col] = piv_row_tid_col_val;
        }

        if (sg_tid == piv_row_idx) {
            local_row[col] = diag_tid_col_val;
        }
    }

    ValueType diag_tid_rhs = sg.shuffle(rhs, diag_pos);
    ValueType piv_row_tid_rhs = sg.shuffle(rhs, piv_row_idx);

    if (sg_tid == diag_pos) {
        rhs = piv_row_tid_rhs;
    }

    if (sg_tid == piv_row_idx) {
        rhs = diag_tid_rhs;
    }
}


template <typename ValueType>
__dpct_inline__ void row_transformation(const int diag_pos, const int size,
                                        ValueType* const local_row,
                                        ValueType& rhs,
                                        sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_tid = sg.get_local_id();

    const ValueType diag_ele = sg.shuffle(local_row[diag_pos], diag_pos);
    assert(abs(diag_ele) != abs(zero<ValueType>()));
    const ValueType multiplier = local_row[diag_pos] / diag_ele;

    for (int col = 0; col < size; col++) {
        const ValueType col_key_val = sg.shuffle(local_row[col], diag_pos);

        if (sg_tid != diag_pos) {
            local_row[col] -= multiplier * col_key_val;
        }
    }

    const ValueType rhs_key_val = sg.shuffle(rhs, diag_pos);
    if (sg_tid != diag_pos) {
        rhs -= multiplier * rhs_key_val;
    }
}

template <typename ValueType>
__dpct_inline__ ValueType solve_general_dense_system(const int size,
                                                     ValueType* const local_row,
                                                     const int rhs_one_idx,
                                                     sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_tid = sg.get_local_id();

    ValueType rhs =
        sg_tid == rhs_one_idx ? one<ValueType>() : zero<ValueType>();

    for (int diag_pos = 0; diag_pos < size; diag_pos++) {
        const int piv_row_idx =
            choose_pivot_row(diag_pos, size, local_row, item_ct1);
        if (piv_row_idx != diag_pos) {
            swap_rows_and_rhs(diag_pos, piv_row_idx, size, local_row, rhs,
                              item_ct1);
        }

        sg.barrier();
        row_transformation(diag_pos, size, local_row, rhs, item_ct1);
        sg.barrier();
    }

    rhs = rhs / local_row[sg_tid];
    return rhs;
}


template <typename ValueType>
__dpct_inline__ void fill_values_dense_mat_and_solve_kernel(
    const int nbatch, const int nrows, const int A_nnz,
    const ValueType* const A_values, const int aiA_nnz,
    const int* const __restrict__ aiA_row_ptrs,
    ValueType* const __restrict__ aiA_values,
    const int* const __restrict__ dense_mat_pattern,
    const int* const __restrict__ rhs_one_idxs,
    const int* const __restrict__ sizes,
    const enum gko::preconditioner::batch_isai_input_matrix_type matrix_type,
    sycl::nd_item<3> item_ct1)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    const auto sg = item_ct1.get_sub_group();
    const int sg_tid = sg.get_local_id();
    const int sg_size = sg.get_max_local_range().size();
    const int sg_global_id = item_ct1.get_global_linear_id() / sg_size;
    const int num_sg_total = item_ct1.get_global_range().size() / sg_size;

    GKO_ASSERT(row_size_limit <= sg_size);  //, "incompatible subwarp size");

    int i = sg_global_id;
    if (i >= nrows * nbatch) return;

    const int row = i % nrows;
    const int batch_id = i / nrows;

    const int* dense_ptr =
        dense_mat_pattern + row_size_limit * row_size_limit * row;
    const int size = sizes[row];
    const int rhs_one_idx = rhs_one_idxs[row];

    if (size > row_size_limit) return;

    ValueType local_row[row_size_limit];
    // Note: The Pattern shouldn't be  a transpose.

    // implicit transpose
    for (int k = 0; k < size; k++)  // coalesced access by subwarp_grp
    {
        if (sg_tid >= size) {
            break;
        }

        const int val_idx = dense_ptr[k * row_size_limit + sg_tid];

        if (val_idx != -1) {
            local_row[k] = A_values[val_idx + batch_id * A_nnz];
        } else {
            local_row[k] = 0;
        }
    }
    // Now the 0th thread of the subwarp contains the 0th row of the dense
    // system, the 1st thread, the 1st row and so on...

    sg.barrier();
    ValueType sol;

    if (matrix_type == gko::preconditioner::batch_isai_input_matrix_type::
                           lower_tri)  // input matrix: lower_tri =>
                                       // tranposed system: uppper_tri
    {
        sol = solve_upper_tri_dense_system(size, local_row, rhs_one_idx,
                                           item_ct1);
    } else if (matrix_type ==
               gko::preconditioner::batch_isai_input_matrix_type::
                   upper_tri)  // input matrix: upper_tri => tranposed
                               // system: lower_tri
    {
        sol = solve_lower_tri_dense_system(size, local_row, rhs_one_idx,
                                           item_ct1);
    } else if (matrix_type ==
               gko::preconditioner::batch_isai_input_matrix_type::
                   general)  // general input matrix
    {
        sol =
            solve_general_dense_system(size, local_row, rhs_one_idx, item_ct1);
    } else {
        // printf("\n No such case: line: %d and file: %s ", __LINE__,
        // __FILE__); //TODO
        assert(false);
    }

    if (sg_tid < size)  // subwarp - coalesced accesses
    {
        aiA_values[aiA_row_ptrs[row] + sg_tid + batch_id * aiA_nnz] = sol;
    }
}

template <typename ValueType>
__dpct_inline__ void match_for_csr(
    const int* const __restrict__ inv_col_idxs, const int inv_row_st,
    const int inv_row_end, const int* const __restrict__ sys_col_idxs,
    const int sys_row_st, const int sys_row_end,
    const int csr_pattern_row_start,
    int* const __restrict__ csr_pattern_col_idxs,
    gko::remove_complex<ValueType>* const __restrict__ csr_pattern_values)
{
    int l = inv_row_st;
    int m = sys_row_st;
    int j = csr_pattern_row_start;

    while (l < inv_row_end && m < sys_row_end) {
        if (inv_col_idxs[l] == sys_col_idxs[m]) {
            csr_pattern_values[j] =
                static_cast<gko::remove_complex<ValueType>>(m);
            csr_pattern_col_idxs[j] = l - inv_row_st;
            j++;

            l++;
            m++;
        } else if (inv_col_idxs[l] > sys_col_idxs[m]) {
            m++;
        } else {
            l++;
        }
    }
}

template <typename ValueType>
__dpct_inline__ void extract_csr_sys_pattern_kernel(
    const int lin_sys_row, const int* const __restrict__ inv_row_ptrs,
    const int* const __restrict__ inv_col_idxs,
    const int* const __restrict__ sys_row_ptrs,
    const int* const __restrict__ sys_col_idxs,
    const int* const __restrict__ csr_pattern_row_ptrs,
    int* const __restrict__ csr_pattern_col_idxs,
    gko::remove_complex<ValueType>* const __restrict__ csr_pattern_values,
    sycl::id<1> gid)
{
    // use one thread per match of the 2 arrays (non-coalseced accesses but data
    // locality)
    const int inv_row_st = inv_row_ptrs[lin_sys_row];
    const int inv_row_end = inv_row_ptrs[lin_sys_row + 1];

    const int col_idx = inv_col_idxs[gid];
    const int sys_row = col_idx;
    const int sys_row_st = sys_row_ptrs[sys_row];
    const int sys_row_end = sys_row_ptrs[sys_row + 1];

    const int csr_pattern_row_start = csr_pattern_row_ptrs[gid - inv_row_st];

    match_for_csr<ValueType>(inv_col_idxs, inv_row_st, inv_row_end,
                             sys_col_idxs, sys_row_st, sys_row_end,
                             csr_pattern_row_start, csr_pattern_col_idxs,
                             csr_pattern_values);
}


template <typename ValueType>
__dpct_inline__ void fill_batch_csr_system_kernel(
    const int nbatch, const int csr_nnz,
    const gko::remove_complex<ValueType>* const __restrict__ csr_pattern_values,
    const int sys_nnz, const ValueType* const __restrict__ sys_csr_values,
    ValueType* const __restrict__ batch_csr_mats_values, sycl::id<1> gid)
{
    const int batch_id = gid / csr_nnz;
    const int csr_nnz_id = gid % csr_nnz;

    const int sys_idx = static_cast<int>(csr_pattern_values[csr_nnz_id]);
    assert(sys_idx >= 0 && sys_idx < sys_nnz);

    const ValueType to_write = sys_csr_values[batch_id * sys_nnz + sys_idx];
    batch_csr_mats_values[batch_id * csr_nnz + csr_nnz_id] = to_write;
}


template <typename ValueType>
__dpct_inline__ void initialize_b_and_x_vectors_kernel(
    const int nbatch, const int size, const int rhs_one_idx,
    ValueType* const __restrict__ b_vals, ValueType* const __restrict__ x_vals,
    sycl::id<1> gid)
{
    const int batch_id = gid / size;
    const int idx = gid % size;

    b_vals[idx + batch_id * size] = zero<ValueType>();
    if (idx == rhs_one_idx) {
        b_vals[idx + batch_id * size] = one<ValueType>();
    }
    x_vals[idx + batch_id * size] = zero<ValueType>();
}

template <typename ValueType>
__dpct_inline__ void write_large_sys_solution_to_inverse_kernel(
    const int nbatch, const int lin_sys_row, const int size,
    const ValueType* const __restrict__ x_vals, const int inv_nnz,
    const int* const __restrict__ inv_row_ptrs,
    ValueType* const __restrict__ inv_vals, sycl::id<1> gid)
{
    assert(size == inv_row_ptrs[lin_sys_row + 1] - inv_row_ptrs[lin_sys_row]);

    const int batch_id = gid / size;
    const int idx = gid % size;

    inv_vals[inv_row_ptrs[lin_sys_row] + idx + batch_id * inv_nnz] =
        x_vals[idx + batch_id * size];
}
