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


template <typename ValueType, typename Group>
__device__ __forceinline__ void broadcast(
    Group& subwarp_grp, const int target_lane, const ValueType& my_a,
    const ValueType& my_b, const ValueType& my_c, const ValueType& my_d,
    ValueType& piv_a, ValueType& piv_b, ValueType& piv_c, ValueType& piv_d)
{
    piv_a = subwarp_grp.shfl(my_a, target_lane);
    piv_b = subwarp_grp.shfl(my_b, target_lane);
    piv_c = subwarp_grp.shfl(my_c, target_lane);
    piv_d = subwarp_grp.shfl(my_d, target_lane);
}

namespace WM_pGE_approach1 {

template <typename ValueType, typename Group>
__device__ __forceinline__ void WM_step(Group& subwarp_grp,
                                        const int curr_group_size,
                                        ValueType& my_a, ValueType& my_b,
                                        ValueType& my_c, ValueType& my_d)
{
    const int lane = subwarp_grp.thread_rank();
    const int curr_grp_idx = lane / curr_group_size;

    const bool is_left_grp = (curr_grp_idx % 2) == 0;
    ValueType piv_a, piv_b, piv_c, piv_d;
    ValueType my_f = zero<ValueType>();

    const int left_grp_last_lane =
        is_left_grp ? ((curr_grp_idx + 1) * curr_group_size) - 1
                    : (curr_grp_idx * curr_group_size) - 1;
    const int right_grp_first_lane = left_grp_last_lane + 1;

    // broadcast last equation of the left group
    broadcast(subwarp_grp, left_grp_last_lane, my_a, my_b, my_c, my_d, piv_a,
              piv_b, piv_c, piv_d);

    if (lane == right_grp_first_lane) {
        my_f = my_b;
    }

    // eliminate a of the right group
    if (!is_left_grp) {
        const ValueType mult = my_a / piv_b;
        my_a = -one<ValueType>() * piv_a * mult;
        my_d -= piv_d * mult;
        my_f -= piv_c * mult;
    }
    if (lane == right_grp_first_lane) {
        my_b = my_f;
    }

    // broadcast first equation of the right group
    broadcast(subwarp_grp, right_grp_first_lane, my_a, my_b, my_c, my_d, piv_a,
              piv_b, piv_c, piv_d);

    // eliminate c of the left group
    if (is_left_grp) {
        const ValueType mult = my_c / piv_b;
        my_a -= piv_a * mult;
        my_d -= piv_d * mult;
        my_c = -one<ValueType>() * piv_c * mult;
    }

    // eliminate fill-in of the right group except for its first row
    if (!is_left_grp && lane != right_grp_first_lane) {
        const ValueType mult = my_f / piv_b;
        my_a -= piv_a * mult;
        my_d -= piv_d * mult;
        my_c -= piv_c * mult;
    }
}

template <typename ValueType, typename Group>
__device__ __forceinline__ void WM_phase(const int num_WM_steps,
                                         Group& subwarp_grp, ValueType& my_a,
                                         ValueType& my_b, ValueType& my_c,
                                         ValueType& my_d, int& curr_group_size)
{
    for (int i = 0; i < num_WM_steps; i++) {
        WM_step(subwarp_grp, curr_group_size, my_a, my_b, my_c, my_d);
        curr_group_size *= 2;
        subwarp_grp.sync();
    }
}


template <typename ValueType, typename Group>
__device__ __forceinline__ void Forward_full_GE_phase(
    Group& subwarp_grp, const int nrows, const int my_row_idx,
    const int tile_size, const int final_group_size,
    ValueType& c_last_row_of_prev_group, ValueType& d_last_row_of_prev_group,
    ValueType& my_a, ValueType& my_b, ValueType& my_c, ValueType& my_d,
    ValueType* __restrict__ c_batch_entry,
    ValueType* __restrict__ d_batch_entry)
{
    const int num_groups_in_tile = tile_size / final_group_size;
    const int lane = subwarp_grp.thread_rank();
    const int grp_idx = lane / final_group_size;

    for (int i = 0; i < num_groups_in_tile; i++) {
        // Forward full GE of group having grp_idx = i
        ValueType my_f = zero<ValueType>();
        const int first_lane_of_ith_group = i * final_group_size;
        const int last_lane_of_ith_group =
            first_lane_of_ith_group + final_group_size - 1;

        // eliminate a (i.e bottom spike) of the group
        if (grp_idx == i) {
            if (lane == first_lane_of_ith_group) {
                my_f = my_b;
            }

            const ValueType mult = my_a;
            my_f -= c_last_row_of_prev_group * mult;
            my_d -= d_last_row_of_prev_group * mult;

            if (lane == first_lane_of_ith_group) {
                my_b = my_f;
            }
        }

        subwarp_grp.sync();
        ValueType piv_f, piv_c, piv_d;
        piv_f = subwarp_grp.shfl(my_f, first_lane_of_ith_group);
        piv_c = subwarp_grp.shfl(my_c, first_lane_of_ith_group);
        piv_d = subwarp_grp.shfl(my_d, first_lane_of_ith_group);

        if (grp_idx == i) {
            // eliminate the fill-in of the group (except for the first lane of
            // the group)
            if (lane != first_lane_of_ith_group) {
                const ValueType mult = my_f / piv_f;
                my_c -= piv_c * mult;
                my_d -= piv_d * mult;
            }

            // divide the row by b and now b is basically 1
            my_c /= my_b;
            my_d /= my_b;
        }

        subwarp_grp.sync();
        c_last_row_of_prev_group =
            subwarp_grp.shfl(my_c, last_lane_of_ith_group);
        d_last_row_of_prev_group =
            subwarp_grp.shfl(my_d, last_lane_of_ith_group);
    }

    if (my_row_idx < nrows) {
        // coalesced accesses while writing data
        c_batch_entry[my_row_idx] = my_c;
        d_batch_entry[my_row_idx] = my_d;

        // printf("\n my_row_idx: %d, my_c: %g, my_d: %g", my_row_idx, my_c,
        // my_d);
    }
}


template <typename ValueType, typename Group>
__device__ __forceinline__ void backward_substitution(
    Group& subwarp_grp, const int nrows, const int my_row_idx,
    const int tile_size, const int final_group_size,
    ValueType& x_first_higher_group,
    const ValueType* const __restrict__ c_batch_entry,
    const ValueType* __restrict__ d_batch_entry,
    ValueType* const __restrict__ x_batch_entry)
{
    const int lane = subwarp_grp.thread_rank();
    const int grp_idx = lane / final_group_size;
    const int num_grps = tile_size / final_group_size;

    ValueType my_c, my_d, my_x;
    if (my_row_idx < nrows) {
        my_c = c_batch_entry[my_row_idx];
        my_d = d_batch_entry[my_row_idx];
        // coalseced accesses while reading

        // printf("\n READ: my_row_idx: %d, my_c: %g, my_d: %g", my_row_idx,
        // my_c,
        //        my_d);
    }

    for (int i = num_grps - 1; i >= 0; i--) {
        if (grp_idx == i) {
            my_x = my_d - x_first_higher_group * my_c;
        }

        subwarp_grp.sync();
        const int target_lane = i * final_group_size;
        x_first_higher_group = subwarp_grp.shfl(my_x, target_lane);
    }

    if (my_row_idx < nrows) {
        x_batch_entry[my_row_idx] = my_x;
        // coalseced accessed while writing

        // printf("\n my_row_idx: %d, my_x: %g", my_row_idx, my_x);
    }
}

}  // namespace WM_pGE_approach1

template <int subwarp_size, typename ValueType>
__global__ __launch_bounds__(default_block_size) void WM_pGE_kernel_approach_1(
    const int num_WM_steps, const size_type nbatch, const int nrows,
    ValueType* const __restrict__ a, ValueType* const __restrict__ b,
    ValueType* const __restrict__ c, ValueType* const __restrict__ d,
    ValueType* const __restrict__ x)
{
    auto subwarpgrp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int subgrpwarp_id_in_grid =
        thread::get_subwarp_id_flat<subwarp_size, int>();
    const int total_num_subwarp_grps_in_grid =
        thread::get_subwarp_num_flat<subwarp_size, int>();
    const int id_within_warp = subwarpgrp.thread_rank();

    // a subwarp per matrix in the batch
    for (size_type batch_idx = subgrpwarp_id_in_grid; batch_idx < nbatch;
         batch_idx += total_num_subwarp_grps_in_grid) {
        // Approach: a thread in the subwarp handles one row of the matrix or to
        // be precise, a row in the matrix tile
        const auto tile_size = subwarp_size;
        const auto num_tiles = ceildiv(nrows, tile_size);
        const bool is_last_tile_similar = ((nrows % tile_size) == 0);
        const int final_group_size = pow(2, num_WM_steps);
        assert(final_group_size <= tile_size);

        ValueType c_last_row_of_prev_group = zero<ValueType>();
        ValueType d_last_row_of_prev_group = zero<ValueType>();


        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            const int row_idx_st_tile = tile_id * tile_size;  // inclusive
            const int row_idx_end_tile =
                tile_id == num_tiles - 1
                    ? nrows
                    : (tile_id + 1) * tile_size;  // exclusive

            ValueType my_a, my_b, my_c, my_d;

            const int my_row_idx = row_idx_st_tile + id_within_warp;

            if (my_row_idx < row_idx_end_tile) {
                my_a = a[batch_idx * nrows + my_row_idx];
                my_b = b[batch_idx * nrows + my_row_idx];
                my_c = c[batch_idx * nrows + my_row_idx];
                my_d = d[batch_idx * nrows + my_row_idx];
                // coalesced accesses while reading data
            }

            if (tile_id < num_tiles - 1 || is_last_tile_similar) {
                // Phase-1 of the alogithm- WM phase
                int curr_group_size = 1;
                WM_pGE_approach1::WM_phase(num_WM_steps, subwarpgrp, my_a, my_b,
                                           my_c, my_d, curr_group_size);
                // In each WM step, the adjacent groups are merged
                // independently.

                // Phase-2 of the algorithm - Full Gaussean elimination of the
                // groups.
                // Now perform full Gaussean elimination on each group of the
                // transformed system to eliminate the bottom spikes
                assert(curr_group_size == final_group_size);

                WM_pGE_approach1::Forward_full_GE_phase(
                    subwarpgrp, nrows, my_row_idx, tile_size, final_group_size,
                    c_last_row_of_prev_group, d_last_row_of_prev_group, my_a,
                    my_b, my_c, my_d, c + batch_idx * nrows,
                    d + batch_idx * nrows);
            } else {
                WM_pGE_approach1::Forward_full_GE_phase(
                    subwarpgrp, nrows, my_row_idx,
                    row_idx_end_tile - row_idx_st_tile, 1,
                    c_last_row_of_prev_group, d_last_row_of_prev_group, my_a,
                    my_b, my_c, my_d, c + batch_idx * nrows,
                    d + batch_idx * nrows);
            }
        }

        // Now backward substitution
        ValueType x_first_higher_group = zero<ValueType>();

        for (int tile_id = num_tiles - 1; tile_id >= 0; tile_id--) {
            const int lane = subwarpgrp.thread_rank();
            const int row_idx_st_tile = tile_id * tile_size;  // inclusive
            const int row_idx_end_tile =
                tile_id == num_tiles - 1
                    ? nrows
                    : (tile_id + 1) * tile_size;  // exclusive
            const int my_row_idx = row_idx_st_tile + lane;

            if (tile_id == num_tiles - 1 && !is_last_tile_similar) {
                WM_pGE_approach1::backward_substitution(
                    subwarpgrp, nrows, my_row_idx,
                    row_idx_end_tile - row_idx_st_tile, 1, x_first_higher_group,
                    c + batch_idx * nrows, d + batch_idx * nrows,
                    x + batch_idx * nrows);
            } else {
                WM_pGE_approach1::backward_substitution(
                    subwarpgrp, nrows, my_row_idx, tile_size, final_group_size,
                    x_first_higher_group, c + batch_idx * nrows,
                    d + batch_idx * nrows, x + batch_idx * nrows);
            }
        }
    }
}


namespace WM_pGE_approach2 {

// template < typename T>
// struct get_type_2
// {

// };

// template<>
// struct get_type_2<double>
// {
//     using type = double2;
// };

// template<>
// struct get_type_2<float>
// {
//     using type = float2;
// };


template <typename ValueType, typename Group>
__device__ __forceinline__ void WM_step(Group& subwarp_grp,
                                        const int curr_group_size,
                                        ValueType& my_a_1, ValueType& my_b_1,
                                        ValueType& my_c_1, ValueType& my_d_1,
                                        ValueType& my_a_2, ValueType& my_b_2,
                                        ValueType& my_c_2, ValueType& my_d_2)
{
    if (curr_group_size == 1) {
        // eliminate a of the right group
        ValueType mult;
        mult = my_a_2 / my_b_1;
        my_a_2 = -one<ValueType>() * my_a_1 * mult;
        my_d_2 -= my_d_1 * mult;
        my_b_2 -= my_c_1 * mult;

        // eliminate c of the left group
        mult = my_c_1 / my_b_2;
        my_a_1 -= my_a_2 * mult;
        my_d_1 -= my_d_2 * mult;
        my_c_1 = -one<ValueType>() * my_c_2 * mult;

    } else {
        const int lane = subwarp_grp.thread_rank();
        const int curr_grp_idx =
            (2 * lane) / curr_group_size;  // both the eqs of the thread are in
                                           // the same group

        const bool is_left_grp = (curr_grp_idx % 2) == 0;
        ValueType piv_a, piv_b, piv_c, piv_d;
        ValueType my_f_1, my_f_2;
        my_f_1 = zero<ValueType>();
        my_f_2 = zero<ValueType>();

        const int left_grp_last_lane =
            is_left_grp ? (((curr_grp_idx + 1) * curr_group_size) / 2) - 1
                        : ((curr_grp_idx * curr_group_size) / 2) - 1;

        const int right_grp_first_lane = left_grp_last_lane + 1;

        // broadcast last equation of the left group
        broadcast(subwarp_grp, left_grp_last_lane, my_a_2, my_b_2, my_c_2,
                  my_d_2, piv_a, piv_b, piv_c, piv_d);

        if (lane == right_grp_first_lane) {
            my_f_1 = my_b_1;
        }

        // eliminate a of the right group
        if (!is_left_grp) {
            const ValueType mult_1 = my_a_1 / piv_b;
            my_a_1 = -one<ValueType>() * piv_a * mult_1;
            my_d_1 -= piv_d * mult_1;
            my_f_1 -= piv_c * mult_1;
            const ValueType mult_2 = my_a_2 / piv_b;
            my_a_2 = -one<ValueType>() * piv_a * mult_2;
            my_d_2 -= piv_d * mult_2;
            my_f_2 -= piv_c * mult_2;
        }

        if (lane == right_grp_first_lane) {
            my_b_1 = my_f_1;
        }

        // broadcast first equation of the right group
        broadcast(subwarp_grp, right_grp_first_lane, my_a_1, my_b_1, my_c_1,
                  my_d_1, piv_a, piv_b, piv_c, piv_d);

        // eliminate c of the left group
        if (is_left_grp) {
            const ValueType mult_1 = my_c_1 / piv_b;
            my_a_1 -= piv_a * mult_1;
            my_d_1 -= piv_d * mult_1;
            my_c_1 = -one<ValueType>() * piv_c * mult_1;

            const ValueType mult_2 = my_c_2 / piv_b;
            my_a_2 -= piv_a * mult_2;
            my_d_2 -= piv_d * mult_2;
            my_c_2 = -one<ValueType>() * piv_c * mult_2;
        }

        // eliminate fill-in of the right group except for its first row
        if (!is_left_grp) {
            const ValueType mult_2 = my_f_2 / piv_b;
            my_a_2 -= piv_a * mult_2;
            my_d_2 -= piv_d * mult_2;
            my_c_2 -= piv_c * mult_2;

            if (lane != right_grp_first_lane) {
                const ValueType mult_1 = my_f_1 / piv_b;
                my_a_1 -= piv_a * mult_1;
                my_d_1 -= piv_d * mult_1;
                my_c_1 -= piv_c * mult_1;
            }
        }
    }
}

template <typename ValueType, typename Group>
__device__ __forceinline__ void WM_phase(const int num_WM_steps,
                                         Group& subwarp_grp, ValueType& my_a_1,
                                         ValueType& my_b_1, ValueType& my_c_1,
                                         ValueType& my_d_1, ValueType& my_a_2,
                                         ValueType& my_b_2, ValueType& my_c_2,
                                         ValueType& my_d_2,
                                         int& curr_group_size)
{
    for (int i = 0; i < num_WM_steps; i++) {
        WM_step(subwarp_grp, curr_group_size, my_a_1, my_b_1, my_c_1, my_d_1,
                my_a_2, my_b_2, my_c_2, my_d_2);
        curr_group_size *= 2;
        subwarp_grp.sync();
    }
}


template <typename ValueType, typename Group>
__device__ __forceinline__ void Forward_full_GE_phase(
    Group& subwarp_grp, const int nrows, const int my_row_idx_1,
    const int tile_size, const int final_group_size,
    ValueType& c_last_row_of_prev_group, ValueType& d_last_row_of_prev_group,
    ValueType& my_a_1, ValueType& my_b_1, ValueType& my_c_1, ValueType& my_d_1,
    ValueType& my_a_2, ValueType& my_b_2, ValueType& my_c_2, ValueType& my_d_2,
    ValueType* __restrict__ c_batch_entry,
    ValueType* __restrict__ d_batch_entry)
{
    const int num_groups_in_tile = tile_size / final_group_size;
    const int lane = subwarp_grp.thread_rank();

    if (final_group_size == 1) {
        const int grp_idx_1 = 2 * lane;
        const int grp_idx_2 = grp_idx_1 + 1;

        for (int i = 0; i < num_groups_in_tile; i++) {
            // Forward full GE of group having grp_idx = i

            const int lane_ith_group = i / 2;

            // eliminate a (i.e bottom spike) of the group
            // and divide the row by b and now b is basically 1
            ValueType c_last_tmp, d_last_tmp;
            if (grp_idx_1 == i) {
                const ValueType mult = my_a_1;
                my_b_1 -= c_last_row_of_prev_group * mult;
                my_d_1 -= d_last_row_of_prev_group * mult;

                my_c_1 /= my_b_1;
                my_d_1 /= my_b_1;

                c_last_tmp = my_c_1;
                d_last_tmp = my_d_1;
            } else if (grp_idx_2 == i) {
                const ValueType mult = my_a_2;
                my_b_2 -= c_last_row_of_prev_group * mult;
                my_d_2 -= d_last_row_of_prev_group * mult;

                my_c_2 /= my_b_2;
                my_d_2 /= my_b_2;

                c_last_tmp = my_c_2;
                d_last_tmp = my_d_2;
            }

            subwarp_grp.sync();

            c_last_row_of_prev_group =
                subwarp_grp.shfl(c_last_tmp, lane_ith_group);
            d_last_row_of_prev_group =
                subwarp_grp.shfl(d_last_tmp, lane_ith_group);
        }

    } else {
        const int grp_idx = (2 * lane) / final_group_size;

        for (int i = 0; i < num_groups_in_tile; i++) {
            // Forward full GE of group having grp_idx = i
            ValueType my_f_1 = zero<ValueType>();
            ValueType my_f_2 = zero<ValueType>();
            const int first_lane_of_ith_group = (i * final_group_size) / 2;
            const int last_lane_of_ith_group =
                first_lane_of_ith_group + (final_group_size / 2) - 1;

            // eliminate a (i.e bottom spike) of the group
            if (grp_idx == i) {
                if (lane == first_lane_of_ith_group) {
                    my_f_1 = my_b_1;
                }

                const ValueType mult_1 = my_a_1;
                my_f_1 -= c_last_row_of_prev_group * mult_1;
                my_d_1 -= d_last_row_of_prev_group * mult_1;

                const ValueType mult_2 = my_a_2;
                my_f_2 -= c_last_row_of_prev_group * mult_2;
                my_d_2 -= d_last_row_of_prev_group * mult_2;

                if (lane == first_lane_of_ith_group) {
                    my_b_1 = my_f_1;
                }
            }

            subwarp_grp.sync();
            ValueType piv_f, piv_c, piv_d;
            piv_f = subwarp_grp.shfl(my_f_1, first_lane_of_ith_group);
            piv_c = subwarp_grp.shfl(my_c_1, first_lane_of_ith_group);
            piv_d = subwarp_grp.shfl(my_d_1, first_lane_of_ith_group);

            if (grp_idx == i) {
                // eliminate the fill-in of the group (except for the first lane
                // of the group)
                const ValueType mult_2 = my_f_2 / piv_f;
                my_c_2 -= piv_c * mult_2;
                my_d_2 -= piv_d * mult_2;

                if (lane != first_lane_of_ith_group) {
                    const ValueType mult_1 = my_f_1 / piv_f;
                    my_c_1 -= piv_c * mult_1;
                    my_d_1 -= piv_d * mult_1;
                }

                // divide the row by b and now b is basically 1
                my_c_1 /= my_b_1;
                my_d_1 /= my_b_1;
                my_c_2 /= my_b_2;
                my_d_2 /= my_b_2;
            }

            subwarp_grp.sync();
            c_last_row_of_prev_group =
                subwarp_grp.shfl(my_c_2, last_lane_of_ith_group);
            d_last_row_of_prev_group =
                subwarp_grp.shfl(my_d_2, last_lane_of_ith_group);
        }
    }

    // TODO: Vector writes to enable coalesced accesses while writing data
    if (my_row_idx_1 < nrows) {
        c_batch_entry[my_row_idx_1] = my_c_1;
        d_batch_entry[my_row_idx_1] = my_d_1;

        // printf("\n App-2 my_row_idx: %d, my_c: %g, my_d: %g", my_row_idx_1,
        //        my_c_1, my_d_1);
    }
    if (my_row_idx_1 + 1 < nrows) {
        c_batch_entry[my_row_idx_1 + 1] = my_c_2;
        d_batch_entry[my_row_idx_1 + 1] = my_d_2;

        // printf("\n App-2 my_row_idx: %d, my_c: %g, my_d: %g", my_row_idx_1 +
        // 1,
        //        my_c_2, my_d_2);
    }
}


template <typename ValueType, typename Group>
__device__ __forceinline__ void backward_substitution(
    Group& subwarp_grp, const int nrows, const int my_row_idx_1,
    const int tile_size, const int final_group_size,
    ValueType& x_first_higher_group,
    const ValueType* const __restrict__ c_batch_entry,
    const ValueType* __restrict__ d_batch_entry,
    ValueType* const __restrict__ x_batch_entry)
{
    ValueType my_c_1, my_d_1, my_x_1, my_c_2, my_d_2, my_x_2;
    // TODO: Vector reads to enable coalseced accesses while reading
    if (my_row_idx_1 < nrows) {
        my_c_1 = c_batch_entry[my_row_idx_1];
        my_d_1 = d_batch_entry[my_row_idx_1];

        // printf("\n READ: tid: %d, App-2 my_row_idx: %d, my_c: %g, my_d: %g",
        //        threadIdx.x, my_row_idx_1, my_c_1, my_d_1);
    }
    if (my_row_idx_1 + 1 < nrows) {
        my_c_2 = c_batch_entry[my_row_idx_1 + 1];
        my_d_2 = d_batch_entry[my_row_idx_1 + 1];

        // printf("\n READ: tid: %d, App-2 my_row_idx: %d, my_c: %g, my_d: %g",
        //        threadIdx.x, my_row_idx_1 + 1, my_c_2, my_d_2);
    }

    const int num_grps = tile_size / final_group_size;
    const int lane = subwarp_grp.thread_rank();

    if (final_group_size == 1) {
        const int grp_idx_1 = 2 * lane;
        const int grp_idx_2 = grp_idx_1 + 1;

        for (int i = num_grps - 1; i >= 0; i--) {
            ValueType x_first_higher_group_temp;
            if (grp_idx_1 == i) {
                my_x_1 = my_d_1 - x_first_higher_group * my_c_1;
                x_first_higher_group_temp = my_x_1;
            } else if (grp_idx_2 == i) {
                my_x_2 = my_d_2 - x_first_higher_group * my_c_2;
                x_first_higher_group_temp = my_x_2;
            }

            subwarp_grp.sync();
            const int target_lane = i / 2;
            x_first_higher_group =
                subwarp_grp.shfl(x_first_higher_group_temp, target_lane);
        }
    } else {
        const int grp_idx = (2 * lane) / final_group_size;

        for (int i = num_grps - 1; i >= 0; i--) {
            if (grp_idx == i) {
                my_x_1 = my_d_1 - x_first_higher_group * my_c_1;
                my_x_2 = my_d_2 - x_first_higher_group * my_c_2;
            }

            subwarp_grp.sync();
            const int target_lane = (i * final_group_size) / 2;
            x_first_higher_group = subwarp_grp.shfl(my_x_1, target_lane);
        }
    }

    // TODO: Vector writes to enable coalseced accessed while writing
    if (my_row_idx_1 < nrows) {
        x_batch_entry[my_row_idx_1] = my_x_1;
        // printf("\n App-2 my_row_idx: %d, my_x: %g", my_row_idx_1, my_x_1);
    }
    if (my_row_idx_1 + 1 < nrows) {
        x_batch_entry[my_row_idx_1 + 1] = my_x_2;
        //  printf("\n App-2 my_row_idx: %d, my_x: %g", my_row_idx_1 + 1,
        //  my_x_2);
    }
}


}  // namespace WM_pGE_approach2

template <int subwarp_size, typename ValueType>
__global__ __launch_bounds__(default_block_size) void WM_pGE_kernel_approach_2(
    const int num_WM_steps, const size_type nbatch, const int nrows,
    ValueType* const __restrict__ a, ValueType* const __restrict__ b,
    ValueType* const __restrict__ c, ValueType* const __restrict__ d,
    ValueType* const __restrict__ x)
{
    auto subwarpgrp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int subgrpwarp_id_in_grid =
        thread::get_subwarp_id_flat<subwarp_size, int>();
    const int total_num_subwarp_grps_in_grid =
        thread::get_subwarp_num_flat<subwarp_size, int>();
    const int id_within_warp = subwarpgrp.thread_rank();

    // a subwarp per matrix in the batch
    for (size_type batch_idx = subgrpwarp_id_in_grid; batch_idx < nbatch;
         batch_idx += total_num_subwarp_grps_in_grid) {
        // Approach: a thread in the subwarp handles two rows of the matrix or
        // to be precise, of the matrix tile
        const auto tile_size = 2 * subwarp_size;
        const auto num_tiles = ceildiv(nrows, tile_size);
        const bool is_last_tile_similar = ((nrows % tile_size) == 0);
        const int final_group_size = pow(2, num_WM_steps);
        assert(final_group_size <= tile_size);

        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("\n subwarp size: %d, tile size: %d, final group size:
        //     %d",
        //            subwarp_size, tile_size, final_group_size);
        // }

        ValueType c_last_row_of_prev_group = zero<ValueType>();
        ValueType d_last_row_of_prev_group = zero<ValueType>();


        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            const int row_idx_st_tile = tile_id * tile_size;  // inclusive
            const int row_idx_end_tile =
                tile_id == num_tiles - 1
                    ? nrows
                    : (tile_id + 1) * tile_size;  // exclusive

            ValueType my_a_1, my_b_1, my_c_1, my_d_1;
            ValueType my_a_2, my_b_2, my_c_2, my_d_2;

            const int my_row_idx_1 = row_idx_st_tile + 2 * id_within_warp;

            // TODO: Change this to vector reads to enable coalesced accesses
            // while reading data
            if (my_row_idx_1 < row_idx_end_tile) {
                my_a_1 = a[batch_idx * nrows + my_row_idx_1];
                my_b_1 = b[batch_idx * nrows + my_row_idx_1];
                my_c_1 = c[batch_idx * nrows + my_row_idx_1];
                my_d_1 = d[batch_idx * nrows + my_row_idx_1];
            }
            if (my_row_idx_1 + 1 < row_idx_end_tile) {
                my_a_2 = a[batch_idx * nrows + my_row_idx_1 + 1];
                my_b_2 = b[batch_idx * nrows + my_row_idx_1 + 1];
                my_c_2 = c[batch_idx * nrows + my_row_idx_1 + 1];
                my_d_2 = d[batch_idx * nrows + my_row_idx_1 + 1];
            }

            if (tile_id < num_tiles - 1 || is_last_tile_similar) {
                // Phase-1 of the alogithm- WM phase
                int curr_group_size = 1;
                WM_pGE_approach2::WM_phase(
                    num_WM_steps, subwarpgrp, my_a_1, my_b_1, my_c_1, my_d_1,
                    my_a_2, my_b_2, my_c_2, my_d_2, curr_group_size);
                // In each WM step, the adjacent groups are merged
                // independently.

                // Phase-2 of the algorithm - Full Gaussean elimination of the
                // groups.
                // Now perform full Gaussean elimination on each group of the
                // transformed system to eliminate the bottom spikes
                assert(curr_group_size == final_group_size);

                WM_pGE_approach2::Forward_full_GE_phase(
                    subwarpgrp, nrows, my_row_idx_1, tile_size,
                    final_group_size, c_last_row_of_prev_group,
                    d_last_row_of_prev_group, my_a_1, my_b_1, my_c_1, my_d_1,
                    my_a_2, my_b_2, my_c_2, my_d_2, c + batch_idx * nrows,
                    d + batch_idx * nrows);
            } else {
                WM_pGE_approach2::Forward_full_GE_phase(
                    subwarpgrp, nrows, my_row_idx_1,
                    row_idx_end_tile - row_idx_st_tile, 1,
                    c_last_row_of_prev_group, d_last_row_of_prev_group, my_a_1,
                    my_b_1, my_c_1, my_d_1, my_a_2, my_b_2, my_c_2, my_d_2,
                    c + batch_idx * nrows, d + batch_idx * nrows);
            }
        }

        // Now backward substitution
        ValueType x_first_higher_group = zero<ValueType>();

        for (int tile_id = num_tiles - 1; tile_id >= 0; tile_id--) {
            const int lane = subwarpgrp.thread_rank();
            const int row_idx_st_tile = tile_id * tile_size;  // inclusive
            const int row_idx_end_tile =
                tile_id == num_tiles - 1
                    ? nrows
                    : (tile_id + 1) * tile_size;  // exclusive
            const int my_row_idx_1 = row_idx_st_tile + 2 * lane;

            if (tile_id == num_tiles - 1 && !is_last_tile_similar) {
                WM_pGE_approach2::backward_substitution(
                    subwarpgrp, nrows, my_row_idx_1,
                    row_idx_end_tile - row_idx_st_tile, 1, x_first_higher_group,
                    c + batch_idx * nrows, d + batch_idx * nrows,
                    x + batch_idx * nrows);
            } else {
                WM_pGE_approach2::backward_substitution(
                    subwarpgrp, nrows, my_row_idx_1, tile_size,
                    final_group_size, x_first_higher_group,
                    c + batch_idx * nrows, d + batch_idx * nrows,
                    x + batch_idx * nrows);
            }
        }
    }
}
