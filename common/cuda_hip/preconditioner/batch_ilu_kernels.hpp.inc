/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

template <typename ValueType>
__device__ __forceinline__ void fill_partial_current_row_array(
    const int nrows, const int curr_row_index,
    ValueType* const __restrict__ current_row_elements_arr,
    const int* const __restrict__ row_ptrs,
    const int* const __restrict__ col_idxs,
    const ValueType* const __restrict__ batch_entry_values,
    const int* const __restrict__ diag_ptrs)
{
    const int diag_ele_loc = diag_ptrs[curr_row_index];
    const int row_end_loc = row_ptrs[curr_row_index + 1];

    for (int i = threadIdx.x + curr_row_index; i < nrows; i += blockDim.x) {
        current_row_elements_arr[i] = zero<ValueType>();
    }

    __syncthreads();

    for (int loc = threadIdx.x + diag_ele_loc; loc < row_end_loc;
         loc += blockDim.x) {
        current_row_elements_arr[col_idxs[loc]] = batch_entry_values[loc];
    }
}


template <typename ValueType>
__device__ __forceinline__ void modify_rows_below_curr_row(
    const int nrows, const int curr_row_index,
    const ValueType* const __restrict__ column_elements_array_for_current_row,
    const int* const __restrict__ row_ptrs,
    const int* const __restrict__ col_idxs,
    ValueType* const __restrict__ batch_entry_values,
    const int* const __restrict__ diag_ptrs,
    ValueType* const __restrict__ row_ele_arr)
{
    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);


    // initilaize it with zeroes
    for (int i = threadIdx.x + curr_row_index + 1; i < nrows; i += blockDim.x) {
        row_ele_arr[i] = zero<ValueType>();
    }
    __syncthreads();


    // one warp per row
    for (int row_below_index = subwarp_grp_id + curr_row_index + 1;
         row_below_index < nrows;
         row_below_index += num_subwarp_grps_per_block) {
        for (int i = subwarp_grp.thread_rank() + row_ptrs[row_below_index];
             i < row_ptrs[row_below_index + 1]; i += subwarp_grp.size()) {
            const int col_index = col_idxs[i];

            if (col_index == curr_row_index) {
                ValueType diag_ele =
                    batch_entry_values[diag_ptrs[curr_row_index]];
                // assert(diag_ele != zero<ValueType>());
                assert(abs(diag_ele) !=
                       abs(zero<ValueType>()));  // Temporary work-around for !=
                                                 // operator error (for complex
                                                 // types) with rocm-4.0
                ValueType row_ele = batch_entry_values[i] / diag_ele;
                row_ele_arr[row_below_index] = row_ele;
                batch_entry_values[i] = row_ele;
            }

#ifdef __CUDACC__
            __syncwarp(__activemask());  // else a warning
#endif

            if (col_index > curr_row_index) {
                ValueType col_ele =
                    column_elements_array_for_current_row[col_index];
                batch_entry_values[i] -= row_ele_arr[row_below_index] * col_ele;
            }
        }
    }
}


template <typename ValueType>
__global__
    __launch_bounds__(default_block_size) void generate_exact_ilu0_kernel(
        const size_type batch_size, const int nrows, const int nnz,
        const int* const __restrict__ diag_ptrs,
        const int* const __restrict__ mat_row_ptrs,
        const int* const __restrict__ mat_col_idxs,
        ValueType* const __restrict__ mat_values)
{
    for (size_type batch_id = blockIdx.x; batch_id < batch_size;
         batch_id += gridDim.x) {
        // Tried out-> Having stuff (mat-row_ptrs, col_idxs, vals and diag_locs)
        // in shared memory slows down the kernel, so don't copy global arrays
        // to shared memory.

        extern __shared__ char local_mem_sh[];
        ValueType* const __restrict__ current_row_elements_arr =
            reinterpret_cast<ValueType*>(local_mem_sh);
        ValueType* const __restrict__ row_ele_arr =
            current_row_elements_arr + nrows;

        for (int curr_row_index = 0; curr_row_index < nrows; curr_row_index++) {
            fill_partial_current_row_array(
                nrows, curr_row_index, current_row_elements_arr, mat_row_ptrs,
                mat_col_idxs, mat_values + nnz * batch_id, diag_ptrs);
            __syncthreads();

            modify_rows_below_curr_row(
                nrows, curr_row_index, current_row_elements_arr, mat_row_ptrs,
                mat_col_idxs, mat_values + nnz * batch_id, diag_ptrs,
                row_ele_arr);
            __syncthreads();
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void parilu0_sweep_for_batch_entry(
    const int nnz, const ValueType* const __restrict__ A_vals,
    ValueType* const __restrict__ fact_mat_vals,
    const int* const __restrict__ dependencies,
    const int* const __restrict__ nz_ptrs)
{
    for (int el = threadIdx.x; el < nnz;
         el += blockDim.x)  // non-coalesced accesses
                            // but atleast data locality
    {
        ValueType diag_val{one<ValueType>()};

        const int st = nz_ptrs[el];
        const int end = nz_ptrs[el + 1] - 1;

        const bool has_diag_dependency = (end + 1 - st) % 2 == 0 ? false : true;

        ValueType sum{zero<ValueType>()};

        for (int idx = st; idx <= end - 1; idx += 2) {
            sum += fact_mat_vals[dependencies[idx]] *
                   fact_mat_vals[dependencies[idx + 1]];
        }

        if (has_diag_dependency == true) {
            diag_val = fact_mat_vals[dependencies[end]];
        }

        ValueType to_write = (A_vals[el] - sum) / diag_val;

        fact_mat_vals[el] = to_write;
    }
}

template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void generate_parilu0_kernel(
    const size_type nbatch, const int num_rows, const int nnz,
    const int* const __restrict__ dependencies,
    const int* const __restrict__ nz_ptrs, const int num_sweeps,
    const ValueType* const __restrict__ A_values,
    ValueType* const __restrict__ fact_mat_values)
{
    for (size_type batch_id = blockIdx.x; batch_id < nbatch;
         batch_id += gridDim.x) {
        extern __shared__ char local_mem_sh[];
        ValueType* const __restrict__ fact_mat_values_sh =
            reinterpret_cast<ValueType*>(local_mem_sh);

        single_copy(nnz, fact_mat_values + batch_id * nnz, fact_mat_values_sh);
        __syncthreads();

        for (int sweep = 0; sweep < num_sweeps; sweep++) {
            parilu0_sweep_for_batch_entry(nnz, A_values + batch_id * nnz,
                                          fact_mat_values_sh, dependencies,
                                          nz_ptrs);
            __syncthreads();
        }

        single_copy(nnz, fact_mat_values_sh, fact_mat_values + batch_id * nnz);
    }
}