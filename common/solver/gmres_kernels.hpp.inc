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

// Must be called with at least `max(stride_b * num_rows, krylov_dim *
// num_cols)` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_1_kernel(
    size_type num_rows, size_type num_cols, size_type krylov_dim,
    const ValueType *__restrict__ b, size_type stride_b,
    ValueType *__restrict__ residual, size_type stride_residual,
    ValueType *__restrict__ givens_sin, size_type stride_sin,
    ValueType *__restrict__ givens_cos, size_type stride_cos,
    stopping_status *__restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();

    const auto row_idx = global_id / stride_b;
    const auto col_idx = global_id % stride_b;

    if (global_id < num_cols) {
        stop_status[global_id].reset();
    }

    if (row_idx < num_rows && col_idx < num_cols) {
        residual[row_idx * stride_residual + col_idx] =
            b[row_idx * stride_b + col_idx];
    }

    if (global_id < krylov_dim * num_cols) {
        const auto row_givens = global_id / num_cols;
        const auto col_givens = global_id % num_cols;

        givens_sin[row_givens * stride_sin + col_givens] = zero<ValueType>();
        givens_cos[row_givens * stride_cos + col_givens] = zero<ValueType>();
    }
}


// Must be called with at least `num_rows * stride_krylov` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_1_kernel(
    size_type num_rows, size_type num_rhs, size_type krylov_dim,
    ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_nc)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_krylov;
    const auto col_idx = global_id % stride_krylov;
    const auto krylov_bases_nrows = (krylov_dim + 1) * num_rows;
    if (row_idx < krylov_bases_nrows && col_idx < num_rhs) {
        krylov_bases[row_idx * stride_krylov + col_idx] = zero<ValueType>();
    }

    if (row_idx < krylov_dim + 1 && col_idx < num_rhs) {
        residual_norm_collection[row_idx * stride_residual_nc + col_idx] =
            zero<ValueType>();
    }
}


// Must be called with at least `num_rows * num_rhs` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_2_kernel(
    size_type num_rows, size_type num_rhs,
    const ValueType *__restrict__ residual, size_type stride_residual,
    const ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    size_type *__restrict__ final_iter_nums)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / num_rhs;
    const auto col_idx = global_id % num_rhs;

    if (global_id < num_rhs) {
        residual_norm_collection[global_id] = residual_norm[global_id];
        final_iter_nums[global_id] = 0;
    }

    if (row_idx < num_rows && col_idx < num_rhs) {
        auto value = residual[row_idx * stride_residual + col_idx] /
                     residual_norm[col_idx];
        krylov_bases[row_idx * stride_krylov + col_idx] = value;
    }
}


__global__
    __launch_bounds__(default_block_size) void increase_final_iteration_numbers_kernel(
        size_type *__restrict__ final_iter_nums,
        const stopping_status *__restrict__ stop_status, size_type total_number)
{
    const auto global_id = thread::get_thread_id_flat();
    if (global_id < total_number) {
        final_iter_nums[global_id] += !stop_status[global_id].has_stopped();
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_dot_size) void multidot_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    const ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = threadIdx.x;
    const auto tidy = threadIdx.y;
    const auto col_idx = blockIdx.x * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, gridDim.y);
    const auto start_row = blockIdx.y * num;
    const auto end_row =
        ((blockIdx.y + 1) * num > num_rows) ? num_rows : (blockIdx.y + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__
        UninitializedArray<ValueType, default_dot_dim *(default_dot_dim + 1)>
            reduction_helper_array;
    ValueType *__restrict__ reduction_helper = reduction_helper_array;

    ValueType local_res = zero<ValueType>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto krylov_idx = i * stride_krylov + col_idx;
            local_res +=
                next_krylov_basis[krylov_idx] * krylov_bases[krylov_idx];
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_res;
    __syncthreads();
    local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto sum =
        reduce(tile_block, local_res,
               [](const ValueType &a, const ValueType &b) { return a + b; });
    const auto new_col_idx = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto hessenberg_idx = k * stride_hessenberg + new_col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, sum);
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_next_krylov_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ next_krylov_basis,
    const ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_krylov;
    const auto col_idx = global_id % stride_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_krylov + col_idx;
        const auto krylov_idx = row_idx * stride_krylov + col_idx;
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;

        next_krylov_basis[next_krylov_idx] -=
            hessenberg_iter[hessenberg_idx] * krylov_bases[krylov_idx];
    }
}


// Must be called with at least `num_cols` blocks, each with `block_size`
// threads. `block_size` must be a power of 2.
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_hessenberg_2_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov, ValueType *__restrict__ hessenberg_iter,
    size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = threadIdx.x;
    const auto col_idx = blockIdx.x;

    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__ UninitializedArray<ValueType, block_size> reduction_helper_array;
    ValueType *__restrict__ reduction_helper = reduction_helper_array;

    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        ValueType local_res{};
        for (size_type i = tidx; i < num_rows; i += block_size) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            const auto next_krylov_value = next_krylov_basis[next_krylov_idx];

            local_res += next_krylov_value * next_krylov_value;
        }

        reduction_helper[tidx] = local_res;

        // Perform thread block reduction. Result is in reduction_helper[0]
        reduce(group::this_thread_block(), reduction_helper,
               [](const ValueType &a, const ValueType &b) { return a + b; });

        if (tidx == 0) {
            hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] =
                sqrt(reduction_helper[0]);
        }
    }
}


// Must be called with at least `num_rows * stride_krylov` threads in
// total.
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_krylov_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_krylov;
    const auto col_idx = global_id % stride_krylov;
    const auto hessenberg =
        hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto krylov_idx = row_idx * stride_krylov + col_idx;

        krylov_bases[krylov_idx] /= hessenberg;
    }
}


template <typename ValueType>
__device__ void calculate_sin_and_cos_kernel(
    size_type col_idx, size_type num_cols, size_type iter,
    const ValueType *hessenberg_iter, size_type stride_hessenberg,
    ValueType *givens_sin, size_type stride_sin, ValueType *givens_cos,
    size_type stride_cos, ValueType &register_sin, ValueType &register_cos)
{
    const auto this_hess = hessenberg_iter[iter * stride_hessenberg + col_idx];
    if (this_hess == zero<ValueType>()) {
        register_cos = zero<ValueType>();
        register_sin = one<ValueType>();

    } else {
        const auto next_hess =
            hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];
        auto hypotenuse = sqrt(this_hess * this_hess + next_hess * next_hess);
        register_cos = abs(this_hess) / hypotenuse;
        register_sin = register_cos * next_hess / this_hess;
    }
    givens_cos[iter * stride_cos + col_idx] = register_cos;
    givens_sin[iter * stride_sin + col_idx] = register_sin;
}


template <typename ValueType>
__device__ void calculate_residual_norm_kernel(
    size_type col_idx, size_type num_cols, size_type iter,
    const ValueType &register_sin, const ValueType &register_cos,
    ValueType *residual_norm, ValueType *residual_norm_collection,
    size_type stride_residual_norm_collection, const ValueType *b_norm)
{
    const auto this_rnc =
        residual_norm_collection[iter * stride_residual_norm_collection +
                                 col_idx];
    const auto next_rnc = -register_sin * this_rnc;
    residual_norm_collection[iter * stride_residual_norm_collection + col_idx] =
        register_cos * this_rnc;
    residual_norm[col_idx] = abs(next_rnc) / b_norm[col_idx];
    residual_norm_collection[(iter + 1) * stride_residual_norm_collection +
                             col_idx] = next_rnc;
}


// Must be called with at least `num_cols` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void givens_rotation_kernel(
    size_type num_rows, size_type num_cols, size_type iter,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    ValueType *__restrict__ givens_sin, size_type stride_sin,
    ValueType *__restrict__ givens_cos, size_type stride_cos,
    ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const ValueType *__restrict__ b_norm,
    const stopping_status *__restrict__ stop_status)
{
    const auto col_idx = thread::get_thread_id_flat();

    if (col_idx >= num_cols || stop_status[col_idx].has_stopped()) {
        return;
    }

    // const auto current_thread_block = group::this_thread_block();

    for (size_type i = 0; i < iter; ++i) {
        const auto tmp =
            givens_cos[i * stride_cos + col_idx] *
                hessenberg_iter[i * stride_hessenberg + col_idx] +
            givens_sin[i * stride_sin + col_idx] *
                hessenberg_iter[(i + 1) * stride_hessenberg + col_idx];
        hessenberg_iter[(i + 1) * stride_hessenberg + col_idx] =
            givens_cos[i * stride_cos + col_idx] *
                hessenberg_iter[(i + 1) * stride_hessenberg + col_idx] -
            givens_sin[i * stride_sin + col_idx] *
                hessenberg_iter[i * stride_hessenberg + col_idx];
        hessenberg_iter[i * stride_hessenberg + col_idx] = tmp;
    }
    // for j in 1:iter - 1
    //     temp             =  cos(j)*hessenberg(j) +
    //                         sin(j)*hessenberg(j+1)
    //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
    //                         cos(j)*hessenberg(j+1)
    //     hessenberg(j)    =  temp;
    // end

    ValueType register_sin;
    ValueType register_cos;
    calculate_sin_and_cos_kernel(
        col_idx, num_cols, iter, hessenberg_iter, stride_hessenberg, givens_sin,
        stride_sin, givens_cos, stride_cos, register_sin, register_cos);
    // Calculate sin and cos

    hessenberg_iter[iter * stride_hessenberg + col_idx] =
        register_cos * hessenberg_iter[iter * stride_hessenberg + col_idx] +
        register_sin *
            hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];
    hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] =
        zero<ValueType>();
    // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
    //                      sin(iter)*hessenberg(iter)
    // hessenberg(iter+1) = 0

    calculate_residual_norm_kernel(
        col_idx, num_cols, iter, register_sin, register_cos, residual_norm,
        residual_norm_collection, stride_residual_norm_collection, b_norm);
    // Calculate residual norm
}


// Must be called with at least `num_rhs` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void solve_upper_triangular_kernel(
    size_type num_cols, size_type num_rhs,
    const ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const ValueType *__restrict__ hessenberg, size_type stride_hessenberg,
    ValueType *__restrict__ y, size_type stride_y,
    const size_type *__restrict__ final_iter_nums)
{
    const auto col_idx = thread::get_thread_id_flat();

    if (col_idx >= num_rhs) {
        return;
    }

    for (int i = final_iter_nums[col_idx] - 1; i >= 0; --i) {
        auto temp =
            residual_norm_collection[i * stride_residual_norm_collection +
                                     col_idx];
        for (size_type j = i + 1; j < final_iter_nums[col_idx]; ++j) {
            temp -= hessenberg[i * stride_hessenberg + j * num_rhs + col_idx] *
                    y[j * stride_y + col_idx];
        }

        y[i * stride_y + col_idx] =
            temp / hessenberg[i * stride_hessenberg + i * num_rhs + col_idx];
    }
    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection
}


// Must be called with at least `stride_preconditioner * num_rows` threads in
// total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void calculate_Qy_kernel(
    size_type num_rows, size_type num_cols, size_type num_rhs,
    const ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    const ValueType *__restrict__ y, size_type stride_y,
    ValueType *__restrict__ before_preconditioner,
    size_type stride_preconditioner,
    const size_type *__restrict__ final_iter_nums)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_id = global_id / stride_preconditioner;
    const auto col_id = global_id % stride_preconditioner;

    if (row_id < num_rows && col_id < num_cols) {
        ValueType temp = zero<ValueType>();

        for (size_type j = 0; j < final_iter_nums[col_id]; ++j) {
            temp +=
                krylov_bases[(row_id + j * num_rows) * stride_krylov + col_id] *
                y[j * stride_y + col_id];
        }
        before_preconditioner[global_id] = temp;
    }
}
