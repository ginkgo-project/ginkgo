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

template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void zero_matrix_kernel(
    size_type m, size_type n, size_type stride, ValueType *__restrict__ array)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < n) {
        auto pos = tidx;
        for (size_type k = 0; k < m; ++k) {
            array[pos] = zero<ValueType>();
            pos += stride;
        }
    }
}


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
template <size_type block_size, typename ValueType,
          typename ValueTypeKrylovBases>
__global__ __launch_bounds__(block_size) void initialize_2_1_kernel(
    size_type num_rows, size_type num_rhs, size_type krylov_dim,
    Accessor3d<ValueTypeKrylovBases, ValueType> krylov_bases,
    ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_nc)
{
    const auto global_id = thread::get_thread_id_flat();
    // krylov indices
    const auto krylov_idx = global_id / krylov_bases.get_stride0();
    const auto reminder = global_id % krylov_bases.get_stride0();
    const auto krylov_row = reminder / krylov_bases.get_stride1();
    const auto rhs = reminder % krylov_bases.get_stride1();

    // residual_norm indices (separated for better coalesced access)
    const auto residual_row = global_id / stride_residual_nc;
    const auto residual_col = global_id % stride_residual_nc;

    if (krylov_idx < krylov_dim + 1 && krylov_row < num_rows && rhs < num_rhs) {
        krylov_bases.write(krylov_idx, krylov_row, rhs, zero<ValueType>());
    }

    if (residual_row < krylov_dim + 1 && residual_col < num_rhs) {
        residual_norm_collection[residual_row * stride_residual_nc +
                                 residual_col] = zero<ValueType>();
    }
}


// Must be called with at least `num_rows * num_rhs` threads in total.
template <size_type block_size, typename ValueType,
          typename ValueTypeKrylovBases>
__global__ __launch_bounds__(block_size) void initialize_2_2_kernel(
    size_type num_rows, size_type num_rhs,
    const ValueType *__restrict__ residual, size_type stride_residual,
    const remove_complex<ValueType> *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    Accessor3d<ValueTypeKrylovBases, ValueType> krylov_bases,
    ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
    size_type *__restrict__ final_iter_nums)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / krylov_bases.get_stride1();
    const auto col_idx = global_id % krylov_bases.get_stride1();

    if (global_id < num_rhs) {
        residual_norm_collection[global_id] = residual_norm[global_id];
        final_iter_nums[global_id] = 0;
    }

    if (row_idx < num_rows && col_idx < num_rhs) {
        auto value = residual[row_idx * stride_residual + col_idx] /
                     residual_norm[col_idx];
        krylov_bases.write({0}, row_idx, col_idx, value);
        next_krylov_basis[row_idx * stride_next_krylov + col_idx] = value;
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
__global__ __launch_bounds__(default_dot_size) void multinorm2_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov, remove_complex<ValueType> *__restrict__ norms,
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
    __shared__ UninitializedArray<remove_complex<ValueType>,
                                  default_dot_dim *(default_dot_dim + 1)>
        reduction_helper_array;
    remove_complex<ValueType> *__restrict__ reduction_helper =
        reduction_helper_array;
    remove_complex<ValueType> local_res = zero<remove_complex<ValueType>>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_res += squared_norm(next_krylov_basis[next_krylov_idx]);
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_res;
    __syncthreads();
    local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto sum =
        reduce(tile_block, local_res,
               [](const remove_complex<ValueType> &a,
                  const remove_complex<ValueType> &b) { return a + b; });
    const auto new_col_idx = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto norms_idx = new_col_idx;
        atomic_add(norms + norms_idx, sum);
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_dot_size) void multinorminf_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov, remove_complex<ValueType> *__restrict__ norms,
    size_type stride_norms, const stopping_status *__restrict__ stop_status)
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
    __shared__ UninitializedArray<remove_complex<ValueType>,
                                  default_dot_dim *(default_dot_dim + 1)>
        reduction_helper_array;
    remove_complex<ValueType> *__restrict__ reduction_helper =
        reduction_helper_array;
    remove_complex<ValueType> local_max = zero<remove_complex<ValueType>>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_max = (local_max >= abs(next_krylov_basis[next_krylov_idx]))
                            ? local_max
                            : abs(next_krylov_basis[next_krylov_idx]);
            // local_max = (local_max >= next_krylov_basis[next_krylov_idx])
            //                 ? local_max
            //                 : next_krylov_basis[next_krylov_idx];
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_max;
    __syncthreads();
    local_max = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto value = reduce(
        tile_block, local_max,
        [](const remove_complex<ValueType> &a,
           const remove_complex<ValueType> &b) { return ((a >= b) ? a : b); });
    const auto new_col_idx = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto norms_idx = new_col_idx;
        atomic_max(norms + norms_idx, value);
    }
}


template <typename ValueType>
__global__
    __launch_bounds__(default_dot_size) void multinorminf_kernel_without_stop(
        size_type num_rows, size_type num_cols,
        const ValueType *__restrict__ next_krylov_basis,
        size_type stride_next_krylov,
        remove_complex<ValueType> *__restrict__ norms, size_type stride_norms)
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
    __shared__ UninitializedArray<remove_complex<ValueType>,
                                  default_dot_dim *(default_dot_dim + 1)>
        reduction_helper_array;
    remove_complex<ValueType> *__restrict__ reduction_helper =
        reduction_helper_array;
    remove_complex<ValueType> local_max = zero<remove_complex<ValueType>>();
    if (col_idx < num_cols) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_max = (local_max >= abs(next_krylov_basis[next_krylov_idx]))
                            ? local_max
                            : abs(next_krylov_basis[next_krylov_idx]);
            // local_max = (local_max >= next_krylov_basis[next_krylov_idx])
            //                 ? local_max
            //                 : next_krylov_basis[next_krylov_idx];
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_max;
    __syncthreads();
    local_max = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto value = reduce(
        tile_block, local_max,
        [](const remove_complex<ValueType> &a,
           const remove_complex<ValueType> &b) { return ((a >= b) ? a : b); });
    const auto new_col_idx = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols) {
        const auto norms_idx = new_col_idx;
        atomic_max(norms + norms_idx, value);
    }
}


template <bool compute_inf, typename ValueType>
__global__ __launch_bounds__(default_dot_size) void multinorm2_inf_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov,
    remove_complex<ValueType> *__restrict__ norms1,
    remove_complex<ValueType> *__restrict__ norms2,
    const stopping_status *__restrict__ stop_status)
{
    using non_complex = remove_complex<ValueType>;
    const auto tidx = threadIdx.x;
    const auto tidy = threadIdx.y;
    const auto col_idx = blockIdx.x * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, gridDim.y);
    const auto start_row = blockIdx.y * num;
    const auto end_row =
        ((blockIdx.y + 1) * num > num_rows) ? num_rows : (blockIdx.y + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__ UninitializedArray<
        non_complex, (1 + compute_inf) * default_dot_dim *(default_dot_dim + 1)>
        reduction_helper_array;
    non_complex *__restrict__ reduction_helper_add = reduction_helper_array;
    non_complex *__restrict__ reduction_helper_max =
        static_cast<non_complex *>(reduction_helper_array) +
        default_dot_dim * (default_dot_dim + 1);
    non_complex local_res = zero<non_complex>();
    non_complex local_max = zero<non_complex>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            const auto num = next_krylov_basis[next_krylov_idx];
            local_res += squared_norm(num);
            if (compute_inf) {
                local_max = ((local_max >= abs(num)) ? local_max : abs(num));
            }
        }
    }
    // Add reduction
    reduction_helper_add[tidx * (default_dot_dim + 1) + tidy] = local_res;
    if (compute_inf) {
        reduction_helper_max[tidx * (default_dot_dim + 1) + tidy] = local_max;
    }
    __syncthreads();
    local_res = reduction_helper_add[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto sum = reduce(
        tile_block, local_res,
        [](const non_complex &a, const non_complex &b) { return a + b; });
    non_complex reduced_max{};
    if (compute_inf) {
        local_max = reduction_helper_max[tidy * (default_dot_dim + 1) + tidx];
        reduced_max = reduce(tile_block, local_max,
                             [](const non_complex &a, const non_complex &b) {
                                 return ((a >= b) ? a : b);
                             });
    }
    const auto new_col_idx = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto norms_idx = new_col_idx;
        atomic_add(norms1 + norms_idx, sum);
        if (compute_inf) {
            atomic_max(norms2 + norms_idx, reduced_max);
        }
    }
}


template <typename ValueType, typename ValueTypeKrylovBases>
__global__ __launch_bounds__(default_dot_size) void multidot_kernel(
    size_type k, size_type num_rows, size_type num_rhs,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov,
    Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = threadIdx.x;
    const auto tidy = threadIdx.y;
    const decltype(num_rows) col_idx = blockIdx.x * default_dot_dim + tidx;
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
    // const auto krylov_col = k * num_rhs + col_idx;
    if (col_idx < num_rhs && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_res += next_krylov_basis[next_krylov_idx] *
                         krylov_bases.read(k, i, col_idx);
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
    if (tidx == 0 && new_col_idx < num_rhs &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto hessenberg_idx = k * stride_hessenberg + new_col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, sum);
    }
}


template <typename ValueType, typename ValueTypeKrylovBases>
__global__ __launch_bounds__(default_dot_size) void multidot_kernel_num_iters(
    size_type num_iters, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov,
    Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = threadIdx.x;
    const auto tidy = threadIdx.y;
    const size_type col_idx = blockIdx.x * default_dot_dim + tidx;
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

    for (size_type k = 0; k < num_iters; ++k) {
        ValueType local_res = zero<ValueType>();
        if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
            for (size_type i = start_row + tidy; i < end_row;
                 i += default_dot_dim) {
                const auto next_krylov_idx = i * stride_next_krylov + col_idx;
                local_res += next_krylov_basis[next_krylov_idx] *
                             krylov_bases.read(k, i, col_idx);
            }
        }
        reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_res;
        __syncthreads();
        local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
        const auto tile_block =
            group::tiled_partition<default_dot_dim>(group::this_thread_block());
        const auto sum = reduce(
            tile_block, local_res,
            [](const ValueType &a, const ValueType &b) { return a + b; });
        const auto new_col_idx = blockIdx.x * default_dot_dim + tidy;
        if (tidx == 0 && new_col_idx < num_cols &&
            !stop_status[new_col_idx].has_stopped()) {
            const auto hessenberg_idx = k * stride_hessenberg + new_col_idx;
            atomic_add(hessenberg_iter + hessenberg_idx, sum);
        }
    }
}


template <typename ValueType, typename ValueTypeKrylovBases>
__global__ __launch_bounds__(default_dot_size) void multidot_kernel_num_iters_1(
    size_type num_iters, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov,
    Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = threadIdx.x;
    const auto tidy = threadIdx.y;
    const auto num_cols_iters = num_cols * num_iters;
    const auto col_idx_iters = blockIdx.x * blockDim.x + tidx;
    const auto col_idx = col_idx_iters % num_cols;
    const auto num = ceildiv(num_rows, gridDim.y);
    const auto start_row = blockIdx.y * num;
    const auto end_row =
        ((blockIdx.y + 1) * num > num_rows) ? num_rows : (blockIdx.y + 1) * num;
    const auto k = col_idx_iters / num_cols;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__
        UninitializedArray<ValueType, default_dot_dim *(default_dot_dim + 1)>
            reduction_helper_array;
    ValueType *__restrict__ reduction_helper = reduction_helper_array;

    //    for (size_type k = 0; k < num_iters; ++k) {
    ValueType local_res = zero<ValueType>();
    if (col_idx_iters < num_cols_iters && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_res += next_krylov_basis[next_krylov_idx] *
                         krylov_bases.read(k, i, col_idx);
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_res;
    __syncthreads();
    /*
    if (((blockIdx.x + blockIdx.y) == 0) && ((tidx + tidy) == 0)) {
        //    if ((blockIdx.x == 0) && ((tidx + tidy) == 0)) {
        for (size_type ii = 0; ii < default_dot_dim; ii++)
            for (size_type jj = 0; jj < default_dot_dim + 1; jj++)
                if (jj <= num)
                    printf("XX_%d(%d,%d) = %g\n", (int)blockIdx.y + 1,
                           (int)ii + 1, (int)jj + 1,
                           reduction_helper[(default_dot_dim + 1) * ii + jj]);
    }
    __syncthreads();
    */
    local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto sum =
        reduce(tile_block, local_res,
               [](const ValueType &a, const ValueType &b) { return a + b; });
    const auto new_col_idx_iters = blockIdx.x * default_dot_dim + tidy;
    const auto new_col_idx = new_col_idx_iters % num_cols;
    const auto new_k = new_col_idx_iters / num_cols;
    if (tidx == 0 && new_col_idx_iters < num_cols_iters &&
        !stop_status[new_col_idx].has_stopped()) {
        //        if ((blockIdx.x + blockIdx.y) == 0) {
        /*
        if (blockIdx.x == 0) {
            printf("sum_%d(%d) = %g\n", (int)blockIdx.y + 1, (int)tidy + 1,
                   sum);
        }
        */
        // const auto hessenberg_idx = k * stride_hessenberg + new_col_idx;
        const auto hessenberg_idx = new_k * stride_hessenberg + new_col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, sum);
    }
    //    }
}

template <int dot_dim, typename ValueType, typename ValueTypeKrylovBases>
__global__ __launch_bounds__(dot_dim *dot_dim) void multidot_kernel_num_iters_2(
    size_type num_iters, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov,
    Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    /*
     * In general in this kernel (thread indices appear to be col-major):
     * grid_dim
     *   x: for col_idx (^= which right hand side)
     *   y: for row_idx
     *   z: for num_iters (number of krylov vectors)
     * block_dim
     *   x: for col_idx
     *   y: for row_idx
     *   (z not used, must be set to 1 in dim)
     */
    const size_type tidx = threadIdx.x;
    const size_type tidy = threadIdx.y;
    const size_type col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type num_rows_per_thread = ceildiv(num_rows, gridDim.y);
    const size_type start_row = blockIdx.y * num_rows_per_thread + threadIdx.y;
    const auto end_row = min((blockIdx.y + 1) * num_rows_per_thread, num_rows);
    const size_type k = blockIdx.z;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__ UninitializedArray<ValueType, dot_dim * dot_dim>
        reduction_helper_array;
    ValueType *__restrict__ reduction_helper = reduction_helper_array;

    ValueType local_res = zero<ValueType>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row; i < end_row; i += blockDim.y) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_res += next_krylov_basis[next_krylov_idx] *
                         krylov_bases.read(k, i, col_idx);
        }
    }
    // Transpose local_res, so each warp contains a local_res from the same
    // right hand side
    reduction_helper[tidx * dot_dim + tidy] = local_res;
    auto thread_block = group::this_thread_block();
    thread_block.sync();
    local_res = reduction_helper[tidy * dot_dim + tidx];
    const auto new_col_idx = blockIdx.x * blockDim.x + tidy;
    const auto tile_block = group::tiled_partition<dot_dim>(thread_block);
    const auto sum =
        reduce(tile_block, local_res,
               [](const ValueType &a, const ValueType &b) { return a + b; });
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto hessenberg_idx = k * stride_hessenberg + new_col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, sum);
    }
}


template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__ __launch_bounds__(block_size) void singledot_kernel_num_iters_2(
    size_type num_iters, size_type num_rows,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov,
    Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    /*
     * In general in this kernel (thread indices appear to be col-major):
     * grid_dim
     *   x: for row_idx
     *   y: for num_iters (number of krylov vectors)
     * block_dim
     *   x: for row_idx (must be block_size)
     *   (y not used, must be set to 1 in dim)
     */
    const size_type tidx = threadIdx.x;
    constexpr size_type col_idx{0};
    const size_type k = blockIdx.y;
    const size_type num_rows_per_thread = ceildiv(num_rows, gridDim.x);
    const size_type start_row = blockIdx.x * num_rows_per_thread + threadIdx.x;
    const auto end_row = min((blockIdx.x + 1) * num_rows_per_thread, num_rows);
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__ UninitializedArray<ValueType, block_size> reduction_helper_array;
    ValueType *__restrict__ reduction_helper = reduction_helper_array;

    ValueType local_res = zero<ValueType>();
    if (!stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row; i < end_row; i += block_size) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_res += next_krylov_basis[next_krylov_idx] *
                         krylov_bases.read(k, i, col_idx);
        }
    }
    // Transpose local_res, so each warp contains a local_res from the same
    // right hand side
    reduction_helper[tidx] = local_res;
    auto thread_block = group::this_thread_block();
    thread_block.sync();
    reduce(thread_block, reduction_helper,
           [](const ValueType &a, const ValueType &b) { return a + b; });
    if (tidx == 0 && !stop_status[col_idx].has_stopped()) {
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, reduction_helper[0]);
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__ __launch_bounds__(block_size) void update_next_krylov_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
    Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;

        next_krylov_basis[next_krylov_idx] -=
            hessenberg_iter[hessenberg_idx] *
            krylov_bases.read(k, row_idx, col_idx);
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__
    __launch_bounds__(block_size) void update_next_krylov_kernel_num_iters(
        size_type num_iters, size_type num_rows, size_type num_cols,
        ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
        Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
        const ValueType *__restrict__ hessenberg_iter,
        size_type stride_hessenberg,
        const stopping_status *__restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        for (size_type k = 0; k < num_iters; ++k) {
            // const auto krylov_idx =
            //    row_idx * stride_krylov + k * num_cols + col_idx;
            const auto hessenberg_idx = k * stride_hessenberg + col_idx;

            next_krylov_basis[next_krylov_idx] -=
                hessenberg_iter[hessenberg_idx] *
                krylov_bases.read(k, row_idx, col_idx);
        }
    }
}


// Must be called with at least `num_vectors * krylov_bases.get_stride0()`
// threads in total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__
    __launch_bounds__(block_size) void update_next_krylov_kernel_num_iters_0(
        size_type num_iters, size_type num_rows, size_type num_cols,
        ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
        Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
        const ValueType *__restrict__ hessenberg_iter,
        size_type stride_hessenberg,
        const stopping_status *__restrict__ stop_status)
{
    // CAREFUL: krylov_bases changes might not be done properly since this
    //          kernel is currently not used!
    const auto global_id = thread::get_thread_id_flat();
    const auto k = global_id / krylov_bases.get_stride0();
    const auto reminder = global_id / krylov_bases.get_stride0();
    const auto row_idx = reminder / krylov_bases.get_stride1();
    const auto col_idx = reminder % krylov_bases.get_stride1();

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;

        auto value = hessenberg_iter[hessenberg_idx] *
                     krylov_bases.read(k, row_idx, col_idx);
        atomic_add(next_krylov_basis + next_krylov_idx, -value);
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__
    __launch_bounds__(block_size) void update_next_krylov_kernel_num_iters_1(
        size_type num_iters, size_type num_rows, size_type num_cols,
        ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
        Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
        const ValueType *__restrict__ hessenberg_iter,
        size_type stride_hessenberg,
        const stopping_status *__restrict__ stop_status)
{
    if (num_cols == 1) {
        const auto tidx = threadIdx.x;
        const size_type row_idx = blockIdx.x;
        constexpr decltype(row_idx) col_idx = 0;

        // Used that way to get around dynamic initialization warning and
        // template error when using `reduction_helper_array` directly in
        // `reduce`
        __shared__ UninitializedArray<ValueType, block_size>
            reduction_helper_array;
        ValueType *__restrict__ reduction_helper = reduction_helper_array;
        if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
            ValueType value = zero<ValueType>();
            for (size_type i = tidx; i < num_iters; i += blockDim.x) {
                const auto hessenberg_idx = i * stride_hessenberg + col_idx;

                value += hessenberg_iter[hessenberg_idx] *
                         krylov_bases.read(i, row_idx, col_idx);
            }
            reduction_helper[tidx] = value;

            // Perform thread block reduction. Result is in
            // reduction_helper[0]
            reduce(
                group::this_thread_block(), reduction_helper,
                [](const ValueType &a, const ValueType &b) { return a + b; });
            if (tidx == 0) {
                const auto next_krylov_idx =
                    row_idx * stride_next_krylov + col_idx;
                next_krylov_basis[next_krylov_idx] -= reduction_helper[0];
            }
        }
    } else {
        const auto global_id = thread::get_thread_id_flat();
        const auto row_idx = global_id / stride_next_krylov;
        const auto col_idx = global_id % stride_next_krylov;

        if (row_idx < num_rows && col_idx < num_cols &&
            !stop_status[col_idx].has_stopped()) {
            const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
            for (size_type k = 0; k < num_iters; ++k) {
                const auto hessenberg_idx = k * stride_hessenberg + col_idx;

                next_krylov_basis[next_krylov_idx] -=
                    hessenberg_iter[hessenberg_idx] *
                    krylov_bases.read(k, row_idx, col_idx);
            }
        }
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__
    __launch_bounds__(block_size) void update_next_krylov_kernel_num_iters_2(
        size_type num_iters, size_type num_blocks, size_type num_rows,
        size_type num_cols, ValueType *__restrict__ next_krylov_basis,
        size_type stride_next_krylov,
        Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
        const ValueType *__restrict__ hessenberg_iter,
        size_type stride_hessenberg,
        const stopping_status *__restrict__ stop_status)
{
    if (num_cols == 1) {
        const auto tidx = threadIdx.x;
        const size_type row_idx = blockIdx.x * num_blocks;
        const auto col_idx = 0;

        // Used that way to get around dynamic initialization warning and
        // template error when using `reduction_helper_array` directly in
        // `reduce`
        __shared__ UninitializedArray<ValueType, block_size * block_size>
            reduction_helper_array;
        ValueType *__restrict__ reduction_helper = reduction_helper_array;
        if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
            //            for (size_type j = 0; j < num_blocks; j++) {
            for (size_type j = row_idx, k = 0; j < num_rows && k < num_blocks;
                 j++, k++) {
                ValueType value = zero<ValueType>();
                for (size_type i = tidx; i < num_iters; i += blockDim.x) {
                    const auto hessenberg_idx = i * stride_hessenberg + col_idx;

                    value += hessenberg_iter[hessenberg_idx] *
                             krylov_bases.read(i, j, col_idx);
                }
                reduction_helper[k * block_size + tidx] = value;
            }

            //            for (size_type j = 0; j < num_blocks; j++) {
            for (size_type j = row_idx, k = 0; j < num_rows && k < num_blocks;
                 j++, k++) {
                // Perform thread block reduction. Result is in
                // reduction_helper[0]
                reduce(group::this_thread_block(),
                       reduction_helper + k * block_size,
                       [](const ValueType &a, const ValueType &b) {
                           return a + b;
                       });
            }
            if (tidx == 0) {
                // for (size_type j = 0; j < num_blocks; j++) {
                for (size_type j = row_idx, k = 0;
                     j < num_rows && k < num_blocks; j++, k++) {
                    // (row_idx + j) * stride_next_krylov + col_idx;
                    const auto next_krylov_idx =
                        j * stride_next_krylov + col_idx;
                    next_krylov_basis[next_krylov_idx] -=
                        reduction_helper[k * block_size];
                    // reduction_helper[j * block_size];
                }
            }
        }
    } else {
        const auto global_id = thread::get_thread_id_flat();
        const auto row_idx = global_id / stride_next_krylov;
        const auto col_idx = global_id % stride_next_krylov;

        if (row_idx < num_rows && col_idx < num_cols &&
            !stop_status[col_idx].has_stopped()) {
            const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
            for (size_type k = 0; k < num_iters; ++k) {
                const auto hessenberg_idx = k * stride_hessenberg + col_idx;

                next_krylov_basis[next_krylov_idx] -=
                    hessenberg_iter[hessenberg_idx] *
                    krylov_bases.read(k, row_idx, col_idx);
            }
        }
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__
    __launch_bounds__(block_size) void update_next_krylov_kernel_num_iters_3(
        size_type num_iters, size_type num_blocks, size_type num_rows,
        size_type num_cols, ValueType *__restrict__ next_krylov_basis,
        size_type stride_next_krylov,
        Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
        const ValueType *__restrict__ hessenberg_iter,
        size_type stride_hessenberg,
        const stopping_status *__restrict__ stop_status)
{
    if (num_cols == 1) {
        const auto tidx = threadIdx.x;
        const size_type row_idx = blockIdx.x * num_blocks;
        const auto col_idx = 0;

        // Used that way to get around dynamic initialization warning and
        // template error when using `reduction_helper_array` directly in
        // `reduce`
        __shared__ UninitializedArray<ValueType, block_size * block_size>
            reduction_helper_array;
        ValueType *__restrict__ reduction_helper = reduction_helper_array;
        if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
            //            for (size_type j = 0; j < num_blocks; j++) {

            for (size_type j = row_idx, k = 0; j < num_rows && k < num_blocks;
                 j++, k++) {
                reduction_helper[k * block_size + tidx] = zero<ValueType>();
            }
            for (size_type i = tidx; i < num_iters; i += blockDim.x) {
                const auto hessenberg_idx = i * stride_hessenberg + col_idx;
                const auto hessenberg_val = hessenberg_iter[hessenberg_idx];
                for (size_type j = row_idx, k = 0;
                     j < num_rows && k < num_blocks; j++, k++) {
                    reduction_helper[k * block_size + tidx] +=
                        hessenberg_val * krylov_bases.read(i, j, col_idx);
                }
            }

            //            for (size_type j = 0; j < num_blocks; j++) {
            for (size_type j = row_idx, k = 0; j < num_rows && k < num_blocks;
                 j++, k++) {
                // Perform thread block reduction. Result is in
                // reduction_helper[0]
                reduce(group::this_thread_block(),
                       reduction_helper + k * block_size,
                       [](const ValueType &a, const ValueType &b) {
                           return a + b;
                       });
            }
            if (tidx == 0) {
                // for (size_type j = 0; j < num_blocks; j++) {
                for (size_type j = row_idx, k = 0;
                     j < num_rows && k < num_blocks; j++, k++) {
                    // (row_idx + j) * stride_next_krylov + col_idx;
                    const auto next_krylov_idx =
                        j * stride_next_krylov + col_idx;
                    next_krylov_basis[next_krylov_idx] -=
                        reduction_helper[k * block_size];
                    // reduction_helper[j * block_size];
                }
            }
        }
    } else {
        const auto global_id = thread::get_thread_id_flat();
        const auto row_idx = global_id / stride_next_krylov;
        const auto col_idx = global_id % stride_next_krylov;

        if (row_idx < num_rows && col_idx < num_cols &&
            !stop_status[col_idx].has_stopped()) {
            const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
            for (size_type k = 0; k < num_iters; ++k) {
                // const auto krylov_idx =
                //    row_idx * stride_krylov + k * num_cols + col_idx;
                const auto hessenberg_idx = k * stride_hessenberg + col_idx;

                next_krylov_basis[next_krylov_idx] -=
                    hessenberg_iter[hessenberg_idx] *
                    krylov_bases.read(k, row_idx, col_idx);
            }
        }
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__
    __launch_bounds__(block_size) void update_next_krylov_kernel_num_iters_4(
        size_type num_iters, size_type num_blocks, size_type num_rows,
        size_type num_cols, ValueType *__restrict__ next_krylov_basis,
        size_type stride_next_krylov,
        Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
        const ValueType *__restrict__ hessenberg_iter,
        size_type stride_hessenberg,
        const stopping_status *__restrict__ stop_status)
{
    if (num_cols == 1) {
        const auto tidx = threadIdx.x;
        const size_type row_idx = blockIdx.x * num_blocks;
        const auto col_idx = 0;

        // Used that way to get around dynamic initialization warning and
        // template error when using `reduction_helper_array` directly in
        // `reduce`
        __shared__ UninitializedArray<ValueType, block_size * block_size>
            reduction_helper_array;
        ValueType *__restrict__ reduction_helper = reduction_helper_array;
        if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
            //            for (size_type j = 0; j < num_blocks; j++) {

            for (size_type j = row_idx, k = 0; j < num_rows && k < num_blocks;
                 j++, k++) {
                reduction_helper[k * block_size + tidx] = zero<ValueType>();
            }
            for (size_type i = tidx; i < num_iters; i += blockDim.x) {
                const auto hessenberg_idx = i * stride_hessenberg + col_idx;
                const auto hessenberg_val = hessenberg_iter[hessenberg_idx];
                for (size_type j = row_idx, k = 0;
                     j < num_rows && k < num_blocks; j++, k++) {
                    reduction_helper[k * block_size + tidx] +=
                        hessenberg_val * krylov_bases.read(i, j, col_idx);
                }
            }
            multireduce(
                group::this_thread_block(), reduction_helper, block_size,
                block_size,
                [](const ValueType &a, const ValueType &b) { return a + b; });

            if (tidx == 0) {
                // for (size_type j = 0; j < num_blocks; j++) {
                for (size_type j = row_idx, k = 0;
                     j < num_rows && k < num_blocks; j++, k++) {
                    // (row_idx + j) * stride_next_krylov + col_idx;
                    const auto next_krylov_idx =
                        j * stride_next_krylov + col_idx;
                    next_krylov_basis[next_krylov_idx] -=
                        reduction_helper[k * block_size];
                    // reduction_helper[j * block_size];
                }
            }
        }
    } else {
        const auto global_id = thread::get_thread_id_flat();
        const auto row_idx = global_id / stride_next_krylov;
        const auto col_idx = global_id % stride_next_krylov;

        if (row_idx < num_rows && col_idx < num_cols &&
            !stop_status[col_idx].has_stopped()) {
            const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
            for (size_type k = 0; k < num_iters; ++k) {
                const auto hessenberg_idx = k * stride_hessenberg + col_idx;

                next_krylov_basis[next_krylov_idx] -=
                    hessenberg_iter[hessenberg_idx] *
                    krylov_bases.read(k, row_idx, col_idx);
            }
        }
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__
    __launch_bounds__(block_size) void update_next_krylov_kernel_and_check(
        size_type k, size_type num_rows, size_type num_cols,
        ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
        Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
        const ValueType *__restrict__ hessenberg_iter,
        size_type stride_hessenberg,
        const remove_complex<ValueType> *__restrict__ arnoldi_norm,
        size_type stride_norm, const stopping_status *__restrict__ stop_status,
        stopping_status *__restrict__ reorth_status,
        size_type *__restrict__ num_reorth)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;
        const auto arnoldi_idx = col_idx;
        const auto reorth_idx = col_idx;

        next_krylov_basis[next_krylov_idx] -=
            hessenberg_iter[hessenberg_idx] *
            krylov_bases.read(k, row_idx, col_idx);
        if (row_idx == 0) {
            if (squared_norm(hessenberg_iter[hessenberg_idx]) >
                (0.99 * sqrt(arnoldi_norm[arnoldi_idx]))) {
                reorth_status[reorth_idx].reset();
                atomic_add(num_reorth, (size_type)1);
            } else {
                reorth_status[reorth_idx].stop(1);
            }
        }
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__ __launch_bounds__(block_size) void update_next_krylov_kernel_and_add(
    size_type k, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
    Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const ValueType *__restrict__ buffer_iter, size_type stride_buffer,
    const stopping_status *__restrict__ stop_status,
    const stopping_status *__restrict__ reorth_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped() &&
        !reorth_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;
        const auto buffer_idx = k * stride_buffer + col_idx;
        next_krylov_basis[next_krylov_idx] -=
            buffer_iter[buffer_idx] * krylov_bases.read(k, row_idx, col_idx);
        if ((row_idx == 0) && !reorth_status[col_idx].has_stopped()) {
            hessenberg_iter[hessenberg_idx] += buffer_iter[buffer_idx];
        }
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__
    __launch_bounds__(block_size) void update_next_krylov_kernel_num_iters_and_add(
        size_type num_iters, size_type num_rows, size_type num_cols,
        ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
        Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
        ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
        const ValueType *__restrict__ buffer_iter, size_type stride_buffer,
        const stopping_status *__restrict__ stop_status,
        const stopping_status *__restrict__ reorth_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped() &&
        !reorth_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        for (size_type k = 0; k < num_iters; ++k) {
            // const auto krylov_idx =
            //    row_idx * stride_krylov + k * num_cols + col_idx;
            const auto hessenberg_idx = k * stride_hessenberg + col_idx;
            const auto buffer_idx = k * stride_buffer + col_idx;
            next_krylov_basis[next_krylov_idx] -=
                buffer_iter[buffer_idx] *
                krylov_bases.read(k, row_idx, col_idx);
            if ((row_idx == 0) && !reorth_status[col_idx].has_stopped()) {
                hessenberg_iter[hessenberg_idx] += buffer_iter[buffer_idx];
            }
        }
    }
}

// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__
    __launch_bounds__(block_size) void update_next_krylov_kernel_num_iters_and_add_1(
        size_type num_iters, size_type num_rows, size_type num_cols,
        ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
        Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
        ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
        const ValueType *__restrict__ buffer_iter, size_type stride_buffer,
        const stopping_status *__restrict__ stop_status,
        const stopping_status *__restrict__ reorth_status)
{
    if (num_cols == 1) {
        const auto tidx = threadIdx.x;
        const size_type row_idx = blockIdx.x;
        constexpr decltype(row_idx) col_idx = 0;

        // Used that way to get around dynamic initialization warning and
        // template error when using `reduction_helper_array` directly in
        // `reduce`
        __shared__ UninitializedArray<ValueType, block_size>
            reduction_helper_array;
        ValueType *__restrict__ reduction_helper = reduction_helper_array;
        if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
            ValueType value = zero<ValueType>();
            for (size_type i = tidx; i < num_iters; i += blockDim.x) {
                const auto hessenberg_idx = i * stride_hessenberg + col_idx;
                const auto buffer_idx = i * stride_buffer + col_idx;

                value += buffer_iter[buffer_idx] *
                         krylov_bases.read(i, row_idx, col_idx);
                if ((row_idx == 0) && !reorth_status[col_idx].has_stopped()) {
                    hessenberg_iter[hessenberg_idx] += buffer_iter[buffer_idx];
                }
            }
            reduction_helper[tidx] = value;

            // Perform thread block reduction. Result is in
            // reduction_helper[0]
            reduce(
                group::this_thread_block(), reduction_helper,
                [](const ValueType &a, const ValueType &b) { return a + b; });
            if (tidx == 0) {
                const auto next_krylov_idx =
                    row_idx * stride_next_krylov + col_idx;
                next_krylov_basis[next_krylov_idx] -= reduction_helper[0];
            }
        }
    } else {
        const auto global_id = thread::get_thread_id_flat();
        const auto row_idx = global_id / stride_next_krylov;
        const auto col_idx = global_id % stride_next_krylov;

        if (row_idx < num_rows && col_idx < num_cols &&
            !stop_status[col_idx].has_stopped() &&
            !reorth_status[col_idx].has_stopped()) {
            const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
            for (size_type k = 0; k < num_iters; ++k) {
                // const auto krylov_idx =
                //    row_idx * stride_krylov + k * num_cols + col_idx;
                const auto hessenberg_idx = k * stride_hessenberg + col_idx;
                const auto buffer_idx = k * stride_buffer + col_idx;
                next_krylov_basis[next_krylov_idx] -=
                    buffer_iter[buffer_idx] *
                    krylov_bases.read(k, row_idx, col_idx);
                if ((row_idx == 0) && !reorth_status[col_idx].has_stopped()) {
                    hessenberg_iter[hessenberg_idx] += buffer_iter[buffer_idx];
                }
            }
        }
    }
}


// Must be called with at least `num_cols` blocks, each with `block_size`
// threads. `block_size` must be a power of 2.
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void check_arnoldi_norms(
    const ValueType *__restrict__ arnoldi_norm0, size_type stride_norm0,
    const ValueType *__restrict__ arnoldi_norm1, size_type stride_norm1,
    const stopping_status *__restrict__ stop_status,
    stopping_status *__restrict__ reorth_status,
    size_type *__restrict__ num_reorth)
{
    const ValueType eta = 1.0 / 2.0;
    const auto global_id = thread::get_thread_id_flat();

    if (global_id < stride_norm0 && !stop_status[global_id].has_stopped()) {
        const auto num0 = (eta * arnoldi_norm0[global_id]);
        // const auto num0 = abs(eta * arnoldi_norm0[global_id]);
        const auto num1 = (arnoldi_norm1[global_id]);
        // const auto num1 = abs(arnoldi_norm1[global_id]);
        // printf("num0 = %f , num1 = %f\n", num0, num1);
        if (num1 < num0) {
            reorth_status[global_id].reset();
            atomic_add(num_reorth, (size_type)1);
        } else {
            reorth_status[global_id].stop(1);
        }
    }
}


// Must be called with at least `num_cols` blocks, each with `block_size`
// threads. `block_size` must be a power of 2.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__ __launch_bounds__(block_size) void check_arnoldi_norms_new(
    remove_complex<ValueType> *__restrict__ arnoldi_norm, size_type stride_norm,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    size_type iter, Accessor3d<ValueTypeKrylovBases, ValueType> krylov_bases,
    const stopping_status *__restrict__ stop_status,
    stopping_status *__restrict__ reorth_status,
    size_type *__restrict__ num_reorth)
{
    constexpr bool has_scale =
        Accessor3d<ValueTypeKrylovBases, ValueType>::has_scale;
    const remove_complex<ValueType> eta = 1.0 / 2.0;
    const auto col_idx = thread::get_thread_id_flat();

    if (col_idx < stride_norm && !stop_status[col_idx].has_stopped()) {
        const auto num0 = (eta * arnoldi_norm[col_idx]);
        // const auto num0 = abs(eta * arnoldi_norm0[col_idx]);
        const auto num11 = (arnoldi_norm[col_idx + stride_norm]);
        const auto num12 = sqrt(num11);
        // const auto num1 = abs(arnoldi_norm1[col_idx]);
        const auto num2 = has_scale ? (arnoldi_norm[col_idx + 2 * stride_norm])
                                    : remove_complex<ValueType>{};
        // printf("num0 = %f , num11 = %f , num12 = %f , num2 = %f\n", num0,
        // num11,
        //        num12, num2);
        if (num11 < num0) {
            reorth_status[col_idx].reset();
            atomic_add(num_reorth, (size_type)1);
        } else {
            reorth_status[col_idx].stop(1);
        }
        arnoldi_norm[col_idx] = num11;
        arnoldi_norm[col_idx + stride_norm] = num12;
        hessenberg_iter[iter * stride_hessenberg + col_idx] = num12;
        // printf("(%d) num1 = %10.5e , num2 = %10.5e\n", cod_idx, num1, num2);
        // printf("%d - %d - %d - %d\n", stride_norm, iter, col_idx,
        //        stride_norm * iter + col_idx);
        helper_functions_accessor<ValueType, ValueTypeKrylovBases>::write_scale(
            krylov_bases, iter, col_idx, num2 / num12);
        // krylov_bases, num_cols * (iter + 1) + col_idx, num2 / num12);
        // krylov_bases, col_idx, num2 / num12);
    }
}


// Must be called with at least `num_cols` blocks, each with `block_size`
// threads. `block_size` must be a power of 2.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__ __launch_bounds__(block_size) void set_scale_kernel(
    size_type num_rhs, size_type num_blocks,
    const remove_complex<ValueType> *__restrict__ residual_norm,
    size_type stride_residual,
    const remove_complex<ValueType> *__restrict__ arnoldi_inf,
    size_type stride_inf,
    Accessor3d<ValueTypeKrylovBases, ValueType> krylov_bases)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto blk_idx = global_id / krylov_bases.get_stride1();
    const auto col_idx = global_id % krylov_bases.get_stride1();

    if (blk_idx < num_blocks && col_idx < num_rhs) {
        if (blk_idx == 0) {
            const auto num1 = residual_norm[col_idx];
            const auto num2 = arnoldi_inf[col_idx];
            // printf("num1 = %10.5e , num2 = %10.5e\n", num1, num2);
            helper_functions_accessor<
                ValueType, ValueTypeKrylovBases>::write_scale(krylov_bases, {0},
                                                              col_idx,
                                                              num2 / num1);
        } else {
            const auto num = one<remove_complex<ValueType>>();
            helper_functions_accessor<
                ValueType, ValueTypeKrylovBases>::write_scale(krylov_bases,
                                                              blk_idx, col_idx,
                                                              num);
        }
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
    // template error when using `reduction_helper_array` directly in
    // `reduce`
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


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename ValueTypeKrylovBases>
__global__ __launch_bounds__(block_size) void update_krylov_next_krylov_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
    Accessor3d<ValueTypeKrylovBases, ValueType> krylov_bases,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;
    const auto hessenberg =
        hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;

        const auto next_krylov_value =
            next_krylov_basis[next_krylov_idx] / hessenberg;

        next_krylov_basis[next_krylov_idx] = next_krylov_value;
        krylov_bases.write(iter + 1, row_idx, col_idx, next_krylov_value);
    }
}


template <typename ValueType>
__device__ void calculate_sin_and_cos_kernel(
    size_type col_idx, size_type num_cols, size_type iter,
    const ValueType *hessenberg_iter, size_type stride_hessenberg,
    ValueType *givens_sin, size_type stride_sin, ValueType *givens_cos,
    size_type stride_cos)
{
    if (hessenberg_iter[iter * stride_hessenberg + col_idx] ==
        zero<ValueType>()) {
        givens_cos[iter * stride_cos + col_idx] = zero<ValueType>();
        givens_sin[iter * stride_sin + col_idx] = one<ValueType>();
    } else {
        auto hypotenuse =
            sqrt(hessenberg_iter[iter * stride_hessenberg + col_idx] *
                     hessenberg_iter[iter * stride_hessenberg + col_idx] +
                 hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] *
                     hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx]);
        givens_cos[iter * stride_cos + col_idx] =
            abs(hessenberg_iter[iter * stride_hessenberg + col_idx]) /
            hypotenuse;
        givens_sin[iter * stride_sin + col_idx] =
            givens_cos[iter * stride_cos + col_idx] *
            hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] /
            hessenberg_iter[iter * stride_hessenberg + col_idx];
    }
}


template <typename ValueType>
__device__ void calculate_residual_norm_kernel(
    size_type col_idx, size_type num_cols, size_type iter,
    const ValueType *givens_sin, size_type stride_sin,
    const ValueType *givens_cos, size_type stride_cos,
    remove_complex<ValueType> *residual_norm,
    ValueType *residual_norm_collection,
    size_type stride_residual_norm_collection,
    const remove_complex<ValueType> *b_norm)
{
    residual_norm_collection[(iter + 1) * stride_residual_norm_collection +
                             col_idx] =
        -givens_sin[iter * stride_sin + col_idx] *
        residual_norm_collection[iter * stride_residual_norm_collection +
                                 col_idx];
    residual_norm_collection[iter * stride_residual_norm_collection + col_idx] =
        givens_cos[iter * stride_cos + col_idx] *
        residual_norm_collection[iter * stride_residual_norm_collection +
                                 col_idx];
    residual_norm[col_idx] =
        abs(residual_norm_collection[(iter + 1) *
                                         stride_residual_norm_collection +
                                     col_idx]) /
        b_norm[col_idx];
}


// Must be called with at least `num_cols` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void givens_rotation_kernel(
    size_type num_rows, size_type num_cols, size_type iter,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    ValueType *__restrict__ givens_sin, size_type stride_sin,
    ValueType *__restrict__ givens_cos, size_type stride_cos,
    remove_complex<ValueType> *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const remove_complex<ValueType> *__restrict__ b_norm,
    const stopping_status *__restrict__ stop_status)
{
    const auto col_idx = thread::get_thread_id_flat();

    if (col_idx >= num_cols || stop_status[col_idx].has_stopped()) {
        return;
    }

    const auto current_thread_block = group::this_thread_block();

    for (size_type i = 0; i < iter; ++i) {
        const auto tmp =
            givens_cos[i * stride_cos + col_idx] *
                hessenberg_iter[i * stride_hessenberg + col_idx] +
            givens_sin[i * stride_sin + col_idx] *
                hessenberg_iter[(i + 1) * stride_hessenberg + col_idx];
        current_thread_block.sync();
        hessenberg_iter[(i + 1) * stride_hessenberg + col_idx] =
            givens_cos[i * stride_cos + col_idx] *
                hessenberg_iter[(i + 1) * stride_hessenberg + col_idx] -
            givens_sin[i * stride_sin + col_idx] *
                hessenberg_iter[i * stride_hessenberg + col_idx];
        hessenberg_iter[i * stride_hessenberg + col_idx] = tmp;
        current_thread_block.sync();
    }
    // for j in 1:iter - 1
    //     temp             =  cos(j)*hessenberg(j) +
    //                         sin(j)*hessenberg(j+1)
    //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
    //                         cos(j)*hessenberg(j+1)
    //     hessenberg(j)    =  temp;
    // end

    calculate_sin_and_cos_kernel(col_idx, num_cols, iter, hessenberg_iter,
                                 stride_hessenberg, givens_sin, stride_sin,
                                 givens_cos, stride_cos);
    // Calculate sin and cos

    hessenberg_iter[iter * stride_hessenberg + col_idx] =
        givens_cos[iter * stride_cos + col_idx] *
            hessenberg_iter[iter * stride_hessenberg + col_idx] +
        givens_sin[iter * stride_sin + col_idx] *
            hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];
    hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] =
        zero<ValueType>();
    // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
    //                      sin(iter)*hessenberg(iter)
    // hessenberg(iter+1) = 0

    calculate_residual_norm_kernel(col_idx, num_cols, iter, givens_sin,
                                   stride_sin, givens_cos, stride_cos,
                                   residual_norm, residual_norm_collection,
                                   stride_residual_norm_collection, b_norm);
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


// Must be called with at least `stride_preconditioner * num_rows` threads
// in total.
template <size_type block_size, typename ValueType,
          typename ValueTypeKrylovBases>
__global__ __launch_bounds__(block_size) void calculate_Qy_kernel(
    size_type num_rows, size_type num_cols,
    Accessor3dConst<ValueTypeKrylovBases, ValueType> krylov_bases,
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
                krylov_bases.read(j, row_id, col_id) * y[j * stride_y + col_id];
        }
        before_preconditioner[global_id] = temp;
    }
}
