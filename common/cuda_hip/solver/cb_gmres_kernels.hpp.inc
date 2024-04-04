// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/solver/common_gmres_kernels.hpp.inc"


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void zero_matrix_kernel(
    size_type m, size_type n, size_type stride, ValueType* __restrict__ array)
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


// Must be called with at least `num_rows * stride_krylov` threads in total.
template <size_type block_size, typename ValueType, typename Accessor3d>
__global__ __launch_bounds__(block_size) void restart_1_kernel(
    size_type num_rows, size_type num_rhs, size_type krylov_dim,
    Accessor3d krylov_bases, ValueType* __restrict__ residual_norm_collection,
    size_type stride_residual_nc)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto krylov_stride =
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::get_stride(
            krylov_bases);
    // krylov indices
    const auto krylov_idx = global_id / krylov_stride[0];
    const auto reminder = global_id % krylov_stride[0];
    const auto krylov_row = reminder / krylov_stride[1];
    const auto rhs = reminder % krylov_stride[1];

    // residual_norm indices (separated for better coalesced access)
    const auto residual_row = global_id / stride_residual_nc;
    const auto residual_col = global_id % stride_residual_nc;

    if (krylov_idx < krylov_dim + 1 && krylov_row < num_rows && rhs < num_rhs) {
        krylov_bases(krylov_idx, krylov_row, rhs) = zero<ValueType>();
    }

    if (residual_row < krylov_dim + 1 && residual_col < num_rhs) {
        residual_norm_collection[residual_row * stride_residual_nc +
                                 residual_col] = zero<ValueType>();
    }
}


// Must be called with at least `num_rows * num_rhs` threads in total.
template <size_type block_size, typename ValueType, typename Accessor3d>
__global__ __launch_bounds__(block_size) void restart_2_kernel(
    size_type num_rows, size_type num_rhs,
    const ValueType* __restrict__ residual, size_type stride_residual,
    const remove_complex<ValueType>* __restrict__ residual_norm,
    ValueType* __restrict__ residual_norm_collection, Accessor3d krylov_bases,
    ValueType* __restrict__ next_krylov_basis, size_type stride_next_krylov,
    size_type* __restrict__ final_iter_nums)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto krylov_stride =
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::get_stride(
            krylov_bases);
    const auto row_idx = global_id / krylov_stride[1];
    const auto col_idx = global_id % krylov_stride[1];

    if (global_id < num_rhs) {
        residual_norm_collection[global_id] = residual_norm[global_id];
        final_iter_nums[global_id] = 0;
    }

    if (row_idx < num_rows && col_idx < num_rhs) {
        auto value = residual[row_idx * stride_residual + col_idx] /
                     residual_norm[col_idx];
        krylov_bases(0, row_idx, col_idx) = value;
        next_krylov_basis[row_idx * stride_next_krylov + col_idx] = value;
    }
}


__global__
__launch_bounds__(default_block_size) void increase_final_iteration_numbers_kernel(
    size_type* __restrict__ final_iter_nums,
    const stopping_status* __restrict__ stop_status, size_type total_number)
{
    const auto global_id = thread::get_thread_id_flat();
    if (global_id < total_number) {
        final_iter_nums[global_id] += !stop_status[global_id].has_stopped();
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_dot_size) void multinorm2_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType* __restrict__ next_krylov_basis,
    size_type stride_next_krylov, remove_complex<ValueType>* __restrict__ norms,
    const stopping_status* __restrict__ stop_status)
{
    using rc_vtype = remove_complex<ValueType>;
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
        uninitialized_array<rc_vtype, default_dot_dim*(default_dot_dim + 1)>
            reduction_helper_array;
    rc_vtype* __restrict__ reduction_helper = reduction_helper_array;
    rc_vtype local_res = zero<rc_vtype>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_res += squared_norm(next_krylov_basis[next_krylov_idx]);
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_res;
    group::this_thread_block().sync();
    local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto sum =
        reduce(tile_block, local_res,
               [](const rc_vtype& a, const rc_vtype& b) { return a + b; });
    const auto new_col_idx = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto norms_idx = new_col_idx;
        atomic_add(norms + norms_idx, sum);
    }
}


template <typename ValueType>
__global__
__launch_bounds__(default_dot_size) void multinorminf_without_stop_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType* __restrict__ next_krylov_basis,
    size_type stride_next_krylov, remove_complex<ValueType>* __restrict__ norms,
    size_type stride_norms)
{
    using rc_vtype = remove_complex<ValueType>;
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
        uninitialized_array<rc_vtype, default_dot_dim*(default_dot_dim + 1)>
            reduction_helper_array;
    rc_vtype* __restrict__ reduction_helper = reduction_helper_array;
    rc_vtype local_max = zero<rc_vtype>();
    if (col_idx < num_cols) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_max = (local_max >= abs(next_krylov_basis[next_krylov_idx]))
                            ? local_max
                            : abs(next_krylov_basis[next_krylov_idx]);
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_max;
    group::this_thread_block().sync();
    local_max = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto value =
        reduce(tile_block, local_max, [](const rc_vtype& a, const rc_vtype& b) {
            return ((a >= b) ? a : b);
        });
    const auto new_col_idx = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols) {
        const auto norms_idx = new_col_idx;
        atomic_max(norms + norms_idx, value);
    }
}


// ONLY computes the inf-norm (into norms2) when compute_inf is true
template <bool compute_inf, typename ValueType>
__global__ __launch_bounds__(default_dot_size) void multinorm2_inf_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType* __restrict__ next_krylov_basis,
    size_type stride_next_krylov,
    remove_complex<ValueType>* __restrict__ norms1,
    remove_complex<ValueType>* __restrict__ norms2,
    const stopping_status* __restrict__ stop_status)
{
    using rc_vtype = remove_complex<ValueType>;
    const auto tidx = threadIdx.x;
    const auto tidy = threadIdx.y;
    const auto col_idx = blockIdx.x * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, gridDim.y);
    const auto start_row = blockIdx.y * num;
    const auto end_row =
        ((blockIdx.y + 1) * num > num_rows) ? num_rows : (blockIdx.y + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__ uninitialized_array<
        rc_vtype, (1 + compute_inf) * default_dot_dim*(default_dot_dim + 1)>
        reduction_helper_array;
    rc_vtype* __restrict__ reduction_helper_add = reduction_helper_array;
    rc_vtype* __restrict__ reduction_helper_max =
        static_cast<rc_vtype*>(reduction_helper_array) +
        default_dot_dim * (default_dot_dim + 1);
    rc_vtype local_res = zero<rc_vtype>();
    rc_vtype local_max = zero<rc_vtype>();
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
    group::this_thread_block().sync();
    local_res = reduction_helper_add[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto sum =
        reduce(tile_block, local_res,
               [](const rc_vtype& a, const rc_vtype& b) { return a + b; });
    rc_vtype reduced_max{};
    if (compute_inf) {
        local_max = reduction_helper_max[tidy * (default_dot_dim + 1) + tidx];
        reduced_max = reduce(tile_block, local_max,
                             [](const rc_vtype& a, const rc_vtype& b) {
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


template <int dot_dim, typename ValueType, typename Accessor3d>
__global__ __launch_bounds__(dot_dim* dot_dim) void multidot_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType* __restrict__ next_krylov_basis,
    size_type stride_next_krylov, const Accessor3d krylov_bases,
    ValueType* __restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status* __restrict__ stop_status)
{
    /*
     * In general in this kernel:
     * grid_dim
     *   x: for col_idx (^= which right hand side)
     *   y: for row_idx
     *   z: for num_iters (number of krylov vectors)
     * block_dim
     *   x: for col_idx (must be < dot_dim)
     *   y: for row_idx (must be < dot_dim)
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
    __shared__ uninitialized_array<ValueType, dot_dim * dot_dim>
        reduction_helper_array;
    ValueType* __restrict__ reduction_helper = reduction_helper_array;

    ValueType local_res = zero<ValueType>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row; i < end_row; i += blockDim.y) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_res += next_krylov_basis[next_krylov_idx] *
                         conj(krylov_bases(k, i, col_idx));
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
               [](const ValueType& a, const ValueType& b) { return a + b; });
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto hessenberg_idx = k * stride_hessenberg + new_col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, sum);
    }
}


template <int block_size, typename ValueType, typename Accessor3d>
__global__ __launch_bounds__(block_size) void singledot_kernel(
    size_type num_rows, const ValueType* __restrict__ next_krylov_basis,
    size_type stride_next_krylov, const Accessor3d krylov_bases,
    ValueType* __restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status* __restrict__ stop_status)
{
    /*
     * In general in this kernel:
     * grid_dim
     *   x: for row_idx
     *   y: for num_iters (number of krylov vectors)
     * block_dim
     *   x: for row_idx (must be block_size)
     *   (y and z not used, must be set to 1 in dim)
     */
    const size_type tidx = threadIdx.x;
    constexpr size_type col_idx{0};
    const size_type k = blockIdx.y;
    const size_type num_rows_per_thread = ceildiv(num_rows, gridDim.x);
    const size_type start_row = blockIdx.x * num_rows_per_thread + threadIdx.x;
    const auto end_row = min((blockIdx.x + 1) * num_rows_per_thread, num_rows);
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__ uninitialized_array<ValueType, block_size>
        reduction_helper_array;
    ValueType* __restrict__ reduction_helper = reduction_helper_array;

    ValueType local_res = zero<ValueType>();
    if (!stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row; i < end_row; i += block_size) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_res += next_krylov_basis[next_krylov_idx] *
                         conj(krylov_bases(k, i, col_idx));
        }
    }
    // Transpose local_res, so each warp contains a local_res from the same
    // right hand side
    reduction_helper[tidx] = local_res;
    auto thread_block = group::this_thread_block();
    thread_block.sync();
    reduce(thread_block, reduction_helper,
           [](const ValueType& a, const ValueType& b) { return a + b; });
    if (tidx == 0 && !stop_status[col_idx].has_stopped()) {
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, reduction_helper[0]);
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename Accessor3d>
__global__ __launch_bounds__(block_size) void update_next_krylov_kernel(
    size_type num_iters, size_type num_rows, size_type num_cols,
    ValueType* __restrict__ next_krylov_basis, size_type stride_next_krylov,
    const Accessor3d krylov_bases,
    const ValueType* __restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        auto local_res = next_krylov_basis[next_krylov_idx];
        for (size_type k = 0; k < num_iters; ++k) {
            const auto hessenberg_idx = k * stride_hessenberg + col_idx;

            local_res -= hessenberg_iter[hessenberg_idx] *
                         krylov_bases(k, row_idx, col_idx);
        }
        next_krylov_basis[next_krylov_idx] = local_res;
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename Accessor3d>
__global__ __launch_bounds__(block_size) void update_next_krylov_and_add_kernel(
    size_type num_iters, size_type num_rows, size_type num_cols,
    ValueType* __restrict__ next_krylov_basis, size_type stride_next_krylov,
    const Accessor3d krylov_bases, ValueType* __restrict__ hessenberg_iter,
    size_type stride_hessenberg, const ValueType* __restrict__ buffer_iter,
    size_type stride_buffer, const stopping_status* __restrict__ stop_status,
    const stopping_status* __restrict__ reorth_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped() &&
        !reorth_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        auto local_res = next_krylov_basis[next_krylov_idx];
        for (size_type k = 0; k < num_iters; ++k) {
            const auto hessenberg_idx = k * stride_hessenberg + col_idx;
            const auto buffer_idx = k * stride_buffer + col_idx;
            local_res -=
                buffer_iter[buffer_idx] * krylov_bases(k, row_idx, col_idx);
            if ((row_idx == 0) && !reorth_status[col_idx].has_stopped()) {
                hessenberg_iter[hessenberg_idx] += buffer_iter[buffer_idx];
            }
        }
        next_krylov_basis[next_krylov_idx] = local_res;
    }
}


// Must be called with at least `num_rhs` threads
template <int block_size, typename ValueType, typename Accessor3d>
__global__ __launch_bounds__(block_size) void check_arnoldi_norms(
    size_type num_rhs, remove_complex<ValueType>* __restrict__ arnoldi_norm,
    size_type stride_norm, ValueType* __restrict__ hessenberg_iter,
    size_type stride_hessenberg, size_type iter, Accessor3d krylov_bases,
    const stopping_status* __restrict__ stop_status,
    stopping_status* __restrict__ reorth_status,
    size_type* __restrict__ num_reorth)
{
    const remove_complex<ValueType> eta_squared = 1.0 / 2.0;
    const auto col_idx = thread::get_thread_id_flat();
    constexpr bool has_scalar =
        gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3d>::value;

    if (col_idx < num_rhs && !stop_status[col_idx].has_stopped()) {
        const auto num0 = (sqrt(eta_squared * arnoldi_norm[col_idx]));
        const auto num11 = sqrt(arnoldi_norm[col_idx + stride_norm]);
        const auto num2 = has_scalar ? (arnoldi_norm[col_idx + 2 * stride_norm])
                                     : remove_complex<ValueType>{};
        if (num11 < num0) {
            reorth_status[col_idx].reset();
            atomic_add(num_reorth, one<size_type>());
        } else {
            reorth_status[col_idx].stop(1);
        }
        arnoldi_norm[col_idx] = num0;
        arnoldi_norm[col_idx + stride_norm] = num11;
        hessenberg_iter[iter * stride_hessenberg + col_idx] = num11;
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
            krylov_bases, iter, col_idx, num2 / num11);
    }
}


template <int block_size, typename RealValueType, typename Accessor3d>
__global__ __launch_bounds__(block_size) void set_scalar_kernel(
    size_type num_rhs, size_type num_blocks,
    const RealValueType* __restrict__ residual_norm, size_type stride_residual,
    const RealValueType* __restrict__ arnoldi_inf, size_type stride_inf,
    Accessor3d krylov_bases)
{
    static_assert(!is_complex_s<RealValueType>::value,
                  "ValueType must not be complex!");
    const auto global_id = thread::get_thread_id_flat();
    const auto krylov_stride =
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::get_stride(
            krylov_bases);
    const auto blk_idx = global_id / krylov_stride[1];
    const auto col_idx = global_id % krylov_stride[1];

    if (blk_idx < num_blocks && col_idx < num_rhs) {
        if (blk_idx == 0) {
            const auto num1 = residual_norm[col_idx];
            const auto num2 = arnoldi_inf[col_idx];
            gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
                krylov_bases, {0}, col_idx, num2 / num1);
        } else {
            const auto num = one<RealValueType>();
            gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
                krylov_bases, blk_idx, col_idx, num);
        }
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename Accessor3d>
__global__ __launch_bounds__(block_size) void update_krylov_next_krylov_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    ValueType* __restrict__ next_krylov_basis, size_type stride_next_krylov,
    Accessor3d krylov_bases, const ValueType* __restrict__ hessenberg_iter,
    size_type stride_hessenberg,
    const stopping_status* __restrict__ stop_status)
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
        krylov_bases(iter + 1, row_idx, col_idx) = next_krylov_value;
    }
}


// Must be called with at least `stride_preconditioner * num_rows` threads
// in total.
template <size_type block_size, typename ValueType, typename Accessor3d>
__global__ __launch_bounds__(block_size) void calculate_Qy_kernel(
    size_type num_rows, size_type num_cols, const Accessor3d krylov_bases,
    const ValueType* __restrict__ y, size_type stride_y,
    ValueType* __restrict__ before_preconditioner,
    size_type stride_preconditioner,
    const size_type* __restrict__ final_iter_nums)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_id = global_id / stride_preconditioner;
    const auto col_id = global_id % stride_preconditioner;

    if (row_id < num_rows && col_id < num_cols) {
        ValueType temp = zero<ValueType>();
        for (size_type j = 0; j < final_iter_nums[col_id]; ++j) {
            temp += krylov_bases(j, row_id, col_id) * y[j * stride_y + col_id];
        }
        before_preconditioner[global_id] = temp;
    }
}
