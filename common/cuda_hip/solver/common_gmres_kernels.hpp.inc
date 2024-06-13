// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// Must be called with at least `max(stride_b * num_rows, krylov_dim *
// num_cols)` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_kernel(
    size_type num_rows, size_type num_cols, size_type krylov_dim,
    const ValueType* __restrict__ b, size_type stride_b,
    ValueType* __restrict__ residual, size_type stride_residual,
    ValueType* __restrict__ givens_sin, size_type stride_sin,
    ValueType* __restrict__ givens_cos, size_type stride_cos,
    stopping_status* __restrict__ stop_status)
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


template <typename ValueType>
__device__ void calculate_sin_and_cos_kernel(
    size_type col_idx, size_type num_cols, size_type iter,
    const ValueType& this_hess, const ValueType& next_hess,
    ValueType* givens_sin, size_type stride_sin, ValueType* givens_cos,
    size_type stride_cos, ValueType& register_sin, ValueType& register_cos)
{
    if (is_zero(this_hess)) {
        register_cos = zero<ValueType>();
        register_sin = one<ValueType>();
    } else {
        const auto scale = abs(this_hess) + abs(next_hess);
        const auto hypotenuse =
            scale * sqrt(abs(this_hess / scale) * abs(this_hess / scale) +
                         abs(next_hess / scale) * abs(next_hess / scale));
        register_cos = conj(this_hess) / hypotenuse;
        register_sin = conj(next_hess) / hypotenuse;
    }
    givens_cos[iter * stride_cos + col_idx] = register_cos;
    givens_sin[iter * stride_sin + col_idx] = register_sin;
}


template <typename ValueType>
__device__ void calculate_residual_norm_kernel(
    size_type col_idx, size_type num_cols, size_type iter,
    const ValueType& register_sin, const ValueType& register_cos,
    remove_complex<ValueType>* residual_norm,
    ValueType* residual_norm_collection,
    size_type stride_residual_norm_collection)
{
    const auto this_rnc =
        residual_norm_collection[iter * stride_residual_norm_collection +
                                 col_idx];
    const auto next_rnc = -conj(register_sin) * this_rnc;
    residual_norm_collection[iter * stride_residual_norm_collection + col_idx] =
        register_cos * this_rnc;
    residual_norm[col_idx] = abs(next_rnc);
    residual_norm_collection[(iter + 1) * stride_residual_norm_collection +
                             col_idx] = next_rnc;
}


// Must be called with at least `num_cols` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void givens_rotation_kernel(
    size_type num_rows, size_type num_cols, size_type iter,
    ValueType* __restrict__ hessenberg_iter, size_type stride_hessenberg,
    ValueType* __restrict__ givens_sin, size_type stride_sin,
    ValueType* __restrict__ givens_cos, size_type stride_cos,
    remove_complex<ValueType>* __restrict__ residual_norm,
    ValueType* __restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const stopping_status* __restrict__ stop_status)
{
    const auto col_idx = thread::get_thread_id_flat();

    if (col_idx >= num_cols || stop_status[col_idx].has_stopped()) {
        return;
    }

    auto this_hess = hessenberg_iter[col_idx];
    auto next_hess = hessenberg_iter[stride_hessenberg + col_idx];
    for (size_type i = 0; i < iter; ++i) {
        const auto cos = givens_cos[i * stride_cos + col_idx];
        const auto sin = givens_sin[i * stride_sin + col_idx];
        hessenberg_iter[i * stride_hessenberg + col_idx] =
            cos * this_hess + sin * next_hess;
        this_hess = conj(cos) * next_hess - conj(sin) * this_hess;
        next_hess = hessenberg_iter[(i + 2) * stride_hessenberg + col_idx];
    }
    // for j in 0:iter - 1
    //     temp             =  cos(j)*hessenberg(j) +
    //                         sin(j)*hessenberg(j+1)
    //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
    //                         cos(j)*hessenberg(j+1)
    //     hessenberg(j)    =  temp;
    // end

    ValueType register_sin;
    ValueType register_cos;
    calculate_sin_and_cos_kernel(col_idx, num_cols, iter, this_hess, next_hess,
                                 givens_sin, stride_sin, givens_cos, stride_cos,
                                 register_sin, register_cos);
    // Calculate sin and cos on hessenberg(iter) and hessenberg(iter+1)

    hessenberg_iter[iter * stride_hessenberg + col_idx] =
        register_cos * this_hess + register_sin * next_hess;
    hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] =
        zero<ValueType>();
    // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
    //                      sin(iter)*hessenberg(iter+1)
    // hessenberg(iter+1) = 0

    calculate_residual_norm_kernel(
        col_idx, num_cols, iter, register_sin, register_cos, residual_norm,
        residual_norm_collection, stride_residual_norm_collection);
    // Calculate residual norm
}


// Must be called with at least `num_rhs` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void solve_upper_triangular_kernel(
    size_type num_cols, size_type num_rhs,
    const ValueType* __restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const ValueType* __restrict__ hessenberg, size_type stride_hessenberg,
    ValueType* __restrict__ y, size_type stride_y,
    const size_type* __restrict__ final_iter_nums)
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
