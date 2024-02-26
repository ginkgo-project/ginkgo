// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_m_kernel(
    size_type subspace_dim, size_type nrhs, ValueType* __restrict__ m_values,
    size_type m_stride, stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row = global_id / m_stride;
    const auto col = global_id % m_stride;

    if (global_id < nrhs) {
        stop_status[global_id].reset();
    }

    if (row < subspace_dim && col < nrhs * subspace_dim) {
        m_values[row * m_stride + col] =
            (row == col / nrhs) ? one<ValueType>() : zero<ValueType>();
    }
}


template <size_type block_size, typename ValueType>
__global__
__launch_bounds__(block_size) void orthonormalize_subspace_vectors_kernel(
    size_type num_rows, size_type num_cols, ValueType* __restrict__ values,
    size_type stride)
{
    const auto tidx = thread::get_thread_id_flat();

    __shared__ uninitialized_array<ValueType, block_size>
        reduction_helper_array;
    // they are not be used in the same time.
    ValueType* reduction_helper = reduction_helper_array;
    auto reduction_helper_real =
        reinterpret_cast<remove_complex<ValueType>*>(reduction_helper);

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type i = 0; i < row; i++) {
            auto dot = zero<ValueType>();
            for (size_type j = tidx; j < num_cols; j += block_size) {
                dot += values[row * stride + j] * conj(values[i * stride + j]);
            }

            // Ensure already finish reading this shared memory
            __syncthreads();
            reduction_helper[tidx] = dot;
            reduce(
                group::this_thread_block(), reduction_helper,
                [](const ValueType& a, const ValueType& b) { return a + b; });
            __syncthreads();

            dot = reduction_helper[0];
            for (size_type j = tidx; j < num_cols; j += block_size) {
                values[row * stride + j] -= dot * values[i * stride + j];
            }
        }

        auto norm = zero<remove_complex<ValueType>>();
        for (size_type j = tidx; j < num_cols; j += block_size) {
            norm += squared_norm(values[row * stride + j]);
        }
        // Ensure already finish reading this shared memory
        __syncthreads();
        reduction_helper_real[tidx] = norm;
        reduce(group::this_thread_block(), reduction_helper_real,
               [](const remove_complex<ValueType>& a,
                  const remove_complex<ValueType>& b) { return a + b; });
        __syncthreads();

        norm = sqrt(reduction_helper_real[0]);
        for (size_type j = tidx; j < num_cols; j += block_size) {
            values[row * stride + j] /= norm;
        }
    }
}


template <typename ValueType>
__global__
__launch_bounds__(default_block_size) void solve_lower_triangular_kernel(
    size_type subspace_dim, size_type nrhs,
    const ValueType* __restrict__ m_values, size_type m_stride,
    const ValueType* __restrict__ f_values, size_type f_stride,
    ValueType* __restrict__ c_values, size_type c_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();

    if (global_id >= nrhs) {
        return;
    }

    if (!stop_status[global_id].has_stopped()) {
        for (size_type row = 0; row < subspace_dim; row++) {
            auto temp = f_values[row * f_stride + global_id];
            for (size_type col = 0; col < row; col++) {
                temp -= m_values[row * m_stride + col * nrhs + global_id] *
                        c_values[col * c_stride + global_id];
            }
            c_values[row * c_stride + global_id] =
                temp / m_values[row * m_stride + row * nrhs + global_id];
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_1_kernel(
    size_type k, size_type num_rows, size_type subspace_dim, size_type nrhs,
    const ValueType* __restrict__ residual_values, size_type residual_stride,
    const ValueType* __restrict__ c_values, size_type c_stride,
    const ValueType* __restrict__ g_values, size_type g_stride,
    ValueType* __restrict__ v_values, size_type v_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row = global_id / nrhs;
    const auto col = global_id % nrhs;

    if (row >= num_rows) {
        return;
    }

    if (!stop_status[col].has_stopped()) {
        auto temp = residual_values[row * residual_stride + col];
        for (size_type j = k; j < subspace_dim; j++) {
            temp -= c_values[j * c_stride + col] *
                    g_values[row * g_stride + j * nrhs + col];
        }
        v_values[row * v_stride + col] = temp;
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_2_kernel(
    size_type k, size_type num_rows, size_type subspace_dim, size_type nrhs,
    const ValueType* __restrict__ omega_values,
    const ValueType* __restrict__ v_values, size_type v_stride,
    const ValueType* __restrict__ c_values, size_type c_stride,
    ValueType* __restrict__ u_values, size_type u_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row = global_id / nrhs;
    const auto col = global_id % nrhs;

    if (row >= num_rows) {
        return;
    }

    if (!stop_status[col].has_stopped()) {
        auto temp = omega_values[col] * v_values[row * v_stride + col];
        for (size_type j = k; j < subspace_dim; j++) {
            temp += c_values[j * c_stride + col] *
                    u_values[row * u_stride + j * nrhs + col];
        }
        u_values[row * u_stride + k * nrhs + col] = temp;
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_dot_size) void multidot_kernel(
    size_type num_rows, size_type nrhs, const ValueType* __restrict__ p_i,
    const ValueType* __restrict__ g_k, size_type g_k_stride,
    ValueType* __restrict__ alpha,
    const stopping_status* __restrict__ stop_status)
{
    const auto tidx = threadIdx.x;
    const auto tidy = threadIdx.y;
    const auto rhs = blockIdx.x * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, gridDim.y);
    const auto start_row = blockIdx.y * num;
    const auto end_row =
        ((blockIdx.y + 1) * num > num_rows) ? num_rows : (blockIdx.y + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__
        uninitialized_array<ValueType, default_dot_dim*(default_dot_dim + 1)>
            reduction_helper_array;
    ValueType* __restrict__ reduction_helper = reduction_helper_array;

    ValueType local_res = zero<ValueType>();
    if (rhs < nrhs && !stop_status[rhs].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto g_idx = i * g_k_stride + rhs;
            local_res += p_i[i] * g_k[g_idx];
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_res;
    __syncthreads();
    local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto sum =
        reduce(tile_block, local_res,
               [](const ValueType& a, const ValueType& b) { return a + b; });
    const auto new_rhs = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_rhs < nrhs && !stop_status[new_rhs].has_stopped()) {
        atomic_add(alpha + new_rhs, sum);
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_g_k_and_u_kernel(
    size_type k, size_type i, size_type size, size_type nrhs,
    const ValueType* __restrict__ alpha, const ValueType* __restrict__ m_values,
    size_type m_stride, const ValueType* __restrict__ g_values,
    size_type g_stride, ValueType* __restrict__ g_k_values,
    size_type g_k_stride, ValueType* __restrict__ u_values, size_type u_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto row = tidx / g_k_stride;
    const auto rhs = tidx % g_k_stride;

    if (row >= size || rhs >= nrhs) {
        return;
    }

    if (!stop_status[rhs].has_stopped()) {
        const auto fact = alpha[rhs] / m_values[i * m_stride + i * nrhs + rhs];
        g_k_values[row * g_k_stride + rhs] -=
            fact * g_values[row * g_stride + i * nrhs + rhs];
        u_values[row * u_stride + k * nrhs + rhs] -=
            fact * u_values[row * u_stride + i * nrhs + rhs];
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_g_kernel(
    size_type k, size_type size, size_type nrhs,
    const ValueType* __restrict__ g_k_values, size_type g_k_stride,
    ValueType* __restrict__ g_values, size_type g_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto row = tidx / g_k_stride;
    const auto rhs = tidx % nrhs;

    if (row >= size || rhs >= nrhs) {
        return;
    }

    if (!stop_status[rhs].has_stopped()) {
        g_values[row * g_stride + k * nrhs + rhs] =
            g_k_values[row * g_k_stride + rhs];
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void update_x_r_and_f_kernel(
    size_type k, size_type size, size_type subspace_dim, size_type nrhs,
    const ValueType* __restrict__ m_values, size_type m_stride,
    const ValueType* __restrict__ g_values, size_type g_stride,
    const ValueType* __restrict__ u_values, size_type u_stride,
    ValueType* __restrict__ f_values, size_type f_stride,
    ValueType* __restrict__ r_values, size_type r_stride,
    ValueType* __restrict__ x_values, size_type x_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row = global_id / x_stride;
    const auto col = global_id % x_stride;

    if (row >= size || col >= nrhs) {
        return;
    }

    if (!stop_status[col].has_stopped()) {
        const auto beta = f_values[k * f_stride + col] /
                          m_values[k * m_stride + k * nrhs + col];
        r_values[row * r_stride + col] -=
            beta * g_values[row * g_stride + k * nrhs + col];
        x_values[row * x_stride + col] +=
            beta * u_values[row * u_stride + k * nrhs + col];

        if (k < row && k + 1 < subspace_dim && row < subspace_dim) {
            f_values[row * f_stride + col] -=
                beta * m_values[row * m_stride + k * nrhs + col];
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(config::warp_size) void compute_omega_kernel(
    size_type nrhs, const remove_complex<ValueType> kappa,
    const ValueType* __restrict__ tht,
    const remove_complex<ValueType>* __restrict__ residual_norm,
    ValueType* __restrict__ omega,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();

    if (global_id >= nrhs) {
        return;
    }

    if (!stop_status[global_id].has_stopped()) {
        auto thr = omega[global_id];
        omega[global_id] /= tht[global_id];
        auto absrho =
            abs(thr / (sqrt(real(tht[global_id])) * residual_norm[global_id]));

        if (absrho < kappa) {
            omega[global_id] *= kappa / absrho;
        }
    }
}
