// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType>
__device__ __forceinline__ void simple_apply(
    const gko::batch::matrix::dense::batch_item<const ValueType>& mat,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ x)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();
    auto subgroup = group::tiled_partition<tile_size>(thread_block);
    const auto subgroup_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subgroups_per_block = ceildiv(blockDim.x, tile_size);

    for (int row = subgroup_id; row < mat.num_rows;
         row += num_subgroups_per_block) {
        ValueType temp = zero<ValueType>();
        for (int j = subgroup.thread_rank(); j < mat.num_cols;
             j += subgroup.size()) {
            const ValueType val = mat.values[row * mat.stride + j];
            temp += val * b[j];
        }

        // subgroup level reduction
        temp = reduce(subgroup, temp, thrust::plus<ValueType>{});

        if (subgroup.thread_rank() == 0) {
            x[row] = temp;
        }
    }
}

template <typename ValueType>
__global__ __launch_bounds__(
    default_block_size,
    sm_oversubscription) void simple_apply_kernel(const gko::batch::matrix::
                                                      dense::uniform_batch<
                                                          const ValueType>
                                                          mat,
                                                  const gko::batch::
                                                      multi_vector::
                                                          uniform_batch<
                                                              const ValueType>
                                                              b,
                                                  const gko::batch::
                                                      multi_vector::
                                                          uniform_batch<
                                                              ValueType>
                                                              x)
{
    for (size_type batch_id = blockIdx.x; batch_id < mat.num_batch_items;
         batch_id += gridDim.x) {
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const auto b_b = gko::batch::extract_batch_item(b, batch_id);
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        simple_apply(mat_b, b_b.values, x_b.values);
    }
}


template <typename ValueType>
__device__ __forceinline__ void advanced_apply(
    const ValueType alpha,
    const gko::batch::matrix::dense::batch_item<const ValueType>& mat,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ x)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();
    auto subgroup = group::tiled_partition<tile_size>(thread_block);
    const auto subgroup_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subgroups_per_block = ceildiv(blockDim.x, tile_size);

    for (int row = subgroup_id; row < mat.num_rows;
         row += num_subgroups_per_block) {
        ValueType temp = zero<ValueType>();
        for (int j = subgroup.thread_rank(); j < mat.num_cols;
             j += subgroup.size()) {
            const ValueType val = mat.values[row * mat.stride + j];
            temp += alpha * val * b[j];
        }

        // subgroup level reduction
        temp = reduce(subgroup, temp, thrust::plus<ValueType>{});

        if (subgroup.thread_rank() == 0) {
            x[row] = temp + beta * x[row];
        }
    }
}

template <typename ValueType>
__global__ __launch_bounds__(
    default_block_size,
    sm_oversubscription) void advanced_apply_kernel(const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                const ValueType>
                                                                alpha,
                                                    const gko::batch::matrix::
                                                        dense::uniform_batch<
                                                            const ValueType>
                                                            mat,
                                                    const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                const ValueType>
                                                                b,
                                                    const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                const ValueType>
                                                                beta,
                                                    const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                ValueType>
                                                                x)
{
    for (size_type batch_id = blockIdx.x; batch_id < mat.num_batch_items;
         batch_id += gridDim.x) {
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const auto b_b = gko::batch::extract_batch_item(b, batch_id);
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        const auto alpha_b = gko::batch::extract_batch_item(alpha, batch_id);
        const auto beta_b = gko::batch::extract_batch_item(beta, batch_id);
        advanced_apply(alpha_b.values[0], mat_b, b_b.values, beta_b.values[0],
                       x_b.values);
    }
}


template <typename ValueType>
__device__ __forceinline__ void scale(const int num_rows, const int stride,
                                      const int num_cols,
                                      const ValueType* const col_scale,
                                      const ValueType* const row_scale,
                                      ValueType* const mat)
{
    for (int iz = threadIdx.x; iz < num_rows * num_cols; iz += blockDim.x) {
        const int row = iz / stride;
        const int col = iz % stride;
        mat[row * stride + col] *= col_scale[col] * row_scale[row];
    }
}


template <typename ValueType>
__global__ void scale_kernel(
    const ValueType* const __restrict__ col_scale_vals,
    const ValueType* const __restrict__ row_scale_vals,
    const gko::batch::matrix::dense::uniform_batch<ValueType> mat)
{
    const int num_rows = mat.num_rows;
    const int stride = mat.stride;
    const int num_cols = mat.num_cols;
    const size_type num_batch_items = mat.num_batch_items;
    for (size_type batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        const auto col_scale_b = col_scale_vals + num_cols * batch_id;
        const auto row_scale_b = row_scale_vals + num_rows * batch_id;
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        scale(num_rows, stride, num_cols, col_scale_b, row_scale_b,
              mat_b.values);
    }
}


template <typename ValueType>
__device__ __forceinline__ void scale_add(
    const ValueType alpha,
    const gko::batch::matrix::dense::batch_item<const ValueType>& mat,
    const gko::batch::matrix::dense::batch_item<ValueType>& in_out)
{
    // TODO: add stride support
    for (int iz = threadIdx.x; iz < mat.num_rows * mat.num_cols;
         iz += blockDim.x) {
        const int row = iz / mat.num_cols;
        const int col = iz % mat.num_cols;
        in_out.values[row * in_out.stride + col] =
            alpha * in_out.values[row * in_out.stride + col] +
            mat.values[row * mat.stride + col];
    }
}


template <typename ValueType>
__global__ void scale_add_kernel(
    const gko::batch::multi_vector::uniform_batch<const ValueType> alpha,
    const gko::batch::matrix::dense::uniform_batch<const ValueType> mat,
    const gko::batch::matrix::dense::uniform_batch<ValueType> in_out)
{
    const size_type num_batch_items = mat.num_batch_items;
    for (size_type batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        const auto alpha_b = gko::batch::extract_batch_item(alpha, batch_id);
        const auto in_out_b =
            gko::batch::matrix::extract_batch_item(in_out, batch_id);
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        scale_add(alpha_b.values[0], mat_b, in_out_b);
    }
}


template <typename ValueType>
__device__ __forceinline__ void add_scaled_identity(
    const ValueType alpha, const ValueType beta,
    const gko::batch::matrix::dense::batch_item<ValueType>& mat)
{
    // TODO: add stride support
    for (int iz = threadIdx.x; iz < mat.num_rows * mat.num_cols;
         iz += blockDim.x) {
        const int row = iz / mat.num_cols;
        const int col = iz % mat.num_cols;
        mat.values[row * mat.stride + col] *= beta;
        if (row == col) {
            mat.values[row * mat.stride + col] += alpha;
        }
    }
}


template <typename ValueType>
__global__ void add_scaled_identity_kernel(
    const gko::batch::multi_vector::uniform_batch<const ValueType> alpha,
    const gko::batch::multi_vector::uniform_batch<const ValueType> beta,
    const gko::batch::matrix::dense::uniform_batch<ValueType> mat)
{
    const size_type num_batch_items = mat.num_batch_items;
    for (size_type batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        const auto alpha_b = gko::batch::extract_batch_item(alpha, batch_id);
        const auto beta_b = gko::batch::extract_batch_item(beta, batch_id);
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        add_scaled_identity(alpha_b.values[0], beta_b.values[0], mat_b);
    }
}
