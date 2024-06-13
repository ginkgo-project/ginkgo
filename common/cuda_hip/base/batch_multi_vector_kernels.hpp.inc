// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType, typename Mapping>
__device__ __forceinline__ void scale(
    const gko::batch::multi_vector::batch_item<const ValueType>& alpha,
    const gko::batch::multi_vector::batch_item<ValueType>& x, Mapping map)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = threadIdx.x; li < max_li; li += blockDim.x) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        x.values[row * x.stride + col] =
            alpha.values[map(row, col, alpha.stride)] *
            x.values[row * x.stride + col];
    }
}


template <typename ValueType, typename Mapping>
__global__
__launch_bounds__(default_block_size, sm_oversubscription) void scale_kernel(
    const gko::batch::multi_vector::uniform_batch<const ValueType> alpha,
    const gko::batch::multi_vector::uniform_batch<ValueType> x, Mapping map)
{
    for (size_type batch_id = blockIdx.x; batch_id < x.num_batch_items;
         batch_id += gridDim.x) {
        const auto alpha_b = gko::batch::extract_batch_item(alpha, batch_id);
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        scale(alpha_b, x_b, map);
    }
}


template <typename ValueType, typename Mapping>
__device__ __forceinline__ void add_scaled(
    const gko::batch::multi_vector::batch_item<const ValueType>& alpha,
    const gko::batch::multi_vector::batch_item<const ValueType>& x,
    const gko::batch::multi_vector::batch_item<ValueType>& y, Mapping map)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = threadIdx.x; li < max_li; li += blockDim.x) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        y.values[row * y.stride + col] +=
            alpha.values[map(col)] * x.values[row * x.stride + col];
    }
}


template <typename ValueType, typename Mapping>
__global__ __launch_bounds__(
    default_block_size,
    sm_oversubscription) void add_scaled_kernel(const gko::batch::multi_vector::
                                                    uniform_batch<
                                                        const ValueType>
                                                        alpha,
                                                const gko::batch::multi_vector::
                                                    uniform_batch<
                                                        const ValueType>
                                                        x,
                                                const gko::batch::multi_vector::
                                                    uniform_batch<ValueType>
                                                        y,
                                                Mapping map)
{
    for (size_type batch_id = blockIdx.x; batch_id < x.num_batch_items;
         batch_id += gridDim.x) {
        const auto alpha_b = gko::batch::extract_batch_item(alpha, batch_id);
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        const auto y_b = gko::batch::extract_batch_item(y, batch_id);
        add_scaled(alpha_b, x_b, y_b, map);
    }
}


template <typename Group, typename ValueType>
__device__ __forceinline__ void single_rhs_compute_conj_dot(Group subgroup,
                                                            const int num_rows,
                                                            const ValueType* x,
                                                            const ValueType* y,
                                                            ValueType& result)

{
    ValueType val = zero<ValueType>();
    for (int r = subgroup.thread_rank(); r < num_rows; r += subgroup.size()) {
        val += conj(x[r]) * y[r];
    }

    // subgroup level reduction
    val = reduce(subgroup, val, thrust::plus<ValueType>{});

    if (subgroup.thread_rank() == 0) {
        result = val;
    }
}


template <typename Group, typename ValueType, typename Mapping>
__device__ __forceinline__ void gen_one_dot(
    const gko::batch::multi_vector::batch_item<const ValueType>& x,
    const gko::batch::multi_vector::batch_item<const ValueType>& y,
    const int rhs_index,
    const gko::batch::multi_vector::batch_item<ValueType>& result,
    Group subgroup, Mapping conj_map)
{
    ValueType val = zero<ValueType>();

    for (int r = subgroup.thread_rank(); r < x.num_rows; r += subgroup.size()) {
        val += conj_map(x.values[r * x.stride + rhs_index]) *
               y.values[r * y.stride + rhs_index];
    }

    // subgroup level reduction
    val = reduce(subgroup, val, thrust::plus<ValueType>{});

    if (subgroup.thread_rank() == 0) {
        result.values[rhs_index] = val;
    }
}


template <typename ValueType, typename Mapping>
__device__ __forceinline__ void compute_gen_dot_product(
    const gko::batch::multi_vector::batch_item<const ValueType>& x,
    const gko::batch::multi_vector::batch_item<const ValueType>& y,
    const gko::batch::multi_vector::batch_item<ValueType>& result,
    Mapping conj_map)
{
    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto subgroup = group::tiled_partition<tile_size>(thread_block);
    const auto subgroup_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subgroups_per_block = ceildiv(blockDim.x, tile_size);

    for (int rhs_index = subgroup_id; rhs_index < x.num_rhs;
         rhs_index += num_subgroups_per_block) {
        gen_one_dot(x, y, rhs_index, result, subgroup, conj_map);
    }
}


template <typename ValueType, typename Mapping>
__global__
__launch_bounds__(default_block_size, sm_oversubscription) void compute_gen_dot_product_kernel(
    const gko::batch::multi_vector::uniform_batch<const ValueType> x,
    const gko::batch::multi_vector::uniform_batch<const ValueType> y,
    const gko::batch::multi_vector::uniform_batch<ValueType> result,
    Mapping map)
{
    for (size_type batch_id = blockIdx.x; batch_id < x.num_batch_items;
         batch_id += gridDim.x) {
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        const auto y_b = gko::batch::extract_batch_item(y, batch_id);
        const auto r_b = gko::batch::extract_batch_item(result, batch_id);
        compute_gen_dot_product(x_b, y_b, r_b, map);
    }
}


template <typename Group, typename ValueType>
__device__ __forceinline__ void single_rhs_compute_norm2(
    Group subgroup, const int num_rows, const ValueType* x,
    remove_complex<ValueType>& result)
{
    using real_type = typename gko::remove_complex<ValueType>;
    real_type val = zero<real_type>();

    for (int r = subgroup.thread_rank(); r < num_rows; r += subgroup.size()) {
        val += squared_norm(x[r]);
    }

    // subgroup level reduction
    val = reduce(subgroup, val, thrust::plus<remove_complex<ValueType>>{});

    if (subgroup.thread_rank() == 0) {
        result = sqrt(val);
    }
}


template <typename Group, typename ValueType>
__device__ __forceinline__ void one_norm2(
    const gko::batch::multi_vector::batch_item<const ValueType>& x,
    const int rhs_index,
    const gko::batch::multi_vector::batch_item<remove_complex<ValueType>>&
        result,
    Group subgroup)
{
    using real_type = typename gko::remove_complex<ValueType>;
    real_type val = zero<real_type>();

    for (int r = subgroup.thread_rank(); r < x.num_rows; r += subgroup.size()) {
        val += squared_norm(x.values[r * x.stride + rhs_index]);
    }

    // subgroup level reduction
    val = reduce(subgroup, val, thrust::plus<remove_complex<ValueType>>{});

    if (subgroup.thread_rank() == 0) {
        result.values[rhs_index] = sqrt(val);
    }
}


/**
 * Computes the 2-norms of some column vectors in global or shared memory.
 *
 * @param x  A row-major multivector with nrhs columns.
 * @param result  Holds norm value for each vector in x.
 */
template <typename ValueType>
__device__ __forceinline__ void compute_norm2(
    const gko::batch::multi_vector::batch_item<const ValueType>& x,
    const gko::batch::multi_vector::batch_item<remove_complex<ValueType>>&
        result)
{
    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto subgroup = group::tiled_partition<tile_size>(thread_block);
    const auto subgroup_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subgroups_per_block = ceildiv(blockDim.x, tile_size);

    for (int rhs_index = subgroup_id; rhs_index < x.num_rhs;
         rhs_index += num_subgroups_per_block) {
        one_norm2(x, rhs_index, result, subgroup);
    }
}


template <typename ValueType>
__global__ __launch_bounds__(
    default_block_size,
    sm_oversubscription) void compute_norm2_kernel(const gko::batch::
                                                       multi_vector::
                                                           uniform_batch<
                                                               const ValueType>
                                                               x,
                                                   const gko::batch::
                                                       multi_vector::
                                                           uniform_batch<
                                                               remove_complex<
                                                                   ValueType>>
                                                               result)
{
    for (size_type batch_id = blockIdx.x; batch_id < x.num_batch_items;
         batch_id += gridDim.x) {
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        const auto r_b = gko::batch::extract_batch_item(result, batch_id);
        compute_norm2(x_b, r_b);
    }
}


template <typename ValueType>
__device__ __forceinline__ void single_rhs_copy(const int num_rows,
                                                const ValueType* in,
                                                ValueType* out)
{
    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        out[iz] = in[iz];
    }
}


/**
 * Copies the values of one multi-vector into another.
 *
 * Note that the output multi-vector should already have memory allocated
 * and stride set.
 */
template <typename ValueType>
__device__ __forceinline__ void copy(
    const gko::batch::multi_vector::batch_item<const ValueType>& in,
    const gko::batch::multi_vector::batch_item<ValueType>& out)
{
    for (int iz = threadIdx.x; iz < in.num_rows * in.num_rhs;
         iz += blockDim.x) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}


template <typename ValueType>
__global__
__launch_bounds__(default_block_size, sm_oversubscription) void copy_kernel(
    const gko::batch::multi_vector::uniform_batch<const ValueType> src,
    const gko::batch::multi_vector::uniform_batch<ValueType> dst)
{
    for (size_type batch_id = blockIdx.x; batch_id < src.num_batch_items;
         batch_id += gridDim.x) {
        const auto dst_b = gko::batch::extract_batch_item(dst, batch_id);
        const auto src_b = gko::batch::extract_batch_item(src, batch_id);
        copy(src_b, dst_b);
    }
}
