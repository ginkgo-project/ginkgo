// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType>
__dpct_inline__ void simple_apply_kernel(
    const gko::batch::matrix::dense::batch_item<const ValueType>& mat,
    const ValueType* b, ValueType* x, sycl::nd_item<3>& item_ct1)
{
    constexpr auto tile_size = config::warp_size;
    auto subg =
        group::tiled_partition<tile_size>(group::this_thread_block(item_ct1));
    const auto subgroup = static_cast<sycl::sub_group>(subg);
    const int subgroup_id = subgroup.get_group_id();
    const int subgroup_size = subgroup.get_local_range().size();
    const int num_subgroups = subgroup.get_group_range().size();

    for (int row = subgroup_id; row < mat.num_rows; row += num_subgroups) {
        ValueType temp = zero<ValueType>();
        for (int j = subgroup.get_local_id(); j < mat.num_cols;
             j += subgroup_size) {
            const ValueType val = mat.values[row * mat.stride + j];
            temp += val * b[j];
        }

        temp = ::gko::kernels::dpcpp::reduce(
            subg, temp, [](ValueType a, ValueType b) { return a + b; });

        if (subgroup.get_local_id() == 0) {
            x[row] = temp;
        }
    }
}


template <typename ValueType>
__dpct_inline__ void advanced_apply_kernel(
    const ValueType alpha,
    const gko::batch::matrix::dense::batch_item<const ValueType>& mat,
    const ValueType* b, const ValueType beta, ValueType* x,
    sycl::nd_item<3>& item_ct1)
{
    constexpr auto tile_size = config::warp_size;
    auto subg =
        group::tiled_partition<tile_size>(group::this_thread_block(item_ct1));
    const auto subgroup = static_cast<sycl::sub_group>(subg);
    const int subgroup_id = subgroup.get_group_id();
    const int subgroup_size = subgroup.get_local_range().size();
    const int num_subgroup = subgroup.get_group_range().size();

    for (int row = subgroup_id; row < mat.num_rows; row += num_subgroup) {
        ValueType temp = zero<ValueType>();
        for (int j = subgroup.get_local_id(); j < mat.num_cols;
             j += subgroup_size) {
            const ValueType val = mat.values[row * mat.stride + j];
            temp += alpha * val * b[j];
        }

        temp = ::gko::kernels::dpcpp::reduce(
            subg, temp, [](ValueType a, ValueType b) { return a + b; });

        if (subgroup.get_local_id() == 0) {
            x[row] = temp + beta * x[row];
        }
    }
}


template <typename ValueType>
__dpct_inline__ void scale_kernel(
    const ValueType* const col_scale, const ValueType* const row_scale,
    gko::batch::matrix::dense::batch_item<ValueType>& mat,
    sycl::nd_item<3>& item_ct1)
{
    constexpr auto tile_size = config::warp_size;
    auto subg =
        group::tiled_partition<tile_size>(group::this_thread_block(item_ct1));
    const auto subgroup = static_cast<sycl::sub_group>(subg);
    const int subgroup_id = subgroup.get_group_id();
    const int subgroup_size = subgroup.get_local_range().size();
    const int num_subgroup = subgroup.get_group_range().size();

    for (int row = subgroup_id; row < mat.num_rows; row += num_subgroup) {
        const ValueType row_scalar = row_scale[row];
        for (int col = subgroup.get_local_id(); col < mat.num_cols;
             col += subgroup_size) {
            mat.values[row * mat.stride + col] *= row_scalar * col_scale[col];
        }
    }
}


template <typename ValueType>
__dpct_inline__ void scale_add_kernel(
    const ValueType alpha,
    const gko::batch::matrix::dense::batch_item<const ValueType>& mat,
    const gko::batch::matrix::dense::batch_item<ValueType>& in_out,
    sycl::nd_item<3>& item_ct1)
{
    constexpr auto tile_size = config::warp_size;
    auto subg =
        group::tiled_partition<tile_size>(group::this_thread_block(item_ct1));
    const auto subgroup = static_cast<sycl::sub_group>(subg);
    const int subgroup_id = subgroup.get_group_id();
    const int subgroup_size = subgroup.get_local_range().size();
    const int num_subgroup = subgroup.get_group_range().size();

    for (int row = subgroup_id; row < mat.num_rows; row += num_subgroup) {
        for (int col = subgroup.get_local_id(); col < mat.num_cols;
             col += subgroup_size) {
            in_out.values[row * in_out.stride + col] =
                alpha * in_out.values[row * in_out.stride + col] +
                mat.values[row * mat.stride + col];
        }
    }
}


template <typename ValueType>
__dpct_inline__ void add_scaled_identity_kernel(
    const ValueType alpha, const ValueType beta,
    const gko::batch::matrix::dense::batch_item<ValueType>& mat,
    sycl::nd_item<3>& item_ct1)
{
    constexpr auto tile_size = config::warp_size;
    auto subg =
        group::tiled_partition<tile_size>(group::this_thread_block(item_ct1));
    const auto subgroup = static_cast<sycl::sub_group>(subg);
    const int subgroup_id = subgroup.get_group_id();
    const int subgroup_size = subgroup.get_local_range().size();
    const int num_subgroup = subgroup.get_group_range().size();

    for (int row = subgroup_id; row < mat.num_rows; row += num_subgroup) {
        for (int col = subgroup.get_local_id(); col < mat.num_cols;
             col += subgroup_size) {
            mat.values[row * mat.stride + col] *= beta;
            if (row == col) {
                mat.values[row * mat.stride + col] += alpha;
            }
        }
    }
}
