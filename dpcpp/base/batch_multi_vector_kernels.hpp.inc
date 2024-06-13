// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType, typename Mapping>
__dpct_inline__ void scale_kernel(
    const gko::batch::multi_vector::batch_item<const ValueType>& alpha,
    const gko::batch::multi_vector::batch_item<ValueType>& x,
    sycl::nd_item<3>& item_ct1, Mapping map)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_linear_id(); li < max_li;
         li += item_ct1.get_local_range().size()) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        x.values[row * x.stride + col] =
            alpha.values[map(row, col, alpha.stride)] *
            x.values[row * x.stride + col];
    }
}


template <typename ValueType, typename Mapping>
__dpct_inline__ void add_scaled_kernel(
    const gko::batch::multi_vector::batch_item<const ValueType>& alpha,
    const gko::batch::multi_vector::batch_item<const ValueType>& x,
    const gko::batch::multi_vector::batch_item<ValueType>& y,
    sycl::nd_item<3>& item_ct1, Mapping map)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_id(2); li < max_li;
         li += item_ct1.get_local_range(2)) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        y.values[row * y.stride + col] +=
            alpha.values[map(col)] * x.values[row * x.stride + col];
    }
}


template <typename ValueType>
__dpct_inline__ void single_rhs_compute_conj_dot(
    const int num_rows, const ValueType* const __restrict__ x,
    const ValueType* const __restrict__ y, ValueType& result,
    sycl::nd_item<3> item_ct1)
{
    const auto group = item_ct1.get_group();
    const auto group_size = item_ct1.get_local_range().size();
    const auto tid = item_ct1.get_local_linear_id();

    ValueType val = zero<ValueType>();

    for (int r = tid; r < num_rows; r += group_size) {
        val += conj(x[r]) * y[r];
    }
    result = sycl::reduce_over_group(group, val, sycl::plus<>());
}


template <int tile_size = config::warp_size, typename ValueType>
__dpct_inline__ void single_rhs_compute_conj_dot_sg(
    const int num_rows, const ValueType* const __restrict__ x,
    const ValueType* const __restrict__ y, ValueType& result,
    sycl::nd_item<3> item_ct1)
{
    auto subg =
        group::tiled_partition<tile_size>(group::this_thread_block(item_ct1));
    const auto subgroup = static_cast<sycl::sub_group>(subg);
    const int subgroup_id = subgroup.get_group_id();
    const int subgroup_size = subgroup.get_local_range().size();
    const auto subgroup_tid = subgroup.get_local_id();

    ValueType val = zero<ValueType>();

    for (int r = subgroup_tid; r < num_rows; r += subgroup_size) {
        val += conj(x[r]) * y[r];
    }

    val = ::gko::kernels::dpcpp::reduce(
        subg, val, [](ValueType a, ValueType b) { return a + b; });

    if (subgroup_tid == 0) {
        result = val;
    }
}


template <typename ValueType, typename Mapping>
__dpct_inline__ void compute_gen_dot_product_kernel(
    const gko::batch::multi_vector::batch_item<const ValueType>& x,
    const gko::batch::multi_vector::batch_item<const ValueType>& y,
    const gko::batch::multi_vector::batch_item<ValueType>& result,
    sycl::nd_item<3>& item_ct1, Mapping conj_map)
{
    constexpr auto tile_size = config::warp_size;
    auto subg =
        group::tiled_partition<tile_size>(group::this_thread_block(item_ct1));
    const auto subgroup = static_cast<sycl::sub_group>(subg);
    const int subgroup_id = subgroup.get_group_id();
    const int subgroup_size = subgroup.get_local_range().size();
    const int num_subgroups = subgroup.get_group_range().size();

    for (int rhs_index = subgroup_id; rhs_index < x.num_rhs;
         rhs_index += num_subgroups) {
        ValueType val = zero<ValueType>();

        for (int r = subgroup.get_local_id(); r < x.num_rows;
             r += subgroup_size) {
            val += conj_map(x.values[r * x.stride + rhs_index]) *
                   y.values[r * y.stride + rhs_index];
        }

        val = ::gko::kernels::dpcpp::reduce(
            subg, val, [](ValueType a, ValueType b) { return a + b; });

        if (subgroup.get_local_id() == 0) {
            result.values[rhs_index] = val;
        }
    }
}


template <int tile_size = config::warp_size, typename ValueType>
__dpct_inline__ void single_rhs_compute_norm2_sg(
    const int num_rows, const ValueType* const __restrict__ x,
    gko::remove_complex<ValueType>& result, sycl::nd_item<3> item_ct1)
{
    auto subg =
        group::tiled_partition<tile_size>(group::this_thread_block(item_ct1));
    const auto subgroup = static_cast<sycl::sub_group>(subg);
    const int subgroup_id = subgroup.get_group_id();
    const int subgroup_size = subgroup.get_local_range().size();

    using real_type = typename gko::remove_complex<ValueType>;
    real_type val = zero<real_type>();

    for (int r = subgroup.get_local_id(); r < num_rows; r += subgroup_size) {
        val += squared_norm(x[r]);
    }

    val = ::gko::kernels::dpcpp::reduce(
        subg, val, [](real_type a, real_type b) { return a + b; });

    if (subgroup.get_local_id() == 0) {
        result = sqrt(val);
    }
}


template <typename ValueType>
__dpct_inline__ void single_rhs_compute_norm2(
    const int num_rows, const ValueType* const __restrict__ x,
    gko::remove_complex<ValueType>& result, sycl::nd_item<3> item_ct1)
{
    const auto group = item_ct1.get_group();
    const auto group_size = item_ct1.get_local_range().size();
    const auto tid = item_ct1.get_local_linear_id();

    using real_type = typename gko::remove_complex<ValueType>;
    real_type val = zero<real_type>();

    for (int r = tid; r < num_rows; r += group_size) {
        val += squared_norm(x[r]);
    }

    val = sycl::reduce_over_group(group, val, sycl::plus<>());

    result = sqrt(val);
}


template <typename ValueType>
__dpct_inline__ void compute_norm2_kernel(
    const gko::batch::multi_vector::batch_item<const ValueType>& x,
    const gko::batch::multi_vector::batch_item<remove_complex<ValueType>>&
        result,
    sycl::nd_item<3>& item_ct1)
{
    constexpr auto tile_size = config::warp_size;
    auto subg =
        group::tiled_partition<tile_size>(group::this_thread_block(item_ct1));
    const auto subgroup = static_cast<sycl::sub_group>(subg);
    const int subgroup_id = subgroup.get_group_id();
    const int subgroup_size = subgroup.get_local_range().size();
    const int num_subgroups = subgroup.get_group_range().size();

    using real_type = typename gko::remove_complex<ValueType>;
    for (int rhs_index = subgroup_id; rhs_index < x.num_rhs;
         rhs_index += num_subgroups) {
        real_type val = zero<real_type>();

        for (int r = subgroup.get_local_id(); r < x.num_rows;
             r += subgroup_size)
            val += squared_norm(x.values[r * x.stride + rhs_index]);

        val = ::gko::kernels::dpcpp::reduce(
            subg, val, [](real_type a, real_type b) { return a + b; });

        if (subgroup.get_local_id() == 0) {
            result.values[rhs_index] = sqrt(val);
        }
    }
}


template <typename ValueType>
__dpct_inline__ void copy_kernel(const int num_rows, const ValueType* in,
                                 ValueType* out, sycl::nd_item<3>& item_ct1)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < num_rows;
         iz += item_ct1.get_local_range().size()) {
        out[iz] = in[iz];
    }
}


template <typename ValueType>
__dpct_inline__ void copy_kernel(
    const gko::batch::multi_vector::batch_item<const ValueType>& in,
    const gko::batch::multi_vector::batch_item<ValueType>& out,
    sycl::nd_item<3>& item_ct1)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < in.num_rows * in.num_rhs;
         iz += item_ct1.get_local_range().size()) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}
