/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
__dpct_inline__ void simple_apply_kernel(
    const gko::batch::matrix::batch_dense::batch_item<const ValueType>& mat,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ x,
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
            temp += val * b[j];
        }
        temp = ::gko::kernels::dpcpp::reduce(
            subg, temp, [](ValueType v1, ValueType v2) { return v1 + v2; });
        if (subgroup.get_local_id() == 0) {
            x[row] = temp;
        }
    }
}


template <typename ValueType>
__dpct_inline__ void advanced_apply_kernel(
    const ValueType alpha,
    const gko::batch::matrix::batch_dense::batch_item<const ValueType>& mat,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ x, sycl::nd_item<3>& item_ct1)
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
            subg, temp, [](ValueType v1, ValueType v2) { return v1 + v2; });
        if (subgroup.get_local_id() == 0) {
            x[row] = temp + beta * x[row];
        }
    }
}
