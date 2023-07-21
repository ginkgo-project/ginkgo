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
__dpct_inline__ void scale_kernel(
    const gko::batch_multi_vector::batch_entry<const ValueType>& alpha,
    const gko::batch_multi_vector::batch_entry<ValueType>& x,
    sycl::nd_item<3>& item_ct1)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_linear_id(); li < max_li;
         li += item_ct1.get_local_range().size()) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            x.values[row * x.stride + col] =
                alpha.values[0] * x.values[row * x.stride + col];
        } else {
            x.values[row * x.stride + col] =
                alpha.values[col] * x.values[row * x.stride + col];
        }
    }
}


template <typename ValueType>
__dpct_inline__ void add_scaled_kernel(
    const gko::batch_multi_vector::batch_entry<const ValueType>& alpha,
    const gko::batch_multi_vector::batch_entry<const ValueType>& x,
    const gko::batch_multi_vector::batch_entry<ValueType>& y,
    sycl::nd_item<3>& item_ct1)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_id(2); li < max_li;
         li += item_ct1.get_local_range(2)) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            y.values[row * y.stride + col] +=
                alpha.values[0] * x.values[row * x.stride + col];
        } else {
            y.values[row * y.stride + col] +=
                alpha.values[col] * x.values[row * x.stride + col];
        }
    }
}


template <typename ValueType>
__dpct_inline__ void compute_dot_product_kernel(
    const gko::batch_multi_vector::batch_entry<const ValueType>& x,
    const gko::batch_multi_vector::batch_entry<const ValueType>& y,
    const gko::batch_multi_vector::batch_entry<ValueType>& result,
    sycl::nd_item<3>& item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int rhs_index = sg_id; rhs_index < x.num_rhs; rhs_index += num_sg) {
        ValueType val = zero<ValueType>();

        for (int r = sg.get_local_id(); r < x.num_rows; r += sg_size) {
            val += conj(x.values[r * x.stride + rhs_index]) *
                   y.values[r * y.stride + rhs_index];
        }

        val = sycl::reduce_over_group(sg, val, sycl::plus<>());

        if (sg.get_local_id() == 0) {
            result.values[rhs_index] = val;
        }
    }
}


template <typename ValueType>
__dpct_inline__ void compute_norm2_kernel(
    const gko::batch_multi_vector::batch_entry<const ValueType>& x,
    const gko::batch_multi_vector::batch_entry<remove_complex<ValueType>>&
        result,
    sycl::nd_item<3>& item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    using real_type = typename gko::remove_complex<ValueType>;
    for (int rhs_index = sg_id; rhs_index < x.num_rhs; rhs_index += num_sg) {
        real_type val = zero<real_type>();

        for (int r = sg.get_local_id(); r < x.num_rows; r += sg_size)
            val += squared_norm(x.values[r * x.stride + rhs_index]);

        val = sycl::reduce_over_group(sg, val, sycl::plus<>());

        if (sg.get_local_id() == 0) result.values[rhs_index] = sqrt(val);
    }
}


template <typename ValueType>
__dpct_inline__ void copy_kernel(
    const gko::batch_multi_vector::batch_entry<const ValueType>& in,
    const gko::batch_multi_vector::batch_entry<ValueType>& out,
    sycl::nd_item<3>& item_ct1)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < in.num_rows * in.num_rhs;
         iz += item_ct1.get_local_range().size()) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}
