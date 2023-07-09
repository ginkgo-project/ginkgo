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

/**
 * Copies the values of vector into another.
 *
 * @param num_rows  Length of vector.
 * @param in  Vector to copy from.
 * @param out  Vector to copy into.
 */
template <typename ValueType>
__dpct_inline__ void copy_kernel(const int num_rows,
                                 const ValueType* const __restrict__ in,
                                 ValueType* const __restrict__ out,
                                 sycl::nd_item<3> item_ct1)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < num_rows;
         iz += item_ct1.get_local_range().size()) {
        out[iz] = in[iz];
    }
}

/**
 * Adds a scaled vector to another.
 *
 * @param num_rows  Common length of both vectors.
 * @param alpha  Scaling factor.
 * @param[in] x  Vector to scale and add.
 * @param[in,out] y  Vector to add to.
 */
template <typename ValueType>
__dpct_inline__ void add_scaled_kernel(const int num_rows,
                                       const ValueType alpha,
                                       const ValueType* const __restrict__ x,
                                       ValueType* const __restrict__ y,
                                       sycl::nd_item<3> item_ct1)
{
    for (int li = item_ct1.get_local_linear_id(); li < num_rows;
         li += item_ct1.get_local_range().size()) {
        y[li] += alpha * x[li];
    }
}

/**
 * Computes the 2-norm of a vector in global or shared memory.
 *
 * @param x  A row-major vector (only 1 column).
 * @param result  Norm value.
 */
template <typename ValueType>
__dpct_inline__ void compute_norm2_sg_kernel(
    const int num_rows, const ValueType* const __restrict__ x,
    gko::remove_complex<ValueType>& result, sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const auto sg_size = sg.get_local_range().size();
    const auto sg_tid = sg.get_local_id();

    using real_type = typename gko::remove_complex<ValueType>;
    real_type val = zero<real_type>();

    for (int r = sg_tid; r < num_rows; r += sg_size) {
        val += squared_norm(x[r]);
    }

    val = sycl::reduce_over_group(sg, val, sycl::plus<>());

    if (sg_tid == 0) {
        result = sqrt(val);
    }
}

template <typename ValueType>
__dpct_inline__ void compute_norm2_kernel(
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


/**
 * Computes the dot product of some column vectors in global or shared memory.
 *
 * @param result  Holds dot product value for vector in x and y.
 */
template <typename ValueType>
__dpct_inline__ void compute_dot_product_sg_kernel(
    const int num_rows, const ValueType* const __restrict__ x,
    const ValueType* const __restrict__ y, ValueType& result,
    sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const auto sg_size = sg.get_local_range().size();
    const auto sg_tid = sg.get_local_id();

    ValueType val = zero<ValueType>();

    for (int r = sg_tid; r < num_rows; r += sg_size) {
        val += conj(x[r]) * y[r];
    }

    val = sycl::reduce_over_group(sg, val, sycl::plus<>());

    if (sg_tid == 0) {
        result = val;
    }
}

template <typename ValueType>
__dpct_inline__ void compute_dot_product_kernel(
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
