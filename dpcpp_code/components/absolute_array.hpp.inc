/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

namespace kernel {


template <typename ValueType>
void inplace_absolute_array_kernel(const size_type n,
                                   ValueType *__restrict__ array,
                                   sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < n) {
        array[tidx] = dpcpp::abs(array[tidx]);
    }
}

template <typename ValueType>
void inplace_absolute_array_kernel(dim3 grid, dim3 block,
                                   size_t dynamic_shared_memory,
                                   sycl::queue *stream, const size_type n,
                                   ValueType *array)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             inplace_absolute_array_kernel(n, array, item_ct1);
                         });
    });
}


template <typename ValueType>
void outplace_absolute_array_kernel(const size_type n,
                                    const ValueType *__restrict__ in,
                                    remove_complex<ValueType> *__restrict__ out,
                                    sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < n) {
        out[tidx] = dpcpp::abs(in[tidx]);
    }
}

template <typename ValueType>
void outplace_absolute_array_kernel(dim3 grid, dim3 block,
                                    size_t dynamic_shared_memory,
                                    sycl::queue *stream, const size_type n,
                                    const ValueType *in,
                                    remove_complex<ValueType> *out)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             outplace_absolute_array_kernel(n, in, out,
                                                            item_ct1);
                         });
    });
}


}  // namespace kernel
