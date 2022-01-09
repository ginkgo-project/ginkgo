/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <cstdlib>

#include <ginkgo/ginkgo.hpp>


#define INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro) \
    template _macro(float);                     \
    template _macro(double);


#define SQRT_KERNEL(_type)                       \
    void sqrt_kernel(std::size_t size, _type* x, \
                     std::shared_ptr<gko::AsyncHandle> handle);


namespace {


// a parallel CUDA kernel that computes the application of a 3 point sqrt
template <typename ValueType>
__global__ void sqrt_kernel_impl(std::size_t size, ValueType* __restrict__ x)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159, i));
    }
}


}  // namespace


template <typename ValueType>
void sqrt_kernel(std::size_t size, ValueType* x,
                 std::shared_ptr<gko::AsyncHandle> handle)
{
    constexpr int block_size = 64;
    const auto grid_size = (size + block_size - 1) / block_size;
    sqrt_kernel_impl<<<grid_size, block_size, 0,
                       gko::as<gko::CudaAsyncHandle>(handle)->get_handle()>>>(
        size, x);
}

INSTANTIATE_FOR_EACH_VALUE_TYPE(SQRT_KERNEL);
