// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdlib>

#include <ginkgo/ginkgo.hpp>


#define INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro) \
    template _macro(float);                     \
    template _macro(double);


#define STENCIL_KERNEL(_type)                                                 \
    void stencil_kernel(std::size_t size, const _type* coefs, const _type* b, \
                        _type* x);


namespace {


// a parallel CUDA kernel that computes the application of a 3 point stencil
template <typename ValueType>
__global__ void stencil_kernel_impl(std::size_t size, const ValueType* coefs,
                                    const ValueType* b, ValueType* x)
{
    const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= size) {
        return;
    }
    auto result = coefs[1] * b[thread_id];
    if (thread_id > 0) {
        result += coefs[0] * b[thread_id - 1];
    }
    if (thread_id < size - 1) {
        result += coefs[2] * b[thread_id + 1];
    }
    x[thread_id] = result;
}


}  // namespace


template <typename ValueType>
void stencil_kernel(std::size_t size, const ValueType* coefs,
                    const ValueType* b, ValueType* x)
{
    constexpr int block_size = 512;
    const auto grid_size = (size + block_size - 1) / block_size;
    stencil_kernel_impl<<<grid_size, block_size>>>(size, coefs, b, x);
}

INSTANTIATE_FOR_EACH_VALUE_TYPE(STENCIL_KERNEL);
