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

#include "cuda/components/cooperative_groups.cuh"


#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/test/utils.hpp"
#include "cuda/base/types.hpp"
#include "cuda/test/utils.hpp"


namespace {


using namespace gko::kernels::cuda;

constexpr int warmup = 10;
constexpr int repeat = 100;
constexpr int inner_loops = 1000;
template <typename T>
class Reduce : public ::testing::Test {
protected:
    using value_type = T;
    Reduce()
        : ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref)),
          result(ref, config::warp_size),
          dresult(cuda)
    {
        for (int i = 0; i < config::warp_size; i++) {
            result.get_data()[i] = static_cast<value_type>(0);
        }
        dresult = result;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    gko::Array<value_type> result;
    gko::Array<value_type> dresult;
};
using VTs = ::testing::Types<int, long, float, double>;
TYPED_TEST_CASE(Reduce, VTs);


template <typename Kernel, typename ValueType>
void test(std::shared_ptr<gko::CudaExecutor> cuda, Kernel kernel,
          std::string name, ValueType *data)
{
    for (int i = 0; i < warmup; i++) {
        kernel<<<1, config::warp_size>>>(as_cuda_type(data), inner_loops);
    }
    cuda->synchronize();
    auto tic = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
        kernel<<<1, config::warp_size>>>(as_cuda_type(data), inner_loops);
    }
    cuda->synchronize();
    auto toc = std::chrono::steady_clock::now();
    auto one_inner_loops =
        std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
    cuda->synchronize();
    tic = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
        kernel<<<1, config::warp_size>>>(as_cuda_type(data), inner_loops * 2);
    }
    cuda->synchronize();
    toc = std::chrono::steady_clock::now();
    auto two_inner_loops =
        std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
    std::cout << name << " with " << inner_loops << " inner_loops time: "
              << static_cast<double>(two_inner_loops - one_inner_loops) /
                     repeat / inner_loops
              << " ( " << two_inner_loops << " - " << one_inner_loops
              << " ) ValueSize: " << sizeof(ValueType) << std::endl;
}

template <int Size, typename ValueType>
__global__ void reduce_legacy(ValueType *__restrict__ data,
                              const int inner_loops)
    __launch_bounds__(config::warp_size)
{
    ValueType local_data = data[threadIdx.x];
    for (int i = 0; i < inner_loops; i++) {
#pragma unroll
        for (int bitmask = 1; bitmask < Size; bitmask <<= 1) {
            const auto remote_data =
                __shfl_xor_sync(0xffffffff, local_data, bitmask, Size);
            local_data = local_data + remote_data;
        }
    }
    data[threadIdx.x] = local_data;
}

template <int Size, typename ValueType>
__global__ void reduce_cg(ValueType *__restrict__ data, const int inner_loops)
    __launch_bounds__(config::warp_size)
{
    ValueType local_data = data[threadIdx.x];
    for (int i = 0; i < inner_loops; i++) {
        auto group = group::tiled_partition<Size>(group::this_thread_block());
#pragma unroll
        for (int bitmask = 1; bitmask < group.size(); bitmask <<= 1) {
            const auto remote_data = group.shfl_xor(local_data, bitmask);
            local_data = local_data + remote_data;
        }
    }
    data[threadIdx.x] = local_data;
}

TYPED_TEST(Reduce, FullWarp)
{
    using value_type = typename TestFixture::value_type;
    test(this->cuda, reduce_legacy<config::warp_size, cuda_type<value_type>>,
         "FullWarp Warm Legacy", this->dresult.get_data());
    test(this->cuda, reduce_legacy<config::warp_size, cuda_type<value_type>>,
         "FullWarp Legacy", this->dresult.get_data());
    test(this->cuda, reduce_cg<config::warp_size, cuda_type<value_type>>,
         "FullWarp Warm Cg", this->dresult.get_data());
    test(this->cuda, reduce_cg<config::warp_size, cuda_type<value_type>>,
         "FullWarp Cg", this->dresult.get_data());
}

TYPED_TEST(Reduce, WarpSize4)
{
    using value_type = typename TestFixture::value_type;
    test(this->cuda, reduce_legacy<4, cuda_type<value_type>>,
         "WarpSize4 Warm Legacy", this->dresult.get_data());
    test(this->cuda, reduce_legacy<4, cuda_type<value_type>>,
         "WarpSize4 Legacy", this->dresult.get_data());
    test(this->cuda, reduce_cg<4, cuda_type<value_type>>, "WarpSize4 Warm Cg",
         this->dresult.get_data());
    test(this->cuda, reduce_cg<4, cuda_type<value_type>>, "WarpSize4 Cg",
         this->dresult.get_data());
}


}  // namespace
