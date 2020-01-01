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

#include "cuda/components/sorting.cuh"


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


namespace {


using gko::kernels::cuda::bitonic_sort;
using gko::kernels::cuda::config;


constexpr auto num_elements = 2048;
constexpr auto num_local = 4;
constexpr auto num_threads = num_elements / num_local;


__global__ void test_sort_shared(gko::int32 *data)
{
    gko::int32 local[num_local];
    __shared__ gko::int32 sh_local[num_elements];
    for (int i = 0; i < num_local; ++i) {
        local[i] = data[threadIdx.x * num_local + i];
    }
    bitonic_sort<num_elements, num_local>(local, sh_local);
    for (int i = 0; i < num_local; ++i) {
        data[threadIdx.x * num_local + i] = local[i];
    }
}


__global__ void test_sort_warp(gko::int32 *data)
{
    gko::int32 local[num_local];
    for (int i = 0; i < num_local; ++i) {
        local[i] = data[threadIdx.x * num_local + i];
    }
    bitonic_sort<config::warp_size * num_local, num_local>(
        local, static_cast<gko::int32 *>(nullptr));
    for (int i = 0; i < num_local; ++i) {
        data[threadIdx.x * num_local + i] = local[i];
    }
}


class Sorting : public ::testing::Test {
protected:
    Sorting()
        : ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref)),
          rng(123456),
          ref_shared(ref, num_elements),
          ref_warp(ref),
          ddata(cuda)
    {
        // we want some duplicate elements
        std::uniform_int_distribution<gko::int32> dist(0, num_elements / 2);
        for (auto i = 0; i < num_elements; ++i) {
            ref_shared.get_data()[i] = dist(rng);
        }
        ddata = gko::Array<gko::int32>{cuda, ref_shared};
        ref_warp = ref_shared;
        std::sort(ref_shared.get_data(), ref_shared.get_data() + num_elements);
        std::sort(ref_warp.get_data(),
                  ref_warp.get_data() + (config::warp_size * num_local));
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::default_random_engine rng;
    gko::Array<gko::int32> ref_shared;
    gko::Array<gko::int32> ref_warp;
    gko::Array<gko::int32> ddata;
};


TEST_F(Sorting, CudaBitonicSortWarp)
{
    test_sort_warp<<<1, config::warp_size>>>(ddata.get_data());
    ddata.set_executor(ref);
    auto data_ptr = ddata.get_const_data();
    auto ref_ptr = ref_warp.get_const_data();

    ASSERT_TRUE(std::equal(data_ptr, data_ptr + (num_local * config::warp_size),
                           ref_ptr));
}


TEST_F(Sorting, CudaBitonicSortShared)
{
    test_sort_shared<<<1, num_threads>>>(ddata.get_data());
    ddata.set_executor(ref);
    auto data_ptr = ddata.get_const_data();
    auto ref_ptr = ref_shared.get_const_data();

    ASSERT_TRUE(std::equal(data_ptr, data_ptr + num_elements, ref_ptr));
}


}  // namespace
