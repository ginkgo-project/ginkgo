// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cuda/components/sorting.cuh"


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "cuda/test/utils.hpp"


namespace {


using gko::kernels::cuda::bitonic_sort;
using gko::kernels::cuda::config;


constexpr int num_elements = 2048;
constexpr int num_local = 4;
constexpr auto num_threads = num_elements / num_local;


__global__ void test_sort_shared(gko::int32* data)
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


__global__ void test_sort_warp(gko::int32* data)
{
    gko::int32 local[num_local];
    for (int i = 0; i < num_local; ++i) {
        local[i] = data[threadIdx.x * num_local + i];
    }
    bitonic_sort<config::warp_size * num_local, num_local>(
        local, static_cast<gko::int32*>(nullptr));
    for (int i = 0; i < num_local; ++i) {
        data[threadIdx.x * num_local + i] = local[i];
    }
}


class Sorting : public CudaTestFixture {
protected:
    Sorting()
        : rng(123456), ref_shared(ref, num_elements), ref_warp(ref), ddata(exec)
    {
        // we want some duplicate elements
        std::uniform_int_distribution<gko::int32> dist(0, num_elements / 2);
        for (int i = 0; i < num_elements; ++i) {
            ref_shared.get_data()[i] = dist(rng);
        }
        ddata = gko::array<gko::int32>{exec, ref_shared};
        ref_warp = ref_shared;
        std::sort(ref_shared.get_data(), ref_shared.get_data() + num_elements);
        std::sort(ref_warp.get_data(),
                  ref_warp.get_data() + (config::warp_size * num_local));
    }

    std::default_random_engine rng;
    gko::array<gko::int32> ref_shared;
    gko::array<gko::int32> ref_warp;
    gko::array<gko::int32> ddata;
};


TEST_F(Sorting, CudaBitonicSortWarp)
{
    test_sort_warp<<<1, config::warp_size, 0, exec->get_stream()>>>(
        ddata.get_data());
    ddata.set_executor(ref);
    auto data_ptr = ddata.get_const_data();
    auto ref_ptr = ref_warp.get_const_data();

    GKO_ASSERT_ARRAY_EQ(ddata, ref_warp);
}


TEST_F(Sorting, CudaBitonicSortShared)
{
    test_sort_shared<<<1, num_threads, 0, exec->get_stream()>>>(
        ddata.get_data());
    ddata.set_executor(ref);
    auto data_ptr = ddata.get_const_data();
    auto ref_ptr = ref_shared.get_const_data();

    GKO_ASSERT_ARRAY_EQ(ddata, ref_shared);
}


}  // namespace
