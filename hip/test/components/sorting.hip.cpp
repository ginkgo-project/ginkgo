// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/components/sorting.hpp"

#include <memory>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/base/index_range.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


using gko::kernels::hip::bitonic_sort;
using gko::kernels::hip::config;


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


class Sorting : public HipTestFixture {
protected:
    Sorting()
        : rng(123456), ref_shared(ref, num_elements), ref_warp(ref), ddata(exec)
    {
        // we want some duplicate elements
        std::uniform_int_distribution<gko::int32> dist(0, num_elements / 2);
        for (auto i = 0; i < num_elements; ++i) {
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


TEST_F(Sorting, HipBitonicSortWarp)
{
    test_sort_warp<<<1, config::warp_size, 0, exec->get_stream()>>>(
        ddata.get_data());
    ddata.set_executor(ref);
    auto data_ptr = ddata.get_const_data();
    auto ref_ptr = ref_warp.get_const_data();

    ASSERT_TRUE(std::equal(data_ptr, data_ptr + (num_local * config::warp_size),
                           ref_ptr));
}


TEST_F(Sorting, HipBitonicSortShared)
{
    test_sort_shared<<<1, num_threads, 0, exec->get_stream()>>>(
        ddata.get_data());
    ddata.set_executor(ref);
    auto data_ptr = ddata.get_const_data();
    auto ref_ptr = ref_shared.get_const_data();

    ASSERT_TRUE(std::equal(data_ptr, data_ptr + num_elements, ref_ptr));
}


constexpr auto num_buckets = 10;

std::array<int, num_buckets + 1> run_bucketsort(
    std::shared_ptr<const gko::HipExecutor> exec, const int* input, int size,
    int* output, gko::array<int>& tmp)
{
    return gko::kernels::hip::bucket_sort<num_buckets>(
        exec, input, input + size, output,
        [] __device__(int i) { return i / 100; }, tmp);
}

TEST_F(Sorting, BucketSort)
{
    for (int size : {0, 1, 10, 100, 1000, 10000, 100000}) {
        SCOPED_TRACE(size);
        const auto proj = [](auto i) { return i / 100; };
        const auto comp = [&proj](auto a, auto b) { return proj(a) < proj(b); };
        gko::array<int> data{ref, static_cast<gko::size_type>(size)};
        std::uniform_int_distribution<int> dist{0, num_buckets * 100 - 1};
        for (auto i : gko::irange{size}) {
            data.get_data()[i] = dist(rng);
        }
        data.set_executor(exec);
        gko::array<int> out_data{exec, static_cast<gko::size_type>(size)};
        gko::array<int> tmp{exec};

        auto offsets = run_bucketsort(exec, data.get_const_data(), size,
                                      out_data.get_data(), tmp);

        data.set_executor(ref);
        out_data.set_executor(ref);
        const auto out_data_ptr = out_data.get_data();
        const auto data_ptr = data.get_data();
        // the output must be sorted by bucket
        ASSERT_TRUE(std::is_sorted(out_data_ptr, out_data_ptr + size, comp));
        // the output offsets must describe the bucket ranges
        for (int bucket = 0; bucket < num_buckets; bucket++) {
            const auto bucket_begin = offsets[bucket];
            const auto bucket_end = offsets[bucket + 1];
            ASSERT_LE(bucket_begin, bucket_end);
            for (const auto i : gko::irange{bucket_begin, bucket_end}) {
                ASSERT_EQ(proj(out_data_ptr[i]), bucket);
            }
        }
        // inside each bucket, the input and output data must be the same
        std::sort(data_ptr, data_ptr + size);
        std::sort(out_data_ptr, out_data_ptr + size);
        std::stable_sort(data_ptr, data_ptr + size, comp);
        std::stable_sort(out_data_ptr, out_data_ptr + size, comp);
        GKO_ASSERT_ARRAY_EQ(data, out_data);
    }
}


}  // namespace
