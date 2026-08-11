// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "omp/components/sorting.hpp"

#include <memory>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/base/index_range.hpp"


class Sorting : public ::testing::Test {
protected:
    Sorting() : exec(gko::OmpExecutor::create()), rng{729854} {}

    std::shared_ptr<const gko::OmpExecutor> exec;
    std::default_random_engine rng;
};


TEST_F(Sorting, BucketSort)
{
    constexpr auto num_buckets = 10;
    for (int size : {0, 1, 10, 100, 1000, 10000, 100000}) {
        SCOPED_TRACE(size);
        const auto proj = [](auto i) { return i / 100; };
        const auto comp = [&proj](auto a, auto b) { return proj(a) < proj(b); };
        std::vector<int> data(size);
        std::uniform_int_distribution<int> dist{0, num_buckets * 100 - 1};
        for (auto& value : data) {
            value = dist(rng);
        }
        std::vector<int> out_data(size);
        gko::array<gko::int64> tmp{exec};

        auto offsets = gko::kernels::omp::bucket_sort<num_buckets>(
            data.cbegin(), data.cend(), out_data.begin(), proj, tmp);

        // the output must be sorted by bucket
        ASSERT_TRUE(std::is_sorted(out_data.begin(), out_data.end(), comp));
        // the output offsets must describe the bucket ranges
        for (int bucket = 0; bucket < num_buckets; bucket++) {
            const auto bucket_begin = offsets[bucket];
            const auto bucket_end = offsets[bucket + 1];
            ASSERT_LE(bucket_begin, bucket_end);
            for (const auto i : gko::irange{bucket_begin, bucket_end}) {
                ASSERT_EQ(proj(out_data[i]), bucket);
            }
        }
        // inside each bucket, the input and output data must be the same
        std::sort(data.begin(), data.end());
        std::sort(out_data.begin(), out_data.end());
        std::stable_sort(data.begin(), data.end(), comp);
        std::stable_sort(out_data.begin(), out_data.end(), comp);
        ASSERT_EQ(data, out_data);
    }
}
