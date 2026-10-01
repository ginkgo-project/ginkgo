// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "omp/components/prefix_sum.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>

#include "core/base/index_range.hpp"
#include "core/test/utils.hpp"


template <typename T>
class PrefixSum : public ::testing::Test {
protected:
    using index_type = T;

    PrefixSum() : exec{gko::OmpExecutor::create()}, rand(293) {}

    std::shared_ptr<const gko::OmpExecutor> exec;
    std::default_random_engine rand;
    gko::size_type total_size;
};

TYPED_TEST_SUITE(PrefixSum, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(PrefixSum, SegmentedPrefixSumWorks)
{
    using index_type = typename TestFixture::index_type;
    const auto max_threads = omp_get_max_threads();
    for (int num_threads = 1; num_threads <= max_threads; num_threads++) {
        SCOPED_TRACE(num_threads);
        omp_set_num_threads(num_threads);
        for (int num_ranges : {10, 100, 1000}) {
            SCOPED_TRACE(num_ranges);
            // repeate multiple times for different random seeds
            for (int repetition : gko::irange{10}) {
                std::uniform_int_distribution<int> count_dist{0, 100};
                std::uniform_int_distribution<index_type> value_dist{-200, 200};
                std::vector<index_type> ref_result;
                std::vector<int> keys;
                std::vector<index_type> input;
                for (int i = 0; i < num_ranges; i++) {
                    const auto start = keys.size();
                    const auto new_count = count_dist(this->rand);
                    keys.insert(keys.end(), new_count, i);
                    std::generate_n(std::back_inserter(input), new_count,
                                    [&] { return value_dist(this->rand); });
                    std::copy(input.begin() + start, input.end(),
                              std::back_inserter(ref_result));
                    std::exclusive_scan(
                        ref_result.begin() + start, ref_result.end(),
                        ref_result.begin() + start, index_type{});
                }

                gko::kernels::omp::components::segmented_prefix_sum(
                    this->exec, keys.cbegin(), input.begin(), keys.size());

                ASSERT_EQ(input, ref_result);
            }
        }
    }
}
