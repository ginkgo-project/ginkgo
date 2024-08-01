// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/segmented_range.hpp"

#include <iterator>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>


TEST(SegmentedRange, WorksByIndex)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<std::vector<int>> result_indices(begins.size());
    gko::segmented_index_range<int> range{begins.data(), ends.data(),
                                          static_cast<int>(begins.size())};

    for (auto row : range.segment_indices()) {
        for (auto nz : range[row]) {
            result_indices[row].push_back(nz);
        }
    }

    ASSERT_EQ(result_indices,
              std::vector<std::vector<int>>(
                  {{}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 5}, {9}}));
}


TEST(SegmentedRange, WorksByRangeFor)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<std::vector<int>> result_indices(begins.size());
    gko::segmented_index_range<int> range{begins.data(), ends.data(),
                                          static_cast<int>(begins.size())};

    for (auto [row, segment] : range) {
        for (auto nz : segment) {
            result_indices[row].push_back(nz);
        }
    }

    ASSERT_EQ(result_indices,
              std::vector<std::vector<int>>(
                  {{}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 5}, {9}}));
}


TEST(SegmentedRange, WorksWithPtrsConstructor)
{
    std::vector<int> ptrs{0, 2, 4, 5, 9};
    std::vector<std::vector<int>> result_indices(ptrs.size() - 1);
    gko::segmented_index_range<int> range{ptrs.data(),
                                          static_cast<int>(ptrs.size() - 1)};

    for (auto row : range.segment_indices()) {
        for (auto nz : range[row]) {
            result_indices[row].push_back(nz);
        }
    }

    ASSERT_EQ(result_indices, std::vector<std::vector<int>>(
                                  {{0, 1}, {2, 3}, {4}, {5, 6, 7, 8}}));
}


TEST(SegmentedValueRange, WorksByIndex)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<int> values(ends.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(begins.size());
    gko::segmented_value_range<int, std::vector<int>::iterator> range{
        begins.data(), ends.data(), values.begin(),
        static_cast<int>(begins.size())};

    for (auto row : gko::irange<int>(begins.size())) {
        for (auto nz : range[row]) {
            result_values[row].push_back(nz);
        }
    }

    ASSERT_EQ(result_values,
              std::vector<std::vector<int>>(
                  {{}, {2, 3, 4, 5, 6, 7, 8, 9, 10}, {5, 6}, {10}}));
}


TEST(SegmentedValueRange, WorksByRangeFor)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<int> values(ends.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(begins.size());
    gko::segmented_value_range<int, std::vector<int>::iterator> range{
        begins.data(), ends.data(), values.begin(),
        static_cast<int>(begins.size())};

    for (auto [row, segment] : range) {
        for (auto nz : segment) {
            result_values[row].push_back(nz);
        }
    }

    ASSERT_EQ(result_values,
              std::vector<std::vector<int>>(
                  {{}, {2, 3, 4, 5, 6, 7, 8, 9, 10}, {5, 6}, {10}}));
}


TEST(SegmentedValueRange, WorksWithPtrsConstructor)
{
    std::vector<int> ptrs{0, 2, 4, 5, 9};
    std::vector<int> values(ptrs.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(ptrs.size() - 1);
    gko::segmented_value_range<int, std::vector<int>::iterator> range{
        ptrs.data(), values.begin(), static_cast<int>(ptrs.size() - 1)};

    for (auto row : range.segment_indices()) {
        for (auto nz : range[row]) {
            result_values[row].push_back(nz);
        }
    }

    ASSERT_EQ(result_values, std::vector<std::vector<int>>(
                                 {{1, 2}, {3, 4}, {5}, {6, 7, 8, 9}}));
}


TEST(SegmentedEnumeratedValueRange, WorksByIndex)
{
    using gko::get;
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<int> values(ends.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(begins.size());
    std::vector<std::vector<int>> result_indices(begins.size());
    gko::segmented_value_range<int, std::vector<int>::iterator> range{
        begins.data(), ends.data(), values.begin(),
        static_cast<int>(begins.size())};

    for (auto row : range.segment_indices()) {
        for (auto tuple : range.enumerated()[row]) {
            result_indices[row].push_back(get<0>(tuple));
            result_values[row].push_back(get<1>(tuple));
        }
    }

    ASSERT_EQ(result_indices,
              std::vector<std::vector<int>>(
                  {{}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 5}, {9}}));
    ASSERT_EQ(result_values,
              std::vector<std::vector<int>>(
                  {{}, {2, 3, 4, 5, 6, 7, 8, 9, 10}, {5, 6}, {10}}));
}


TEST(SegmentedEnumeratedValueRange, WorksByRangeFor)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<int> values(ends.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(begins.size());
    std::vector<std::vector<int>> result_indices(begins.size());
    gko::segmented_value_range<int, std::vector<int>::iterator> range{
        begins.data(), ends.data(), values.begin(),
        static_cast<int>(begins.size())};
    auto enumerated_range = range.enumerated();

    for (auto [row, segment] : enumerated_range) {
        for (auto [index, value] : segment) {
            result_indices[row].push_back(index);
            result_values[row].push_back(value);
        }
    }

    ASSERT_EQ(result_indices,
              std::vector<std::vector<int>>(
                  {{}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 5}, {9}}));
    ASSERT_EQ(result_values,
              std::vector<std::vector<int>>(
                  {{}, {2, 3, 4, 5, 6, 7, 8, 9, 10}, {5, 6}, {10}}));
}


#ifndef NDEBUG


bool check_assertion_exit_code(int exit_code)
{
#ifdef _MSC_VER
    // MSVC picks up the exit code incorrectly,
    // so we can only check that it exits
    return true;
#else
    return exit_code != 0;
#endif
}


TEST(DeathTest, Assertions)
{
    using range_t = gko::segmented_index_range<int>;
    using vrange_t = gko::segmented_value_range<int, int*>;
    using range_it_t = range_t::iterator;
    using vrange_it_t = vrange_t::iterator;
    std::vector<int> ptrs{0, 1};
    std::vector<int> values{0, 1};
    range_t range{ptrs.data(), static_cast<int>(ptrs.size() - 1)};
    range_t range2{ptrs.data(), 0};
    vrange_t vrange{ptrs.data(), values.data(),
                    static_cast<int>(ptrs.size() - 1)};
    vrange_t vrange2{ptrs.data(), values.data(), 0};
    // gko::segmented_index_range::iterator
    EXPECT_EXIT((void)*(range_it_t{range, -1}), check_assertion_exit_code, "");
    EXPECT_EXIT((void)*(range_it_t{range, 1}), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(range_it_t{range, 0} == range_it_t{range2, 0}),
                check_assertion_exit_code, "");
    // gko::segmented_index_range
    EXPECT_EXIT((void)(range_t{nullptr, -1}), check_assertion_exit_code, "");
    EXPECT_EXIT((void)range[-1], check_assertion_exit_code, "");
    EXPECT_EXIT((void)range[1], check_assertion_exit_code, "");
    EXPECT_EXIT((void)range.begin_index(-1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)range.begin_index(1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)range.end_index(-1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)range.end_index(1), check_assertion_exit_code, "");
    // gko::segmented_value_range::iterator
    EXPECT_EXIT((void)*(vrange_it_t{vrange, -1}), check_assertion_exit_code,
                "");
    EXPECT_EXIT((void)*(vrange_it_t{vrange, 1}), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(vrange_it_t{vrange, 0} == vrange_it_t{vrange2, 0}),
                check_assertion_exit_code, "");
    // gko::segmented_value_range
    EXPECT_EXIT((void)(vrange_t{nullptr, nullptr, -1}),
                check_assertion_exit_code, "");
    EXPECT_EXIT((void)vrange[-1], check_assertion_exit_code, "");
    EXPECT_EXIT((void)vrange[1], check_assertion_exit_code, "");
    EXPECT_EXIT((void)vrange.begin_index(-1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)vrange.begin_index(1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)vrange.end_index(-1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)vrange.end_index(1), check_assertion_exit_code, "");
}


#endif
