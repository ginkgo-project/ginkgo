// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iterator>
#include <numeric>
#include <vector>


#include <gtest/gtest.h>


#include "core/base/segmented_range.hpp"


TEST(IndexedIterator, IteratorProperties)
{
    using iterator_type =
        gko::indexed_iterator<std::vector<int>::const_iterator,
                              gko::irange<int>::iterator>;
    gko::irange_strided<int> range(0, 10, 2);
    std::vector<int> values(10);
    std::iota(values.begin(), values.end(), 1);

    auto it = iterator_type{values.begin(), range.begin()};
    it += 4;
    ASSERT_EQ(*it, 9);
    it -= 4;
    ASSERT_EQ(*it, 1);
    ASSERT_EQ(*it++, 1);
    ASSERT_EQ(*++it, 5);
    ASSERT_EQ(*it--, 5);
    ASSERT_EQ(*--it, 1);
    ASSERT_EQ(*(it + 1), 3);
    ASSERT_EQ(*(it - -1), 3);
    ASSERT_EQ(it[2], 5);
    ASSERT_EQ((it + 4) - it, 4);
    ASSERT_TRUE(it == it);
    ASSERT_TRUE(it != (it + 1));
    ASSERT_TRUE(it < (it + 1));
    ASSERT_TRUE(it <= (it + 1));
    ASSERT_TRUE(it <= it);
    ASSERT_TRUE((it + 1) > it);
    ASSERT_TRUE((it + 1) >= it);
    ASSERT_TRUE(it >= it);
    ASSERT_FALSE(it != it);
    ASSERT_FALSE(it == (it + 1));
    ASSERT_FALSE(it > it);
    ASSERT_FALSE(it > (it + 1));
    ASSERT_FALSE(it < it);
    ASSERT_FALSE((it + 1) < it);
    ASSERT_FALSE(it >= (it + 1));
    ASSERT_FALSE((it + 1) <= it);
}


TEST(IndexedIterator, CanAccessMemberOfStruct)
{
    struct value {
        int i;
    };
    using iterator_type =
        gko::indexed_iterator<std::vector<value>::const_iterator,
                              gko::irange<int>::iterator>;
    gko::irange_strided<int> range(0, 10, 2);
    std::vector<value> values(10, value{4});

    auto it = iterator_type{values.begin(), range.begin()};
    ASSERT_EQ(it->i, 4);
}


TEST(IndexedRange, RangeForLoop)
{
    gko::irange_strided<int> index_range(0, 10, 2);
    std::vector<int> values(10);
    std::iota(values.begin(), values.end(), 1);
    std::vector<int> result;
    gko::indexed_range<std::vector<int>::const_iterator,
                       gko::irange<int>::iterator>
        range{values.begin(), index_range.begin(), index_range.end()};

    for (auto value : range) {
        result.push_back(value);
    }

    ASSERT_EQ(result, std::vector<int>({1, 3, 5, 7, 9}));
}


TEST(IndexedRange, WorksInAlgorithm)
{
    gko::irange_strided<int> index_range(0, 10, 2);
    std::vector<int> values(10);
    std::iota(values.begin(), values.end(), 1);
    std::vector<int> result;
    gko::indexed_range<std::vector<int>::const_iterator,
                       gko::irange<int>::iterator>
        range{values.begin(), index_range.begin(), index_range.end()};

    auto it = std::lower_bound(range.begin(), range.end(), 6);

    ASSERT_EQ(*it, 7);
    ASSERT_EQ(it - range.begin(), 3);
}


TEST(EnumeratingIndexedIterator, IteratorProperties)
{
    using iterator_type =
        gko::enumerating_indexed_iterator<std::vector<int>::const_iterator,
                                          gko::irange<int>::iterator>;
    using tuple_type = typename iterator_type::enumerated;
    gko::irange_strided<int> range(0, 10, 2);
    std::vector<int> values(10);
    std::iota(values.begin(), values.end(), 1);

    auto it = iterator_type{values.begin(), range.begin()};
    it += 4;
    ASSERT_EQ(it->value, 9);
    ASSERT_EQ(it->index, 8);
    ASSERT_EQ(*it, tuple_type(8, 9));
    it -= 4;
    ASSERT_EQ(*it, tuple_type(0, 1));
    ASSERT_EQ(*it++, tuple_type(0, 1));
    ASSERT_EQ(*++it, tuple_type(4, 5));
    ASSERT_EQ(*it--, tuple_type(4, 5));
    ASSERT_EQ(*--it, tuple_type(0, 1));
    ASSERT_EQ(*(it + 1), tuple_type(2, 3));
    ASSERT_EQ(*(it - -1), tuple_type(2, 3));
    ASSERT_EQ(it[2], tuple_type(4, 5));
    ASSERT_EQ((it + 4) - it, 4);
    ASSERT_TRUE(it == it);
    ASSERT_TRUE(it != (it + 1));
    ASSERT_TRUE(it < (it + 1));
    ASSERT_TRUE(it <= (it + 1));
    ASSERT_TRUE(it <= it);
    ASSERT_TRUE((it + 1) > it);
    ASSERT_TRUE((it + 1) >= it);
    ASSERT_TRUE(it >= it);
    ASSERT_FALSE(it != it);
    ASSERT_FALSE(it == (it + 1));
    ASSERT_FALSE(it > it);
    ASSERT_FALSE(it > (it + 1));
    ASSERT_FALSE(it < it);
    ASSERT_FALSE((it + 1) < it);
    ASSERT_FALSE(it >= (it + 1));
    ASSERT_FALSE((it + 1) <= it);
}


TEST(EnumeratingIndexedRange, RangeForLoop)
{
    using range_type =
        gko::enumerating_indexed_range<std::vector<int>::const_iterator,
                                       gko::irange<int>::iterator>;
    using iterator_type = typename range_type::iterator;
    using tuple_type = typename iterator_type::enumerated;
    gko::irange_strided<int> index_range(0, 10, 2);
    std::vector<int> values(10);
    std::iota(values.begin(), values.end(), 1);
    std::vector<int> result;
    range_type range{values.begin(), index_range.begin(), index_range.end()};

    for (auto tuple : range) {
        ASSERT_EQ(tuple.value, tuple.index + 1);
        result.push_back(tuple.value);
    }

    ASSERT_EQ(result, std::vector<int>({1, 3, 5, 7, 9}));
}


TEST(EnumeratingIndexedRange, WorksInAlgorithm)
{
    using range_type =
        gko::enumerating_indexed_range<std::vector<int>::const_iterator,
                                       gko::irange<int>::iterator>;
    using iterator_type = typename range_type::iterator;
    using tuple_type = typename iterator_type::enumerated;
    gko::irange_strided<int> index_range(0, 10, 2);
    std::vector<int> values(10);
    std::iota(values.begin(), values.end(), 1);
    std::vector<tuple_type> result;
    range_type range{values.begin(), index_range.begin(), index_range.end()};

    std::copy(range.begin(), range.end(), std::back_inserter(result));

    ASSERT_EQ(result, std::vector<tuple_type>(
                          {tuple_type{0, 1}, tuple_type{2, 3}, tuple_type{4, 5},
                           tuple_type{6, 7}, tuple_type{8, 9}}));
}


TEST(SegmentedRange, Works)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<std::vector<int>> result_indices(begins.size());
    gko::segmented_range<std::vector<int>::iterator> range{
        begins.begin(), ends.begin(), static_cast<int>(begins.size())};

    for (auto row : gko::irange<int>(begins.size())) {
        for (auto nz : range[row]) {
            result_indices[row].push_back(nz);
        }
    }

    ASSERT_EQ(result_indices,
              std::vector<std::vector<int>>(
                  {{}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 5}, {9}}));
}


TEST(SegmentedValueRange, Works)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<int> values(ends.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(begins.size());
    gko::segmented_value_range<std::vector<int>::iterator,
                               std::vector<int>::iterator>
        range{values.begin(), begins.begin(), ends.begin(),
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


TEST(SegmentedEnumeratedValueRange, Works)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<int> values(ends.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(begins.size());
    std::vector<std::vector<int>> result_indices(begins.size());
    gko::segmented_value_range<std::vector<int>::iterator,
                               std::vector<int>::iterator>
        range{values.begin(), begins.begin(), ends.begin(),
              static_cast<int>(begins.size())};

    for (auto row : gko::irange<int>(begins.size())) {
        for (auto tuple : range.enumerate(row)) {
            result_values[row].push_back(tuple.value);
            result_indices[row].push_back(tuple.index);
        }
    }

    ASSERT_EQ(result_indices,
              std::vector<std::vector<int>>(
                  {{}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 5}, {9}}));
    ASSERT_EQ(result_values,
              std::vector<std::vector<int>>(
                  {{}, {2, 3, 4, 5, 6, 7, 8, 9, 10}, {5, 6}, {10}}));
}
