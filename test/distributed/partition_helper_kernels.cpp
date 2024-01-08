// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/base/iterator_factory.hpp"
#include "core/distributed/partition_helpers_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


using gko::experimental::distributed::comm_index_type;


// TODO: remove with c++17
template <typename T>
T clamp(const T& v, const T& lo, const T& hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}


template <typename IndexType>
std::vector<IndexType> create_iota(IndexType min, IndexType max)
{
    std::vector<IndexType> iota(clamp(max - min, IndexType(0), max));
    std::iota(iota.begin(), iota.end(), min);
    return iota;
}


template <typename IndexType>
std::vector<IndexType> create_range_offsets(gko::size_type num_ranges)
{
    std::default_random_engine engine;
    std::uniform_int_distribution<IndexType> dist(5, 10);
    std::vector<IndexType> range_sizes(num_ranges);
    std::generate(range_sizes.begin(), range_sizes.end(),
                  [&]() { return dist(engine); });

    std::vector<IndexType> range_offsets(num_ranges + 1, 0);
    std::partial_sum(range_sizes.begin(), range_sizes.end(),
                     range_offsets.begin() + 1);
    return range_offsets;
}


template <typename IndexType>
std::vector<IndexType> create_ranges(
    const std::vector<IndexType>& range_offsets)
{
    assert(range_offsets.size() >= 2);
    gko::size_type num_ranges = range_offsets.size() - 1;
    std::vector<IndexType> ranges(num_ranges * 2, 0);
    for (gko::size_type i = 1; i < num_ranges; ++i) {
        ranges[2 * i - 1] = range_offsets[i];
        ranges[2 * i] = range_offsets[i];
    }
    ranges.back() = range_offsets.back();
    return ranges;
}


template <typename IndexType>
std::vector<IndexType> create_ranges(gko::size_type num_ranges)
{
    auto range_offsets = create_range_offsets<IndexType>(num_ranges);

    return create_ranges(range_offsets);
}


std::vector<std::size_t> sample_unique(std::size_t min, std::size_t max,
                                       gko::size_type n)
{
    std::default_random_engine engine;
    auto values = create_iota(min, max);
    std::shuffle(values.begin(), values.end(), engine);
    values.erase(values.begin() + clamp(n, gko::size_type(0), values.size()),
                 values.end());
    return values;
}


template <typename IndexType>
std::vector<IndexType> remove_indices(const std::vector<IndexType>& source,
                                      std::vector<std::size_t> idxs)
{
    std::sort(idxs.begin(), idxs.end(), std::greater<>{});
    auto result = source;
    for (auto idx : idxs) {
        result.erase(result.begin() + 2 * idx, result.begin() + 2 * idx + 1);
    }
    return result;
}


template <typename IndexType>
gko::array<IndexType> make_array(std::shared_ptr<const gko::Executor> exec,
                                 const std::vector<IndexType>& v)
{
    return gko::array<IndexType>(exec, v.begin(), v.end());
}


template <typename IndexType>
std::pair<std::vector<IndexType>, std::vector<comm_index_type>>
shuffle_range_and_pid(const std::vector<IndexType>& ranges,
                      const std::vector<comm_index_type>& pid)
{
    std::default_random_engine engine;

    auto result = std::make_pair(ranges, pid);

    auto num_ranges = result.second.size();
    auto range_start_it = gko::detail::make_permute_iterator(
        result.first.begin(), [](const auto i) { return 2 * i; });
    auto range_end_it = gko::detail::make_permute_iterator(
        result.first.begin() + 1, [](const auto i) { return 2 * i; });
    auto zip_it = gko::detail::make_zip_iterator(range_start_it, range_end_it,
                                                 result.second.begin());
    std::shuffle(zip_it, zip_it + num_ranges, engine);

    return result;
}


template <typename IndexType>
class PartitionHelpers : public CommonTestFixture {
protected:
    using index_type = IndexType;
};

TYPED_TEST_SUITE(PartitionHelpers, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(PartitionHelpers, CanCheckConsecutiveRanges)
{
    using index_type = typename TestFixture::index_type;
    auto offsets = make_array(this->exec, create_ranges<index_type>(100));
    bool result = false;

    gko::kernels::EXEC_NAMESPACE::partition_helpers::check_consecutive_ranges(
        this->exec, offsets, result);

    ASSERT_TRUE(result);
}


TYPED_TEST(PartitionHelpers, CanCheckNonConsecutiveRanges)
{
    using index_type = typename TestFixture::index_type;
    auto full_range_ends = create_ranges<index_type>(100);
    auto removal_idxs = sample_unique(0, full_range_ends.size() / 2, 4);
    auto start_ends =
        make_array(this->exec, remove_indices(full_range_ends, removal_idxs));
    bool result = true;

    gko::kernels::EXEC_NAMESPACE::partition_helpers::check_consecutive_ranges(
        this->exec, start_ends, result);

    ASSERT_FALSE(result);
}


TYPED_TEST(PartitionHelpers, CanCheckConsecutiveRangesWithSingleRange)
{
    using index_type = typename TestFixture::index_type;
    auto start_ends = make_array(this->ref, create_ranges<index_type>(1));
    bool result = false;

    gko::kernels::EXEC_NAMESPACE::partition_helpers::check_consecutive_ranges(
        this->exec, start_ends, result);

    ASSERT_TRUE(result);
}


TYPED_TEST(PartitionHelpers, CanCheckConsecutiveRangesWithSingleElement)
{
    using index_type = typename TestFixture::index_type;
    auto start_ends = gko::array<index_type>(this->exec, {1});
    bool result = false;

    gko::kernels::EXEC_NAMESPACE::partition_helpers::check_consecutive_ranges(
        this->exec, start_ends, result);

    ASSERT_TRUE(result);
}


TYPED_TEST(PartitionHelpers, CanSortConsecutiveRanges)
{
    using index_type = typename TestFixture::index_type;
    auto start_ends = make_array(this->exec, create_ranges<index_type>(100));
    auto part_ids = create_iota<comm_index_type>(0, 100);
    auto part_ids_arr = gko::array<comm_index_type>(
        this->exec, part_ids.begin(), part_ids.end());
    auto expected_start_ends = start_ends;
    auto expected_part_ids = part_ids_arr;

    gko::kernels::EXEC_NAMESPACE::partition_helpers::sort_by_range_start(
        this->exec, start_ends, part_ids_arr);

    GKO_ASSERT_ARRAY_EQ(expected_start_ends, start_ends);
    GKO_ASSERT_ARRAY_EQ(expected_part_ids, part_ids_arr);
}


TYPED_TEST(PartitionHelpers, CanSortNonConsecutiveRanges)
{
    using index_type = typename TestFixture::index_type;
    auto ranges = create_ranges<index_type>(100);
    auto part_ids = create_iota(0, 100);
    auto shuffled = shuffle_range_and_pid(ranges, part_ids);
    auto expected_start_ends = make_array(this->exec, ranges);
    auto expected_part_ids = gko::array<comm_index_type>(
        this->exec, part_ids.begin(), part_ids.end());
    auto start_ends = make_array(this->exec, shuffled.first);
    auto part_ids_arr = gko::array<comm_index_type>(
        this->exec, shuffled.second.begin(), shuffled.second.end());

    gko::kernels::EXEC_NAMESPACE::partition_helpers::sort_by_range_start(
        this->exec, start_ends, part_ids_arr);

    GKO_ASSERT_ARRAY_EQ(expected_start_ends, start_ends);
    GKO_ASSERT_ARRAY_EQ(expected_part_ids, part_ids_arr);
}


TYPED_TEST(PartitionHelpers, CanCompressRanges)
{
    using index_type = typename TestFixture::index_type;
    auto expected_offsets = create_range_offsets<index_type>(100);
    auto ranges = make_array(this->exec, create_ranges(expected_offsets));
    gko::array<index_type> offsets{this->exec, expected_offsets.size()};

    gko::kernels::EXEC_NAMESPACE::partition_helpers::compress_ranges(
        this->exec, ranges, offsets);

    GKO_ASSERT_ARRAY_EQ(offsets, make_array(this->exec, expected_offsets));
}
