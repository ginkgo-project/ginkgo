/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/distributed/partition_helpers_kernels.hpp"


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/base/iterator_factory.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


using gko::experimental::distributed::comm_index_type;

template <typename IndexType>
using range_container =
    std::pair<std::vector<IndexType>, std::vector<IndexType>>;


// TODO: remove with c++17
template <typename T>
T clamp(const T& v, const T& lo, const T& hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}


template <typename IndexType>
std::vector<IndexType> create_iota(IndexType min, IndexType max)
{
    std::vector<IndexType> iota(
        clamp(max - min, static_cast<IndexType>(0), max));
    std::iota(iota.begin(), iota.end(), min);
    return iota;
}


template <typename IndexType>
range_container<IndexType> create_ranges(gko::size_type num_ranges)
{
    std::default_random_engine engine;
    std::uniform_int_distribution<IndexType> dist(5, 10);
    std::vector<IndexType> range_sizes(num_ranges);
    std::generate(range_sizes.begin(), range_sizes.end(),
                  [&]() { return dist(engine); });

    std::vector<IndexType> range_offsets(num_ranges + 1, 0);
    std::partial_sum(range_sizes.begin(), range_sizes.end(),
                     range_offsets.begin() + 1);

    std::vector<IndexType> range_starts(num_ranges);
    std::vector<IndexType> range_ends(num_ranges);
    std::copy_n(range_offsets.begin(), num_ranges, range_starts.begin());
    std::copy_n(range_offsets.begin() + 1, num_ranges, range_ends.begin());

    return {std::move(range_starts), std::move(range_ends)};
}


std::vector<std::size_t> sample_unique(std::size_t min, std::size_t max,
                                       gko::size_type n)
{
    std::default_random_engine engine;
    auto values = create_iota(min, max);
    std::shuffle(values.begin(), values.end(), engine);
    values.erase(values.begin() + clamp(n, 0ul, values.size()), values.end());
    return values;
}


template <typename IndexType>
std::vector<IndexType> remove_indices(const std::vector<IndexType>& source,
                                      std::vector<std::size_t> idxs)
{
    std::sort(idxs.begin(), idxs.end(), std::greater<>{});
    auto result = source;
    for (auto idx : idxs) {
        result.erase(result.begin() + idx);
    }
    return result;
}


template <typename IndexType>
gko::array<IndexType> concat_start_end(
    std::shared_ptr<const gko::Executor> exec,
    const range_container<IndexType>& start_ends)
{
    gko::size_type num_ranges = start_ends.first.size();
    gko::array<IndexType> concat(exec, num_ranges * 2);

    exec->copy_from(exec->get_master().get(), num_ranges,
                    start_ends.first.data(), concat.get_data());
    exec->copy_from(exec->get_master().get(), num_ranges,
                    start_ends.second.data(), concat.get_data() + num_ranges);

    return concat;
}


template <typename IndexType>
std::pair<range_container<IndexType>, std::vector<comm_index_type>>
shuffle_range_and_pid(const range_container<IndexType>& ranges,
                      const std::vector<comm_index_type>& pid)
{
    std::default_random_engine engine;

    auto result = std::make_pair(ranges, pid);

    auto num_ranges = result.second.size();
    auto zip_it = gko::detail::make_zip_iterator(
        result.first.first.begin(),
        result.first.second.begin(),
        result.second.begin());
    std::shuffle(zip_it, zip_it + num_ranges, engine);

    return result;
}

template <typename IndexType>
class PartitionHelpers : public CommonTestFixture {
protected:
    using index_type = IndexType;
};

TYPED_TEST_SUITE(PartitionHelpers, gko::test::IndexTypes);


TYPED_TEST(PartitionHelpers, CanCheckConsecutiveRanges)
{
    using index_type = typename TestFixture::index_type;
    auto start_ends =
        concat_start_end(this->exec, create_ranges<index_type>(100));
    bool result = false;

    gko::kernels::EXEC_NAMESPACE::partition_helpers::check_consecutive_ranges(
        this->exec, start_ends, &result);

    ASSERT_TRUE(result);
}


TYPED_TEST(PartitionHelpers, CanCheckNonConsecutiveRanges)
{
    using index_type = typename TestFixture::index_type;
    auto full_range_ends = create_ranges<index_type>(100);
    auto removal_idxs = sample_unique(0, full_range_ends.first.size(), 4);
    auto start_ends = concat_start_end(
        this->exec,
        std::make_pair(remove_indices(full_range_ends.first, removal_idxs),
                       remove_indices(full_range_ends.second, removal_idxs)));
    bool result = true;

    gko::kernels::EXEC_NAMESPACE::partition_helpers::check_consecutive_ranges(
        this->exec, start_ends, &result);

    ASSERT_FALSE(result);
}


TYPED_TEST(PartitionHelpers, CanCheckConsecutiveRangesWithSingleRange)
{
    using index_type = typename TestFixture::index_type;
    auto start_ends = concat_start_end(this->ref, create_ranges<index_type>(1));
    bool result = false;

    gko::kernels::EXEC_NAMESPACE::partition_helpers::check_consecutive_ranges(
        this->exec, start_ends, &result);

    ASSERT_TRUE(result);
}


TYPED_TEST(PartitionHelpers, CanCheckConsecutiveRangesWithSingleElement)
{
    using index_type = typename TestFixture::index_type;
    auto start_ends = gko::array<index_type>(this->exec, {1});
    bool result = false;

    gko::kernels::EXEC_NAMESPACE::partition_helpers::check_consecutive_ranges(
        this->exec, start_ends, &result);

    ASSERT_TRUE(result);
}


TYPED_TEST(PartitionHelpers, CanSortConsecutiveRanges)
{
    using index_type = typename TestFixture::index_type;
    auto start_ends =
        concat_start_end(this->exec, create_ranges<index_type>(100));
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
    auto expected_start_ends = concat_start_end(this->exec, ranges);
    auto expected_part_ids = gko::array<comm_index_type>(
        this->exec, part_ids.begin(), part_ids.end());
    auto start_ends = concat_start_end(this->exec, shuffled.first);
    auto part_ids_arr = gko::array<comm_index_type>(
        this->exec, shuffled.second.begin(), shuffled.second.end());

    gko::kernels::EXEC_NAMESPACE::partition_helpers::sort_by_range_start(
        this->exec, start_ends, part_ids_arr);

    GKO_ASSERT_ARRAY_EQ(expected_start_ends, start_ends);
    GKO_ASSERT_ARRAY_EQ(expected_part_ids, part_ids_arr);
}
