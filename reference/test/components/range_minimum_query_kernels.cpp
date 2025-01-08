// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query_kernels.hpp"

#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "core/base/index_range.hpp"
#include "core/components/range_minimum_query.hpp"
#include "core/test/utils.hpp"


template <typename IndexType>
class RangeMinimumQuery : public ::testing::Test {
protected:
    using index_type = IndexType;
    using word_type = std::make_unsigned_t<index_type>;
    using block_argmin_storage_type =
        gko::kernels::reference::range_minimum_query::block_argmin_storage_type<
            IndexType>;
    using superblock_storage_type =
        gko::range_minimum_query_superblocks<index_type>;
    RangeMinimumQuery() : ref{gko::ReferenceExecutor::create()}, rng{167349} {}

    std::vector<index_type> create_random_values(index_type size)
    {
        std::vector<index_type> values(size);
        std::uniform_int_distribution<index_type> dist(0, 10000);
        for (auto& value : values) {
            value = dist(this->rng);
        }
        return values;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::default_random_engine rng;
};

TYPED_TEST_SUITE(RangeMinimumQuery, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(RangeMinimumQuery, ComputeLookupSmall)
{
    using index_type = typename TestFixture::index_type;
    using block_argmin_storage_type =
        typename TestFixture::block_argmin_storage_type;
    constexpr auto block_size =
        gko::kernels::reference::range_minimum_query::small_block_size;
    constexpr auto block_argmin_num_bits = gko::ceil_log2_constexpr(block_size);
    // keep these in sync with small_block_size
    for (index_type size :
         {0, 1, 2, 3, 10, 15, 16, 17, 25, 127, 128, 129, 1023}) {
        SCOPED_TRACE(size);
        const auto values = this->create_random_values(size);
        const auto num_blocks = static_cast<index_type>(gko::ceildiv(
            size,
            gko::kernels::reference::range_minimum_query::small_block_size));
        std::vector<gko::uint32> block_argmin_storage(
            block_argmin_storage_type::storage_size(num_blocks,
                                                    block_argmin_num_bits));
        block_argmin_storage_type block_argmin{
            block_argmin_storage.data(), block_argmin_num_bits, num_blocks};
        std::vector<index_type> block_min(num_blocks);
        std::vector<gko::uint16> block_type(num_blocks);
        gko::block_range_minimum_query_lookup_table<block_size> small_lut;

        gko::kernels::reference::range_minimum_query::compute_lookup_small(
            this->ref, values.data(), size, block_argmin, block_min.data(),
            block_type.data());

        for (auto block : gko::irange{num_blocks}) {
            SCOPED_TRACE(block);
            const auto block_begin = values.begin() + block * block_size;
            const auto block_end =
                values.begin() + std::min(size, (block + 1) * block_size);
            const auto block_local_size = block_end - block_begin;
            const auto min_it = std::min_element(block_begin, block_end);
            const auto min_value = *min_it;
            const auto min_pos = min_it - block_begin;
            ASSERT_EQ(min_pos, block_argmin.get(block));
            ASSERT_EQ(min_value, block_min[block]);
            const auto tree = block_type[block];
            for (auto first : gko::irange{block_local_size}) {
                for (auto last : gko::irange{first, block_local_size}) {
                    const auto argmin = std::distance(
                        block_begin, std::min_element(block_begin + first,
                                                      block_begin + last + 1));
                    ASSERT_EQ(argmin, small_lut.lookup(static_cast<int>(tree),
                                                       first, last))
                        << "in range [" << first << "," << last << "]";
                }
            }
        }
    }
}

TYPED_TEST(RangeMinimumQuery, ComputeLookupLarge)
{
    using index_type = typename TestFixture::index_type;
    using superblock_storage_type =
        typename TestFixture::superblock_storage_type;
    using word_type = typename TestFixture::word_type;
    for (index_type num_blocks :
         {2, 3, 10, 15, 16, 17, 25, 127, 128, 129, 1023}) {
        SCOPED_TRACE(num_blocks);
        const auto block_min = this->create_random_values(num_blocks);
        std::vector<word_type> superblock_storage(
            superblock_storage_type::compute_storage_size(num_blocks));
        superblock_storage_type superblocks(
            block_min.data(), superblock_storage.data(), num_blocks);

        gko::kernels::reference::range_minimum_query::compute_lookup_large(
            this->ref, block_min.data(), num_blocks, superblocks);

        for (auto level : gko::irange(superblocks.num_levels())) {
            SCOPED_TRACE(level);
            const auto block_size =
                superblock_storage_type::block_size_for_level(level);
            for (auto block : gko::irange(num_blocks)) {
                const auto begin = block_min.begin() + block;
                const auto end = block_min.begin() +
                                 std::min(block + block_size, num_blocks);
                const auto argmin = std::min_element(begin, end) - begin;
                ASSERT_EQ(superblocks.block_argmin(level, block), argmin);
            }
        }
    }
}


TYPED_TEST(RangeMinimumQuery, Query)
{
    using index_type = typename TestFixture::index_type;
    using superblock_storage_type =
        typename TestFixture::superblock_storage_type;
    using word_type = typename TestFixture::word_type;
    for (index_type num_blocks :
         {2, 3, 10, 15, 16, 17, 25, 127, 128, 129, 1023}) {
        SCOPED_TRACE(num_blocks);
        const auto block_min = this->create_random_values(num_blocks);
        std::vector<word_type> superblock_storage(
            superblock_storage_type::compute_storage_size(num_blocks));
        superblock_storage_type superblocks(
            block_min.data(), superblock_storage.data(), num_blocks);
        gko::kernels::reference::range_minimum_query::compute_lookup_large(
            this->ref, block_min.data(), num_blocks, superblocks);
        for (auto first : gko::irange{num_blocks}) {
            SCOPED_TRACE(first);
            for (auto last : gko::irange{first, num_blocks}) {
                SCOPED_TRACE(last);
                const auto begin = block_min.begin() + first;
                const auto end = block_min.begin() + last + 1;
                const auto min_it = std::min_element(begin, end);
                const auto argmin = std::distance(block_min.begin(), min_it);
                const auto min = *min_it;

                const auto result = superblocks.query(first, last);

                ASSERT_EQ(min, result.min);
                ASSERT_EQ(argmin, result.argmin);
            }
        }
    }
}
