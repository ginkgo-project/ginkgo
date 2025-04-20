// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/bitvector.hpp"

#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "core/base/index_range.hpp"
#include "core/components/bitvector.hpp"
#include "core/test/utils.hpp"
#include "reference/components/bitvector.hpp"


template <typename IndexType>
class Bitvector : public ::testing::Test {
protected:
    using index_type = IndexType;
    using device_type = gko::bitvector<index_type>;
    using storage_type = typename device_type::storage_type;
    constexpr static auto block_size = device_type::block_size;
    Bitvector()
        : ref{gko::ReferenceExecutor::create()}, rng{67593}, sizes{0,    1,
                                                                   2,    16,
                                                                   31,   32,
                                                                   33,   40,
                                                                   63,   64,
                                                                   65,   127,
                                                                   128,  129,
                                                                   1000, 1024,
                                                                   2000}
    {}

    std::vector<index_type> create_random_values(index_type num_values,
                                                 index_type size)
    {
        std::vector<index_type> values(num_values);
        std::uniform_int_distribution<index_type> dist(
            0, std::max(size - 1, index_type{}));
        for (auto& value : values) {
            value = dist(this->rng);
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::default_random_engine rng;
    std::vector<index_type> sizes;
};

TYPED_TEST_SUITE(Bitvector, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(Bitvector, ComputeBitsAndRanks)
{
    using index_type = typename TestFixture::index_type;
    using storage_type = typename TestFixture::storage_type;
    constexpr auto block_size = TestFixture::block_size;
    for (auto size : this->sizes) {
        SCOPED_TRACE(size);
        for (auto num_values :
             {index_type{}, size / 10, size / 4, size / 2, size}) {
            SCOPED_TRACE(num_values);
            auto values = this->create_random_values(num_values, size);
            num_values = values.size();
            auto num_blocks = gko::ceildiv(size, block_size);

            auto bv = gko::kernels::reference::bitvector::from_sorted_indices(
                this->ref, values.data(), num_values, size);
            auto dbv = bv.device_view();

            // check bits and ranks are correct
            ASSERT_EQ(bv.get_size(), size);
            ASSERT_EQ(dbv.size(), size);
            ASSERT_EQ(bv.get_num_blocks(), num_blocks);
            ASSERT_EQ(dbv.num_blocks(), num_blocks);
            auto it = values.begin();
            index_type rank{};
            for (auto i : gko::irange{size}) {
                const auto block = i / block_size;
                const auto local = i % block_size;
                ASSERT_EQ(dbv.rank(i), rank);
                if (it != values.end() && *it == i) {
                    ASSERT_TRUE(bool(bv.get_bits()[block] &
                                     (storage_type{1} << local)));
                    ASSERT_TRUE(dbv.get(i));
                    ++rank;
                    ++it;
                } else {
                    ASSERT_FALSE(bool(bv.get_bits()[block] &
                                      (storage_type{1} << local)));
                    ASSERT_FALSE(dbv.get(i));
                }
            }
        }
    }
}
