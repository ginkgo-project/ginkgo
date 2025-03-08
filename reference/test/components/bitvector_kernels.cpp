// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/bitvector.hpp"

#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "core/base/index_range.hpp"
#include "core/components/bitvector_kernels.hpp"
#include "core/test/utils.hpp"


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
            const auto num_blocks = (size + block_size - 1) / block_size;
            std::vector<storage_type> bits(num_blocks, ~storage_type{});
            std::vector<index_type> ranks(num_blocks, -1);

            gko::kernels::reference::bitvector::compute_bits_and_ranks(
                this->ref, values.data(), num_values, size, bits.data(),
                ranks.data());

            // check bits and ranks are correct
            gko::device_bitvector<index_type> bv(bits.data(), ranks.data(),
                                                 size);
            ASSERT_EQ(bv.size(), size);
            ASSERT_EQ(bv.num_blocks(), num_blocks);
            auto it = values.begin();
            index_type rank{};
            for (auto i : gko::irange{size}) {
                const auto block = i / block_size;
                const auto local = i % block_size;
                ASSERT_EQ(bv.rank(i), rank);
                if (it != values.end() && *it == i) {
                    ASSERT_TRUE(bool(bits[block] & (storage_type{1} << local)));
                    ASSERT_TRUE(bv.get(i));
                    ++rank;
                    ++it;
                } else {
                    ASSERT_FALSE(
                        bool(bits[block] & (storage_type{1} << local)));
                    ASSERT_FALSE(bv.get(i));
                }
            }
        }
    }
}
