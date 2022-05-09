/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/components/sparse_bitset.hpp"


#include <memory>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename LocalGlobalIndexType>
class SparseBitset : public ::testing::Test {
protected:
    using local_index_type =
        typename std::tuple_element<0, decltype(LocalGlobalIndexType())>::type;
    using global_index_type =
        typename std::tuple_element<1, decltype(LocalGlobalIndexType())>::type;
    using array_type = gko::array<global_index_type>;
    using flat_type =
        gko::sparse_bitset<0, local_index_type, global_index_type>;
    using hierarchical_type1 =
        gko::sparse_bitset<1, local_index_type, global_index_type>;
    using hierarchical_type2 =
        gko::sparse_bitset<2, local_index_type, global_index_type>;

    SparseBitset()
        : ref{gko::ReferenceExecutor::create()},
          indices{ref, {2, 3, 42, 43, 94}}
    {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::array<global_index_type> indices;
};

TYPED_TEST_SUITE(SparseBitset, gko::test::LocalGlobalIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(SparseBitset, GenerateFlatSortedWorks)
{
    using flat_type = typename TestFixture::flat_type;

    const auto set = flat_type::from_indices_sorted(this->indices, 100);

    const auto device_bitset = set.to_device();
    ASSERT_EQ(device_bitset.bitmaps[0], 0b1100u);
    ASSERT_EQ(device_bitset.bitmaps[1], 0b11u << 10);
    ASSERT_EQ(device_bitset.bitmaps[2], 1u << 30);
    ASSERT_EQ(device_bitset.bitmaps[3], 0);
    ASSERT_EQ(device_bitset.ranks[0], 0);
    ASSERT_EQ(device_bitset.ranks[1], 2);
    ASSERT_EQ(device_bitset.ranks[2], 4);
    ASSERT_EQ(device_bitset.ranks[3], 5);
}


TYPED_TEST(SparseBitset, GenerateFlatUnsortedWorks)
{
    using flat_type = typename TestFixture::flat_type;

    const auto set = flat_type::from_indices_unsorted(this->indices, 100);

    const auto device_bitset = set.to_device();
    ASSERT_EQ(device_bitset.bitmaps[0], 0b1100u);
    ASSERT_EQ(device_bitset.bitmaps[1], 0b11u << 10);
    ASSERT_EQ(device_bitset.bitmaps[2], 1u << 30);
    ASSERT_EQ(device_bitset.bitmaps[3], 0);
    ASSERT_EQ(device_bitset.ranks[0], 0);
    ASSERT_EQ(device_bitset.ranks[1], 2);
    ASSERT_EQ(device_bitset.ranks[2], 4);
    ASSERT_EQ(device_bitset.ranks[3], 5);
}


TYPED_TEST(SparseBitset, DeviceSparseBitset)
{
    using flat_type = typename TestFixture::flat_type;
    const auto set = flat_type::from_indices_sorted(this->indices, 100);
    const auto device_bitset = set.to_device();

    ASSERT_EQ(device_bitset.rank(0), 0);
    ASSERT_EQ(device_bitset.rank(2), 0);
    ASSERT_EQ(device_bitset.rank(3), 1);
    ASSERT_EQ(device_bitset.rank(32), 2);
    ASSERT_EQ(device_bitset.rank(42), 2);
    ASSERT_EQ(device_bitset.rank(43), 3);
    ASSERT_EQ(device_bitset.rank(44), 4);
    ASSERT_EQ(device_bitset.rank(64), 4);
    ASSERT_EQ(device_bitset.rank(94), 4);
    ASSERT_EQ(device_bitset.rank(99), 5);
    ASSERT_FALSE(device_bitset.contains(0));
    ASSERT_TRUE(device_bitset.contains(2));
    ASSERT_TRUE(device_bitset.contains(3));
    ASSERT_FALSE(device_bitset.contains(32));
    ASSERT_TRUE(device_bitset.contains(42));
    ASSERT_TRUE(device_bitset.contains(43));
    ASSERT_FALSE(device_bitset.contains(44));
    ASSERT_FALSE(device_bitset.contains(64));
    ASSERT_TRUE(device_bitset.contains(94));
    ASSERT_FALSE(device_bitset.contains(99));
}


}  // namespace
