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

#include "core/distributed/partition_kernels.hpp"


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/distributed/partition.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename LocalGlobalIndexType>
class Partition : public CommonTestFixture {
protected:
    using local_index_type =
        typename std::tuple_element<0, decltype(LocalGlobalIndexType())>::type;
    using global_index_type =
        typename std::tuple_element<1, decltype(LocalGlobalIndexType())>::type;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;

    Partition() : rand_engine(96457) {}

    void assert_equal(std::unique_ptr<part_type>& part,
                      std::unique_ptr<part_type>& dpart)
    {
        ASSERT_EQ(part->get_size(), dpart->get_size());
        ASSERT_EQ(part->get_num_ranges(), dpart->get_num_ranges());
        ASSERT_EQ(part->get_num_parts(), dpart->get_num_parts());
        ASSERT_EQ(part->get_num_empty_parts(), dpart->get_num_empty_parts());
        GKO_ASSERT_ARRAY_EQ(
            gko::make_array_view(
                this->ref, part->get_num_ranges() + 1,
                const_cast<global_index_type*>(part->get_range_bounds())),
            gko::make_array_view(
                this->exec, dpart->get_num_ranges() + 1,
                const_cast<global_index_type*>(dpart->get_range_bounds())));
        GKO_ASSERT_ARRAY_EQ(
            gko::make_array_view(
                this->ref, part->get_num_ranges(),
                const_cast<comm_index_type*>(part->get_part_ids())),
            gko::make_array_view(
                this->exec, dpart->get_num_ranges(),
                const_cast<comm_index_type*>(dpart->get_part_ids())));
        GKO_ASSERT_ARRAY_EQ(
            gko::make_array_view(this->ref, part->get_num_ranges(),
                                 const_cast<local_index_type*>(
                                     part->get_range_starting_indices())),
            gko::make_array_view(this->exec, dpart->get_num_ranges(),
                                 const_cast<local_index_type*>(
                                     dpart->get_range_starting_indices())));
        GKO_ASSERT_ARRAY_EQ(
            gko::make_array_view(
                this->ref, part->get_num_parts(),
                const_cast<local_index_type*>(part->get_part_sizes())),
            gko::make_array_view(
                this->exec, dpart->get_num_parts(),
                const_cast<local_index_type*>(dpart->get_part_sizes())));
    }

    std::default_random_engine rand_engine;
};

TYPED_TEST_SUITE(Partition, gko::test::LocalGlobalIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(Partition, BuildsFromMapping)
{
    using part_type = typename TestFixture::part_type;
    comm_index_type num_parts = 7;
    std::uniform_int_distribution<comm_index_type> part_dist{0, num_parts - 1};
    auto mapping = gko::test::generate_random_array<comm_index_type>(
        10000, part_dist, this->rand_engine, this->ref);
    gko::array<comm_index_type> dmapping{this->exec, mapping};

    auto part = part_type::build_from_mapping(this->ref, mapping, num_parts);
    auto dpart = part_type::build_from_mapping(this->exec, dmapping, num_parts);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromMappingWithEmptyPart)
{
    using part_type = typename TestFixture::part_type;
    comm_index_type num_parts = 7;
    // skip part 0
    std::uniform_int_distribution<comm_index_type> part_dist{1, num_parts - 1};
    auto mapping = gko::test::generate_random_array<comm_index_type>(
        10000, part_dist, this->rand_engine, this->ref);
    gko::array<comm_index_type> dmapping{this->exec, mapping};

    auto part = part_type::build_from_mapping(this->ref, mapping, num_parts);
    auto dpart = part_type::build_from_mapping(this->exec, dmapping, num_parts);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromMappingWithAlmostAllPartsEmpty)
{
    using part_type = typename TestFixture::part_type;
    comm_index_type num_parts = 7;
    // return only part 1
    std::uniform_int_distribution<comm_index_type> part_dist{1, 1};
    auto mapping = gko::test::generate_random_array<comm_index_type>(
        10000, part_dist, this->rand_engine, this->ref);
    gko::array<comm_index_type> dmapping{this->exec, mapping};

    auto part = part_type::build_from_mapping(this->ref, mapping, num_parts);
    auto dpart = part_type::build_from_mapping(this->exec, dmapping, num_parts);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromMappingWithAllPartsEmpty)
{
    using part_type = typename TestFixture::part_type;
    comm_index_type num_parts = 7;
    gko::array<comm_index_type> mapping{this->ref, 0};
    gko::array<comm_index_type> dmapping{this->exec, 0};

    auto part = part_type::build_from_mapping(this->ref, mapping, num_parts);
    auto dpart = part_type::build_from_mapping(this->exec, dmapping, num_parts);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromMappingWithOnePart)
{
    using part_type = typename TestFixture::part_type;
    comm_index_type num_parts = 1;
    gko::array<comm_index_type> mapping{this->ref, 10000};
    mapping.fill(0);
    gko::array<comm_index_type> dmapping{this->exec, mapping};

    auto part = part_type::build_from_mapping(this->ref, mapping, num_parts);
    auto dpart = part_type::build_from_mapping(this->exec, dmapping, num_parts);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguous)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    gko::array<global_index_type> ranges{this->ref,
                                         {0, 1234, 3134, 4578, 16435, 60000}};
    gko::array<global_index_type> dranges{this->exec, ranges};

    auto part = part_type::build_from_contiguous(this->ref, ranges);
    auto dpart = part_type::build_from_contiguous(this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguousWithSomeEmptyParts)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    gko::array<global_index_type> ranges{
        this->ref, {0, 1234, 3134, 3134, 4578, 16435, 16435, 60000}};
    gko::array<global_index_type> dranges{this->exec, ranges};

    auto part = part_type::build_from_contiguous(this->ref, ranges);
    auto dpart = part_type::build_from_contiguous(this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguousWithMostlyEmptyParts)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    gko::array<global_index_type> ranges{
        this->ref, {0, 0, 3134, 4578, 4578, 4578, 4578, 4578}};
    gko::array<global_index_type> dranges{this->exec, ranges};

    auto part = part_type::build_from_contiguous(this->ref, ranges);
    auto dpart = part_type::build_from_contiguous(this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguousWithOnlyEmptyParts)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    gko::array<global_index_type> ranges{this->ref, {0, 0, 0, 0, 0, 0, 0}};
    gko::array<global_index_type> dranges{this->exec, ranges};

    auto part = part_type::build_from_contiguous(this->ref, ranges);
    auto dpart = part_type::build_from_contiguous(this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguousWithOnlyOneEmptyPart)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    gko::array<global_index_type> ranges{this->ref, {0, 0}};
    gko::array<global_index_type> dranges{this->exec, ranges};

    auto part = part_type::build_from_contiguous(this->ref, ranges);
    auto dpart = part_type::build_from_contiguous(this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguousWithSingleEntry)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    gko::array<global_index_type> ranges{this->ref, {0}};
    gko::array<global_index_type> dranges{this->exec, ranges};

    auto part = part_type::build_from_contiguous(this->ref, ranges);
    auto dpart = part_type::build_from_contiguous(this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguousWithPartId)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    gko::array<global_index_type> ranges{this->ref,
                                         {0, 1234, 3134, 4578, 16435, 60000}};
    gko::array<comm_index_type> part_id{this->ref, {0, 4, 3, 1, 2}};
    gko::array<global_index_type> dranges{this->exec, ranges};

    auto part = part_type::build_from_contiguous(this->ref, ranges, part_id);
    auto dpart = part_type::build_from_contiguous(this->exec, dranges, part_id);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromGlobalSize)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    const int num_parts = 7;
    const global_index_type global_size = 708;

    auto part = part_type::build_from_global_size_uniform(this->ref, num_parts,
                                                          global_size);
    auto dpart = part_type::build_from_global_size_uniform(
        this->exec, num_parts, global_size);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromGlobalSizeEmpty)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    const int num_parts = 7;
    const global_index_type global_size = 0;

    auto part = part_type::build_from_global_size_uniform(this->ref, num_parts,
                                                          global_size);
    auto dpart = part_type::build_from_global_size_uniform(
        this->exec, num_parts, global_size);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromGlobalSizeMorePartsThanSize)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    const int num_parts = 77;
    const global_index_type global_size = 13;

    auto part = part_type::build_from_global_size_uniform(this->ref, num_parts,
                                                          global_size);
    auto dpart = part_type::build_from_global_size_uniform(
        this->exec, num_parts, global_size);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, IsOrderedTrue)
{
    using part_type = typename TestFixture::part_type;
    comm_index_type num_parts = 7;
    gko::size_type size_per_part = 1000;
    gko::size_type global_size = num_parts * size_per_part;
    gko::array<comm_index_type> mapping{this->ref, global_size};
    for (comm_index_type i = 0; i < num_parts; ++i) {
        std::fill(mapping.get_data() + i * size_per_part,
                  mapping.get_data() + (i + 1) * size_per_part, i);
    }
    auto dpart = part_type::build_from_mapping(this->exec, mapping, num_parts);

    ASSERT_TRUE(dpart->has_ordered_parts());
}


TYPED_TEST(Partition, IsOrderedFail)
{
    using part_type = typename TestFixture::part_type;
    comm_index_type num_parts = 7;
    gko::size_type size_per_part = 1000;
    gko::size_type global_size = num_parts * size_per_part;
    gko::array<comm_index_type> mapping{this->ref, global_size};
    for (comm_index_type i = 0; i < num_parts; ++i) {
        std::fill(mapping.get_data() + i * size_per_part,
                  mapping.get_data() + (i + 1) * size_per_part,
                  num_parts - 1 - i);
    }
    auto dpart = part_type::build_from_mapping(this->exec, mapping, num_parts);

    ASSERT_FALSE(dpart->has_ordered_parts());
}


TYPED_TEST(Partition, IsOrderedRandom)
{
    using part_type = typename TestFixture::part_type;
    comm_index_type num_parts = 7;
    std::uniform_int_distribution<comm_index_type> part_dist{0, num_parts - 1};
    auto mapping = gko::test::generate_random_array<comm_index_type>(
        10000, part_dist, this->rand_engine, this->ref);
    auto part = part_type::build_from_mapping(this->ref, mapping, num_parts);
    auto dpart = part_type::build_from_mapping(this->exec, mapping, num_parts);

    ASSERT_EQ(part->has_ordered_parts(), dpart->has_ordered_parts());
}
