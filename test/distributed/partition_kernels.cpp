/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/distributed/partition.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/distributed/partition_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


namespace {


using global_index_type = gko::distributed::global_index_type;
using comm_index_type = gko::distributed::comm_index_type;


template <typename LocalIndexType>
class Partition : public ::testing::Test {
protected:
    using local_index_type = LocalIndexType;
    Partition() : rand_engine(96457) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    void assert_equal(
        std::unique_ptr<gko::distributed::Partition<local_index_type>>& part,
        std::unique_ptr<gko::distributed::Partition<local_index_type>>& dpart)
    {
        ASSERT_EQ(part->get_size(), dpart->get_size());
        ASSERT_EQ(part->get_num_ranges(), dpart->get_num_ranges());
        ASSERT_EQ(part->get_num_parts(), dpart->get_num_parts());
        GKO_ASSERT_ARRAY_EQ(
            gko::make_array_view(this->ref, part->get_num_ranges() + 1,
                                 part->get_range_bounds()),
            gko::make_array_view(this->exec, dpart->get_num_ranges() + 1,
                                 dpart->get_range_bounds()));
        GKO_ASSERT_ARRAY_EQ(
            gko::make_array_view(this->ref, part->get_num_ranges(),
                                 part->get_part_ids()),
            gko::make_array_view(this->exec, dpart->get_num_ranges(),
                                 dpart->get_part_ids()));
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

    std::ranlux48 rand_engine;

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
};

TYPED_TEST_SUITE(Partition, gko::test::IndexTypes);


TYPED_TEST(Partition, BuildsFromMapping)
{
    using local_index_type = typename TestFixture::local_index_type;
    comm_index_type num_parts = 7;
    std::uniform_int_distribution<comm_index_type> part_dist{0, num_parts - 1};
    gko::Array<comm_index_type> mapping{this->ref, 10000};
    for (gko::size_type i = 0; i < mapping.get_num_elems(); i++) {
        mapping.get_data()[i] = part_dist(this->rand_engine);
    }
    gko::Array<comm_index_type> dmapping{this->exec, mapping};

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, mapping, num_parts);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->exec, dmapping, num_parts);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromMappingWithEmptyPart)
{
    using local_index_type = typename TestFixture::local_index_type;
    comm_index_type num_parts = 7;
    // skip part 0
    std::uniform_int_distribution<comm_index_type> part_dist{1, num_parts - 1};
    gko::Array<comm_index_type> mapping{this->ref, 10000};
    for (gko::size_type i = 0; i < mapping.get_num_elems(); i++) {
        mapping.get_data()[i] = part_dist(this->rand_engine);
    }
    gko::Array<comm_index_type> dmapping{this->exec, mapping};

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, mapping, num_parts);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->exec, dmapping, num_parts);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromMappingWithAlmostAllPartsEmpty)
{
    using local_index_type = typename TestFixture::local_index_type;
    comm_index_type num_parts = 7;
    // return only part 1
    std::uniform_int_distribution<comm_index_type> part_dist{1, 1};
    gko::Array<comm_index_type> mapping{this->ref, 10000};
    for (gko::size_type i = 0; i < mapping.get_num_elems(); i++) {
        mapping.get_data()[i] = part_dist(this->rand_engine);
    }
    gko::Array<comm_index_type> dmapping{this->exec, mapping};

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, mapping, num_parts);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->exec, dmapping, num_parts);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromMappingWithAllPartsEmpty)
{
    using local_index_type = typename TestFixture::local_index_type;
    comm_index_type num_parts = 7;
    gko::Array<comm_index_type> mapping{this->ref, 0};
    gko::Array<comm_index_type> dmapping{this->exec, 0};

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, mapping, num_parts);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->exec, dmapping, num_parts);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromMappingWithOnePart)
{
    using local_index_type = typename TestFixture::local_index_type;
    comm_index_type num_parts = 1;
    gko::Array<comm_index_type> mapping{this->ref, 10000};
    mapping.fill(0);
    gko::Array<comm_index_type> dmapping{this->exec, mapping};

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, mapping, num_parts);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->exec, dmapping, num_parts);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguous)
{
    using local_index_type = typename TestFixture::local_index_type;
    gko::Array<global_index_type> ranges{this->ref,
                                         {0, 1234, 3134, 4578, 16435, 60000}};
    gko::Array<global_index_type> dranges{this->exec, ranges};

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->ref, ranges);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguousWithSomeEmptyParts)
{
    using local_index_type = typename TestFixture::local_index_type;
    gko::Array<global_index_type> ranges{
        this->ref, {0, 1234, 3134, 3134, 4578, 16435, 16435, 60000}};
    gko::Array<global_index_type> dranges{this->exec, ranges};

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->ref, ranges);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguousWithSomeMostlyEmptyParts)
{
    using local_index_type = typename TestFixture::local_index_type;
    gko::Array<global_index_type> ranges{
        this->ref, {0, 0, 3134, 4578, 4578, 4578, 4578, 4578}};
    gko::Array<global_index_type> dranges{this->exec, ranges};

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->ref, ranges);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguousWithOnlyEmptyParts)
{
    using local_index_type = typename TestFixture::local_index_type;
    gko::Array<global_index_type> ranges{this->ref, {0, 0, 0, 0, 0, 0, 0}};
    gko::Array<global_index_type> dranges{this->exec, ranges};

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->ref, ranges);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromContiguousWithOnlyOneEmptyPart)
{
    using local_index_type = typename TestFixture::local_index_type;
    gko::Array<global_index_type> ranges{this->ref, {0, 0}};
    gko::Array<global_index_type> dranges{this->exec, ranges};

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->ref, ranges);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->exec, dranges);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromGlobalSize)
{
    using local_index_type = typename TestFixture::local_index_type;
    const int num_parts = 7;
    const global_index_type global_size = 708;

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_global_size(
            this->ref, num_parts, global_size);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_global_size(
            this->exec, num_parts, global_size);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromGlobalSizeEmpty)
{
    using local_index_type = typename TestFixture::local_index_type;
    const int num_parts = 7;
    const global_index_type global_size = 0;

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_global_size(
            this->ref, num_parts, global_size);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_global_size(
            this->exec, num_parts, global_size);

    this->assert_equal(part, dpart);
}


TYPED_TEST(Partition, BuildsFromGlobalSizeMorePartsThanSize)
{
    using local_index_type = typename TestFixture::local_index_type;
    const int num_parts = 77;
    const global_index_type global_size = 13;

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_global_size(
            this->ref, num_parts, global_size);
    auto dpart =
        gko::distributed::Partition<local_index_type>::build_from_global_size(
            this->exec, num_parts, global_size);

    this->assert_equal(part, dpart);
}


}  // namespace
