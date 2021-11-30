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

#include "core/distributed/vector_kernels.hpp"


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


#include "core/test/utils.hpp"


namespace {

using comm_index_type = gko::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class Vector : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using local_entry = gko::matrix_data_entry<value_type, local_index_type>;
    using global_entry = gko::matrix_data_entry<value_type, global_index_type>;

    Vector()
        : ref(gko::ReferenceExecutor::create()),
          mapping{ref},
          input{ref},
          output{ref}
    {}

    void validate(const gko::distributed::Partition<
                      local_index_type, global_index_type>* partition,
                  std::initializer_list<global_entry> input_entries,
                  std::initializer_list<std::initializer_list<local_entry>>
                      output_entries)
    {
        std::vector<gko::Array<local_entry>> ref_outputs;

        input = gko::Array<global_entry>{ref, input_entries};
        for (auto entry : output_entries) {
            ref_outputs.push_back(gko::Array<local_entry>{ref, entry});
        }

        for (comm_index_type part = 0; part < partition->get_num_parts();
             ++part) {
            gko::kernels::reference::distributed_vector::build_local(
                ref, input, partition, part, output, value_type{});

            GKO_ASSERT_ARRAY_EQ(output, ref_outputs[part]);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::Array<comm_index_type> mapping;
    gko::Array<global_entry> input;
    gko::Array<local_entry> output;
};

TYPED_TEST_SUITE(Vector, gko::test::ValueLocalGlobalIndexTypes);


TYPED_TEST(Vector, BuildsLocalEmpty)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {1, 0, 2, 2, 0, 1, 1, 2}};
    comm_index_type num_parts = 3;
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);

    this->validate(partition.get(), {}, {{}, {}, {}});
}


TYPED_TEST(Vector, BuildsLocalSmall)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {1, 0}};
    comm_index_type num_parts = 2;
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);

    this->validate(partition.get(),
                   {{0, 0, 1}, {0, 1, 2}, {1, 0, 3}, {1, 1, 4}},
                   {{{0, 0, 3}, {0, 1, 4}}, {{0, 0, 1}, {0, 1, 2}}});
}


TYPED_TEST(Vector, BuildsLocal)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);

    this->validate(partition.get(),
                   {{0, 0, 1},
                    {0, 1, 2},
                    {1, 2, 3},
                    {1, 3, 4},
                    {2, 4, 5},
                    {3, 5, 6},
                    {4, 6, 7},
                    {5, 7, 8}},
                   {{{0, 4, 5}, {1, 5, 6}},
                    {{0, 0, 1}, {0, 1, 2}, {1, 7, 8}},
                    {{0, 2, 3}, {0, 3, 4}, {1, 6, 7}}});
}


}  // namespace
