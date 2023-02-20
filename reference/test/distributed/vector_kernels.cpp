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

#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


#include "core/distributed/vector_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class Vector : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using local_index_type =
        typename std::tuple_element<1, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using global_index_type =
        typename std::tuple_element<2, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using mtx = gko::matrix::Dense<value_type>;

    Vector() : ref(gko::ReferenceExecutor::create()) {}

    void validate(
        const gko::dim<2> size,
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            partition,
        I<global_index_type> input_rows, I<global_index_type> input_cols,
        I<value_type> input_vals, I<I<I<value_type>>> output_entries)
    {
        std::vector<I<I<value_type>>> ref_outputs;
        auto input = gko::device_matrix_data<value_type, global_index_type>{
            ref, size, input_rows, input_cols, input_vals};
        for (auto entry : output_entries) {
            ref_outputs.emplace_back(entry);
        }
        for (comm_index_type part = 0; part < partition->get_num_parts();
             ++part) {
            auto num_rows =
                static_cast<gko::size_type>(partition->get_part_size(part));
            auto output = mtx::create(ref, gko::dim<2>{num_rows, size[1]});
            output->fill(gko::zero<value_type>());

            gko::kernels::reference::distributed_vector::build_local(
                ref, input, partition.get(), part, output.get());

            GKO_ASSERT_MTX_NEAR(output, ref_outputs[part], 0);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(Vector, gko::test::ValueLocalGlobalIndexTypes);


TYPED_TEST(Vector, BuildsLocalEmpty)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<comm_index_type> mapping{this->ref, {1, 0, 2, 2, 0, 1, 1, 2}};
    comm_index_type num_parts = 3;
    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);

    this->validate(gko::dim<2>{0, 0}, partition, {}, {}, {},
                   {{{}, {}}, {{}, {}, {}}, {{}, {}, {}}});
}


TYPED_TEST(Vector, BuildsLocalSmall)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<comm_index_type> mapping{this->ref, {1, 0}};
    comm_index_type num_parts = 2;
    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);

    this->validate(gko::dim<2>{2, 2}, partition, {0, 0, 1, 1}, {0, 1, 0, 1},
                   {1, 2, 3, 4}, {{{3, 4}}, {{1, 2}}});
}


TYPED_TEST(Vector, BuildsLocal)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<comm_index_type> mapping{this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);

    this->validate(gko::dim<2>{6, 8}, partition, {0, 0, 1, 1, 2, 3, 4, 5},
                   {0, 1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 7, 8},
                   {{{0, 0, 0, 0, 5, 0, 0, 0}, {0, 0, 0, 0, 0, 6, 0, 0}},
                    {{1, 2, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 8}},
                    {{0, 0, 3, 4, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 7, 0}}});
}


}  // namespace
