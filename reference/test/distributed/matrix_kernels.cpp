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

#include "core/distributed/matrix_kernels.hpp"


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


#include "core/test/utils.hpp"


namespace {


using global_index_type = gko::distributed::global_index_type;
using comm_index_type = gko::distributed::comm_index_type;


template <typename ValueLocalIndexType>
class Matrix : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueLocalIndexType())>::type;
    using local_index_type =
        typename std::tuple_element<1, decltype(ValueLocalIndexType())>::type;
    using local_entry = gko::matrix_data_entry<value_type, local_index_type>;
    using global_entry = gko::matrix_data_entry<value_type, global_index_type>;

    Matrix()
        : ref(gko::ReferenceExecutor::create()),
          mapping{ref},
          input{ref},
          diag{ref},
          offdiag{ref},
          gather_idxs{ref},
          recv_offsets{ref}
    {}

    void validate(
        const gko::distributed::Partition<local_index_type> *partition,
        std::initializer_list<global_entry> input_entries,
        std::initializer_list<std::initializer_list<local_entry>> diag_entries,
        std::initializer_list<std::initializer_list<local_entry>>
            offdiag_entries,
        std::initializer_list<std::initializer_list<local_index_type>>
            gather_idx_entries,
        std::initializer_list<std::initializer_list<comm_index_type>>
            recv_offset_entries)
    {
        std::vector<gko::Array<local_entry>> ref_diags;
        std::vector<gko::Array<local_entry>> ref_offdiags;
        std::vector<gko::Array<local_index_type>> ref_gather_idxs;
        std::vector<gko::Array<comm_index_type>> ref_recv_offsets;

        this->recv_offsets.resize_and_reset(
            static_cast<gko::size_type>(partition->get_num_parts() + 1));
        input = gko::Array<global_entry>{ref, input_entries};
        for (auto entry : diag_entries) {
            ref_diags.push_back(gko::Array<local_entry>{ref, entry});
        }
        for (auto entry : offdiag_entries) {
            ref_offdiags.push_back(gko::Array<local_entry>{ref, entry});
        }
        for (auto entry : gather_idx_entries) {
            ref_gather_idxs.push_back(gko::Array<local_index_type>{ref, entry});
        }
        for (auto entry : recv_offset_entries) {
            ref_recv_offsets.push_back(gko::Array<comm_index_type>{ref, entry});
        }

        for (comm_index_type part = 0; part < partition->get_num_parts();
             ++part) {
            gko::kernels::reference::distributed_matrix::build_diag_offdiag(
                ref, input, partition, part, diag, offdiag, gather_idxs,
                recv_offsets.get_data(), value_type{});

            GKO_ASSERT_ARRAY_EQ(diag, ref_diags[part]);
            GKO_ASSERT_ARRAY_EQ(offdiag, ref_offdiags[part]);
            GKO_ASSERT_ARRAY_EQ(gather_idxs, ref_gather_idxs[part]);
            GKO_ASSERT_ARRAY_EQ(recv_offsets, ref_recv_offsets[part]);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::Array<comm_index_type> mapping;
    gko::Array<global_entry> input;
    gko::Array<local_entry> diag;
    gko::Array<local_entry> offdiag;
    gko::Array<local_index_type> gather_idxs;
    gko::Array<comm_index_type> recv_offsets;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueIndexTypes);


TYPED_TEST(Matrix, BuildsDiagOffdiagEmpty)
{
    using local_index_type = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {1, 0, 2, 2, 0, 1, 1, 2}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->validate(partition.get(), {}, {{}, {}, {}}, {{}, {}, {}},
                   {{}, {}, {}}, {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
}


TYPED_TEST(Matrix, BuildsDiagOffdiagSmall)
{
    using local_index_type = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {1, 0}};
    comm_index_type num_parts = 2;
    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->validate(partition.get(),
                   {{0, 0, 1}, {0, 1, 2}, {1, 0, 3}, {1, 1, 4}},
                   {{{0, 0, 4}}, {{0, 0, 1}}}, {{{0, 0, 3}}, {{0, 0, 2}}},
                   {{0}, {0}}, {{0, 0, 1}, {0, 1, 1}});
}


TYPED_TEST(Matrix, BuildsDiagOffdiagNoOffdiag)
{
    using local_index_type = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->validate(partition.get(),
                   {{0, 0, 1},
                    {0, 5, 2},
                    {1, 1, 3},
                    {1, 4, 4},
                    {2, 3, 5},
                    {3, 2, 6},
                    {4, 4, 7},
                    {5, 0, 8}},
                   {{{0, 1, 5}, {1, 0, 6}},
                    {{0, 0, 1}, {0, 1, 2}, {1, 0, 8}},
                    {{0, 0, 3}, {0, 1, 4}, {1, 1, 7}}},
                   {{}, {}, {}}, {{}, {}, {}},
                   {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
}


TYPED_TEST(Matrix, BuildsDiagOffdiagNoDiag)
{
    using local_index_type = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->validate(
        partition.get(),
        {{0, 1, 1}, {0, 3, 2}, {1, 5, 5}, {3, 1, 6}, {4, 3, 7}, {5, 2, 8}},
        {{}, {}, {}},
        {{{1, 0, 6}},
         {{0, 2, 1}, {0, 1, 2}, {1, 0, 8}},
         {{0, 1, 5}, {1, 0, 7}}},
        {{0}, {0, 1, 0}, {1, 1}}, {{0, 0, 0, 1}, {0, 2, 2, 3}, {0, 1, 2, 2}});
}


TYPED_TEST(Matrix, BuildsDiagOffdiagMixed)
{
    using local_index_type = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->validate(partition.get(),
                   {{0, 0, 11},
                    {0, 1, 1},
                    {0, 3, 2},
                    {0, 5, 12},
                    {1, 1, 13},
                    {1, 4, 14},
                    {1, 5, 5},
                    {2, 3, 15},
                    {3, 1, 6},
                    {3, 2, 16},
                    {4, 3, 7},
                    {4, 4, 17},
                    {5, 0, 18},
                    {5, 2, 8}},
                   {{{0, 1, 15}, {1, 0, 16}},
                    {{0, 0, 11}, {0, 1, 12}, {1, 0, 18}},
                    {{0, 0, 13}, {0, 1, 14}, {1, 1, 17}}},
                   {{{1, 0, 6}},
                    {{0, 2, 1}, {0, 1, 2}, {1, 0, 8}},
                    {{0, 1, 5}, {1, 0, 7}}},
                   {{0}, {0, 1, 0}, {1, 1}},
                   {{0, 0, 0, 1}, {0, 2, 2, 3}, {0, 1, 2, 2}});
}


}  // namespace
