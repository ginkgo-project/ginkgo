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

#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/distributed/matrix_kernels.hpp"
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
    using Mtx = gko::matrix::Csr<value_type, local_index_type>;
    using GMtx = gko::matrix::Csr<value_type, global_index_type>;

    Matrix()
        : ref(gko::ReferenceExecutor::create()),
          mapping{ref},
          input{ref},
          diag{ref},
          offdiag{ref},
          gather_idxs{ref},
          recv_offsets{ref},
          local_to_global_row{ref},
          local_to_global_col{ref},
          mat_diag{GMtx::create(ref)},
          mat_offdiag{GMtx::create(ref)},
          mat_merged{GMtx::create(ref)}
    {
        mat_diag->read({{0, 0, 1}, {0, 1, 2}, {1, 1, 3}});
        mat_offdiag->read({{0, 0, 4}, {0, 1, 5}, {1, 0, 6}, {1, 1, 7}});
        mat_merged->read({{0, 0, 1},
                          {0, 1, 2},
                          {1, 1, 3},
                          {0, 2, 4},
                          {0, 3, 5},
                          {1, 2, 6},
                          {1, 3, 7}});
    }

    void validate(
        const gko::distributed::Partition<local_index_type>* partition,
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
                recv_offsets.get_data(), local_to_global_row,
                local_to_global_col, value_type{});

            GKO_ASSERT_ARRAY_EQ(diag, ref_diags[part]);
            GKO_ASSERT_ARRAY_EQ(offdiag, ref_offdiags[part]);
            GKO_ASSERT_ARRAY_EQ(gather_idxs, ref_gather_idxs[part]);
            GKO_ASSERT_ARRAY_EQ(recv_offsets, ref_recv_offsets[part]);
        }
    }

    gko::Array<global_entry> create_input_not_full_rank()
    {
        return gko::Array<global_entry>{
            this->ref, std::initializer_list<global_entry>{{0, 0, 1},
                                                           {0, 3, 2},
                                                           {2, 2, 5},
                                                           {3, 0, 6},
                                                           {3, 3, 7},
                                                           {4, 4, 8},
                                                           {4, 6, 9},
                                                           {5, 4, 10},
                                                           {5, 5, 11},
                                                           {6, 5, 12}}};
    }

    gko::Array<global_entry> create_input_full_rank()
    {
        return gko::Array<global_entry>{
            this->ref, std::initializer_list<global_entry>{{0, 0, 1},
                                                           {0, 3, 2},
                                                           {1, 1, 3},
                                                           {1, 2, 4},
                                                           {2, 2, 5},
                                                           {3, 0, 6},
                                                           {3, 3, 7},
                                                           {4, 4, 8},
                                                           {4, 6, 9},
                                                           {5, 4, 10},
                                                           {5, 5, 11},
                                                           {6, 5, 12}}};
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::Array<comm_index_type> mapping;
    gko::Array<global_entry> input;
    gko::Array<local_entry> diag;
    gko::Array<local_entry> offdiag;
    gko::Array<local_index_type> gather_idxs;
    gko::Array<comm_index_type> recv_offsets;
    gko::Array<global_index_type> local_to_global_row;
    gko::Array<global_index_type> local_to_global_col;
    std::unique_ptr<gko::matrix::Csr<value_type, global_index_type>> mat_diag;
    std::unique_ptr<gko::matrix::Csr<value_type, global_index_type>>
        mat_offdiag;
    std::unique_ptr<gko::matrix::Csr<value_type, global_index_type>> mat_merged;
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

TYPED_TEST(Matrix, BuildRowMapContinuous)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {0, 0, 0, 1, 1, 2, 2}};
    constexpr comm_index_type num_parts = 3;
    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    this->recv_offsets.resize_and_reset(num_parts + 1);
    gko::Array<global_index_type> result[num_parts] = {
        {this->ref, {0, 1, 2}}, {this->ref, {3, 4}}, {this->ref, {5, 6}}};

    for (int local_id = 0; local_id < num_parts; ++local_id) {
        gko::kernels::reference::distributed_matrix::build_diag_offdiag(
            this->ref, this->create_input_full_rank(), partition.get(),
            local_id, this->diag, this->offdiag, this->gather_idxs,
            this->recv_offsets.get_data(), this->local_to_global_row,
            this->local_to_global_col, value_type{});

        GKO_ASSERT_ARRAY_EQ(result[local_id], this->local_to_global_row);
    }
}


TYPED_TEST(Matrix, BuildRowMapScattered)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {0, 1, 2, 0, 1, 2, 0}};
    constexpr comm_index_type num_parts = 3;
    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    this->recv_offsets.resize_and_reset(num_parts + 1);
    gko::Array<global_index_type> result[num_parts] = {
        {this->ref, {0, 3, 6}}, {this->ref, {1, 4}}, {this->ref, {2, 5}}};

    for (int local_id = 0; local_id < num_parts; ++local_id) {
        gko::kernels::reference::distributed_matrix::build_diag_offdiag(
            this->ref, this->create_input_full_rank(), partition.get(),
            local_id, this->diag, this->offdiag, this->gather_idxs,
            this->recv_offsets.get_data(), this->local_to_global_row,
            this->local_to_global_col, value_type{});

        GKO_ASSERT_ARRAY_EQ(result[local_id], this->local_to_global_row);
    }
}

TYPED_TEST(Matrix, BuildRowMapNotFullRank)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {0, 0, 0, 1, 1, 2, 2}};
    constexpr comm_index_type num_parts = 3;
    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    this->recv_offsets.resize_and_reset(num_parts + 1);
    gko::Array<global_index_type> result[num_parts] = {
        {this->ref, {0, -1, 2}}, {this->ref, {3, 4}}, {this->ref, {5, 6}}};

    for (int local_id = 0; local_id < num_parts; ++local_id) {
        gko::kernels::reference::distributed_matrix::build_diag_offdiag(
            this->ref, this->create_input_not_full_rank(), partition.get(),
            local_id, this->diag, this->offdiag, this->gather_idxs,
            this->recv_offsets.get_data(), this->local_to_global_row,
            this->local_to_global_col, value_type{});

        GKO_ASSERT_ARRAY_EQ(result[local_id], this->local_to_global_row);
    }
}


TYPED_TEST(Matrix, BuildColMapContinuous)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {0, 0, 0, 1, 1, 2, 2}};
    constexpr comm_index_type num_parts = 3;
    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    this->recv_offsets.resize_and_reset(num_parts + 1);
    gko::Array<global_index_type> result[num_parts] = {
        {this->ref, {3}}, {this->ref, {0, 6}}, {this->ref, {4}}};

    for (int local_id = 0; local_id < num_parts; ++local_id) {
        gko::kernels::reference::distributed_matrix::build_diag_offdiag(
            this->ref, this->create_input_full_rank(), partition.get(),
            local_id, this->diag, this->offdiag, this->gather_idxs,
            this->recv_offsets.get_data(), this->local_to_global_row,
            this->local_to_global_col, value_type{});

        GKO_ASSERT_ARRAY_EQ(result[local_id], this->local_to_global_col);
    }
}

TYPED_TEST(Matrix, BuildColMapScattered)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {0, 1, 2, 0, 1, 2, 0}};
    constexpr comm_index_type num_parts = 3;
    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    this->recv_offsets.resize_and_reset(num_parts + 1);
    gko::Array<global_index_type> result[num_parts] = {
        {this->ref, {5}},
        {this->ref, {6, 2}},
        {this->ref, {4}}};  // the columns are sorted by their part_id

    for (int local_id = 0; local_id < num_parts; ++local_id) {
        gko::kernels::reference::distributed_matrix::build_diag_offdiag(
            this->ref, this->create_input_full_rank(), partition.get(),
            local_id, this->diag, this->offdiag, this->gather_idxs,
            this->recv_offsets.get_data(), this->local_to_global_row,
            this->local_to_global_col, value_type{});

        GKO_ASSERT_ARRAY_EQ(result[local_id], this->local_to_global_col);
    }
}

TYPED_TEST(Matrix, MergeLocalMatricesOffdiagRight)
{
    this->mat_diag->read({gko::dim<2>{2, 4},
                          {
                              {0, 0, 1},
                              {1, 1, 3},
                          }});
    this->mat_offdiag->read({gko::dim<2>{2, 4}, {{0, 3, 2}, {1, 2, 4}}});
    this->mat_merged =
        TestFixture::GMtx::create(this->ref, gko::dim<2>{2, 4}, 4);
    auto result = TestFixture::GMtx::create(this->ref);
    result->read(
        {gko::dim<2>{2, 4}, {{0, 0, 1}, {0, 3, 2}, {1, 1, 3}, {1, 2, 4}}});

    gko::kernels::reference::distributed_matrix::merge_diag_offdiag(
        this->ref, this->mat_diag.get(), this->mat_offdiag.get(),
        this->mat_merged.get());

    GKO_ASSERT_MTX_NEAR(result.get(), this->mat_merged.get(), 0);
}

TYPED_TEST(Matrix, MergeLocalMatricesOffdiagLeft)
{
    this->mat_diag->read({gko::dim<2>{2, 4},
                          {
                              {0, 2, 1},
                              {1, 3, 3},
                          }});
    this->mat_offdiag->read({gko::dim<2>{2, 4}, {{0, 0, 2}, {1, 1, 4}}});
    this->mat_merged =
        TestFixture::GMtx::create(this->ref, gko::dim<2>{2, 4}, 4);
    auto result = TestFixture::GMtx::create(this->ref);
    result->read(
        {gko::dim<2>{2, 4}, {{0, 0, 2}, {0, 2, 1}, {1, 1, 4}, {1, 3, 3}}});

    gko::kernels::reference::distributed_matrix::merge_diag_offdiag(
        this->ref, this->mat_diag.get(), this->mat_offdiag.get(),
        this->mat_merged.get());

    GKO_ASSERT_MTX_NEAR(result.get(), this->mat_merged.get(), 0);
}

TYPED_TEST(Matrix, MergeLocalMatricesOffdiagBoth)
{
    this->mat_diag->read({gko::dim<2>{2, 4},
                          {
                              {0, 1, 1},
                              {1, 2, 3},
                          }});
    this->mat_offdiag->read({gko::dim<2>{2, 4}, {{0, 3, 2}, {1, 0, 4}}});
    this->mat_merged =
        TestFixture::GMtx::create(this->ref, gko::dim<2>{2, 4}, 4);
    auto result = TestFixture::GMtx::create(this->ref);
    result->read(
        {gko::dim<2>{2, 4}, {{0, 1, 1}, {0, 3, 2}, {1, 0, 4}, {1, 2, 3}}});

    gko::kernels::reference::distributed_matrix::merge_diag_offdiag(
        this->ref, this->mat_diag.get(), this->mat_offdiag.get(),
        this->mat_merged.get());

    GKO_ASSERT_MTX_NEAR(result.get(), this->mat_merged.get(), 0);
}

TYPED_TEST(Matrix, MergeLocalMatricesEmptyDiag)
{
    this->mat_diag->read({gko::dim<2>{2, 4}, {}});
    this->mat_offdiag->read({gko::dim<2>{2, 4}, {{0, 3, 2}, {1, 2, 4}}});
    this->mat_merged =
        TestFixture::GMtx::create(this->ref, gko::dim<2>{2, 4}, 4);

    gko::kernels::reference::distributed_matrix::merge_diag_offdiag(
        this->ref, this->mat_diag.get(), this->mat_offdiag.get(),
        this->mat_merged.get());

    GKO_ASSERT_MTX_NEAR(this->mat_offdiag.get(), this->mat_merged.get(), 0);
}

TYPED_TEST(Matrix, MergeLocalMatricesEmptyOffdiag)
{
    this->mat_diag->read({gko::dim<2>{2, 4},
                          {
                              {0, 0, 1},
                              {1, 1, 3},
                          }});
    this->mat_offdiag->read({gko::dim<2>{2, 4}, {}});
    this->mat_merged =
        TestFixture::GMtx::create(this->ref, gko::dim<2>{2, 4}, 4);

    gko::kernels::reference::distributed_matrix::merge_diag_offdiag(
        this->ref, this->mat_diag.get(), this->mat_offdiag.get(),
        this->mat_merged.get());

    GKO_ASSERT_MTX_NEAR(this->mat_diag.get(), this->mat_merged.get(), 0);
}

}  // namespace
