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

#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/distributed/matrix_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


using comm_index_type = gko::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class Matrix : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, local_index_type>;

    Matrix()
        : ref(gko::ReferenceExecutor::create()),
          mapping{ref},
          diag{ref},
          offdiag{ref},
          gather_idxs{ref},
          recv_offsets{ref},
          local_to_global_ghost{ref}
    {}

    void validate(
        gko::dim<2> size,
        const gko::distributed::Partition<local_index_type, global_index_type>*
            partition,
        std::initializer_list<global_index_type> input_rows,
        std::initializer_list<global_index_type> input_cols,
        std::initializer_list<value_type> input_vals,
        std::initializer_list<
            std::initializer_list<std::initializer_list<value_type>>>
            diag_entries,
        std::initializer_list<
            std::initializer_list<std::initializer_list<value_type>>>
            offdiag_entries,
        std::initializer_list<std::initializer_list<local_index_type>>
            gather_idx_entries,
        std::initializer_list<std::initializer_list<comm_index_type>>
            recv_offset_entries)
    {
        using local_d_md_type =
            gko::device_matrix_data<value_type, local_index_type>;
        using md_type = typename local_d_md_type::host_type;
        std::vector<gko::device_matrix_data<value_type, local_index_type>>
            ref_diags;
        std::vector<gko::device_matrix_data<value_type, local_index_type>>
            ref_offdiags;
        std::vector<gko::Array<local_index_type>> ref_gather_idxs;
        std::vector<gko::Array<comm_index_type>> ref_recv_offsets;

        auto input = gko::device_matrix_data<value_type, global_index_type>{
            ref, size, input_rows, input_cols, input_vals};
        this->recv_offsets.resize_and_reset(
            static_cast<gko::size_type>(partition->get_num_parts() + 1));
        for (auto entry : diag_entries) {
            ref_diags.push_back(
                local_d_md_type::create_from_host(ref, md_type{entry}));
        }
        for (auto entry : offdiag_entries) {
            ref_offdiags.push_back(
                local_d_md_type::create_from_host(ref, md_type{entry}));
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
                recv_offsets.get_data(), local_to_global_ghost);

            assert_device_matrix_data_equal(diag, ref_diags[part]);
            assert_device_matrix_data_equal(offdiag, ref_offdiags[part]);
            GKO_ASSERT_ARRAY_EQ(gather_idxs, ref_gather_idxs[part]);
            GKO_ASSERT_ARRAY_EQ(recv_offsets, ref_recv_offsets[part]);
        }
    }

    template <typename Data1, typename Data2>
    void assert_device_matrix_data_equal(const Data1& first,
                                         const Data2& second)
    {
        auto dense_first =
            gko::matrix::Dense<value_type>::create(first.get_executor());
        dense_first->read(first);
        auto dense_second =
            gko::matrix::Dense<value_type>::create(second.get_executor());
        dense_second->read(second);
    }

    gko::device_matrix_data<value_type, global_index_type>
    create_input_not_full_rank()
    {
        return gko::device_matrix_data<value_type, global_index_type>{
            this->ref, gko::dim<2>{7, 7},
            I<global_index_type>{0, 0, 2, 3, 3, 4, 4, 5, 5, 6},
            I<global_index_type>{0, 3, 2, 0, 3, 4, 6, 4, 5, 5},
            I<value_type>{1, 2, 5, 6, 7, 8, 9, 10, 11, 12}};
    }

    gko::device_matrix_data<value_type, global_index_type>
    create_input_full_rank()
    {
        return gko::device_matrix_data<value_type, global_index_type>{
            this->ref, gko::dim<2>{7, 7},
            I<global_index_type>{0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6},
            I<global_index_type>{0, 3, 1, 2, 2, 0, 3, 4, 6, 4, 5, 5},
            I<value_type>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::Array<comm_index_type> mapping;
    gko::device_matrix_data<value_type, local_index_type> diag;
    gko::device_matrix_data<value_type, local_index_type> offdiag;
    gko::Array<local_index_type> gather_idxs;
    gko::Array<comm_index_type> recv_offsets;
    gko::Array<global_index_type> local_to_global_ghost;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueLocalGlobalIndexTypes);


TYPED_TEST(Matrix, BuildsDiagOffdiagEmpty)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {1, 0, 2, 2, 0, 1, 1, 2}};
    comm_index_type num_parts = 3;
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);

    this->validate(gko::dim<2>{0, 0}, partition.get(), {}, {}, {}, {{}, {}, {}},
                   {{}, {}, {}}, {{}, {}, {}},
                   {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
}


TYPED_TEST(Matrix, BuildsDiagOffdiagSmall)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {1, 0}};
    comm_index_type num_parts = 2;
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);

    this->validate(gko::dim<2>{2, 2}, partition.get(), {0, 0, 1, 1},
                   {0, 1, 0, 1}, {1, 2, 3, 4}, {{{4}}, {{1}}}, {{{3}}, {{2}}},
                   {{0}, {0}}, {{0, 0, 1}, {0, 1, 1}});
}


TYPED_TEST(Matrix, BuildsDiagOffdiagNoOffdiag)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);

    this->validate(gko::dim<2>{6, 6}, partition.get(), {0, 0, 1, 1, 2, 3, 4, 5},
                   {0, 5, 1, 4, 3, 2, 4, 0}, {1, 2, 3, 4, 5, 6, 7, 8},
                   {{{0, 5}, {6, 0}}, {{1, 2}, {8, 0}}, {{3, 4}, {0, 7}}},
                   {{}, {}, {}}, {{}, {}, {}},
                   {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
}


TYPED_TEST(Matrix, BuildsDiagOffdiagNoDiag)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);

    this->validate(gko::dim<2>{6, 6}, partition.get(), {0, 0, 1, 3, 4, 5},
                   {1, 3, 5, 1, 3, 2}, {1, 2, 5, 6, 7, 8}, {{{}}, {{}}, {{}}},
                   {{{0}, {6}}, {{1, 0, 2}, {0, 8, 0}}, {{0, 5}, {7, 0}}},
                   {{0}, {0, 1, 0}, {1, 1}},
                   {{0, 0, 0, 1}, {0, 2, 2, 3}, {0, 1, 2, 2}});
}


TYPED_TEST(Matrix, BuildsDiagOffdiagMixed)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);

    this->validate(
        gko::dim<2>{6, 6}, partition.get(),
        {0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5},
        {0, 1, 3, 5, 1, 4, 5, 3, 1, 2, 3, 4, 0, 2},
        {11, 1, 2, 12, 13, 14, 5, 15, 6, 16, 7, 17, 18, 8},
        {{{0, 15}, {16, 0}}, {{11, 12}, {18, 0}}, {{13, 14}, {0, 17}}},
        {{{0}, {6}}, {{0, 1, 2}, {8, 0, 0}}, {{0, 5}, {7, 0}}},
        {{0}, {0, 1, 0}, {1, 1}}, {{0, 0, 0, 1}, {0, 2, 2, 3}, {0, 1, 2, 2}});
}


TYPED_TEST(Matrix, BuildGhostMapContinuous)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {0, 0, 0, 1, 1, 2, 2}};
    constexpr comm_index_type num_parts = 3;
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);
    this->recv_offsets.resize_and_reset(num_parts + 1);
    gko::Array<global_index_type> result[num_parts] = {
        {this->ref, {3}}, {this->ref, {0, 6}}, {this->ref, {4}}};

    for (int local_id = 0; local_id < num_parts; ++local_id) {
        gko::kernels::reference::distributed_matrix::build_diag_offdiag(
            this->ref, this->create_input_full_rank(), partition.get(),
            local_id, this->diag, this->offdiag, this->gather_idxs,
            this->recv_offsets.get_data(), this->local_to_global_ghost);

        GKO_ASSERT_ARRAY_EQ(result[local_id], this->local_to_global_ghost);
    }
}

TYPED_TEST(Matrix, BuildGhostMapScattered)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {0, 1, 2, 0, 1, 2, 0}};
    constexpr comm_index_type num_parts = 3;
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);
    this->recv_offsets.resize_and_reset(num_parts + 1);
    gko::Array<global_index_type> result[num_parts] = {
        {this->ref, {5}},
        {this->ref, {6, 2}},
        {this->ref, {4}}};  // the columns are sorted by their part_id

    for (int local_id = 0; local_id < num_parts; ++local_id) {
        gko::kernels::reference::distributed_matrix::build_diag_offdiag(
            this->ref, this->create_input_full_rank(), partition.get(),
            local_id, this->diag, this->offdiag, this->gather_idxs,
            this->recv_offsets.get_data(), this->local_to_global_ghost);

        GKO_ASSERT_ARRAY_EQ(result[local_id], this->local_to_global_ghost);
    }
}

}  // namespace
