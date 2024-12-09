// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/assembly_kernels.hpp"

#include <vector>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/test/utils.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class AssemblyHelpers : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, local_index_type>;

    AssemblyHelpers() : ref(gko::ReferenceExecutor::create()), mapping{ref} {}

    gko::device_matrix_data<value_type, global_index_type> create_input()
    {
        return gko::device_matrix_data<value_type, global_index_type>{
            this->ref, gko::dim<2>{7, 7},
            gko::array<global_index_type>{ref,
                                          {0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6}},
            gko::array<global_index_type>{ref,
                                          {0, 3, 1, 2, 2, 0, 3, 4, 6, 4, 5, 5}},
            gko::array<value_type>{ref,
                                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}};
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::array<comm_index_type> mapping;
};

TYPED_TEST_SUITE(AssemblyHelpers, gko::test::ValueLocalGlobalIndexTypesBase,
                 TupleTypenameNameGenerator);


TYPED_TEST(AssemblyHelpers, CountOverlapEntries)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using ca = gko::array<comm_index_type>;
    using ga = gko::array<git>;
    this->mapping = {this->ref, {1, 0, 2, 2, 0, 1, 1}};
    std::vector<ca> send_count_ref{ca{this->ref, I<comm_index_type>{0, 5, 3}},
                                   ca{this->ref, I<comm_index_type>{4, 0, 3}},
                                   ca{this->ref, I<comm_index_type>{4, 5, 0}}};
    std::vector<ga> send_pos_ref{
        ga{this->ref, I<git>{0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7}},
        ga{this->ref, I<git>{0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 6}},
        ga{this->ref, I<git>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9}}};
    std::vector<ga> original_pos_ref{
        ga{this->ref, I<git>{-1, -1, -1, -1, 0, 1, 9, 10, 11, 4, 5, 6}},
        ga{this->ref, I<git>{2, 3, 7, 8, -1, -1, -1, -1, -1, 4, 5, 6}},
        ga{this->ref, I<git>{2, 3, 7, 8, 0, 1, 9, 10, 11, -1, -1, -1}}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    auto input = this->create_input();

    ca send_count{this->ref, static_cast<gko::size_type>(num_parts)};
    ga send_positions{this->ref, input.get_num_stored_elements()};
    ga original_positions{this->ref, input.get_num_stored_elements()};
    for (gko::size_type i = 0; i < num_parts; i++) {
        send_count.fill(0);

        gko::kernels::reference::assembly::count_non_owning_entries(
            this->ref, input, partition.get(), i, send_count, send_positions,
            original_positions);

        GKO_ASSERT_ARRAY_EQ(send_count, send_count_ref[i]);
        GKO_ASSERT_ARRAY_EQ(send_positions, send_pos_ref[i]);
        GKO_ASSERT_ARRAY_EQ(original_positions, original_pos_ref[i]);
    }
}


TYPED_TEST(AssemblyHelpers, FillOverlapSendBuffers)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    using ga = gko::array<git>;
    using va = gko::array<vt>;
    this->mapping = {this->ref, {1, 0, 2, 2, 0, 1, 1}};
    std::vector<ga> send_positions{
        ga{this->ref, I<git>{0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7}},
        ga{this->ref, I<git>{0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 6}},
        ga{this->ref, I<git>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9}}};
    std::vector<ga> original_positions{
        ga{this->ref, I<git>{-1, -1, -1, -1, 0, 1, 9, 10, 11, 4, 5, 6}},
        ga{this->ref, I<git>{2, 3, 7, 8, -1, -1, -1, -1, -1, 4, 5, 6}},
        ga{this->ref, I<git>{2, 3, 7, 8, 0, 1, 9, 10, 11, -1, -1, -1}}};
    std::vector<ga> send_row_idxs_ref{
        ga{this->ref, I<git>{0, 0, 5, 5, 6, 2, 3, 3}},
        ga{this->ref, I<git>{1, 1, 4, 4, 2, 3, 3}},
        ga{this->ref, I<git>{1, 1, 4, 4, 0, 0, 5, 5, 6}}};
    std::vector<ga> send_col_idxs_ref{
        ga{this->ref, I<git>{0, 3, 4, 5, 5, 2, 0, 3}},
        ga{this->ref, I<git>{1, 2, 4, 6, 2, 0, 3}},
        ga{this->ref, I<git>{1, 2, 4, 6, 0, 3, 4, 5, 5}}};
    std::vector<va> send_values_ref{
        va{this->ref, I<vt>{1, 2, 10, 11, 12, 5, 6, 7}},
        va{this->ref, I<vt>{3, 4, 8, 9, 5, 6, 7}},
        va{this->ref, I<vt>{3, 4, 8, 9, 1, 2, 10, 11, 12}}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    auto input = this->create_input();

    gko::array<git> send_row_idxs{this->ref};
    gko::array<git> send_col_idxs{this->ref};
    gko::array<vt> send_values{this->ref};
    for (gko::size_type i = 0; i < num_parts; i++) {
        auto num_entries = send_row_idxs_ref[i].get_size();
        send_row_idxs.resize_and_reset(num_entries);
        send_col_idxs.resize_and_reset(num_entries);
        send_values.resize_and_reset(num_entries);

        gko::kernels::reference::assembly::fill_send_buffers(
            this->ref, input, partition.get(), i, send_positions[i],
            original_positions[i], send_row_idxs, send_col_idxs, send_values);

        GKO_ASSERT_ARRAY_EQ(send_row_idxs, send_row_idxs_ref[i]);
        GKO_ASSERT_ARRAY_EQ(send_col_idxs, send_col_idxs_ref[i]);
        GKO_ASSERT_ARRAY_EQ(send_values, send_values_ref[i]);
    }
}
