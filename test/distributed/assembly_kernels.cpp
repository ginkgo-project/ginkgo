// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/assembly_kernels.hpp"

#include <algorithm>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class AssemblyHelpers : public CommonTestFixture {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;

    AssemblyHelpers() : engine(42) {}

    void validate(
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            row_partition,
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            d_row_partition,
        gko::device_matrix_data<value_type, global_index_type> input)
    {
        gko::device_matrix_data<value_type, global_index_type> d_input{exec,
                                                                       input};
        gko::size_type num_parts = row_partition->get_num_parts();
        gko::size_type num_entries = input.get_num_stored_elements();
        for (comm_index_type part = 0; part < num_parts; ++part) {
            gko::array<comm_index_type> send_count{ref, num_parts};
            send_count.fill(0);
            gko::array<comm_index_type> d_send_count{exec, num_parts};
            d_send_count.fill(0);
            gko::array<global_index_type> send_positions{ref, num_entries};
            gko::array<global_index_type> d_send_positions{exec, num_entries};
            gko::array<global_index_type> original_positions{ref, num_entries};
            gko::array<global_index_type> d_original_positions{exec,
                                                               num_entries};

            gko::kernels::reference::assembly::count_non_owning_entries(
                ref, input, row_partition.get(), part, send_count,
                send_positions, original_positions);
            gko::kernels::GKO_DEVICE_NAMESPACE::assembly::
                count_non_owning_entries(exec, d_input, d_row_partition.get(),
                                         part, d_send_count, d_send_positions,
                                         d_original_positions);

            gko::array<global_index_type> send_offsets{ref, num_parts + 1};
            std::partial_sum(send_count.get_data(),
                             send_count.get_data() + num_parts,
                             send_offsets.get_data() + 1);
            send_offsets.get_data()[0] = 0;
            gko::array<global_index_type> d_send_offsets{exec, send_offsets};
            gko::size_type num_send_entries =
                send_offsets.get_data()[num_parts];
            gko::array<global_index_type> send_row_idxs{ref, num_send_entries};
            gko::array<global_index_type> send_col_idxs{ref, num_send_entries};
            gko::array<value_type> send_values{ref, num_send_entries};
            gko::array<global_index_type> d_send_row_idxs{exec,
                                                          num_send_entries};
            gko::array<global_index_type> d_send_col_idxs{exec,
                                                          num_send_entries};
            gko::array<value_type> d_send_values{exec, num_send_entries};

            gko::kernels::reference::assembly::fill_send_buffers(
                ref, input, row_partition.get(), part, send_positions,
                original_positions, send_row_idxs, send_col_idxs, send_values);
            gko::kernels::GKO_DEVICE_NAMESPACE::assembly::fill_send_buffers(
                exec, d_input, d_row_partition.get(), part, d_send_positions,
                d_original_positions, d_send_row_idxs, d_send_col_idxs,
                d_send_values);

            GKO_ASSERT_ARRAY_EQ(send_positions, d_send_positions);
            GKO_ASSERT_ARRAY_EQ(original_positions, d_original_positions);
            GKO_ASSERT_ARRAY_EQ(send_count, d_send_count);
            GKO_ASSERT_ARRAY_EQ(send_row_idxs, d_send_row_idxs);
            GKO_ASSERT_ARRAY_EQ(send_col_idxs, d_send_col_idxs);
            GKO_ASSERT_ARRAY_EQ(send_values, d_send_values);
        }
    }

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(AssemblyHelpers, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(AssemblyHelpers, AddNonLocalEntriesEmptyIsSameAsRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<comm_index_type> mapping{this->ref, {1, 0, 2, 2, 0, 1, 1, 2}};
    comm_index_type num_parts = 3;

    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(
        partition, d_partition,
        gko::device_matrix_data<value_type, global_index_type>{this->ref});
}


TYPED_TEST(AssemblyHelpers, AddNonLocalEntriesLocalSmallIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::experimental::distributed::comm_index_type num_parts = 3;
    gko::size_type num_rows = 10;
    gko::size_type num_cols = 10;
    auto mapping = gko::test::generate_random_array<
        gko::experimental::distributed::comm_index_type>(
        num_rows,
        std::uniform_int_distribution<
            gko::experimental::distributed::comm_index_type>(0, num_parts - 1),
        this->engine, this->ref);
    auto input = gko::test::generate_random_device_matrix_data<
        value_type, global_index_type>(
        num_rows, num_cols,
        std::uniform_int_distribution<int>(0, static_cast<int>(num_cols - 1)),
        std::uniform_real_distribution<>(0, 1), this->engine, this->ref);

    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(partition, d_partition, input);
}


TYPED_TEST(AssemblyHelpers, AddNonLocalEntriesLocalIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::experimental::distributed::comm_index_type num_parts = 13;
    gko::size_type num_rows = 67;
    gko::size_type num_cols = 67;
    auto mapping = gko::test::generate_random_array<
        gko::experimental::distributed::comm_index_type>(
        num_rows,
        std::uniform_int_distribution<
            gko::experimental::distributed::comm_index_type>(0, num_parts - 1),
        this->engine, this->ref);
    auto input =
        gko::test::generate_random_device_matrix_data<value_type,
                                                      global_index_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<int>(static_cast<int>(1),
                                               static_cast<int>(num_cols - 1)),
            std::uniform_real_distribution<>(0, 1), this->engine, this->ref);

    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(partition, d_partition, input);
}
