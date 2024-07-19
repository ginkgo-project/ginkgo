// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/matrix_kernels.hpp"

#include <algorithm>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class Matrix : public CommonTestFixture {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;

    Matrix() : engine(42) {}

    void validate(
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            row_partition,
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            col_partition,
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            d_row_partition,
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            d_col_partition,
        gko::device_matrix_data<value_type, global_index_type> input)
    {
        gko::device_matrix_data<value_type, global_index_type> d_input{exec,
                                                                       input};
        gko::size_type num_parts = row_partition->get_num_parts();
        gko::size_type num_entries = input.get_num_stored_elements();
        for (comm_index_type part = 0; part < num_parts; ++part) {
            gko::array<local_index_type> local_row_idxs{ref};
            gko::array<local_index_type> local_col_idxs{ref};
            gko::array<value_type> local_values{ref};
            gko::array<local_index_type> d_local_row_idxs{exec};
            gko::array<local_index_type> d_local_col_idxs{exec};
            gko::array<value_type> d_local_values{exec};
            gko::array<local_index_type> non_local_row_idxs{ref};
            gko::array<global_index_type> non_local_col_idxs{ref};
            gko::array<value_type> non_local_values{ref};
            gko::array<local_index_type> d_non_local_row_idxs{exec};
            gko::array<global_index_type> d_non_local_col_idxs{exec};
            gko::array<value_type> d_non_local_values{exec};
            gko::array<comm_index_type> overlap_count{ref, num_parts};
            overlap_count.fill(0);
            gko::array<comm_index_type> d_overlap_count{exec, num_parts};
            d_overlap_count.fill(0);
            gko::array<global_index_type> overlap_positions{ref, num_entries};
            gko::array<global_index_type> d_overlap_positions{exec,
                                                              num_entries};
            gko::array<global_index_type> original_positions{ref, num_entries};
            gko::array<global_index_type> d_original_positions{exec,
                                                               num_entries};

            gko::kernels::reference::distributed_matrix::count_overlap_entries(
                ref, input, row_partition.get(), part, overlap_count,
                overlap_positions, original_positions);
            gko::kernels::GKO_DEVICE_NAMESPACE::distributed_matrix::
                count_overlap_entries(
                    exec, d_input, d_row_partition.get(), part, d_overlap_count,
                    d_overlap_positions, d_original_positions);

            gko::array<global_index_type> overlap_offsets{ref, num_parts + 1};
            std::partial_sum(overlap_count.get_data(),
                             overlap_count.get_data() + num_parts,
                             overlap_offsets.get_data() + 1);
            overlap_offsets.get_data()[0] = 0;
            gko::array<global_index_type> d_overlap_offsets{exec,
                                                            overlap_offsets};
            gko::size_type num_overlap_entries =
                overlap_offsets.get_data()[num_parts];
            gko::array<global_index_type> overlap_row_idxs{ref,
                                                           num_overlap_entries};
            gko::array<global_index_type> overlap_col_idxs{ref,
                                                           num_overlap_entries};
            gko::array<value_type> overlap_values{ref, num_overlap_entries};
            gko::array<global_index_type> d_overlap_row_idxs{
                exec, num_overlap_entries};
            gko::array<global_index_type> d_overlap_col_idxs{
                exec, num_overlap_entries};
            gko::array<value_type> d_overlap_values{exec, num_overlap_entries};

            gko::kernels::reference::distributed_matrix::
                fill_overlap_send_buffers(ref, input, row_partition.get(), part,
                                          overlap_positions, original_positions,
                                          overlap_row_idxs, overlap_col_idxs,
                                          overlap_values);
            gko::kernels::GKO_DEVICE_NAMESPACE::distributed_matrix::
                fill_overlap_send_buffers(
                    exec, d_input, d_row_partition.get(), part,
                    d_overlap_positions, d_original_positions,
                    d_overlap_row_idxs, d_overlap_col_idxs, d_overlap_values);

            gko::kernels::reference::distributed_matrix::
                separate_local_nonlocal(
                    ref, input, row_partition.get(), col_partition.get(), part,
                    local_row_idxs, local_col_idxs, local_values,
                    non_local_row_idxs, non_local_col_idxs, non_local_values);
            gko::kernels::GKO_DEVICE_NAMESPACE::distributed_matrix::
                separate_local_nonlocal(
                    exec, d_input, d_row_partition.get(), d_col_partition.get(),
                    part, d_local_row_idxs, d_local_col_idxs, d_local_values,
                    d_non_local_row_idxs, d_non_local_col_idxs,
                    d_non_local_values);

            GKO_ASSERT_ARRAY_EQ(overlap_positions, d_overlap_positions);
            GKO_ASSERT_ARRAY_EQ(original_positions, d_original_positions);
            GKO_ASSERT_ARRAY_EQ(overlap_count, d_overlap_count);
            GKO_ASSERT_ARRAY_EQ(overlap_row_idxs, d_overlap_row_idxs);
            GKO_ASSERT_ARRAY_EQ(overlap_col_idxs, d_overlap_col_idxs);
            GKO_ASSERT_ARRAY_EQ(overlap_values, d_overlap_values);
            GKO_ASSERT_ARRAY_EQ(local_row_idxs, d_local_row_idxs);
            GKO_ASSERT_ARRAY_EQ(local_col_idxs, d_local_col_idxs);
            GKO_ASSERT_ARRAY_EQ(local_values, d_local_values);
            GKO_ASSERT_ARRAY_EQ(non_local_row_idxs, d_non_local_row_idxs);
            GKO_ASSERT_ARRAY_EQ(non_local_col_idxs, d_non_local_col_idxs);
            GKO_ASSERT_ARRAY_EQ(non_local_values, d_non_local_values);
        }
    }

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(Matrix, BuildsDiagOffdiagEmptyIsSameAsRef)
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
        partition, partition, d_partition, d_partition,
        gko::device_matrix_data<value_type, global_index_type>{this->ref});
}


TYPED_TEST(Matrix, BuildsLocalSmallIsEquivalentToRef)
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
        std::uniform_real_distribution<gko::remove_complex<value_type>>(0, 1),
        this->engine, this->ref);

    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(partition, partition, d_partition, d_partition, input);
}


TYPED_TEST(Matrix, BuildsLocalIsEquivalentToRef)
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
    auto input = gko::test::generate_random_device_matrix_data<
        value_type, global_index_type>(
        num_rows, num_cols,
        std::uniform_int_distribution<int>(static_cast<int>(num_cols - 1),
                                           static_cast<int>(num_cols - 1)),
        std::uniform_real_distribution<gko::remove_complex<value_type>>(0, 1),
        this->engine, this->ref);

    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(partition, partition, d_partition, d_partition, input);
}


TYPED_TEST(Matrix, BuildsDiagOffdiagEmptyWithColPartitionIsSameAsRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<comm_index_type> row_mapping{this->ref,
                                            {1, 0, 2, 2, 0, 1, 1, 2}};
    gko::array<comm_index_type> col_mapping{this->ref,
                                            {0, 0, 2, 2, 2, 1, 1, 1}};
    comm_index_type num_parts = 3;

    auto row_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 row_mapping,
                                                                 num_parts);
    auto d_row_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 row_mapping,
                                                                 num_parts);
    auto col_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 col_mapping,
                                                                 num_parts);
    auto d_col_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 col_mapping,
                                                                 num_parts);

    this->validate(
        row_partition, col_partition, d_row_partition, d_col_partition,
        gko::device_matrix_data<value_type, global_index_type>{this->ref});
}


TYPED_TEST(Matrix, BuildsLocalSmallWithColPartitionIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::experimental::distributed::comm_index_type num_parts = 3;
    gko::size_type num_rows = 10;
    gko::size_type num_cols = 10;
    auto row_mapping = gko::test::generate_random_array<
        gko::experimental::distributed::comm_index_type>(
        num_rows,
        std::uniform_int_distribution<
            gko::experimental::distributed::comm_index_type>(0, num_parts - 1),
        this->engine, this->ref);
    auto col_mapping = gko::test::generate_random_array<
        gko::experimental::distributed::comm_index_type>(
        num_rows,
        std::uniform_int_distribution<
            gko::experimental::distributed::comm_index_type>(0, num_parts - 1),
        this->engine, this->ref);
    auto input = gko::test::generate_random_device_matrix_data<
        value_type, global_index_type>(
        num_rows, num_cols,
        std::uniform_int_distribution<int>(0, static_cast<int>(num_cols - 1)),
        std::uniform_real_distribution<gko::remove_complex<value_type>>(0, 1),
        this->engine, this->ref);

    auto row_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 row_mapping,
                                                                 num_parts);
    auto d_row_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 row_mapping,
                                                                 num_parts);
    auto col_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 col_mapping,
                                                                 num_parts);
    auto d_col_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 col_mapping,
                                                                 num_parts);

    this->validate(row_partition, col_partition, d_row_partition,
                   d_col_partition, input);
}


TYPED_TEST(Matrix, BuildsLocalWithColPartitionIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::experimental::distributed::comm_index_type num_parts = 13;
    gko::size_type num_rows = 67;
    gko::size_type num_cols = 67;
    auto row_mapping = gko::test::generate_random_array<
        gko::experimental::distributed::comm_index_type>(
        num_rows,
        std::uniform_int_distribution<
            gko::experimental::distributed::comm_index_type>(0, num_parts - 1),
        this->engine, this->ref);
    auto col_mapping = gko::test::generate_random_array<
        gko::experimental::distributed::comm_index_type>(
        num_rows,
        std::uniform_int_distribution<
            gko::experimental::distributed::comm_index_type>(0, num_parts - 1),
        this->engine, this->ref);
    auto input = gko::test::generate_random_device_matrix_data<
        value_type, global_index_type>(
        num_rows, num_cols,
        std::uniform_int_distribution<int>(static_cast<int>(num_cols),
                                           static_cast<int>(num_cols)),
        std::uniform_real_distribution<gko::remove_complex<value_type>>(0, 1),
        this->engine, this->ref);

    auto row_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 row_mapping,
                                                                 num_parts);
    auto d_row_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 row_mapping,
                                                                 num_parts);
    auto col_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 col_mapping,
                                                                 num_parts);
    auto d_col_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 col_mapping,
                                                                 num_parts);

    this->validate(row_partition, col_partition, d_row_partition,
                   d_col_partition, input);
}
