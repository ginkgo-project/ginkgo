// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/dd_matrix_kernels.hpp"

#include <algorithm>
#include <random>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "test/utils/common_fixture.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class DdMatrix : public CommonTestFixture {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;

    DdMatrix() : engine(42) {}

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
        for (comm_index_type part = 0; part < row_partition->get_num_parts();
             ++part) {
            gko::array<global_index_type> non_local_row_idxs{ref};
            gko::array<global_index_type> non_local_col_idxs{ref};
            gko::array<global_index_type> d_non_local_row_idxs{exec};
            gko::array<global_index_type> d_non_local_col_idxs{exec};

            gko::kernels::reference::distributed_dd_matrix::
                filter_non_owning_idxs(ref, input, row_partition.get(),
                                       col_partition.get(), part,
                                       non_local_row_idxs, non_local_col_idxs);
            gko::kernels::GKO_DEVICE_NAMESPACE::distributed_dd_matrix::
                filter_non_owning_idxs(
                    exec, d_input, d_row_partition.get(), d_col_partition.get(),
                    part, d_non_local_row_idxs, d_non_local_col_idxs);

            GKO_ASSERT_ARRAY_EQ(d_non_local_row_idxs, non_local_row_idxs);
            GKO_ASSERT_ARRAY_EQ(d_non_local_col_idxs, non_local_col_idxs);
        }

        std::default_random_engine engine;
    }

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(DdMatrix, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(DdMatrix, BuildsEmptyIsSameAsRef)
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


TYPED_TEST(DdMatrix, BuildsLocalSmallIsEquivalentToRef)
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


TYPED_TEST(DdMatrix, BuildsLocalIsEquivalentToRef)
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


TYPED_TEST(DdMatrix, BuildsLocalSmallWithColPartitionIsEquivalentToRef)
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


TYPED_TEST(DdMatrix, BuildsLocalWithColPartitionIsEquivalentToRef)
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
