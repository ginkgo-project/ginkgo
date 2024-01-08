// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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


using comm_index_type = gko::experimental::distributed::comm_index_type;


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
          local_row_idxs{ref},
          local_col_idxs{ref},
          local_values{ref},
          non_local_row_idxs{ref},
          non_local_col_idxs{ref},
          non_local_values{ref},
          gather_idxs{ref},
          recv_sizes{ref},
          non_local_to_global{ref}
    {}

    void validate(
        gko::dim<2> size,
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            row_partition,
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            col_partition,
        std::initializer_list<global_index_type> input_rows,
        std::initializer_list<global_index_type> input_cols,
        std::initializer_list<value_type> input_vals,
        std::initializer_list<
            std::tuple<gko::dim<2>, std::initializer_list<global_index_type>,
                       std::initializer_list<global_index_type>,
                       std::initializer_list<value_type>>>
            local_entries,
        std::initializer_list<
            std::tuple<gko::dim<2>, std::initializer_list<global_index_type>,
                       std::initializer_list<global_index_type>,
                       std::initializer_list<value_type>>>
            non_local_entries,
        std::initializer_list<std::initializer_list<local_index_type>>
            gather_idx_entries,
        std::initializer_list<std::initializer_list<comm_index_type>>
            recv_sizes_entries)
    {
        std::vector<gko::device_matrix_data<value_type, local_index_type>>
            ref_locals;
        std::vector<gko::device_matrix_data<value_type, local_index_type>>
            ref_non_locals;
        std::vector<gko::array<local_index_type>> ref_gather_idxs;
        std::vector<gko::array<comm_index_type>> ref_recv_sizes;

        auto input = gko::device_matrix_data<value_type, global_index_type>{
            ref, size, input_rows, input_cols, input_vals};
        this->recv_sizes.resize_and_reset(
            static_cast<gko::size_type>(row_partition->get_num_parts()));
        for (auto entry : local_entries) {
            ref_locals.emplace_back(ref, std::get<0>(entry), std::get<1>(entry),
                                    std::get<2>(entry), std::get<3>(entry));
        }
        for (auto entry : non_local_entries) {
            ref_non_locals.emplace_back(ref, std::get<0>(entry),
                                        std::get<1>(entry), std::get<2>(entry),
                                        std::get<3>(entry));
        }
        for (auto entry : gather_idx_entries) {
            ref_gather_idxs.emplace_back(ref, entry);
        }
        for (auto entry : recv_sizes_entries) {
            ref_recv_sizes.emplace_back(ref, entry);
        }

        for (comm_index_type part = 0; part < row_partition->get_num_parts();
             ++part) {
            gko::kernels::reference::distributed_matrix::build_local_nonlocal(
                ref, input, row_partition.get(), col_partition.get(), part,
                local_row_idxs, local_col_idxs, local_values,
                non_local_row_idxs, non_local_col_idxs, non_local_values,
                gather_idxs, recv_sizes, non_local_to_global);

            assert_device_matrix_data_equal(local_row_idxs, local_col_idxs,
                                            local_values, ref_locals[part]);
            assert_device_matrix_data_equal(
                non_local_row_idxs, non_local_col_idxs, non_local_values,
                ref_non_locals[part]);
            GKO_ASSERT_ARRAY_EQ(gather_idxs, ref_gather_idxs[part]);
            GKO_ASSERT_ARRAY_EQ(recv_sizes, ref_recv_sizes[part]);
        }
    }

    template <typename A1, typename A2, typename A3, typename Data2>
    void assert_device_matrix_data_equal(A1& row_idxs, A2& col_idxs, A3& values,
                                         Data2& second)
    {
        auto array_second = second.empty_out();

        GKO_ASSERT_ARRAY_EQ(row_idxs, array_second.row_idxs);
        GKO_ASSERT_ARRAY_EQ(col_idxs, array_second.col_idxs);
        GKO_ASSERT_ARRAY_EQ(values, array_second.values);
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
    gko::array<comm_index_type> mapping;
    gko::array<local_index_type> local_row_idxs;
    gko::array<local_index_type> local_col_idxs;
    gko::array<value_type> local_values;
    gko::array<local_index_type> non_local_row_idxs;
    gko::array<local_index_type> non_local_col_idxs;
    gko::array<value_type> non_local_values;
    gko::array<local_index_type> gather_idxs;
    gko::array<comm_index_type> recv_sizes;
    gko::array<global_index_type> non_local_to_global;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(Matrix, BuildsLocalNonLocalEmpty)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 0, 2, 2, 0, 1, 1, 2}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->validate(
        gko::dim<2>{8, 8}, partition, partition, {}, {}, {},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 3}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 3}, I<git>{}, I<git>{}, I<vt>{})},
        {std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 0}, I<git>{}, I<git>{}, I<vt>{})},
        {{}, {}, {}}, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
}


TYPED_TEST(Matrix, BuildsLocalNonLocalSmall)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 0}};
    comm_index_type num_parts = 2;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->validate(
        gko::dim<2>{2, 2}, partition, partition, {0, 0, 1, 1}, {0, 1, 0, 1},
        {1, 2, 3, 4},
        {std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{4}),
         std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{1})},
        {std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{3}),
         std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{2})},
        {{0}, {0}}, {{0, 1}, {1, 0}});
}


TYPED_TEST(Matrix, BuildsLocalNonLocalNoNonLocal)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->validate(
        gko::dim<2>{6, 6}, partition, partition, {0, 0, 1, 1, 2, 3, 4, 5},
        {0, 5, 1, 4, 3, 2, 4, 0}, {1, 2, 3, 4, 5, 6, 7, 8},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{1, 0},
                         I<vt>{5, 6}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0, 1}, I<git>{0, 1, 0},
                         I<vt>{1, 2, 8}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0, 1}, I<git>{0, 1, 1},
                         I<vt>{3, 4, 7})},
        {std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{})},
        {{}, {}, {}}, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
}


TYPED_TEST(Matrix, BuildsLocalNonLocalNoLocal)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->validate(
        gko::dim<2>{6, 6}, partition, partition, {0, 0, 1, 3, 4, 5},
        {1, 3, 5, 1, 3, 2}, {1, 2, 5, 6, 7, 8},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{})},
        {std::make_tuple(gko::dim<2>{2, 1}, I<git>{1}, I<git>{0}, I<vt>{6}),
         std::make_tuple(gko::dim<2>{2, 3}, I<git>{0, 0, 1}, I<git>{2, 1, 0},
                         I<vt>{1, 2, 8}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{1, 0},
                         I<vt>{5, 7})},
        {{0}, {0, 1, 0}, {1, 1}}, {{0, 0, 1}, {2, 0, 1}, {1, 1, 0}});
}


TYPED_TEST(Matrix, BuildsLocalNonLocalMixed)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->validate(
        gko::dim<2>{6, 6}, partition, partition,
        {0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5},
        {0, 1, 3, 5, 1, 4, 5, 3, 1, 2, 3, 4, 0, 2},
        {11, 1, 2, 12, 13, 14, 5, 15, 6, 16, 7, 17, 18, 8},

        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{1, 0},
                         I<vt>{15, 16}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0, 1}, I<git>{0, 1, 0},
                         I<vt>{11, 12, 18}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0, 1}, I<git>{0, 1, 1},
                         I<vt>{13, 14, 17})},
        {std::make_tuple(gko::dim<2>{2, 1}, I<git>{1}, I<git>{0}, I<vt>{6}),
         std::make_tuple(gko::dim<2>{2, 3}, I<git>{0, 0, 1}, I<git>{2, 1, 0},
                         I<vt>{1, 2, 8}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{1, 0},
                         I<vt>{5, 7})},
        {{0}, {0, 1, 0}, {1, 1}}, {{0, 0, 1}, {2, 0, 1}, {1, 1, 0}});
}


TYPED_TEST(Matrix, BuildsLocalNonLocalEmptyWithColPartition)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 0, 2, 2, 0, 1, 1, 2}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    gko::array<comm_index_type> col_mapping{this->ref,
                                            {0, 0, 2, 2, 2, 1, 1, 1}};
    auto col_partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, col_mapping, num_parts);

    this->validate(
        gko::dim<2>{8, 8}, partition, col_partition, {}, {}, {},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 3}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 3}, I<git>{}, I<git>{}, I<vt>{})},
        {std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 0}, I<git>{}, I<git>{}, I<vt>{})},
        {{}, {}, {}}, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
}


TYPED_TEST(Matrix, BuildsLocalNonLocalSmallWithColPartition)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 0}};
    comm_index_type num_parts = 2;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    gko::array<comm_index_type> col_mapping{this->ref, {0, 1}};
    auto col_partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, col_mapping, num_parts);

    this->validate(
        gko::dim<2>{2, 2}, partition, col_partition, {0, 0, 1, 1}, {0, 1, 0, 1},
        {1, 2, 3, 4},
        {std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{3}),
         std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{2})},
        {std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{4}),
         std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{1})},
        {{0}, {0}}, {{0, 1}, {1, 0}});
}

TYPED_TEST(Matrix, BuildsLocalNonLocalNoNonLocalWithColPartition)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    gko::array<comm_index_type> col_mapping{this->ref, {0, 0, 2, 2, 1, 1}};
    auto col_partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, col_mapping, num_parts);

    this->validate(
        gko::dim<2>{6, 6}, partition, col_partition, {3, 0, 5, 1, 1, 4},
        {1, 4, 5, 2, 3, 3}, {1, 2, 3, 4, 5, 6},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{1}, I<git>{1}, I<vt>{1}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{0, 1},
                         I<vt>{2, 3}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0, 1}, I<git>{0, 1, 1},
                         I<vt>{4, 5, 6})},
        {std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{})},
        {{}, {}, {}}, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
}


TYPED_TEST(Matrix, BuildsLocalNonLocalNoLocalWithColPartition)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    gko::array<comm_index_type> col_mapping{this->ref, {0, 0, 2, 2, 1, 1}};
    auto col_partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, col_mapping, num_parts);

    this->validate(
        gko::dim<2>{6, 6}, partition, col_partition, {2, 3, 2, 0, 5, 1, 1},
        {2, 3, 5, 0, 1, 1, 4}, {1, 2, 3, 4, 5, 6, 7},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{})},
        {std::make_tuple(gko::dim<2>{2, 3}, I<git>{0, 1, 0}, I<git>{1, 2, 0},
                         I<vt>{1, 2, 3}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{0, 1},
                         I<vt>{4, 5}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0}, I<git>{0, 1},
                         I<vt>{6, 7})},
        {{1, 0, 1}, {0, 1}, {1, 0}}, {{0, 1, 2}, {2, 0, 0}, {1, 1, 0}});
}


TYPED_TEST(Matrix, BuildsLocalNonLocalMixedWithColPartition)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    gko::array<comm_index_type> col_mapping{this->ref, {0, 0, 2, 2, 1, 1}};
    auto col_partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, col_mapping, num_parts);

    this->validate(gko::dim<2>{6, 6}, partition, col_partition,
                   {2, 3, 3, 0, 5, 1, 4, 2, 3, 2, 0, 0, 1, 1, 4, 4},
                   {0, 0, 1, 5, 4, 2, 2, 3, 2, 4, 1, 2, 4, 5, 0, 5},
                   {11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1, 1},
                                    I<git>{0, 0, 1}, I<vt>{11, 12, 13}),
                    std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1},
                                    I<git>{1, 0}, I<vt>{14, 15}),
                    std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1},
                                    I<git>{0, 0}, I<vt>{16, 17})},
                   {std::make_tuple(gko::dim<2>{2, 3}, I<git>{0, 1, 0},
                                    I<git>{2, 1, 0}, I<vt>{1, 2, 3}),
                    std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0},
                                    I<git>{0, 1}, I<vt>{4, 5}),
                    std::make_tuple(gko::dim<2>{2, 3}, I<git>{0, 0, 1, 1},
                                    I<git>{1, 2, 0, 2}, I<vt>{6, 7, 8, 9})},
                   {{0, 0, 1}, {1, 0}, {0, 0, 1}},
                   {{0, 1, 2}, {1, 0, 1}, {1, 2, 0}});
}


TYPED_TEST(Matrix, BuildsLocalNonLocalNonSquare)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    gko::array<comm_index_type> row_mapping{this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, row_mapping, num_parts);
    gko::array<comm_index_type> col_mapping{this->ref, {0, 2, 2, 1}};
    auto col_partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, col_mapping, num_parts);

    this->validate(
        gko::dim<2>{6, 4}, partition, col_partition,
        {2, 3, 0, 1, 4, 3, 3, 0, 1, 4}, {0, 0, 3, 2, 1, 2, 3, 0, 3, 3},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        {std::make_tuple(gko::dim<2>{2, 1}, I<git>{0, 1}, I<git>{0, 0},
                         I<vt>{1, 2}),
         std::make_tuple(gko::dim<2>{2, 1}, I<git>{0}, I<git>{0}, I<vt>{3}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{1, 0},
                         I<vt>{4, 5})},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{1, 1}, I<git>{1, 0},
                         I<vt>{6, 7}),
         std::make_tuple(gko::dim<2>{2, 1}, I<git>{0}, I<git>{0}, I<vt>{8}),
         std::make_tuple(gko::dim<2>{2, 1}, I<git>{0, 1}, I<git>{0, 0},
                         I<vt>{9, 10})},
        {{0, 1}, {0}, {0}}, {{0, 1, 1}, {1, 0, 0}, {0, 1, 0}});
}


TYPED_TEST(Matrix, BuildGhostMapContinuous)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {0, 0, 0, 1, 1, 2, 2}};
    constexpr comm_index_type num_parts = 3;
    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);
    this->recv_sizes.resize_and_reset(num_parts + 1);
    gko::array<global_index_type> result[num_parts] = {
        {this->ref, {3}}, {this->ref, {0, 6}}, {this->ref, {4}}};

    for (int local_id = 0; local_id < num_parts; ++local_id) {
        gko::kernels::reference::distributed_matrix::build_local_nonlocal(
            this->ref, this->create_input_full_rank(), partition.get(),
            partition.get(), local_id, this->local_row_idxs,
            this->local_col_idxs, this->local_values, this->non_local_row_idxs,
            this->non_local_col_idxs, this->non_local_values, this->gather_idxs,
            this->recv_sizes, this->non_local_to_global);

        GKO_ASSERT_ARRAY_EQ(result[local_id], this->non_local_to_global);
    }
}

TYPED_TEST(Matrix, BuildGhostMapScattered)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    this->mapping = {this->ref, {0, 1, 2, 0, 1, 2, 0}};
    constexpr comm_index_type num_parts = 3;
    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 this->mapping,
                                                                 num_parts);
    this->recv_sizes.resize_and_reset(num_parts + 1);
    gko::array<global_index_type> result[num_parts] = {
        {this->ref, {5}},
        {this->ref, {6, 2}},
        {this->ref, {4}}};  // the columns are sorted by their part_id

    for (int local_id = 0; local_id < num_parts; ++local_id) {
        gko::kernels::reference::distributed_matrix::build_local_nonlocal(
            this->ref, this->create_input_full_rank(), partition.get(),
            partition.get(), local_id, this->local_row_idxs,
            this->local_col_idxs, this->local_values, this->non_local_row_idxs,
            this->non_local_col_idxs, this->non_local_values, this->gather_idxs,
            this->recv_sizes, this->non_local_to_global);

        GKO_ASSERT_ARRAY_EQ(result[local_id], this->non_local_to_global);
    }
}

}  // namespace
