// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/matrix_kernels.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>

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
          non_local_values{ref}
    {}

    /**
     * apply the `separate_local_nonlocal` kernel and validate the result
     * against provided reference values
     *
     * @param size  the expected global matrix size
     * @param row_partition  the row partition passed to the kernel
     * @param col_partition  the column partition passed to the kernel
     * @param input_rows  the row indices passed to the kernel
     * @param input_cols  the column indices passed to the kernel
     * @param input_vals  the values passed to the kernel
     * @param local_entries  the reference local matrix data. It is provided
     *                       as a list of tuples for each part of the row
     *                       partition. Each tuple consists of the size of
     *                       the local matrix, a list of row indices,
     *                       a list of column indices, and a list of values.
     *                       The indices are mapped to local indexing.
     * @param non_local_entries  the reference non-local matrix data. It is
     *                           provided as a list of tuples for each part
     *                           of the row partition. Each tuple contains
     *                           the size of the non-local matrix, a list of
     *                           row indices (mapped to local indexing), a
     *                           list of column indices (NOT mapped to local
     *                           indexing), and a list of values.
     */
    void act_and_assert(
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
            non_local_entries)
    {
        std::vector<gko::device_matrix_data<value_type, local_index_type>>
            ref_locals;
        std::vector<
            std::tuple<gko::dim<2>, gko::array<local_index_type>,
                       gko::array<global_index_type>, gko::array<value_type>>>
            ref_non_locals;

        auto input = gko::device_matrix_data<value_type, global_index_type>{
            ref, size, gko::array<global_index_type>{ref, input_rows},
            gko::array<global_index_type>{ref, input_cols},
            gko::array<value_type>{ref, input_vals}};
        for (auto entry : local_entries) {
            ref_locals.emplace_back(ref, std::get<0>(entry), std::get<1>(entry),
                                    std::get<2>(entry), std::get<3>(entry));
        }
        for (auto entry : non_local_entries) {
            ref_non_locals.emplace_back(
                std::get<0>(entry),
                gko::array<local_index_type>{ref, std::get<1>(entry)},
                gko::array<global_index_type>{ref, std::get<2>(entry)},
                gko::array<value_type>{ref, std::get<3>(entry)});
        }

        for (comm_index_type part = 0; part < row_partition->get_num_parts();
             ++part) {
            gko::kernels::reference::distributed_matrix::
                separate_local_nonlocal(
                    ref, input, row_partition.get(), col_partition.get(), part,
                    local_row_idxs, local_col_idxs, local_values,
                    non_local_row_idxs, non_local_col_idxs, non_local_values);


            auto local_arrays = ref_locals[part].empty_out();
            GKO_ASSERT_ARRAY_EQ(local_row_idxs, local_arrays.row_idxs);
            GKO_ASSERT_ARRAY_EQ(local_col_idxs, local_arrays.col_idxs);
            GKO_ASSERT_ARRAY_EQ(local_values, local_arrays.values);
            GKO_ASSERT_ARRAY_EQ(non_local_row_idxs,
                                std::get<1>(ref_non_locals[part]));
            GKO_ASSERT_ARRAY_EQ(non_local_col_idxs,
                                std::get<2>(ref_non_locals[part]));
            GKO_ASSERT_ARRAY_EQ(non_local_values,
                                std::get<3>(ref_non_locals[part]));
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
            gko::array<global_index_type>{ref, {0, 0, 2, 3, 3, 4, 4, 5, 5, 6}},
            gko::array<global_index_type>{ref, {0, 3, 2, 0, 3, 4, 6, 4, 5, 5}},
            gko::array<value_type>{ref, {1, 2, 5, 6, 7, 8, 9, 10, 11, 12}}};
    }

    gko::device_matrix_data<value_type, global_index_type>
    create_input_full_rank()
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
    gko::array<local_index_type> local_row_idxs;
    gko::array<local_index_type> local_col_idxs;
    gko::array<value_type> local_values;
    gko::array<local_index_type> non_local_row_idxs;
    gko::array<global_index_type> non_local_col_idxs;
    gko::array<value_type> non_local_values;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(Matrix, CountOverlapEntries)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    using ca = gko::array<comm_index_type>;
    using ga = gko::array<git>;
    this->mapping = {this->ref, {1, 0, 2, 2, 0, 1, 1}};
    std::vector<ca> overlap_count_ref{
        ca{this->ref, I<comm_index_type>{0, 5, 3}},
        ca{this->ref, I<comm_index_type>{4, 0, 3}},
        ca{this->ref, I<comm_index_type>{4, 5, 0}}};
    std::vector<ga> overlap_pos_ref{
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
    auto input = this->create_input_full_rank();

    ca overlap_count{this->ref, static_cast<gko::size_type>(num_parts)};
    ga overlap_positions{this->ref, input.get_num_stored_elements()};
    ga original_positions{this->ref, input.get_num_stored_elements()};
    for (gko::size_type i = 0; i < num_parts; i++) {
        overlap_count.fill(0);
        gko::kernels::reference::distributed_matrix::count_overlap_entries(
            this->ref, input, partition.get(), i, overlap_count,
            overlap_positions, original_positions);
        GKO_ASSERT_ARRAY_EQ(overlap_count, overlap_count_ref[i]);
        GKO_ASSERT_ARRAY_EQ(overlap_positions, overlap_pos_ref[i]);
        GKO_ASSERT_ARRAY_EQ(original_positions, original_pos_ref[i]);
    }
}


TYPED_TEST(Matrix, FillOverlapSendBuffers)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    using ca = gko::array<comm_index_type>;
    using ga = gko::array<git>;
    using va = gko::array<vt>;
    this->mapping = {this->ref, {1, 0, 2, 2, 0, 1, 1}};
    std::vector<ga> overlap_positions{
        ga{this->ref, I<git>{0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7}},
        ga{this->ref, I<git>{0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 6}},
        ga{this->ref, I<git>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9}}};
    std::vector<ga> original_positions{
        ga{this->ref, I<git>{-1, -1, -1, -1, 0, 1, 9, 10, 11, 4, 5, 6}},
        ga{this->ref, I<git>{2, 3, 7, 8, -1, -1, -1, -1, -1, 4, 5, 6}},
        ga{this->ref, I<git>{2, 3, 7, 8, 0, 1, 9, 10, 11, -1, -1, -1}}};
    std::vector<ga> overlap_row_idxs_ref{
        ga{this->ref, I<git>{0, 0, 5, 5, 6, 2, 3, 3}},
        ga{this->ref, I<git>{1, 1, 4, 4, 2, 3, 3}},
        ga{this->ref, I<git>{1, 1, 4, 4, 0, 0, 5, 5, 6}}};
    std::vector<ga> overlap_col_idxs_ref{
        ga{this->ref, I<git>{0, 3, 4, 5, 5, 2, 0, 3}},
        ga{this->ref, I<git>{1, 2, 4, 6, 2, 0, 3}},
        ga{this->ref, I<git>{1, 2, 4, 6, 0, 3, 4, 5, 5}}};
    std::vector<va> overlap_values_ref{
        va{this->ref, I<vt>{1, 2, 10, 11, 12, 5, 6, 7}},
        va{this->ref, I<vt>{3, 4, 8, 9, 5, 6, 7}},
        va{this->ref, I<vt>{3, 4, 8, 9, 1, 2, 10, 11, 12}}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);
    auto input = this->create_input_full_rank();

    gko::array<git> overlap_row_idxs{this->ref};
    gko::array<git> overlap_col_idxs{this->ref};
    gko::array<vt> overlap_values{this->ref};
    for (gko::size_type i = 0; i < num_parts; i++) {
        auto num_entries = overlap_row_idxs_ref[i].get_size();
        overlap_row_idxs.resize_and_reset(num_entries);
        overlap_col_idxs.resize_and_reset(num_entries);
        overlap_values.resize_and_reset(num_entries);
        gko::kernels::reference::distributed_matrix::fill_overlap_send_buffers(
            this->ref, input, partition.get(), i, overlap_positions[i],
            original_positions[i], overlap_row_idxs, overlap_col_idxs,
            overlap_values);
        GKO_ASSERT_ARRAY_EQ(overlap_row_idxs, overlap_row_idxs_ref[i]);
        GKO_ASSERT_ARRAY_EQ(overlap_col_idxs, overlap_col_idxs_ref[i]);
        GKO_ASSERT_ARRAY_EQ(overlap_values, overlap_values_ref[i]);
    }
}


TYPED_TEST(Matrix, SeparateLocalNonLocalEmpty)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 0, 2, 2, 0, 1, 1, 2}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->act_and_assert(
        gko::dim<2>{8, 8}, partition, partition, {}, {}, {},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 3}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 3}, I<git>{}, I<git>{}, I<vt>{})},
        {std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 0}, I<git>{}, I<git>{}, I<vt>{})});
}


TYPED_TEST(Matrix, SeparateLocalNonLocalSmall)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 0}};
    comm_index_type num_parts = 2;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->act_and_assert(
        gko::dim<2>{2, 2}, partition, partition, {0, 0, 1, 1}, {0, 1, 0, 1},
        {1, 2, 3, 4},
        {std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{4}),
         std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{1})},
        {std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{3}),
         std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{1}, I<vt>{2})});
}


TYPED_TEST(Matrix, SeparateLocalNonLocalNoNonLocal)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->act_and_assert(
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
         std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{})});
}


TYPED_TEST(Matrix, SeparateLocalNonLocalNoLocal)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->act_and_assert(
        gko::dim<2>{6, 6}, partition, partition, {0, 0, 1, 3, 4, 5},
        {1, 3, 5, 1, 3, 2}, {1, 2, 5, 6, 7, 8},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{})},
        {std::make_tuple(gko::dim<2>{2, 1}, I<git>{1}, I<git>{1}, I<vt>{6}),
         std::make_tuple(gko::dim<2>{2, 3}, I<git>{0, 0, 1}, I<git>{1, 3, 2},
                         I<vt>{1, 2, 8}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{5, 3},
                         I<vt>{5, 7})});
}


TYPED_TEST(Matrix, SeparateLocalNonLocalMixed)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    this->mapping = {this->ref, {1, 2, 0, 0, 2, 1}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->act_and_assert(
        gko::dim<2>{6, 6}, partition, partition,
        // clang-format on
        {0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5},
        {0, 1, 3, 5, 1, 4, 5, 3, 1, 2, 3, 4, 0, 2},
        {11, 1, 2, 12, 13, 14, 5, 15, 6, 16, 7, 17, 18, 8},
        // clang-format off
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{1, 0},
                         I<vt>{15, 16}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0, 1}, I<git>{0, 1, 0},
                         I<vt>{11, 12, 18}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0, 1}, I<git>{0, 1, 1},
                         I<vt>{13, 14, 17})},
        {std::make_tuple(gko::dim<2>{2, 1}, I<git>{1}, I<git>{1}, I<vt>{6}),
         std::make_tuple(gko::dim<2>{2, 3}, I<git>{0, 0, 1}, I<git>{1, 3, 2},
                         I<vt>{1, 2, 8}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{5, 3},
                         I<vt>{5, 7})});
}


TYPED_TEST(Matrix, SeparateLocalNonLocalEmptyWithColPartition)
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

    this->act_and_assert(
        gko::dim<2>{8, 8}, partition, col_partition, {}, {}, {},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 3}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 3}, I<git>{}, I<git>{}, I<vt>{})},
        {std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{3, 0}, I<git>{}, I<git>{}, I<vt>{})});
}


TYPED_TEST(Matrix, SeparateLocalNonLocalSmallWithColPartition)
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

    this->act_and_assert(
        gko::dim<2>{2, 2}, partition, col_partition, {0, 0, 1, 1}, {0, 1, 0, 1},
        {1, 2, 3, 4},
        {std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{3}),
         std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{2})},
        {std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{1}, I<vt>{4}),
         std::make_tuple(gko::dim<2>{1, 1}, I<git>{0}, I<git>{0}, I<vt>{1})});
}

TYPED_TEST(Matrix, SeparateLocalNonLocalNoNonLocalWithColPartition)
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

    this->act_and_assert(
        gko::dim<2>{6, 6}, partition, col_partition, {3, 0, 5, 1, 1, 4},
        {1, 4, 5, 2, 3, 3}, {1, 2, 3, 4, 5, 6},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{1}, I<git>{1}, I<vt>{1}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{0, 1},
                         I<vt>{2, 3}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0, 1}, I<git>{0, 1, 1},
                         I<vt>{4, 5, 6})},
        {std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 0}, I<git>{}, I<git>{}, I<vt>{})});
}


TYPED_TEST(Matrix, SeparateLocalNonLocalNoLocalWithColPartition)
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

    this->act_and_assert(
        gko::dim<2>{6, 6}, partition, col_partition, {2, 3, 2, 0, 5, 1, 1},
        {2, 3, 5, 0, 1, 1, 4}, {1, 2, 3, 4, 5, 6, 7},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{}, I<git>{}, I<vt>{})},
        {std::make_tuple(gko::dim<2>{2, 3}, I<git>{0, 1, 0}, I<git>{2, 3, 5},
                         I<vt>{1, 2, 3}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{0, 1},
                         I<vt>{4, 5}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0}, I<git>{1, 4},
                         I<vt>{6, 7})});
}


TYPED_TEST(Matrix, SeparateLocalNonLocalMixedWithColPartition)
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

    this->act_and_assert(gko::dim<2>{6, 6}, partition, col_partition,
        // clang-format off
                   {2, 3, 3, 0, 5, 1, 4, 2, 3, 2, 0, 0, 1, 1, 4, 4},
                   { 0,  0,  1,  5,  4,  2,  2, 3, 2, 4, 1, 2, 4, 5, 0, 5},
                   {11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        // clang-format on
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1, 1}, I<git>{0, 0, 1},
                         I<vt>{11, 12, 13}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{1, 0},
                         I<vt>{14, 15}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{0, 0},
                         I<vt>{16, 17})},
        {std::make_tuple(gko::dim<2>{2, 3}, I<git>{0, 1, 0}, I<git>{3, 2, 4},
                         I<vt>{1, 2, 3}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 0}, I<git>{1, 2},
                         I<vt>{4, 5}),
         std::make_tuple(gko::dim<2>{2, 3}, I<git>{0, 0, 1, 1},
                         I<git>{4, 5, 0, 5}, I<vt>{6, 7, 8, 9})});
}


TYPED_TEST(Matrix, SeparateLocalNonLocalNonSquare)
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

    this->act_and_assert(
        gko::dim<2>{6, 4}, partition, col_partition,
        // clang-format off
        {2, 3, 0, 1, 4, 3, 3, 0, 1, 4},
        {0, 0, 3, 2, 1, 2, 3, 0, 3, 3},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        // clang-format on
        {std::make_tuple(gko::dim<2>{2, 1}, I<git>{0, 1}, I<git>{0, 0},
                         I<vt>{1, 2}),
         std::make_tuple(gko::dim<2>{2, 1}, I<git>{0}, I<git>{0}, I<vt>{3}),
         std::make_tuple(gko::dim<2>{2, 2}, I<git>{0, 1}, I<git>{1, 0},
                         I<vt>{4, 5})},
        {std::make_tuple(gko::dim<2>{2, 2}, I<git>{1, 1}, I<git>{2, 3},
                         I<vt>{6, 7}),
         std::make_tuple(gko::dim<2>{2, 1}, I<git>{0}, I<git>{0}, I<vt>{8}),
         std::make_tuple(gko::dim<2>{2, 1}, I<git>{0, 1}, I<git>{3, 3},
                         I<vt>{9, 10})});
}


}  // namespace
