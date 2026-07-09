// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
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
          diag_row_idxs{ref},
          diag_col_idxs{ref},
          diag_values{ref},
          off_diag_row_idxs{ref},
          off_diag_col_idxs{ref},
          off_diag_values{ref}
    {}

    /**
     * apply the `separate_diag_off_diag` kernel and validate the result
     * against provided reference values
     *
     * @param size  the expected global matrix size
     * @param row_partition  the row partition passed to the kernel
     * @param col_partition  the column partition passed to the kernel
     * @param input_rows  the row indices passed to the kernel
     * @param input_cols  the column indices passed to the kernel
     * @param input_vals  the values passed to the kernel
     * @param diag_entries  the reference diag matrix data. It is provided
     *                       as a list of tuples for each part of the row
     *                       partition. Each tuple consists of the size of
     *                       the diag matrix, a list of row indices,
     *                       a list of column indices, and a list of values.
     *                       The indices are mapped to local indexing.
     * @param off_diag_entries  the reference off-diag matrix data. It is
     *                           provided as a list of tuples for each part
     *                           of the row partition. Each tuple contains
     *                           the size of the off-diag matrix, a list of
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
            diag_entries,
        std::initializer_list<
            std::tuple<gko::dim<2>, std::initializer_list<global_index_type>,
                       std::initializer_list<global_index_type>,
                       std::initializer_list<value_type>>>
            off_diag_entries)
    {
        std::vector<gko::device_matrix_data<value_type, local_index_type>>
            ref_diags;
        std::vector<
            std::tuple<gko::dim<2>, gko::array<local_index_type>,
                       gko::array<global_index_type>, gko::array<value_type>>>
            ref_off_diags;

        auto input = gko::device_matrix_data<value_type, global_index_type>{
            ref, size, gko::array<global_index_type>{ref, input_rows},
            gko::array<global_index_type>{ref, input_cols},
            gko::array<value_type>{ref, input_vals}};
        for (auto entry : diag_entries) {
            ref_diags.emplace_back(ref, std::get<0>(entry), std::get<1>(entry),
                                   std::get<2>(entry), std::get<3>(entry));
        }
        for (auto entry : off_diag_entries) {
            ref_off_diags.emplace_back(
                std::get<0>(entry),
                gko::array<local_index_type>{ref, std::get<1>(entry)},
                gko::array<global_index_type>{ref, std::get<2>(entry)},
                gko::array<value_type>{ref, std::get<3>(entry)});
        }

        for (comm_index_type part = 0; part < row_partition->get_num_parts();
             ++part) {
            gko::kernels::reference::distributed_matrix::separate_diag_off_diag(
                ref, input, row_partition.get(), col_partition.get(), part,
                diag_row_idxs, diag_col_idxs, diag_values, off_diag_row_idxs,
                off_diag_col_idxs, off_diag_values);


            auto diag_arrays = ref_diags[part].empty_out();
            GKO_ASSERT_ARRAY_EQ(diag_row_idxs, diag_arrays.row_idxs);
            GKO_ASSERT_ARRAY_EQ(diag_col_idxs, diag_arrays.col_idxs);
            GKO_ASSERT_ARRAY_EQ(diag_values, diag_arrays.values);
            GKO_ASSERT_ARRAY_EQ(off_diag_row_idxs,
                                std::get<1>(ref_off_diags[part]));
            GKO_ASSERT_ARRAY_EQ(off_diag_col_idxs,
                                std::get<2>(ref_off_diags[part]));
            GKO_ASSERT_ARRAY_EQ(off_diag_values,
                                std::get<3>(ref_off_diags[part]));
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
    gko::array<local_index_type> diag_row_idxs;
    gko::array<local_index_type> diag_col_idxs;
    gko::array<value_type> diag_values;
    gko::array<local_index_type> off_diag_row_idxs;
    gko::array<global_index_type> off_diag_col_idxs;
    gko::array<value_type> off_diag_values;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(Matrix, SeparateDiagOffDiagEmpty)
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


TYPED_TEST(Matrix, SeparateDiagOffDiagSmall)
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


TYPED_TEST(Matrix, SeparateDiagOffDiagNoOffDiag)
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


TYPED_TEST(Matrix, SeparateDiagOffDiagNoDiag)
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


TYPED_TEST(Matrix, SeparateDiagOffDiagMixed)
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


TYPED_TEST(Matrix, SeparateDiagOffDiagEmptyWithColPartition)
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


TYPED_TEST(Matrix, SeparateDiagOffDiagSmallWithColPartition)
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

TYPED_TEST(Matrix, SeparateDiagOffDiagNoOffDiagWithColPartition)
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


TYPED_TEST(Matrix, SeparateDiagOffDiagNoDiagWithColPartition)
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


TYPED_TEST(Matrix, SeparateDiagOffDiagMixedWithColPartition)
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


TYPED_TEST(Matrix, SeparateDiagOffDiagNonSquare)
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


TYPED_TEST(Matrix, SeparateDiagOffDiagLocalRowsSplitsByColumn)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    using vt = typename TestFixture::value_type;
    auto ref = this->ref;
    // columns partitioned into 3 contiguous parts of size 2 over [0,6)
    auto col_partition = gko::experimental::distributed::Partition<
        lit, git>::build_from_contiguous(ref,
                                         gko::array<git>{ref, {0, 2, 4, 6}});
    // two nonzeros in local row 0, columns given as compact indices into
    // col_map: compact 0 -> global col 3 (owned by part 1), compact 1 ->
    // global col 5 (owned by part 2).
    gko::array<lit> row_idxs{ref, {0, 0}};
    gko::array<lit> col_idxs{ref, {0, 1}};
    gko::array<git> col_map{ref, {3, 5}};
    gko::array<vt> values{ref, {vt{10}, vt{20}}};

    gko::kernels::reference::distributed_matrix::
        separate_diag_off_diag_local_rows(
            ref, row_idxs, col_idxs, col_map, values, col_partition.get(),
            /*local_part=*/1, this->diag_row_idxs, this->diag_col_idxs,
            this->diag_values, this->off_diag_row_idxs, this->off_diag_col_idxs,
            this->off_diag_values);

    // diag: the col=3 entry, local col = 3 - 2 = 1, row 0
    GKO_ASSERT_ARRAY_EQ(this->diag_row_idxs, I<lit>({0}));
    GKO_ASSERT_ARRAY_EQ(this->diag_col_idxs, I<lit>({1}));
    GKO_ASSERT_ARRAY_EQ(this->diag_values, I<vt>({vt{10}}));
    // off-diag: the col=5 entry, kept global (5), row 0
    GKO_ASSERT_ARRAY_EQ(this->off_diag_row_idxs, I<lit>({0}));
    GKO_ASSERT_ARRAY_EQ(this->off_diag_col_idxs, I<git>({5}));
    GKO_ASSERT_ARRAY_EQ(this->off_diag_values, I<vt>({vt{20}}));
}


TYPED_TEST(Matrix, CompressColumnsBuildsCompactMap)
{
    using lit = typename TestFixture::local_index_type;
    using git = typename TestFixture::global_index_type;
    auto ref = this->ref;
    // global columns with duplicates and gaps -> distinct {2, 5, 8}
    gko::array<git> global_cols{ref, {5, 2, 5, 8, 2}};
    gko::array<lit> compact_cols{ref};
    gko::array<git> distinct_cols{ref};

    gko::kernels::reference::distributed_matrix::compress_columns(
        ref, global_cols, compact_cols, distinct_cols);

    // distinct sorted unique, compact = position of each input in distinct
    GKO_ASSERT_ARRAY_EQ(distinct_cols, I<git>({2, 5, 8}));
    GKO_ASSERT_ARRAY_EQ(compact_cols, I<lit>({1, 0, 1, 2, 0}));
}


}  // namespace
