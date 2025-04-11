// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/dd_matrix_kernels.hpp"

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
class DdMatrix : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, local_index_type>;

    DdMatrix()
        : ref(gko::ReferenceExecutor::create()),
          mapping{ref},
          non_owning_row_idxs{ref},
          non_owning_col_idxs{ref}
    {
        gko::size_type num_rows = 12;

        // Matrix on rank 0
        local_contributions.emplace_back(
            gko::device_matrix_data<value_type, global_index_type>{
                ref, gko::dim<2>{num_rows, num_rows},
                gko::array<global_index_type>{
                    ref, I<global_index_type>{0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                              3, 3, 3, 4, 4, 4, 4, 5, 5, 5}},
                gko::array<global_index_type>{
                    ref, I<global_index_type>{0, 1, 3, 0, 1, 2, 4, 1, 2, 5,
                                              0, 3, 4, 1, 3, 4, 5, 2, 4, 5}},
                gko::array<value_type>{
                    ref, I<value_type>{2,    -1, -1,   -1, 3,    -1,   -1,
                                       -1,   2,  -1,   -1, 1.5,  -0.5, -1,
                                       -0.5, 2,  -0.5, -1, -0.5, 1.5}}});

        // Matrix on rank 1
        local_contributions.emplace_back(
            gko::device_matrix_data<value_type, global_index_type>{
                ref, gko::dim<2>{num_rows, num_rows},
                gko::array<global_index_type>{
                    ref, I<global_index_type>{3, 3, 3, 4, 4, 4, 4, 5, 5, 5,
                                              6, 6, 6, 7, 7, 7, 7, 8, 8, 8}},
                gko::array<global_index_type>{
                    ref, I<global_index_type>{3, 4, 6, 3, 4, 5, 7, 4, 5, 8,
                                              3, 6, 7, 4, 6, 7, 8, 5, 7, 8}},
                gko::array<value_type>{
                    ref, I<value_type>{1.5,  -0.5, -1,   -0.5, 2,    -0.5, -1,
                                       -0.5, 1.5,  -1,   -1,   1.5,  -0.5, -1,
                                       -0.5, 2,    -0.5, -1,   -0.5, 1.5}}});

        // Matrix on rank 2
        local_contributions.emplace_back(
            gko::device_matrix_data<value_type, global_index_type>{
                ref, gko::dim<2>{num_rows, num_rows},
                gko::array<global_index_type>{
                    ref,
                    I<global_index_type>{6, 6, 6, 7,  7,  7,  7,  8,  8,  8,
                                         9, 9, 9, 10, 10, 10, 10, 11, 11, 11}},
                gko::array<global_index_type>{
                    ref,
                    I<global_index_type>{6, 7, 9,  6, 7, 8,  10, 7, 8,  11,
                                         6, 9, 10, 7, 9, 10, 11, 8, 10, 11}},
                gko::array<value_type>{
                    ref, I<value_type>{1.5,  -0.5, -1, -0.5, 2,  -0.5, -1,
                                       -0.5, 1.5,  -1, -1,   2,  -1,   -1,
                                       -1,   3,    -1, -1,   -1, 2}}});
    }

    /**
     * apply the `filter_non_owning_idxs` kernel and validate the result
     * against provided reference values
     *
     * @param size  the expected global matrix size
     * @param row_partition  the row partition passed to the kernel
     * @param col_partition  the column partition passed to the kernel
     * @param input_rows  the row indices passed to the kernel
     * @param input_cols  the column indices passed to the kernel
     * @param input_vals  the values passed to the kernel
     * @param non_owning_rows  the reference non owning row idxs.
     * @param non_owning_cols  the reference non owning col idxs.
     */
    void act_and_assert_filter_non_owning(
        gko::dim<2> size,
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            row_partition,
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            col_partition,
        std::initializer_list<std::initializer_list<global_index_type>>
            non_owning_rows,
        std::initializer_list<std::initializer_list<global_index_type>>
            non_owning_cols)
    {
        std::vector<gko::array<global_index_type>> ref_non_owning_rows;
        std::vector<gko::array<global_index_type>> ref_non_owning_cols;

        for (auto entry : non_owning_rows) {
            ref_non_owning_rows.emplace_back(
                gko::array<global_index_type>{ref, entry});
        }
        for (auto entry : non_owning_cols) {
            ref_non_owning_cols.emplace_back(
                gko::array<global_index_type>{ref, entry});
        }

        for (comm_index_type part = 0; part < row_partition->get_num_parts();
             ++part) {
            gko::kernels::reference::distributed_dd_matrix::
                filter_non_owning_idxs(ref, this->local_contributions[part],
                                       row_partition.get(), col_partition.get(),
                                       part, non_owning_row_idxs,
                                       non_owning_col_idxs);

            GKO_ASSERT_ARRAY_EQ(non_owning_col_idxs, ref_non_owning_cols[part]);
            GKO_ASSERT_ARRAY_EQ(non_owning_row_idxs, ref_non_owning_rows[part]);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::array<comm_index_type> mapping;
    gko::array<global_index_type> non_owning_row_idxs;
    gko::array<global_index_type> non_owning_col_idxs;
    std::vector<gko::device_matrix_data<value_type, global_index_type>>
        local_contributions;
};

TYPED_TEST_SUITE(DdMatrix, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(DdMatrix, FilterNonOwningIdxs)
{
    using git = typename TestFixture::global_index_type;
    using lit = typename TestFixture::local_index_type;
    this->mapping = {this->ref, {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}};
    comm_index_type num_parts = 3;
    auto partition =
        gko::experimental::distributed::Partition<lit, git>::build_from_mapping(
            this->ref, this->mapping, num_parts);

    this->act_and_assert_filter_non_owning(
        gko::dim<2>{12, 12}, partition, partition,
        {I<git>{4, 4, 4, 4, 5, 5, 5}, I<git>{3, 3, 3, 8, 8, 8},
         I<git>{6, 6, 6, 7, 7, 7, 7}},
        {I<git>{4, 5, 4, 4, 5, 4, 5}, I<git>{3, 3, 8, 3, 8, 8},
         I<git>{6, 7, 6, 7, 7, 6, 7}});
}


}  // namespace
