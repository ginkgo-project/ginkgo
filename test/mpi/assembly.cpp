// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <memory>
#include <random>

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/assembly.hpp>
#include <ginkgo/core/distributed/partition.hpp>

#include "core/test/utils.hpp"
#include "ginkgo/core/base/exception.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#ifndef GKO_COMPILING_DPCPP


template <typename ValueLocalGlobalIndexType>
class AssemblyHelpers : public CommonMpiTestFixture {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;


    AssemblyHelpers()
        : size{5, 5},
          dist_input{
              {{size,
                {{0, 1, 1},
                 {0, 3, 2},
                 {1, 1, 3},
                 {1, 2, 4},
                 {2, 0, 1},
                 {2, 3, 1}}},
               {size, {{0, 0, 1}, {2, 1, 5}, {2, 2, 6}, {3, 3, 8}, {3, 4, 7}}},
               {size, {{2, 2, 1}, {3, 3, -1}, {4, 0, 9}, {4, 4, 10}}}}},
          res_row_idxs{{{exec, {0, 0, 0, 1, 1, 2, 2}},
                        {exec, {0, 2, 2, 2, 2, 3, 3}},
                        {exec, {2, 3, 4, 4}}}},
          res_col_idxs{{{exec, {0, 1, 3, 1, 2, 0, 3}},
                        {exec, {0, 0, 1, 2, 3, 3, 4}},
                        {exec, {2, 3, 0, 4}}}},
          res_values{{{exec, {1, 1, 2, 3, 4, 1, 1}},
                      {exec, {1, 1, 5, 7, 1, 7, 7}},
                      {exec, {1, -1, 9, 10}}}},
          engine(42)
    {
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 2, 4, 5}));
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }


    gko::dim<2> size;
    std::shared_ptr<Partition> row_part;

    gko::matrix_data<value_type, global_index_type> mat_input;
    std::array<matrix_data, 3> dist_input;
    std::array<gko::array<global_index_type>, 3> res_row_idxs;
    std::array<gko::array<global_index_type>, 3> res_col_idxs;
    std::array<gko::array<value_type>, 3> res_values;

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(AssemblyHelpers, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(AssemblyHelpers, AddsNonLocalEntries)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    auto rank = this->comm.rank();
    auto input = gko::device_matrix_data<value_type, global_index_type>::
        create_from_host(this->exec, this->dist_input[rank]);

    auto result = gko::experimental::distributed::assemble_rows_from_neighbors<
        value_type, local_index_type, global_index_type>(this->comm, input,
                                                         this->row_part);

    auto result_arrays = result.empty_out();
    GKO_ASSERT_ARRAY_EQ(result_arrays.row_idxs, this->res_row_idxs[rank]);
    GKO_ASSERT_ARRAY_EQ(result_arrays.col_idxs, this->res_col_idxs[rank]);
    GKO_ASSERT_ARRAY_EQ(result_arrays.values, this->res_values[rank]);
}

#endif
