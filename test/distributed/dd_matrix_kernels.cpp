// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/dd_matrix_kernels.hpp"

#include <algorithm>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/test/utils.hpp"
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
        //  ++part) {
        // gko::array<local_index_type> local_row_idxs{ref};
        // gko::array<local_index_type> local_col_idxs{ref};
        // gko::array<value_type> local_values{ref};
        // gko::array<local_index_type> d_local_row_idxs{exec};
        // gko::array<local_index_type> d_local_col_idxs{exec};
        // gko::array<value_type> d_local_values{exec};
        // gko::array<local_index_type> non_local_row_idxs{ref};
        // gko::array<global_index_type> non_local_col_idxs{ref};
        // gko::array<value_type> non_local_values{ref};
        // gko::array<local_index_type> d_non_local_row_idxs{exec};
        // gko::array<global_index_type> d_non_local_col_idxs{exec};
        // gko::array<value_type> d_non_local_values{exec};

        // gko::kernels::reference::distributed_matrix::
        //     separate_local_nonlocal(
        //         ref, input, row_partition.get(), col_partition.get(), part,
        //         local_row_idxs, local_col_idxs, local_values,
        //         non_local_row_idxs, non_local_col_idxs, non_local_values);
        // gko::kernels::GKO_DEVICE_NAMESPACE::distributed_matrix::
        //     separate_local_nonlocal(
        //         exec, d_input, d_row_partition.get(), d_col_partition.get(),
        //         part, d_local_row_idxs, d_local_col_idxs, d_local_values,
        //         d_non_local_row_idxs, d_non_local_col_idxs,
        //         d_non_local_values);

        // GKO_ASSERT_ARRAY_EQ(local_row_idxs, d_local_row_idxs);
        // GKO_ASSERT_ARRAY_EQ(local_col_idxs, d_local_col_idxs);
        // GKO_ASSERT_ARRAY_EQ(local_values, d_local_values);
        // GKO_ASSERT_ARRAY_EQ(non_local_row_idxs, d_non_local_row_idxs);
        // GKO_ASSERT_ARRAY_EQ(non_local_col_idxs, d_non_local_col_idxs);
        // GKO_ASSERT_ARRAY_EQ(non_local_values, d_non_local_values);
    }
}

std::default_random_engine engine;
}
;

TYPED_TEST_SUITE(DdMatrix, gko::test::ValueLocalGlobalIndexTypesBase,
                 TupleTypenameNameGenerator);


TYPED_TEST(DdMatrix, BuildsDiagOffdiagEmptyIsSameAsRef)
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
