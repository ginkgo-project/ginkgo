// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"


#include <algorithm>
#include <memory>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/distributed/partition_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


class IndexMap : public CommonTestFixture {
protected:
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using map_type =
        gko::experimental::distributed::index_map<local_index_type,
                                                  global_index_type>;

    std::default_random_engine engine;
};

TEST_F(IndexMap, BuildMappingSameAsRef)
{
    gko::array<comm_index_type> target_ids(ref);
    gko::collection::array<local_index_type> remote_local_idxs(ref);
    gko::collection::array<global_index_type> remote_global_idxs(ref);
    gko::array<comm_index_type> dtarget_ids(exec);
    gko::collection::array<local_index_type> dremote_local_idxs(exec);
    gko::collection::array<global_index_type> dremote_global_idxs(exec);
    auto part = part_type::build_from_global_size_uniform(ref, 13, 37);
    auto dpart = gko::clone(exec, part);
    gko::kernels::cuda::partition::comm_index_type part_id = 5;
    auto query = gko::test::generate_random_array<global_index_type>(
        13,
        std::uniform_int_distribution<global_index_type>(
            0, part->get_size() - part->get_part_size(part_id)),
        engine, ref);
    for (int i = 0; i < query.get_size(); ++i) {
        query.get_data()[i] =
            (query.get_data()[i] + part->get_range_bounds()[part_id + 1]) %
            part->get_size();
    }
    auto dquery = gko::array(exec, query);

    gko::kernels::reference::index_map::build_mapping(
        ref, part.get(), query, target_ids, remote_local_idxs,
        remote_global_idxs);
    gko::kernels::EXEC_NAMESPACE::index_map::build_mapping(
        exec, dpart.get(), dquery, dtarget_ids, dremote_local_idxs,
        dremote_global_idxs);

    GKO_ASSERT_ARRAY_EQ(target_ids, dtarget_ids);
    GKO_ASSERT_ARRAY_EQ(remote_local_idxs.get_flat(),
                        dremote_local_idxs.get_flat());
    GKO_ASSERT_ARRAY_EQ(remote_global_idxs.get_flat(),
                        dremote_global_idxs.get_flat());
}
