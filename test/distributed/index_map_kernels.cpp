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

    IndexMap()
    {
        auto connections = generate_connection_idxs(ref, this_rank, 11);
        auto dconnections = gko::array<global_index_type>(exec, connections);
        gko::kernels::reference::index_map::build_mapping(
            ref, part.get(), connections, target_ids, remote_local_idxs,
            remote_global_idxs);
        gko::kernels::EXEC_NAMESPACE::index_map::build_mapping(
            exec, dpart.get(), dconnections, dtarget_ids, dremote_local_idxs,
            dremote_global_idxs);
    }

    gko::array<global_index_type> generate_connection_idxs(
        std::shared_ptr<const gko::Executor> exec, comm_index_type excluded_pid,
        gko::size_type num_connections)
    {
        // create vector with [0, ..., num_parts) excluding excluded_pid
        std::vector<gko::size_type> part_ids(num_parts - 1);
        std::iota(part_ids.begin(), part_ids.end(), excluded_pid + 1);
        std::transform(part_ids.begin(), part_ids.end(), part_ids.begin(),
                       [&](const auto pid) { return pid % num_parts; });
        // get random connections
        std::shuffle(part_ids.begin(), part_ids.end(), engine);
        std::vector<gko::size_type> connected_ids(
            part_ids.begin(), part_ids.begin() + num_connections);
        // create global index space of connections
        std::vector<global_index_type> connections_index_space;
        for (auto pid : connected_ids) {
            for (global_index_type i = 0; i < local_size; ++i) {
                connections_index_space.push_back(
                    i + static_cast<global_index_type>(pid * local_size));
            }
        }
        // generate query from connection_index_space
        std::uniform_int_distribution<> dist(
            0, connections_index_space.size() - 1);
        gko::array<global_index_type> connection_idxs{ref, 11};
        std::generate_n(connection_idxs.get_data(), connection_idxs.get_size(),
                        [&] { return connections_index_space[dist(engine)]; });
        return {std::move(exec), std::move(connection_idxs)};
    }

    gko::array<global_index_type> generate_query(
        std::shared_ptr<const gko::Executor> exec,
        const gko::array<global_index_type>& connection_idxs,
        gko::size_type num_queries)
    {
        auto host_connection_idxs =
            gko::make_temporary_clone(ref, &connection_idxs);
        // generate query from connection_index_space
        std::uniform_int_distribution<> dist(0, connection_idxs.get_size() - 1);
        gko::array<global_index_type> query{ref, num_queries};
        std::generate_n(query.get_data(), query.get_size(), [&] {
            return host_connection_idxs->get_const_data()[dist(engine)];
        });
        return {std::move(exec), std::move(query)};
    }

    gko::array<global_index_type> generate_complement_idxs(
        std::shared_ptr<const gko::Executor> exec,
        const gko::array<global_index_type>& idxs)
    {
        auto host_idxs = gko::make_temporary_clone(ref, &idxs);
        std::vector<global_index_type> full_idxs(part->get_size());
        std::iota(full_idxs.begin(), full_idxs.end(), 0);

        std::set<global_index_type> idxs_set(
            host_idxs->get_const_data(),
            host_idxs->get_const_data() + host_idxs->get_size());

        auto end = std::remove_if(
            full_idxs.begin(), full_idxs.end(),
            [&](const auto v) { return idxs_set.find(v) != idxs_set.end(); });
        auto complement_size = std::distance(full_idxs.begin(), end);
        return {std::move(exec), full_idxs.begin(), end};
    }


    gko::array<global_index_type> combine_arrays(
        std::shared_ptr<const gko::Executor> exec,
        const gko::array<global_index_type>& a,
        const gko::array<global_index_type>& b)
    {
        gko::array<global_index_type> result(exec, a.get_size() + b.get_size());
        exec->copy_from(a.get_executor(), a.get_size(), a.get_const_data(),
                        result.get_data());
        exec->copy_from(b.get_executor(), b.get_size(), b.get_const_data(),
                        result.get_data() + a.get_size());
        return result;
    }

    gko::array<global_index_type> take_random(
        const gko::array<global_index_type>& a, gko::size_type n)
    {
        auto copy = gko::array<global_index_type>(ref, a);
        std::shuffle(copy.get_data(), copy.get_data() + copy.get_size(),
                     engine);

        return {a.get_executor(), copy.get_const_data(),
                copy.get_const_data() + n};
    }

    gko::array<comm_index_type> target_ids{ref};
    gko::collection::array<local_index_type> remote_local_idxs{ref};
    gko::collection::array<global_index_type> remote_global_idxs{ref};
    gko::array<comm_index_type> dtarget_ids{exec};
    gko::collection::array<local_index_type> dremote_local_idxs{exec};
    gko::collection::array<global_index_type> dremote_global_idxs{exec};

    comm_index_type num_parts = 13;
    global_index_type local_size = 41;
    comm_index_type this_rank = 5;

    std::shared_ptr<part_type> part = part_type::build_from_global_size_uniform(
        ref, num_parts, num_parts* local_size);
    std::shared_ptr<part_type> dpart = gko::clone(exec, part);

    std::default_random_engine engine;
};

TEST_F(IndexMap, BuildMappingSameAsRef)
{
    auto query = generate_connection_idxs(ref, this_rank, 11);
    auto dquery = gko::array<global_index_type>(exec, query);
    gko::array<comm_index_type> target_ids{ref};
    gko::collection::array<local_index_type> remote_local_idxs{ref};
    gko::collection::array<global_index_type> remote_global_idxs{ref};
    gko::array<comm_index_type> dtarget_ids{exec};
    gko::collection::array<local_index_type> dremote_local_idxs{exec};
    gko::collection::array<global_index_type> dremote_global_idxs{exec};

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


TEST_F(IndexMap, GetLocalWithLocalIndexSpaceSameAsRef)
{
    auto local_space = gko::array<global_index_type>(ref, local_size);
    std::iota(local_space.get_data(), local_space.get_data() + local_size,
              this_rank * local_size);
    auto query = generate_query(ref, local_space, 33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::get_local(
        ref, part.get(), target_ids, remote_global_idxs, this_rank, query,
        gko::experimental::distributed::index_space::local, result);
    gko::kernels::EXEC_NAMESPACE::index_map::get_local(
        exec, dpart.get(), dtarget_ids, dremote_global_idxs, this_rank, dquery,
        gko::experimental::distributed::index_space::local, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetLocalWithLocalIndexSpaceWithInvalidIndexSameAsRef)
{
    auto local_space = gko::array<global_index_type>(ref, local_size);
    std::iota(local_space.get_data(), local_space.get_data() + local_size,
              this_rank * local_size);
    auto query = generate_query(
        ref,
        combine_arrays(
            ref, local_space,
            take_random(generate_complement_idxs(ref, local_space), 12)),
        33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::get_local(
        ref, part.get(), target_ids, remote_global_idxs, this_rank, query,
        gko::experimental::distributed::index_space::local, result);
    gko::kernels::EXEC_NAMESPACE::index_map::get_local(
        exec, dpart.get(), dtarget_ids, dremote_global_idxs, this_rank, dquery,
        gko::experimental::distributed::index_space::local, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetLocalWithNonLocalIndexSpaceSameAsRef)
{
    auto query = generate_query(ref, remote_global_idxs.get_flat(), 33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::get_local(
        ref, part.get(), target_ids, remote_global_idxs, this_rank, query,
        gko::experimental::distributed::index_space::non_local, result);
    gko::kernels::EXEC_NAMESPACE::index_map::get_local(
        exec, dpart.get(), dtarget_ids, dremote_global_idxs, this_rank, dquery,
        gko::experimental::distributed::index_space::non_local, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetLocalWithNonLocalIndexSpaceWithInvalidIndexSameAsRef)
{
    auto query = generate_query(
        ref,
        combine_arrays(ref, remote_global_idxs.get_flat(),
                       take_random(generate_complement_idxs(
                                       ref, remote_global_idxs.get_flat()),
                                   12)),
        33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::get_local(
        ref, part.get(), target_ids, remote_global_idxs, this_rank, query,
        gko::experimental::distributed::index_space::non_local, result);
    gko::kernels::EXEC_NAMESPACE::index_map::get_local(
        exec, dpart.get(), dtarget_ids, dremote_global_idxs, this_rank, dquery,
        gko::experimental::distributed::index_space::non_local, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetLocalWithCombinedIndexSpaceSameAsRef)
{
    auto local_space = gko::array<global_index_type>(ref, local_size);
    std::iota(local_space.get_data(), local_space.get_data() + local_size,
              this_rank * local_size);
    auto combined_space =
        combine_arrays(ref, local_space, remote_global_idxs.get_flat());
    auto query = generate_query(ref, combined_space, 33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::get_local(
        ref, part.get(), target_ids, remote_global_idxs, this_rank, query,
        gko::experimental::distributed::index_space::combined, result);
    gko::kernels::EXEC_NAMESPACE::index_map::get_local(
        exec, dpart.get(), dtarget_ids, dremote_global_idxs, this_rank, dquery,
        gko::experimental::distributed::index_space::combined, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetLocalWithCombinedIndexSpaceWithInvalidIndexSameAsRef)
{
    auto local_space = gko::array<global_index_type>(ref, local_size);
    std::iota(local_space.get_data(), local_space.get_data() + local_size,
              this_rank * local_size);
    auto combined_space =
        combine_arrays(ref, local_space, remote_global_idxs.get_flat());
    auto query = generate_query(
        ref,
        combine_arrays(
            ref, combined_space,
            take_random(generate_complement_idxs(ref, combined_space), 12)),
        33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::get_local(
        ref, part.get(), target_ids, remote_global_idxs, this_rank, query,
        gko::experimental::distributed::index_space::non_local, result);
    gko::kernels::EXEC_NAMESPACE::index_map::get_local(
        exec, dpart.get(), dtarget_ids, dremote_global_idxs, this_rank, dquery,
        gko::experimental::distributed::index_space::non_local, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}
