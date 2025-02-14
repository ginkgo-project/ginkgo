// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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
#include "test/utils/common_fixture.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename LocalIndexType, typename GlobalIndexType>
gko::array<GlobalIndexType> generate_connection_idxs(
    const std::shared_ptr<const gko::Executor>& exec, comm_index_type rank,
    std::shared_ptr<gko::experimental::distributed::Partition<LocalIndexType,
                                                              GlobalIndexType>>
        partition,
    std::default_random_engine engine, gko::size_type num_connections)
{
    auto ref = exec->get_master();
    auto num_parts = partition->get_num_parts();
    auto local_size =
        static_cast<GlobalIndexType>(partition->get_part_size(rank));
    // create vector with [0, ..., num_parts) excluding excluded_pid
    std::vector<gko::size_type> part_ids(num_parts - 1);
    std::iota(part_ids.begin(), part_ids.end(), rank + 1);
    std::transform(part_ids.begin(), part_ids.end(), part_ids.begin(),
                   [&](const auto pid) { return pid % num_parts; });
    // get random connections
    std::shuffle(part_ids.begin(), part_ids.end(), engine);
    std::vector<gko::size_type> connected_ids(
        part_ids.begin(), part_ids.begin() + num_connections);
    // create global index space of connections
    std::vector<GlobalIndexType> connections_index_space;
    for (auto pid : connected_ids) {
        for (GlobalIndexType i = 0; i < local_size; ++i) {
            connections_index_space.push_back(
                i + static_cast<GlobalIndexType>(pid * local_size));
        }
    }
    // generate query from connection_index_space
    std::uniform_int_distribution<> dist(0, connections_index_space.size() - 1);
    gko::array<GlobalIndexType> connection_idxs{ref, 11};
    std::generate_n(connection_idxs.get_data(), connection_idxs.get_size(),
                    [&] { return connections_index_space[dist(engine)]; });
    return {exec, std::move(connection_idxs)};
}


class IndexMapBuildMapping : public CommonTestFixture {};


TEST_F(IndexMapBuildMapping, BuildMappingSameAsRef)
{
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    std::default_random_engine engine;
    comm_index_type num_parts = 13;
    global_index_type local_size = 41;
    comm_index_type this_rank = 5;
    std::shared_ptr<part_type> part = part_type::build_from_global_size_uniform(
        ref, num_parts, num_parts * local_size);
    std::shared_ptr<part_type> dpart = gko::clone(exec, part);
    auto query = generate_connection_idxs(ref, this_rank, part, engine, 11);
    auto dquery = gko::array<global_index_type>(exec, query);
    gko::array<comm_index_type> target_ids{ref};
    gko::array<local_index_type> remote_local_idxs{ref};
    gko::array<global_index_type> remote_global_idxs{ref};
    gko::array<gko::int64> remote_sizes{ref};
    gko::array<comm_index_type> dtarget_ids{exec};
    gko::array<local_index_type> dremote_local_idxs{exec};
    gko::array<global_index_type> dremote_global_idxs{exec};
    gko::array<gko::int64> dremote_sizes{exec};

    gko::kernels::reference::index_map::build_mapping(
        ref, part.get(), query, target_ids, remote_local_idxs,
        remote_global_idxs, remote_sizes);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::build_mapping(
        exec, dpart.get(), dquery, dtarget_ids, dremote_local_idxs,
        dremote_global_idxs, dremote_sizes);

    GKO_ASSERT_ARRAY_EQ(remote_sizes, dremote_sizes);
    GKO_ASSERT_ARRAY_EQ(target_ids, dtarget_ids);
    GKO_ASSERT_ARRAY_EQ(remote_local_idxs, dremote_local_idxs);
    GKO_ASSERT_ARRAY_EQ(remote_global_idxs, dremote_global_idxs);
}


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
        engine.seed(490729788);

        auto connections =
            generate_connection_idxs(ref, this_rank, part, engine, 11);
        auto dconnections = gko::array<global_index_type>(exec, connections);

        auto flat_remote_local_idxs = gko::array<local_index_type>(ref);
        auto flat_remote_global_idxs = gko::array<global_index_type>(ref);
        auto dflat_remote_local_idxs = gko::array<local_index_type>(exec);
        auto dflat_remote_global_idxs = gko::array<global_index_type>(exec);

        auto remote_sizes = gko::array<gko::int64>(ref);
        auto dremote_sizes = gko::array<gko::int64>(exec);

        gko::kernels::reference::index_map::build_mapping(
            ref, part.get(), connections, target_ids, flat_remote_local_idxs,
            flat_remote_global_idxs, remote_sizes);
        gko::kernels::GKO_DEVICE_NAMESPACE::index_map::build_mapping(
            exec, dpart.get(), dconnections, dtarget_ids,
            dflat_remote_local_idxs, dflat_remote_global_idxs, dremote_sizes);

        remote_local_idxs =
            gko::segmented_array<local_index_type>::create_from_sizes(
                std::move(flat_remote_local_idxs), remote_sizes);
        remote_global_idxs =
            gko::segmented_array<global_index_type>::create_from_sizes(
                std::move(flat_remote_global_idxs), remote_sizes);
        dremote_local_idxs =
            gko::segmented_array<local_index_type>::create_from_sizes(
                std::move(dflat_remote_local_idxs), dremote_sizes);
        dremote_global_idxs =
            gko::segmented_array<global_index_type>::create_from_sizes(
                std::move(dflat_remote_global_idxs), dremote_sizes);
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

    gko::array<local_index_type> generate_to_global_query(
        std::shared_ptr<const gko::Executor> exec, gko::size_type size,
        gko::size_type num_queries)
    {
        std::uniform_int_distribution<local_index_type> dist(0, size - 1);
        gko::array<local_index_type> query{ref, num_queries};
        std::generate_n(query.get_data(), query.get_size(),
                        [&] { return dist(engine); });
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
    gko::segmented_array<local_index_type> remote_local_idxs{ref};
    gko::segmented_array<global_index_type> remote_global_idxs{ref};
    gko::array<comm_index_type> dtarget_ids{exec};
    gko::segmented_array<local_index_type> dremote_local_idxs{exec};
    gko::segmented_array<global_index_type> dremote_global_idxs{exec};

    comm_index_type num_parts = 13;
    global_index_type local_size = 41;
    comm_index_type this_rank = 5;

    std::shared_ptr<part_type> part = part_type::build_from_global_size_uniform(
        ref, num_parts, num_parts* local_size);
    std::shared_ptr<part_type> dpart = gko::clone(exec, part);

    std::default_random_engine engine;
};


TEST_F(IndexMap, GetLocalWithLocalIndexSpaceSameAsRef)
{
    auto local_space = gko::array<global_index_type>(ref, local_size);
    std::iota(local_space.get_data(), local_space.get_data() + local_size,
              this_rank * local_size);
    auto query = generate_query(ref, local_space, 33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), target_ids, to_device_const(remote_global_idxs),
        this_rank, query, gko::experimental::distributed::index_space::local,
        result);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, dquery, gko::experimental::distributed::index_space::local,
        dresult);

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

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), target_ids, to_device_const(remote_global_idxs),
        this_rank, query, gko::experimental::distributed::index_space::local,
        result);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, dquery, gko::experimental::distributed::index_space::local,
        dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


template <typename T>
gko::array<T> get_flat_array(const gko::segmented_array<T>& arr)
{
    return gko::make_const_array_view(arr.get_executor(), arr.get_size(),
                                      arr.get_const_flat_data())
        .copy_to_array();
}


TEST_F(IndexMap, GetLocalWithNonLocalIndexSpaceSameAsRef)
{
    auto query = generate_query(ref, get_flat_array(remote_global_idxs), 33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), target_ids, to_device_const(remote_global_idxs),
        this_rank, query,
        gko::experimental::distributed::index_space::non_local, result);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, dquery,
        gko::experimental::distributed::index_space::non_local, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetLocalWithNonLocalIndexSpaceWithInvalidIndexSameAsRef)
{
    auto query = generate_query(
        ref,
        combine_arrays(ref, get_flat_array(remote_global_idxs),
                       take_random(generate_complement_idxs(
                                       ref, get_flat_array(remote_global_idxs)),
                                   12)),
        33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), target_ids, to_device_const(remote_global_idxs),
        this_rank, query,
        gko::experimental::distributed::index_space::non_local, result);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, dquery,
        gko::experimental::distributed::index_space::non_local, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetLocalWithCombinedIndexSpaceSameAsRef)
{
    auto local_space = gko::array<global_index_type>(ref, local_size);
    std::iota(local_space.get_data(), local_space.get_data() + local_size,
              this_rank * local_size);
    auto combined_space =
        combine_arrays(ref, local_space, get_flat_array(remote_global_idxs));
    auto query = generate_query(ref, combined_space, 33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), target_ids, to_device_const(remote_global_idxs),
        this_rank, query, gko::experimental::distributed::index_space::combined,
        result);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, dquery,
        gko::experimental::distributed::index_space::combined, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetLocalWithCombinedIndexSpaceWithInvalidIndexSameAsRef)
{
    auto local_space = gko::array<global_index_type>(ref, local_size);
    std::iota(local_space.get_data(), local_space.get_data() + local_size,
              this_rank * local_size);
    auto combined_space =
        combine_arrays(ref, local_space, get_flat_array(remote_global_idxs));
    auto query = generate_query(
        ref,
        combine_arrays(
            ref, combined_space,
            take_random(generate_complement_idxs(ref, combined_space), 12)),
        33);
    auto dquery = gko::array<global_index_type>(exec, query);
    auto result = gko::array<local_index_type>(ref);
    auto dresult = gko::array<local_index_type>(exec);

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), target_ids, to_device_const(remote_global_idxs),
        this_rank, query,
        gko::experimental::distributed::index_space::non_local, result);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, dquery,
        gko::experimental::distributed::index_space::non_local, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetGlobalWithLocalIndexSpaceSameAsRef)
{
    auto query = generate_to_global_query(ref, local_size * 2, 33);
    auto dquery = gko::array<local_index_type>(exec, query);
    auto result = gko::array<global_index_type>(ref);
    auto dresult = gko::array<global_index_type>(exec);

    gko::kernels::reference::index_map::map_to_global(
        ref, to_device_const(part.get()), to_device_const(remote_global_idxs),
        this_rank, query, gko::experimental::distributed::index_space::local,
        result);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_global(
        exec, to_device_const(dpart.get()),
        to_device_const(dremote_global_idxs), this_rank, dquery,
        gko::experimental::distributed::index_space::local, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetGlobalWithNonLocalIndexSpaceSameAsRef)
{
    auto query =
        generate_to_global_query(ref, remote_global_idxs.get_size() * 2, 33);
    auto dquery = gko::array<local_index_type>(exec, query);
    auto result = gko::array<global_index_type>(ref);
    auto dresult = gko::array<global_index_type>(exec);

    gko::kernels::reference::index_map::map_to_global(
        ref, to_device_const(part.get()), to_device_const(remote_global_idxs),
        this_rank, query,
        gko::experimental::distributed::index_space::non_local, result);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_global(
        exec, to_device_const(dpart.get()),
        to_device_const(dremote_global_idxs), this_rank, dquery,
        gko::experimental::distributed::index_space::non_local, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, GetGlobalWithCombinedIndexSpaceSameAsRef)
{
    auto query = generate_to_global_query(
        ref, (local_size + remote_global_idxs.get_size()) * 2, 33);
    auto dquery = gko::array<local_index_type>(exec, query);
    auto result = gko::array<global_index_type>(ref);
    auto dresult = gko::array<global_index_type>(exec);

    gko::kernels::reference::index_map::map_to_global(
        ref, to_device_const(part.get()), to_device_const(remote_global_idxs),
        this_rank, query, gko::experimental::distributed::index_space::combined,
        result);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_global(
        exec, to_device_const(dpart.get()),
        to_device_const(dremote_global_idxs), this_rank, dquery,
        gko::experimental::distributed::index_space::combined, dresult);

    GKO_ASSERT_ARRAY_EQ(result, dresult);
}


TEST_F(IndexMap, RoundTripGlobalWithLocalIndexSpace)
{
    auto local_space = gko::array<global_index_type>(ref, local_size);
    std::iota(local_space.get_data(), local_space.get_data() + local_size,
              this_rank * local_size);
    auto query = generate_query(exec, local_space, 33);
    auto local = gko::array<local_index_type>(exec);
    auto global = gko::array<global_index_type>(exec);

    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, query, gko::experimental::distributed::index_space::combined,
        local);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_global(
        exec, to_device_const(dpart.get()),
        to_device_const(dremote_global_idxs), this_rank, local,
        gko::experimental::distributed::index_space::combined, global);

    GKO_ASSERT_ARRAY_EQ(global, query);
}


TEST_F(IndexMap, RoundTripLocalWithLocalIndexSpace)
{
    auto query = generate_to_global_query(exec, local_size, 333);
    auto local = gko::array<local_index_type>(exec);
    auto global = gko::array<global_index_type>(exec);

    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_global(
        exec, to_device_const(dpart.get()),
        to_device_const(dremote_global_idxs), this_rank, query,
        gko::experimental::distributed::index_space::combined, global);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, global,
        gko::experimental::distributed::index_space::combined, local);

    GKO_ASSERT_ARRAY_EQ(local, query);
}


TEST_F(IndexMap, RoundTripGlobalWithNonLocalIndexSpace)
{
    auto query = generate_query(exec, get_flat_array(remote_global_idxs), 333);
    auto local = gko::array<local_index_type>(exec);
    auto global = gko::array<global_index_type>(exec);

    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, query, gko::experimental::distributed::index_space::combined,
        local);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_global(
        exec, to_device_const(dpart.get()),
        to_device_const(dremote_global_idxs), this_rank, local,
        gko::experimental::distributed::index_space::combined, global);

    GKO_ASSERT_ARRAY_EQ(global, query);
}


TEST_F(IndexMap, RoundTripLocalWithNonLocalIndexSpace)
{
    auto query =
        generate_to_global_query(exec, remote_global_idxs.get_size(), 33);
    auto local = gko::array<local_index_type>(exec);
    auto global = gko::array<global_index_type>(exec);

    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_global(
        exec, to_device_const(dpart.get()),
        to_device_const(dremote_global_idxs), this_rank, query,
        gko::experimental::distributed::index_space::combined, global);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, global,
        gko::experimental::distributed::index_space::combined, local);

    GKO_ASSERT_ARRAY_EQ(local, query);
}


TEST_F(IndexMap, RoundTripGlobalWithCombinedIndexSpace)
{
    auto local_space = gko::array<global_index_type>(ref, local_size);
    std::iota(local_space.get_data(), local_space.get_data() + local_size,
              this_rank * local_size);
    auto combined_space =
        combine_arrays(ref, local_space, get_flat_array(remote_global_idxs));
    auto query = generate_query(exec, combined_space, 333);
    auto local = gko::array<local_index_type>(exec);
    auto global = gko::array<global_index_type>(exec);

    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, query, gko::experimental::distributed::index_space::combined,
        local);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_global(
        exec, to_device_const(dpart.get()),
        to_device_const(dremote_global_idxs), this_rank, local,
        gko::experimental::distributed::index_space::combined, global);

    GKO_ASSERT_ARRAY_EQ(global, query);
}


TEST_F(IndexMap, RoundTripLocalWithCombinedIndexSpace)
{
    auto local_space = gko::array<global_index_type>(ref, local_size);
    std::iota(local_space.get_data(), local_space.get_data() + local_size,
              this_rank * local_size);
    auto combined_space =
        combine_arrays(ref, local_space, get_flat_array(remote_global_idxs));
    auto query = generate_to_global_query(
        exec, local_size + remote_global_idxs.get_size(), 333);
    auto local = gko::array<local_index_type>(exec);
    auto global = gko::array<global_index_type>(exec);

    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_global(
        exec, to_device_const(dpart.get()),
        to_device_const(dremote_global_idxs), this_rank, query,
        gko::experimental::distributed::index_space::combined, global);
    gko::kernels::GKO_DEVICE_NAMESPACE::index_map::map_to_local(
        exec, dpart.get(), dtarget_ids, to_device_const(dremote_global_idxs),
        this_rank, global,
        gko::experimental::distributed::index_space::combined, local);

    GKO_ASSERT_ARRAY_EQ(local, query);
}
