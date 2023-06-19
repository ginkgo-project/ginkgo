#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/overlapping_vector.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/log/logger.hpp>


#include "core/base/utils.hpp"
#include "core/test/utils.hpp"
#include "test/utils/mpi/executor.hpp"


using namespace gko::experimental::distributed;

class VectorCreation : public CommonMpiTestFixture {
public:
    using value_type = double;
    using index_type = gko::int32;

    using part_type = overlapping_partition<index_type>;
    using vector_type = overlapping_vector<value_type, index_type>;
    using md_type = gko::matrix_data<value_type, index_type>;
    using local_vector_type = gko::matrix::Dense<value_type>;


    std::array<std::unique_ptr<local_vector_type>, 3> md{
        {gko::initialize<local_vector_type>({0, 0, 1, 2}, this->exec),
         gko::initialize<local_vector_type>({1, 1, 0, 2}, this->exec),
         gko::initialize<local_vector_type>({2, 2, 0, 1}, this->exec)}};
    std::array<std::unique_ptr<local_vector_type>, 3> md_i{
        {gko::initialize<local_vector_type>({0, 0, 0}, this->exec),
         gko::initialize<local_vector_type>({1, 1, 1}, this->exec),
         gko::initialize<local_vector_type>({2, 2, 2}, this->exec)}};
};


TEST_F(VectorCreation, CanCreatePartition)
{
    std::array<gko::array<comm_index_type>, 3> targets_ids = {
        {{exec, {1, 2}}, {exec, {0, 2}}, {exec, {0, 1}}}};
    gko::array<gko::size_type> group_sizes{exec, {1, 1}};

    auto rank = comm.rank();

    auto part = part_type::build_from_grouped_recv1(
        exec, 2, {}, targets_ids[rank], group_sizes);

    auto vec =
        vector_type::create(exec, comm, part, gko::make_dense_view(md[rank]));

    auto non_local = vec->extract_non_local();

    std::array<I<I<value_type>>, 3> ref_non_local = {
        {{{1}, {2}}, {{0}, {2}}, {{0}, {1}}}};

    GKO_ASSERT_MTX_NEAR(non_local, ref_non_local[rank], 0);
}


TEST_F(VectorCreation, CanMakeConsistent)
{
    using index_set = gko::index_set<index_type>;
    auto rank = comm.rank();
    gko::array<comm_index_type> targets_ids{
        exec, {(rank + comm.size() - 1) % comm.size()}};
    gko::array<gko::size_type> group_sizes{exec, {1}};
    auto part = part_type::build_from_grouped_recv1(
        exec, 2,
        {std::make_pair(index_set{exec, {0}}, (rank + 1) % comm.size())},
        targets_ids, group_sizes);

    auto neighbor_comm =
        gko::experimental::distributed::create_neighborhood_comm(comm,
                                                                 part.get());

    auto vec = vector_type::create(exec, neighbor_comm, part,
                                   gko::make_dense_view(md_i[rank]));

    vec->make_consistent(gko::experimental::distributed::transformation::set);


    auto non_local = vec->extract_non_local();

    std::array<I<I<value_type>>, 3> ref_non_local = {{{{2}}, {{0}}, {{1}}}};

    GKO_ASSERT_MTX_NEAR(non_local, ref_non_local[rank], 0);
}


template <typename T, typename... Rest>
auto make_vector(T&& first, Rest&&... rest)
{
    return std::vector<T>{first, std::forward<Rest>(rest)...};
}

template <typename... Args>
constexpr std::array<std::common_type_t<Args...>, sizeof...(Args)> to_std_array(
    Args&&... args)
{
    using T = std::common_type_t<Args...>;
    return {static_cast<T>(args)...};
}


TEST_F(VectorCreation, CanMakeConsistentLarge)
{
    ASSERT_EQ(comm.size(), 6);
    using index_set = gko::index_set<index_type>;
    auto rank = comm.rank();

    auto send_idxs = to_std_array(
        make_vector(std::make_pair(index_set(exec, gko::span{9, 12}), 1),
                    std::make_pair(index_set(exec, gko::span{0, 2}), 4)),
        make_vector(std::make_pair(index_set(exec, gko::span{0, 3}), 0),
                    std::make_pair(index_set(exec, gko::span{3, 6}), 2),
                    std::make_pair(index_set(exec, gko::span{8, 10}), 3),
                    std::make_pair(index_set(exec, gko::span{10, 12}), 5)),
        make_vector(std::make_pair(index_set(exec, gko::span{2, 4}), 4),
                    std::make_pair(index_set(exec, gko::span{6, 9}), 1)),
        make_vector(std::make_pair(index_set(exec, gko::span{7, 9}), 1)),
        make_vector(std::make_pair(index_set(exec, gko::span{0, 2}), 0),
                    std::make_pair(index_set(exec, gko::span{1, 3}), 2)),
        make_vector(std::make_pair(index_set(exec, gko::span{5, 7}), 1)));

    std::array<gko::array<comm_index_type>, 6> targets_ids = {{
        {exec, {1, 4}},
        {exec, {5, 3, 2, 0}},
        {exec, {1, 4}},
        {exec, {1}},
        {exec, {0, 2}},
        {exec, {1}},
    }};
    std::array<gko::array<gko::size_type>, 6> group_sizes = {
        {{exec, {3, 2}},
         {exec, {2, 2, 3, 3}},
         {exec, {3, 2}},
         {exec, {2}},
         {exec, {2, 2}},
         {exec, {2}}}};
    std::array<int, 6> recv_sizes = {5, 10, 5, 2, 4, 2};

    auto part = part_type::build_from_grouped_recv1(
        exec, 12, send_idxs[rank], targets_ids[rank], group_sizes[rank]);

    auto neighbor_comm =
        gko::experimental::distributed::create_neighborhood_comm(comm,
                                                                 part.get());

    auto init_vector =
        local_vector_type ::create(ref, gko::dim<2>{12 + recv_sizes[rank], 1});
    init_vector->fill(0.0);
    for (int i = 0; i < 12; ++i) {
        init_vector->at(i) = i + 100 * (rank + 1);
    }
    auto vec = vector_type::create(exec, neighbor_comm, part,
                                   gko::make_dense_view(init_vector));

    vec->make_consistent(gko::experimental::distributed::transformation::set);


    auto non_local = vec->extract_non_local();

    std::array<std::unique_ptr<local_vector_type>, 6> ref_non_local = {
        gko::initialize<local_vector_type>({200, 201, 202, 500, 501}, exec),
        gko::initialize<local_vector_type>(
            {605, 606, 407, 408, 306, 307, 308, 109, 110, 111}, exec),
        gko::initialize<local_vector_type>({203, 204, 205, 501, 502}, exec),
        gko::initialize<local_vector_type>({208, 209}, exec),
        gko::initialize<local_vector_type>({100, 101, 302, 303}, exec),
        gko::initialize<local_vector_type>({210, 211}, exec)};

    GKO_ASSERT_MTX_NEAR(non_local, ref_non_local[rank], 0);
}
