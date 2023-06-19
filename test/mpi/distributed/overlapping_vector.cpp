#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/overlapping_vector.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/log/logger.hpp>


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
