#include <ginkgo/core/distributed/partition_helpers.hpp>
#include <ginkgo/core/distributed/partition.hpp>


#include "core/test/utils.hpp"
#include "test/utils/mpi/executor.hpp"


template<typename IndexType>
class PartitionHelpers : public CommonMpiTestFixture{
protected:
    using index_type = IndexType;

};

TYPED_TEST_SUITE(PartitionHelpers, gko::test::IndexTypes);


TYPED_TEST(PartitionHelpers, CanBuildFromLocalRanges){
    using itype = typename TestFixture::index_type ;
    gko::span local_range[] = {{0u, 4u}, {4u, 9u}, {9u, 11u}};
    gko::array<itype> expects{this->exec, {0, 4, 9, 11}};

    auto part = gko::experimental::distributed::build_partition_from_local_range<gko::int32, itype>(this->exec, local_range[this->comm.rank()], this->comm);

    GKO_ASSERT_ARRAY_EQ(expects,
                        gko::make_const_array_view(this->exec, expects.get_num_elems(), part->get_range_bounds()));
}
