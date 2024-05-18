// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <memory>

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename ValueLocalGlobalIndexType>
class Pgm : public CommonMpiTestFixture {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using dist_mtx_type =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;
    using pgm = gko::multigrid::Pgm<value_type, local_index_type>;

    Pgm()
        : size{8, 8}, mat_input{size, {{0, 0, 5},  {0, 1, -1}, {1, 0, -1},
                                       {1, 1, 5},  {2, 2, 5},  {3, 3, 5},
                                       {4, 4, 5},  {4, 6, -2}, {5, 5, 5},
                                       {5, 7, -2}, {6, 4, -2}, {6, 6, 5},
                                       {7, 5, -2}, {7, 7, 5},  {0, 2, -3},
                                       {0, 4, 1},  {0, 5, 2},  {0, 6, 3},
                                       {1, 3, -4}, {1, 5, 4},  {1, 6, 5},
                                       {1, 7, 6},  {2, 0, -3}, {2, 5, -1},
                                       {2, 6, -2}, {3, 1, -4}, {3, 7, -5},
                                       {4, 0, 1},  {5, 0, 2},  {5, 1, 4},
                                       {5, 2, -1}, {6, 0, 3},  {6, 1, 5},
                                       {6, 2, -2}, {7, 1, 6},  {7, 3, -5}}}
    {
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 2, 4, 8}));

        dist_mat = dist_mtx_type::create(exec, comm);
        dist_mat->read_distributed(mat_input, row_part);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    gko::dim<2> size;
    std::shared_ptr<Partition> row_part;

    gko::matrix_data<value_type, global_index_type> mat_input;

    std::shared_ptr<dist_mtx_type> dist_mat;
};

TYPED_TEST_SUITE(Pgm, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(Pgm, CanGenerateFromDistributedMatrix)
{
    using pgm = typename TestFixture::pgm;
    using value_type = typename TestFixture::value_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    auto pgm_factory = pgm::build().on(this->exec);
    auto rank = this->comm.rank();
    I<I<value_type>> res_local[] = {{{8}}, {{5, 0}, {0, 5}}, {{6, 0}, {0, 6}}};
    // the non_local new index should follow the local matrix
    // For example, we only store the nonzeros part like [* -1 -2 *] in
    // matrix[2, 4:8], whose * is not in the storage. The 1st and 3rd elements
    // are aggregated to the first group but the rest are aggregated to the
    // second group. Although the stored elements are not aggregated to the same
    // group, we still need to reorder to fit the local matrix ordering. i.e.
    // [-1 -2] -> [-2 -1] after aggregation.
    I<I<value_type>> res_non_local[] = {{{-3, -4, 9, 12}},
                                        {{-3, -2, -1}, {-4, 0, -5}},
                                        {{9, -2, 0}, {12, -1, -5}}};

    auto result = pgm_factory->generate(this->dist_mat);

    auto coarse = gko::as<dist_mtx_type>(result->get_coarse_op());
    GKO_ASSERT_MTX_NEAR(gko::as<local_matrix_type>(coarse->get_local_matrix()),
                        res_local[rank], r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(
        gko::as<local_matrix_type>(coarse->get_non_local_matrix()),
        res_non_local[rank], r<value_type>::value);
}
