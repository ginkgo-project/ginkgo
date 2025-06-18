// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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
#include <ginkgo/core/multigrid/fixed_coarsening.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename ValueLocalGlobalIndexType>
class FixedCoarsening : public CommonMpiTestFixture {
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
    using fixed_coarsening =
        gko::multigrid::FixedCoarsening<value_type, local_index_type>;

    FixedCoarsening()
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

TYPED_TEST_SUITE(FixedCoarsening, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(FixedCoarsening, CanGenerateFromDistributedMatrix)
{
    using fixed_coarsening = typename TestFixture::fixed_coarsening;
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    I<local_index_type> coarse_rows_list[] = {I<local_index_type>{0},
                                              I<local_index_type>{0, 1},
                                              I<local_index_type>{1, 3}};
    auto rank = this->comm.rank();
    auto coarse_rows =
        gko::array<local_index_type>(this->exec, coarse_rows_list[rank]);
    auto fixed_coarsening_factory =
        fixed_coarsening::build().with_coarse_rows(coarse_rows).on(this->exec);
    I<I<value_type>> res_local[] = {
        {{5}}, {{5, 0}, {0, 5}}, {{5, -2}, {-2, 5}}};

    // the size is decide before dropping
    I<I<value_type>> res_non_local[] = {
        {{-3, 0, 2, 0}}, {{-3, -1, 0}, {0, 0, -5}}, {{2, -1, 0}, {0, 0, -5}}};

    auto result = fixed_coarsening_factory->generate(this->dist_mat);

    auto coarse = gko::as<dist_mtx_type>(result->get_coarse_op());
    GKO_ASSERT_MTX_NEAR(gko::as<local_matrix_type>(coarse->get_local_matrix()),
                        res_local[rank], r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(
        gko::as<local_matrix_type>(coarse->get_non_local_matrix()),
        res_non_local[rank], r<value_type>::value);
    auto restrict_op = gko::as<dist_mtx_type>(result->get_restrict_op());
    I<I<value_type>> restrict_local[] = {
        {{1, 0}}, {{1, 0}, {0, 1}}, {{0, 1, 0, 0}, {0, 0, 0, 1}}};
    GKO_ASSERT_MTX_NEAR(
        gko::as<local_matrix_type>(restrict_op->get_local_matrix()),
        restrict_local[rank], r<value_type>::value);
    auto prolong_op = gko::as<dist_mtx_type>(result->get_prolong_op());
    I<I<value_type>> prolong_local[] = {
        {{1}, {0}}, {{1, 0}, {0, 1}}, {{0, 0}, {1, 0}, {0, 0}, {0, 1}}};
    GKO_ASSERT_MTX_NEAR(
        gko::as<local_matrix_type>(prolong_op->get_local_matrix()),
        prolong_local[rank], r<value_type>::value);
}
