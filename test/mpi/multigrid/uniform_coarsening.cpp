// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
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
#include <ginkgo/core/multigrid/uniform_coarsening.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename ValueLocalGlobalIndexType>
class UniformCoarsening : public CommonMpiTestFixture {
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
    using uniform_coarsening =
        gko::multigrid::UniformCoarsening<value_type, local_index_type>;

    // 8x8 symmetric matrix with partition [0,2), [2,4), [4,8).
    // coarse_skip=2 either selects rows 0,2,4,6 (injection mode) or
    // aggregates {0,1},{2,3},{4,5},{6,7} (aggregation mode). Non-local
    // entries live only in the "leader" row of each aggregate (0,2,4,6),
    // so the off-block coarse matrices are identical in both modes.
    UniformCoarsening()
        : size{8, 8},
          mat_input{
              size,
              {// diagonal
               {0, 0, 5},
               {1, 1, 5},
               {2, 2, 5},
               {3, 3, 5},
               {4, 4, 5},
               {5, 5, 5},
               {6, 6, 5},
               {7, 7, 5},
               // local connections within partitions
               {0, 1, -1},
               {1, 0, -1},  // rank 0
               {2, 3, -2},
               {3, 2, -2},  // rank 1
               {4, 5, -1},
               {5, 4, -1},  // rank 2
               {6, 7, -1},
               {7, 6, -1},  // rank 2
               {4, 6, -2},
               {6, 4, -2},  // rank 2 cross
                            // non-local connections (only between coarse rows)
               {0, 2, -1},
               {2, 0, -1},  // rank 0 <-> rank 1
               {0, 4, -1},
               {4, 0, -1},  // rank 0 <-> rank 2
               {2, 6, -1},
               {6, 2, -1}}}  // rank 1 <-> rank 2
    {
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 2, 4, 8}));

        mat_input.sort_row_major();
        dist_mat = dist_mtx_type::create(exec, comm);
        dist_mat->read_distributed(mat_input, row_part);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    gko::dim<2> size;
    std::shared_ptr<Partition> row_part;

    gko::matrix_data<value_type, global_index_type> mat_input;

    std::shared_ptr<dist_mtx_type> dist_mat;
};

TYPED_TEST_SUITE(UniformCoarsening, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(UniformCoarsening, CanGenerateFromDistributedMatrix)
{
    using uc = typename TestFixture::uniform_coarsening;
    using value_type = typename TestFixture::value_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    auto uc_factory =
        uc::build().with_coarse_skip(2).with_aggregation(false).on(this->exec);
    auto rank = this->comm.rank();

    // Injection mode. Expected coarse local matrices per rank:
    // Rank 0: coarse rows={0}, local A[0,0]=5 -> [[5]]
    // Rank 1: coarse rows={0}, local A[2,2]=5 -> [[5]]
    // Rank 2: coarse rows={0,2}, local submatrix at {4,6}:
    //         [[5,-2],[-2,5]]
    I<I<value_type>> res_local[] = {{{5}}, {{5}}, {{5, -2}, {-2, 5}}};

    // Expected coarse non-local matrices per rank:
    // Rank 0: 1 coarse row, non-local to coarse global 1(row2), 2(row4)
    //         A[0,2]=-1, A[0,4]=-1 -> [[-1, -1]]
    // Rank 1: 1 coarse row, non-local to coarse global 0(row0), 3(row6)
    //         A[2,0]=-1, A[2,6]=-1 -> [[-1, -1]]
    // Rank 2: 2 coarse rows, non-local to coarse global 0(row0), 1(row2)
    //         Coarse row 0(fine 4): A[4,0]=-1, A[4,2]=0
    //         Coarse row 1(fine 6): A[6,0]=0,  A[6,2]=-1
    //         -> [[-1, 0], [0, -1]]
    I<I<value_type>> res_non_local[] = {
        {{-1, -1}}, {{-1, -1}}, {{-1, 0}, {0, -1}}};

    auto result = uc_factory->generate(this->dist_mat);

    auto coarse = gko::as<dist_mtx_type>(result->get_coarse_op());
    GKO_ASSERT_MTX_NEAR(gko::as<local_matrix_type>(coarse->get_diag_matrix()),
                        res_local[rank], r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(
        gko::as<local_matrix_type>(coarse->get_off_diag_matrix()),
        res_non_local[rank], r<value_type>::value);
}


TYPED_TEST(UniformCoarsening, CanGenerateAggregationFromDistributedMatrix)
{
    using uc = typename TestFixture::uniform_coarsening;
    using value_type = typename TestFixture::value_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    auto uc_factory =
        uc::build().with_coarse_skip(2).with_aggregation(true).on(this->exec);
    auto rank = this->comm.rank();

    // Aggregation mode. R aggregates rows pairwise per rank, P = R^T,
    // local Ac = R*A_local*P.
    // Rank 0: agg of rows {0,1}; A_local=[[5,-1],[-1,5]] -> Ac=[[8]]
    // Rank 1: agg of rows {2,3}; A_local=[[5,-2],[-2,5]] -> Ac=[[6]]
    // Rank 2: aggs {4,5},{6,7}; A_local 4x4 -> Ac=[[8,-2],[-2,8]]
    I<I<value_type>> res_local[] = {{{8}}, {{6}}, {{8, -2}, {-2, 8}}};

    // Off-block entries only exist in aggregate leader rows (0,2,4,6),
    // so the non-local sums per aggregate equal the injection-mode values.
    I<I<value_type>> res_non_local[] = {
        {{-1, -1}}, {{-1, -1}}, {{-1, 0}, {0, -1}}};

    auto result = uc_factory->generate(this->dist_mat);

    auto coarse = gko::as<dist_mtx_type>(result->get_coarse_op());
    GKO_ASSERT_MTX_NEAR(gko::as<local_matrix_type>(coarse->get_diag_matrix()),
                        res_local[rank], r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(
        gko::as<local_matrix_type>(coarse->get_off_diag_matrix()),
        res_non_local[rank], r<value_type>::value);
}
