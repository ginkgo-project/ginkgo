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


template <typename ValueLocalGlobalIndexType>
class UniformCoarseningOffDiagAgg : public CommonMpiTestFixture {
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
    using uniform_coarsening =
        gko::multigrid::UniformCoarsening<value_type, local_index_type>;

    UniformCoarseningOffDiagAgg()
        : size{8, 8},
          mat_input{size,
                    {// diagonal
                     {0, 0, 5},
                     {1, 1, 5},
                     {2, 2, 5},
                     {3, 3, 5},
                     {4, 4, 5},
                     {5, 5, 5},
                     {6, 6, 5},
                     {7, 7, 5},
                     // local
                     {0, 1, -1},
                     {1, 0, -1},
                     {2, 3, -2},
                     {3, 2, -2},
                     {4, 5, -1},
                     {5, 4, -1},
                     {6, 7, -1},
                     {7, 6, -1},
                     {4, 6, -2},
                     {6, 4, -2},
                     // rank0 <-> rank1, both rows of each agg (row-sum +
                     // column-collapse: cols 2 and 3 are both in rank1 agg 0)
                     {0, 2, -1},
                     {2, 0, -1},
                     {1, 3, -1},
                     {3, 1, -1},
                     // rank0 <-> rank2 agg 0 (rows 4,5)
                     {0, 4, -1},
                     {4, 0, -1},
                     {1, 5, -1},
                     {5, 1, -1},
                     // rank1 <-> rank2 agg 1 (rows 6,7)
                     {2, 6, -1},
                     {6, 2, -1},
                     {3, 7, -1},
                     {7, 3, -1}}}
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

TYPED_TEST_SUITE(UniformCoarseningOffDiagAgg,
                 gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(UniformCoarseningOffDiagAgg, AggregatesOffDiagonalEntries)
{
    using uc = typename TestFixture::uniform_coarsening;
    using value_type = typename TestFixture::value_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    auto uc_factory =
        uc::build().with_coarse_skip(2).with_aggregation(true).on(this->exec);
    auto rank = this->comm.rank();

    // Local Ac is unchanged by the extra non-local entries (matches
    // UniformCoarsening.CanGenerateAggregationFromDistributedMatrix).
    I<I<value_type>> res_local[] = {{{8}}, {{6}}, {{8, -2}, {-2, 8}}};

    // Off-diagonal aggregation:
    // - Rank 0 coarse row 0 sums over both local rows. Cols collapse:
    //   rank1 rows {2,3} -> one coarse col; rank2 rows {4,5} -> one coarse
    //   col. Sums: A[0,2]+A[1,3]=-2 and A[0,4]+A[1,5]=-2.
    // - Rank 1 symmetric.
    // - Rank 2 has 2 coarse rows: coarse row 0 (rows 4,5) only touches
    //   rank 0 (sum -2); coarse row 1 (rows 6,7) only touches rank 1
    //   (sum -2). Off-diagonal sees 2 coarse non-local cols (rank0 agg 0,
    //   rank1 agg 0).
    I<I<value_type>> res_non_local[] = {
        {{-2, -2}}, {{-2, -2}}, {{-2, 0}, {0, -2}}};

    auto result = uc_factory->generate(this->dist_mat);

    auto coarse = gko::as<dist_mtx_type>(result->get_coarse_op());
    GKO_ASSERT_MTX_NEAR(gko::as<local_matrix_type>(coarse->get_diag_matrix()),
                        res_local[rank], r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(
        gko::as<local_matrix_type>(coarse->get_off_diag_matrix()),
        res_non_local[rank], r<value_type>::value);
}


TYPED_TEST(UniformCoarseningOffDiagAgg, InjectionDropsNonLeaderEntries)
{
    using uc = typename TestFixture::uniform_coarsening;
    using value_type = typename TestFixture::value_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    auto uc_factory =
        uc::build().with_coarse_skip(2).with_aggregation(false).on(this->exec);
    auto rank = this->comm.rank();

    // Injection keeps only leader rows (0,2,4,6). Local Ac is the
    // row-selection submatrix; entries from non-leader rows vanish.
    I<I<value_type>> res_local[] = {{{5}}, {{5}}, {{5, -2}, {-2, 5}}};

    // Off-diagonals: only entries from leader rows that also point to
    // leader rows on the remote side survive. The fixture's non-leader-row
    // non-local entries must be dropped.
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


// 10-row tri-diagonal matrix (plus extra cross-rank edges 0<->3) with
// partition {0,3,7,10}. With coarse_skip=3, rank 1 has 4 local rows which
// is not divisible by 3, producing 2 coarse rows on rank 1 (0..2 -> 0,
// 3 -> 1). Ranks 0 and 2 have 3 local rows each (one coarse row).
template <typename ValueLocalGlobalIndexType>
class UniformCoarseningNonDivisible : public CommonMpiTestFixture {
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
    using uniform_coarsening =
        gko::multigrid::UniformCoarsening<value_type, local_index_type>;

    UniformCoarseningNonDivisible()
        : size{10, 10},
          mat_input{size,
                    {// diagonal
                     {0, 0, 5},
                     {1, 1, 5},
                     {2, 2, 5},
                     {3, 3, 5},
                     {4, 4, 5},
                     {5, 5, 5},
                     {6, 6, 5},
                     {7, 7, 5},
                     {8, 8, 5},
                     {9, 9, 5},
                     // tri-diag couplings (-1 on each off-diagonal)
                     {0, 1, -1},
                     {1, 0, -1},
                     {1, 2, -1},
                     {2, 1, -1},
                     {2, 3, -1},
                     {3, 2, -1},
                     {3, 4, -1},
                     {4, 3, -1},
                     {4, 5, -1},
                     {5, 4, -1},
                     {5, 6, -1},
                     {6, 5, -1},
                     {6, 7, -1},
                     {7, 6, -1},
                     {7, 8, -1},
                     {8, 7, -1},
                     {8, 9, -1},
                     {9, 8, -1},
                     // extra cross-rank edge: rank 0 <-> rank 1 leader pair
                     {0, 3, -1},
                     {3, 0, -1}}}
    {
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 3, 7, 10}));

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

TYPED_TEST_SUITE(UniformCoarseningNonDivisible,
                 gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(UniformCoarseningNonDivisible, AggregationWithCoarseSkipThree)
{
    using uc = typename TestFixture::uniform_coarsening;
    using value_type = typename TestFixture::value_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    auto uc_factory =
        uc::build().with_coarse_skip(3).with_aggregation(true).on(this->exec);
    auto rank = this->comm.rank();

    // Aggregation, coarse_skip=3:
    // Rank 0 (3 rows): all -> coarse 0. R*A*R^T on tri-diag(3x3) = [[11]].
    // Rank 1 (4 rows): rows 0..2 -> coarse 0, row 3 -> coarse 1.
    //   Ac = [[11,-1],[-1,5]].
    // Rank 2 (3 rows): all -> coarse 0. Ac = [[11]].
    I<I<value_type>> res_local[] = {{{11}}, {{11, -1}, {-1, 5}}, {{11}}};

    // Off-diagonals (aggregated):
    // Rank 0 coarse 0 -> rank 1 coarse 0: edges (0,3) + (2,3) = -2.
    // Rank 1 coarse 0 -> rank 0 coarse 0: edges (3,0) + (3,2) = -2.
    // Rank 1 coarse 1 -> rank 2 coarse 0: edge (6,7) = -1.
    // Rank 2 coarse 0 -> rank 1 coarse 1: edge (7,6) = -1.
    I<I<value_type>> res_non_local[] = {{{-2}}, {{-2, 0}, {0, -1}}, {{-1}}};

    auto result = uc_factory->generate(this->dist_mat);

    auto coarse = gko::as<dist_mtx_type>(result->get_coarse_op());
    GKO_ASSERT_MTX_NEAR(gko::as<local_matrix_type>(coarse->get_diag_matrix()),
                        res_local[rank], r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(
        gko::as<local_matrix_type>(coarse->get_off_diag_matrix()),
        res_non_local[rank], r<value_type>::value);
}


TYPED_TEST(UniformCoarseningNonDivisible, InjectionWithCoarseSkipThree)
{
    using uc = typename TestFixture::uniform_coarsening;
    using value_type = typename TestFixture::value_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    auto uc_factory =
        uc::build().with_coarse_skip(3).with_aggregation(false).on(this->exec);
    auto rank = this->comm.rank();

    // Injection, coarse_skip=3:
    // Rank 0 leaders {local 0 = global 0}. Ac = [[5]].
    // Rank 1 leaders {local 0 = global 3, local 3 = global 6}. Diagonal
    //   submatrix has no leader-leader local edges, so Ac = [[5,0],[0,5]].
    // Rank 2 leaders {local 0 = global 7}. Ac = [[5]].
    I<I<value_type>> res_local[] = {{{5}}, {{5, 0}, {0, 5}}, {{5}}};

    // Off-diagonals: only leader-leader cross-rank edges survive.
    // Rank 0 leader 0 -> rank 1 leader 0 via edge (0,3) -> -1.
    // Rank 1 leader 0 -> rank 0 leader 0 via edge (3,0) -> -1.
    //   (edge (3,2) drops: col 2 is not a leader.)
    // Rank 1 leader 1 -> rank 2 leader 0 via edge (6,7) -> -1.
    // Rank 2 leader 0 -> rank 1 leader 1 via edge (7,6) -> -1.
    I<I<value_type>> res_non_local[] = {{{-1}}, {{-1, 0}, {0, -1}}, {{-1}}};

    auto result = uc_factory->generate(this->dist_mat);

    auto coarse = gko::as<dist_mtx_type>(result->get_coarse_op());
    GKO_ASSERT_MTX_NEAR(gko::as<local_matrix_type>(coarse->get_diag_matrix()),
                        res_local[rank], r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(
        gko::as<local_matrix_type>(coarse->get_off_diag_matrix()),
        res_non_local[rank], r<value_type>::value);
}
