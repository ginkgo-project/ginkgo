// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/pmis.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


// A_c = R (A P) runs a local csr::spgemm, which has no 64-bit local index
// support on rocSPARSE or on cuSPARSE before CUDA 13. Mirrors the guard in
// test/mpi/distributed/spgemm.cpp; GKO_CUDA_TOOLKIT_VERSION_MAJOR comes from
// this test's CMakeLists, its absence meaning no support.
#if defined(GKO_COMPILING_HIP)
#define GKO_DEVICE_HAS_INT64_SPGEMM 0
#elif defined(GKO_COMPILING_CUDA)
#if defined(GKO_CUDA_TOOLKIT_VERSION_MAJOR) && \
    (GKO_CUDA_TOOLKIT_VERSION_MAJOR >= 13)
#define GKO_DEVICE_HAS_INT64_SPGEMM 1
#else
#define GKO_DEVICE_HAS_INT64_SPGEMM 0
#endif
#else
#define GKO_DEVICE_HAS_INT64_SPGEMM 1
#endif


#if !GKO_DEVICE_HAS_INT64_SPGEMM
#define SKIP_IF_DEVICE_NO_INT64_SPGEMM(local_index_type)                     \
    if (sizeof(local_index_type) > 4) {                                      \
        GTEST_SKIP() << "distributed spgemm with 64-bit local indices is "   \
                        "unsupported on this backend (rocSPARSE has no "     \
                        "64-bit spgemm, cuSPARSE requires CUDA 13)";         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#else
#define SKIP_IF_DEVICE_NO_INT64_SPGEMM(local_index_type)                     \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#endif


template <typename ValueLocalGlobalIndexType>
class Pmis : public CommonMpiTestFixture {
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
    using pmis = gko::multigrid::Pmis<value_type, local_index_type>;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using dense_type = gko::matrix::Dense<value_type>;
    using real_type = gko::remove_complex<value_type>;

    // 1D Laplacian: every off-diagonal is strong at the default threshold, so
    // the weights alone decide the splitting
    template <typename IndexType>
    static gko::matrix_data<value_type, IndexType> laplace_data(int n)
    {
        gko::matrix_data<value_type, IndexType> data{gko::dim<2>(n, n)};
        for (int i = 0; i < n; i++) {
            data.nonzeros.emplace_back(i, i, value_type{2});
            if (i > 0) {
                data.nonzeros.emplace_back(i, i - 1, value_type{-1});
            }
            if (i < n - 1) {
                data.nonzeros.emplace_back(i, i + 1, value_type{-1});
            }
        }
        data.sort_row_major();
        return data;
    }

    // A chain whose strongest coupling crosses a rank boundary of the
    // {0, 4, 8, 12} partition, so rows 3, 4, 7 and 8 have their row maximum in
    // the off-diagonal block. Their weak local neighbour is not a strong
    // dependence, but would become one under a diagonal-block-only maximum.
    template <typename IndexType>
    static gko::matrix_data<value_type, IndexType> cross_boundary_data(int n)
    {
        // coupling(i) joins rows i and i + 1
        const auto coupling = [](int i) {
            return (i == 3 || i == 7) ? value_type{-10} : value_type{-1};
        };
        gko::matrix_data<value_type, IndexType> data{gko::dim<2>(n, n)};
        for (int i = 0; i < n; i++) {
            // strictly diagonally dominant
            auto diag = value_type{1};
            if (i > 0) {
                data.nonzeros.emplace_back(i, i - 1, coupling(i - 1));
                diag += gko::abs(coupling(i - 1));
            }
            if (i < n - 1) {
                data.nonzeros.emplace_back(i, i + 1, coupling(i));
                diag += gko::abs(coupling(i));
            }
            data.nonzeros.emplace_back(i, i, diag);
        }
        data.sort_row_major();
        return data;
    }

    Pmis() : size{12, 12}
    {
        const int n = static_cast<int>(size[0]);
        auto dist_input = laplace_data<global_index_type>(n);
        auto serial_input = laplace_data<local_index_type>(n);

        // contiguous and rank-ordered, which is what makes the distributed
        // coarse numbering match the serial one
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 4, 8, 12}));
        dist_mat = dist_mtx_type::create(exec, comm);
        dist_mat->read_distributed(dist_input, row_part);

        serial_mat = local_matrix_type::create(exec);
        serial_mat->read(serial_input);

        // a larger stencil for the solve and hierarchy tests, in which 12
        // rows are coarsened away in one level. It follows the stencil used
        // by CgWithMg in test/mpi/solver/solver.cpp.
        const int nb = static_cast<int>(big_size[0]);
        auto big_dist = laplace_data<global_index_type>(nb);
        auto big_serial = laplace_data<local_index_type>(nb);
        gko::matrix_data<value_type, global_index_type> rhs{
            gko::dim<2>(big_size[0], 1)};
        gko::matrix_data<value_type, global_index_type> zero{
            gko::dim<2>(big_size[0], 1)};
        for (int i = 0; i < nb; i++) {
            rhs.nonzeros.emplace_back(i, 0, value_type{1});
            zero.nonzeros.emplace_back(i, 0, value_type{0});
        }
        big_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 32, 64, 96}));
        laplace_mat = dist_mtx_type::create(exec, comm);
        laplace_mat->read_distributed(big_dist, big_part);
        serial_laplace = local_matrix_type::create(exec);
        serial_laplace->read(big_serial);

        const auto nloc =
            static_cast<gko::size_type>(big_part->get_part_size(comm.rank()));
        b = dist_vec_type::create(exec, comm, gko::dim<2>{big_size[0], 1},
                                  gko::dim<2>{nloc, 1});
        b->read_distributed(rhs, big_part);
        x = dist_vec_type::create(exec, comm, gko::dim<2>{big_size[0], 1},
                                  gko::dim<2>{nloc, 1});
        x->read_distributed(zero, big_part);
    }

    // identical multigrid settings for the distributed and the serial run
    template <typename MgLevelFactory>
    static auto mg_settings(MgLevelFactory level)
    {
        return gko::solver::Multigrid::build()
            .with_max_levels(3u)
            .with_min_coarse_rows(8u)
            .with_mg_level(level)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u));
    }

    // No smoother: this compares coarse operators, and Multigrid's default
    // Jacobi uses int32 indices, so generating it against a serial
    // Csr<value_type, int64> throws. This is a pre-existing limitation that
    // also reproduces with Pgm.
    template <typename MgLevelFactory>
    auto mg_hierarchy_settings(MgLevelFactory level)
    {
        return gko::solver::Multigrid::build()
            .with_max_levels(3u)
            .with_min_coarse_rows(8u)
            .with_mg_level(level)
            .with_pre_smoother(nullptr)
            .with_mid_smoother(nullptr)
            .with_coarsest_solver(
                gko::solver::Cg<value_type>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u))
                    .on(exec))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u));
    }

    // For a contiguous, rank-ordered partition the coarse operator has to
    // agree with the serial one entry for entry on the rows this rank owns:
    // the draw is a pure function of the global index and the measure is
    // accumulated globally, so both runs pick the same C-points.
    void assert_coarse_matches_serial(const gko::LinOp* dist_coarse_op,
                                      const gko::LinOp* serial_coarse_op)
    {
        using entry = std::tuple<gko::size_type, gko::size_type, value_type>;
        auto dist_coarse = gko::as<dist_mtx_type>(dist_coarse_op);
        auto serial_coarse = gko::as<local_matrix_type>(serial_coarse_op);
        ASSERT_EQ(dist_coarse->get_size()[0], serial_coarse->get_size()[0]);

        // write() emits only the rows this rank owns, with global indices
        gko::matrix_data<value_type, global_index_type> dist_data;
        dist_coarse->write(dist_data);
        gko::matrix_data<value_type, local_index_type> serial_data;
        serial_coarse->write(serial_data);

        auto n_local =
            gko::as<local_matrix_type>(dist_coarse->get_diag_matrix())
                ->get_size()[0];
        gko::size_type scan = 0;
        comm.scan(ref, &n_local, &scan, 1, MPI_SUM);
        const auto offset = scan - n_local;

        std::vector<entry> mine;
        std::vector<entry> theirs;
        for (const auto& e : dist_data.nonzeros) {
            mine.emplace_back(static_cast<gko::size_type>(e.row),
                              static_cast<gko::size_type>(e.column), e.value);
        }
        for (const auto& e : serial_data.nonzeros) {
            const auto r = static_cast<gko::size_type>(e.row);
            if (r >= offset && r < offset + n_local) {
                theirs.emplace_back(r, static_cast<gko::size_type>(e.column),
                                    e.value);
            }
        }
        auto by_position = [](const entry& a, const entry& b) {
            return std::tie(std::get<0>(a), std::get<1>(a)) <
                   std::tie(std::get<0>(b), std::get<1>(b));
        };
        std::sort(mine.begin(), mine.end(), by_position);
        std::sort(theirs.begin(), theirs.end(), by_position);

        // a rank may own no rows, but across all ranks at least one entry
        // has to be compared
        gko::size_type local_entries = mine.size();
        gko::size_type global_entries = 0;
        comm.all_reduce(ref, &local_entries, &global_entries, 1, MPI_SUM);
        ASSERT_GT(global_entries, 0);
        ASSERT_EQ(mine.size(), theirs.size());
        for (gko::size_type i = 0; i < mine.size(); i++) {
            ASSERT_EQ(std::get<0>(mine[i]), std::get<0>(theirs[i]));
            ASSERT_EQ(std::get<1>(mine[i]), std::get<1>(theirs[i]));
            ASSERT_NEAR(gko::abs(std::get<2>(mine[i]) - std::get<2>(theirs[i])),
                        0.0, r<value_type>::value);
        }
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    gko::dim<2> size;
    std::shared_ptr<Partition> row_part;
    std::shared_ptr<dist_mtx_type> dist_mat;
    std::shared_ptr<local_matrix_type> serial_mat;

    gko::dim<2> big_size{96, 96};
    std::shared_ptr<Partition> big_part;
    std::shared_ptr<dist_mtx_type> laplace_mat;
    std::shared_ptr<local_matrix_type> serial_laplace;
    std::shared_ptr<dist_vec_type> b;
    std::shared_ptr<dist_vec_type> x;
};

TYPED_TEST_SUITE(Pmis, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(Pmis, CanGenerateFromDistributedMatrix)
{
    SKIP_IF_DEVICE_NO_INT64_SPGEMM(typename TestFixture::local_index_type);
    using pmis = typename TestFixture::pmis;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;

    auto level = pmis::build().on(this->exec)->generate(this->dist_mat);

    auto prolong = gko::as<dist_mtx_type>(level->get_prolong_op());
    auto restrict_op = gko::as<dist_mtx_type>(level->get_restrict_op());
    auto coarse = gko::as<dist_mtx_type>(level->get_coarse_op());
    const auto n_coarse = coarse->get_size()[0];
    ASSERT_GT(n_coarse, 0);
    ASSERT_LT(n_coarse, this->size[0]);
    ASSERT_EQ(prolong->get_size(), gko::dim<2>(this->size[0], n_coarse));
    ASSERT_EQ(restrict_op->get_size(), gko::dim<2>(n_coarse, this->size[0]));
    ASSERT_EQ(coarse->get_size(), gko::dim<2>(n_coarse, n_coarse));
}


TYPED_TEST(Pmis, EveryLocalRowOfProlongationIsNonEmpty)
{
    SKIP_IF_DEVICE_NO_INT64_SPGEMM(typename TestFixture::local_index_type);
    using pmis = typename TestFixture::pmis;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;

    auto level = pmis::build().on(this->exec)->generate(this->dist_mat);

    auto p = gko::as<dist_mtx_type>(level->get_prolong_op());
    auto host_diag =
        gko::clone(this->ref, gko::as<local_matrix_type>(p->get_diag_matrix()));
    auto host_off = gko::clone(
        this->ref, gko::as<local_matrix_type>(p->get_off_diag_matrix()));
    // a coarse row carries its identity entry, a fine row interpolates from at
    // least one strong C-neighbour, possibly on another rank
    for (gko::size_type row = 0; row < host_diag->get_size()[0]; row++) {
        const auto n_diag = host_diag->get_const_row_ptrs()[row + 1] -
                            host_diag->get_const_row_ptrs()[row];
        const auto n_off = host_off->get_const_row_ptrs()[row + 1] -
                           host_off->get_const_row_ptrs()[row];
        ASSERT_GT(n_diag + n_off, 0)
            << "row " << row << " has no coarse parent";
    }
}


TYPED_TEST(Pmis, DistributedCoarseOperatorMatchesSerial)
{
    SKIP_IF_DEVICE_NO_INT64_SPGEMM(typename TestFixture::local_index_type);
    using pmis = typename TestFixture::pmis;

    auto dist_level = pmis::build().on(this->exec)->generate(this->dist_mat);
    auto serial_level =
        pmis::build().on(this->exec)->generate(this->serial_mat);

    this->assert_coarse_matches_serial(dist_level->get_coarse_op().get(),
                                       serial_level->get_coarse_op().get());
}


TYPED_TEST(Pmis, RowMaximumSpansBothBlocks)
{
    SKIP_IF_DEVICE_NO_INT64_SPGEMM(typename TestFixture::local_index_type);
    using value_type = typename TestFixture::value_type;
    using global_index_type = typename TestFixture::global_index_type;
    using local_index_type = typename TestFixture::local_index_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using pmis = typename TestFixture::pmis;
    const int n = static_cast<int>(this->size[0]);
    // rows 3, 4, 7 and 8 hold their largest off-diagonal entry in the
    // off-diagonal block, so a truncated row maximum would call their weak
    // local neighbour strong and coarsen differently
    auto cross_dist = gko::share(dist_mtx_type::create(this->exec, this->comm));
    cross_dist->read_distributed(
        TestFixture::template cross_boundary_data<global_index_type>(n),
        this->row_part);
    auto cross_serial = gko::share(local_matrix_type::create(this->exec));
    cross_serial->read(
        TestFixture::template cross_boundary_data<local_index_type>(n));

    auto dist_level = pmis::build().on(this->exec)->generate(cross_dist);
    auto serial_level = pmis::build().on(this->exec)->generate(cross_serial);

    this->assert_coarse_matches_serial(dist_level->get_coarse_op().get(),
                                       serial_level->get_coarse_op().get());
}


TYPED_TEST(Pmis, WorksWithAnEmptyLocalRange)
{
    SKIP_IF_DEVICE_NO_INT64_SPGEMM(typename TestFixture::local_index_type);
    using global_index_type = typename TestFixture::global_index_type;
    using local_index_type = typename TestFixture::local_index_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using Partition = typename TestFixture::Partition;
    using pmis = typename TestFixture::pmis;
    const int n = static_cast<int>(this->size[0]);
    // rank 0 owns no rows: zero-length prefix sums, an empty halo and an
    // empty share of the coarse partition, while still participating in every
    // collective
    auto skewed = gko::share(Partition::build_from_contiguous(
        this->exec, gko::array<global_index_type>(
                        this->exec, I<global_index_type>{0, 0, 6, 12})));
    auto skewed_dist =
        gko::share(dist_mtx_type::create(this->exec, this->comm));
    skewed_dist->read_distributed(
        TestFixture::template laplace_data<global_index_type>(n), skewed);
    auto serial = gko::share(local_matrix_type::create(this->exec));
    serial->read(TestFixture::template laplace_data<local_index_type>(n));

    auto dist_level = pmis::build().on(this->exec)->generate(skewed_dist);
    auto serial_level = pmis::build().on(this->exec)->generate(serial);

    // still contiguous and rank-ordered, so the numbering matches the serial
    // one
    this->assert_coarse_matches_serial(dist_level->get_coarse_op().get(),
                                       serial_level->get_coarse_op().get());
    // P keeps the share of the empty rank empty while spanning all rows
    auto p = gko::as<dist_mtx_type>(dist_level->get_prolong_op());
    auto p_diag = gko::as<local_matrix_type>(p->get_diag_matrix());
    ASSERT_EQ(p->get_size()[0], this->size[0]);
    ASSERT_EQ(p_diag->get_size()[0], this->comm.rank() == 0 ? 0u : 6u);
}


TYPED_TEST(Pmis, DistributedMultigridPreconditionedCgConverges)
{
    SKIP_IF_DEVICE_NO_INT64_SPGEMM(typename TestFixture::local_index_type);
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using dense_type = typename TestFixture::dense_type;
    using pmis = typename TestFixture::pmis;
    using cg = gko::solver::Cg<value_type>;
    const auto tol = r<value_type>::value * real_type{1e3};

    auto solver =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(100u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_baseline(gko::stop::mode::rhs_norm)
                               .with_reduction_factor(tol))
            .with_preconditioner(gko::share(
                TestFixture::mg_settings(pmis::build()).on(this->exec)))
            .on(this->exec)
            ->generate(this->laplace_mat);
    auto residual = gko::clone(this->b);
    auto one = gko::initialize<dense_type>({value_type{1}}, this->exec);
    auto neg_one = gko::initialize<dense_type>({value_type{-1}}, this->exec);
    auto res_norm = gko::initialize<gko::matrix::Dense<real_type>>(
        {real_type{0}}, this->exec);
    auto rhs_norm = gko::initialize<gko::matrix::Dense<real_type>>(
        {real_type{0}}, this->exec);

    solver->apply(this->b, this->x);

    // residual = b - A x, measured relative to ||b||
    this->laplace_mat->apply(neg_one, this->x, one, residual);
    residual->compute_norm2(res_norm);
    this->b->compute_norm2(rhs_norm);
    const auto rel =
        this->exec->copy_val_to_host(res_norm->get_const_values()) /
        this->exec->copy_val_to_host(rhs_norm->get_const_values());
    ASSERT_LT(rel, tol * real_type{10});
}


TYPED_TEST(Pmis, HierarchyIsIndependentOfRankCount)
{
    SKIP_IF_DEVICE_NO_INT64_SPGEMM(typename TestFixture::local_index_type);
    using pmis = typename TestFixture::pmis;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;

    auto dist_mg = this->mg_hierarchy_settings(pmis::build())
                       .on(this->exec)
                       ->generate(this->laplace_mat);
    auto serial_mg = this->mg_hierarchy_settings(pmis::build())
                         .on(this->exec)
                         ->generate(this->serial_laplace);

    auto dist_levels = dist_mg->get_mg_level_list();
    auto serial_levels = serial_mg->get_mg_level_list();
    // a hierarchy that stopped at level 0 would pass vacuously
    ASSERT_GE(dist_levels.size(), 2);
    ASSERT_EQ(dist_levels.size(), serial_levels.size());
    for (gko::size_type lvl = 0; lvl < dist_levels.size(); lvl++) {
        auto d_coarse =
            gko::as<dist_mtx_type>(dist_levels.at(lvl)->get_coarse_op());
        auto s_coarse =
            gko::as<local_matrix_type>(serial_levels.at(lvl)->get_coarse_op());
        // same number of coarse points at every level
        ASSERT_EQ(d_coarse->get_size(), s_coarse->get_size())
            << "level " << lvl;
        // the same global nnz, so the operator complexity matches as well
        gko::size_type local_nnz =
            gko::as<local_matrix_type>(d_coarse->get_diag_matrix())
                ->get_num_stored_elements() +
            gko::as<local_matrix_type>(d_coarse->get_off_diag_matrix())
                ->get_num_stored_elements();
        gko::size_type global_nnz = 0;
        this->comm.all_reduce(this->ref, &local_nnz, &global_nnz, 1, MPI_SUM);
        ASSERT_EQ(global_nnz, s_coarse->get_num_stored_elements())
            << "level " << lvl;
    }
}
