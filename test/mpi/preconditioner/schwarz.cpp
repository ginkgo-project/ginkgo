// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <memory>
#include <random>


#include <mpi.h>


#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/distributed/preconditioner/schwarz_ovlp.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/mpi/executor.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename ValueLocalGlobalIndexType>
class SchwarzPreconditioner : public CommonMpiTestFixture {
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
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using local_vec_type = gko::matrix::Dense<value_type>;
    using dist_prec_type =
        gko::experimental::distributed::preconditioner::Schwarz<
            value_type, local_index_type, global_index_type>;
    using solver_type = gko::solver::Bicgstab<value_type>;
    using local_prec_type =
        gko::preconditioner::Jacobi<value_type, local_index_type>;
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;
    using non_dist_matrix_type =
        gko::matrix::Csr<value_type, global_index_type>;
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;


    SchwarzPreconditioner()
        : CommonMpiTestFixture(),
          size{8, 8},
          mat_input{size,
                    {{0, 0, 2},  {0, 1, -1}, {1, 0, -1}, {1, 1, 2},  {1, 2, -1},
                     {2, 1, -1}, {2, 2, 2},  {2, 3, -1}, {3, 2, -1}, {3, 3, 2},
                     {3, 4, -1}, {4, 3, -1}, {4, 4, 2},  {4, 5, -1}, {5, 4, -1},
                     {5, 5, 2},  {5, 6, -1}, {6, 5, -1}, {6, 6, 2},  {6, 7, -1},
                     {7, 6, -1}, {7, 7, 2}}}
    {
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 2, 4, 8}));

        dist_mat = dist_mtx_type::create(exec, comm);
        dist_mat->read_distributed(mat_input, row_part);
        non_dist_mat = non_dist_matrix_type::create(exec);
        non_dist_mat->read(mat_input);

        auto nrhs = 1;
        auto global_size =
            gko::dim<2>{size[0], static_cast<gko::size_type>(nrhs)};
        auto local_size = gko::dim<2>{
            static_cast<gko::size_type>(row_part->get_part_size(comm.rank())),
            static_cast<gko::size_type>(nrhs)};
        auto dist_result =
            dist_vec_type::create(ref, comm, global_size, local_size, nrhs);
        dist_b = gko::share(gko::clone(exec, dist_result));
        dist_b->fill(-gko::one<value_type>());
        dist_x = gko::share(gko::clone(exec, dist_result));
        dist_x->fill(gko::zero<value_type>());
        auto non_dist_result = local_vec_type::create(ref, global_size, nrhs);
        non_dist_result->fill(-gko::one<value_type>());
        non_dist_b = gko::share(gko::clone(exec, non_dist_result));
        non_dist_x = gko::share(gko::clone(exec, non_dist_result));
        non_dist_x->fill(gko::zero<value_type>());

        local_solver_factory =
            local_prec_type::build().with_max_block_size(1u).on(exec);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    gko::dim<2> size;
    std::shared_ptr<Partition> row_part;

    gko::matrix_data<value_type, global_index_type> mat_input;

    std::shared_ptr<dist_mtx_type> dist_mat;
    std::shared_ptr<dist_vec_type> dist_b;
    std::shared_ptr<dist_vec_type> dist_x;
    std::shared_ptr<non_dist_matrix_type> non_dist_mat;
    std::shared_ptr<local_vec_type> non_dist_b;
    std::shared_ptr<local_vec_type> non_dist_x;
    std::shared_ptr<gko::LinOpFactory> non_dist_solver_factory;
    std::shared_ptr<gko::LinOpFactory> dist_solver_factory;
    std::shared_ptr<gko::LinOpFactory> local_solver_factory;

    void assert_equal_to_non_distributed_vector(
        std::shared_ptr<dist_vec_type> dist_vec,
        std::shared_ptr<local_vec_type> local_vec)
    {
        auto host_row_part = row_part->clone(ref);
        auto l_dist_vec = dist_vec->get_local_vector();
        auto vec_view = local_vec_type::create_const(
            exec, l_dist_vec->get_size(),
            gko::array<value_type>::const_view(
                exec, l_dist_vec->get_size()[0],
                local_vec->get_const_values() +
                    host_row_part->get_range_bounds()[comm.rank()]),
            l_dist_vec->get_size()[1]);
        GKO_ASSERT_MTX_NEAR(l_dist_vec, vec_view.get(), r<value_type>::value);
    }
};

TYPED_TEST_SUITE(SchwarzPreconditioner, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);

TYPED_TEST(SchwarzPreconditioner, GenerateFailsIfInvalidState)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using local_prec_type =
        gko::preconditioner::Jacobi<value_type, local_index_type>;
    using prec = typename TestFixture::dist_prec_type;

    auto local_solver = gko::share(local_prec_type::build()
                                       .with_max_block_size(1u)
                                       .on(this->exec)
                                       ->generate(this->non_dist_mat));
    auto schwarz = prec::build()
                       .with_local_solver(this->local_solver_factory)
                       .with_generated_local_solver(local_solver)
                       .on(this->exec);

    ASSERT_THROW(schwarz->generate(this->dist_mat), gko::InvalidStateError);
}


TYPED_TEST(SchwarzPreconditioner, GenerateFailsIfNoSolverProvided)
{
    using prec = typename TestFixture::dist_prec_type;
    auto schwarz_no_solver = prec::build().on(this->exec);

    ASSERT_THROW(schwarz_no_solver->generate(this->dist_mat),
                 gko::InvalidStateError);
}


TYPED_TEST(SchwarzPreconditioner, CanApplyPreconditionedSolver)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    using cg = typename TestFixture::solver_type;
    using prec = typename TestFixture::dist_prec_type;
    constexpr double tolerance = 1e-20;
    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(200u).on(this->exec));
    auto tol_stop = gko::share(
        gko::stop::ResidualNorm<value_type>::build()
            .with_reduction_factor(
                static_cast<gko::remove_complex<value_type>>(tolerance))
            .on(this->exec));
    this->dist_solver_factory =
        cg::build()
            .with_preconditioner(
                prec::build()
                    .with_local_solver(this->local_solver_factory)
                    .on(this->exec))
            .with_criteria(iter_stop, tol_stop)
            .on(this->exec);
    auto dist_solver = this->dist_solver_factory->generate(this->dist_mat);
    this->non_dist_solver_factory =
        cg::build()
            .with_preconditioner(this->local_solver_factory)
            .with_criteria(iter_stop, tol_stop)
            .on(this->exec);
    auto non_dist_solver =
        this->non_dist_solver_factory->generate(this->non_dist_mat);

    dist_solver->apply(this->dist_b.get(), this->dist_x.get());
    non_dist_solver->apply(this->non_dist_b.get(), this->non_dist_x.get());

    this->assert_equal_to_non_distributed_vector(this->dist_x,
                                                 this->non_dist_x);
}


TYPED_TEST(SchwarzPreconditioner, CanApplyPreconditionedSolverWithPregenSolver)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using local_prec_type =
        gko::preconditioner::Jacobi<value_type, local_index_type>;
    using csr = typename TestFixture::local_matrix_type;
    using cg = typename TestFixture::solver_type;
    using prec = typename TestFixture::dist_prec_type;

    auto local_solver =
        gko::share(local_prec_type::build()
                       .with_max_block_size(1u)
                       .on(this->exec)
                       ->generate(this->dist_mat->get_local_matrix()));
    auto precond = prec::build()
                       .with_local_solver(this->local_solver_factory)
                       .on(this->exec)
                       ->generate(this->dist_mat);
    auto precond_pregen = prec::build()
                              .with_generated_local_solver(local_solver)
                              .on(this->exec)
                              ->generate(this->dist_mat);
    auto dist_x = gko::share(this->dist_x->clone());
    auto dist_x_pregen = gko::share(this->dist_x->clone());

    precond->apply(this->dist_b.get(), dist_x.get());
    precond_pregen->apply(this->dist_b.get(), dist_x_pregen.get());

    GKO_ASSERT_MTX_NEAR(dist_x->get_local_vector(),
                        dist_x_pregen->get_local_vector(),
                        r<value_type>::value);
}


TYPED_TEST(SchwarzPreconditioner, CanApplyPreconditioner)
{
    using value_type = typename TestFixture::value_type;
    using prec = typename TestFixture::dist_prec_type;

    auto precond_factory = prec::build()
                               .with_local_solver(this->local_solver_factory)
                               .on(this->exec);
    auto local_precond =
        this->local_solver_factory->generate(this->non_dist_mat);
    auto precond = precond_factory->generate(this->dist_mat);

    precond->apply(this->dist_b.get(), this->dist_x.get());
    local_precond->apply(this->non_dist_b.get(), this->non_dist_x.get());

    this->assert_equal_to_non_distributed_vector(this->dist_x,
                                                 this->non_dist_x);
}


TYPED_TEST(SchwarzPreconditioner, CanAdvancedApplyPreconditioner)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    using vec = typename TestFixture::local_vec_type;
    using cg = typename TestFixture::solver_type;
    using prec = typename TestFixture::dist_prec_type;

    auto precond_factory = prec::build()
                               .with_local_solver(this->local_solver_factory)
                               .on(this->exec);
    auto local_precond =
        this->local_solver_factory->generate(this->non_dist_mat);
    auto precond = precond_factory->generate(this->dist_mat);
    auto one = gko::initialize<vec>({gko::one<value_type>()}, this->exec);
    auto two = gko::initialize<vec>({2.0}, this->exec);

    precond->apply(two.get(), this->dist_b.get(), one.get(),
                   this->dist_x.get());
    local_precond->apply(two.get(), this->non_dist_b.get(), one.get(),
                         this->non_dist_x.get());

    this->assert_equal_to_non_distributed_vector(this->dist_x,
                                                 this->non_dist_x);
}


class Overlap : public CommonMpiTestFixture {
public:
    using value_type = double;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using map_type =
        gko::experimental::distributed::index_map<local_index_type,
                                                  global_index_type>;
    using csr_mtx_type = gko::matrix::Csr<value_type, global_index_type>;
    using dist_mtx_type =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;
    using dense_vec_type = gko::matrix::Dense<value_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;

    Overlap()
    {
        part = part_type::build_from_global_size_uniform(exec, 3, 6);

        dist_mat = dist_mtx_type::create(exec, comm);

        gko::matrix_data<value_type, global_index_type> mat_input{
            {{2, -1, 0, 0, 0, 0},
             {-1, 2, -1, 0, 0, 0},
             {0, -1, 2, -1, 0, 0},
             {0, 0, -1, 2, -1, 0},
             {0, 0, 0, -1, 2, -1},
             {0, 0, 0, 0, -1, 2}}};
        imap = dist_mat->read_distributed(mat_input, this->part);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }


    gko::dim<2> size;

    std::shared_ptr<part_type> part;

    std::unique_ptr<dist_mtx_type> dist_mat;
    map_type imap;
};


TEST_F(Overlap, CanGetNonLocalRows)
{
    auto result = gko::experimental::distributed::preconditioner::get_recv_rows(
        dist_mat.get(), imap);
    std::sort(result.begin(), result.end());

    auto rank = comm.rank();
    std::vector<matrix_data::nonzero_type> expected[] = {
        {{2, 1, -1}, {2, 2, 2}, {2, 3, -1}},
        {{1, 0, -1}, {1, 1, 2}, {1, 2, -1}, {4, 3, -1}, {4, 4, 2}, {4, 5, -1}},
        {{3, 2, -1}, {3, 3, 2}, {3, 4, -1}}};
    std::sort(expected[rank].begin(), expected[rank].end());
    EXPECT_EQ(result.size(), expected[rank].size());
    for (std::size_t i = 0; i < std::max(result.size(), expected[rank].size());
         ++i) {
        auto& a = result[std::min(i, result.size() - 1)];
        auto& b = expected[rank][std::min(i, expected[rank].size() - 1)];

        EXPECT_EQ(a, b);
    }
}


TEST_F(Overlap, CanFilterNonRelevant)
{
    auto recv_rows =
        gko::experimental::distributed::preconditioner::get_recv_rows(
            dist_mat.get(), imap);

    auto result =
        gko::experimental::distributed::preconditioner::filter_non_relevant(
            recv_rows, imap);
    std::sort(result.begin(), result.end());

    auto rank = comm.rank();
    std::vector<matrix_data::nonzero_type> expected[] = {
        {{2, 1, -1}, {2, 2, 2}},
        {{1, 1, 2}, {1, 2, -1}, {4, 3, -1}, {4, 4, 2}},
        {{3, 3, 2}, {3, 4, -1}}};
    std::sort(expected[rank].begin(), expected[rank].end());
    EXPECT_EQ(result.size(), expected[rank].size());
    for (std::size_t i = 0; i < std::max(result.size(), expected[rank].size());
         ++i) {
        auto& a = result[std::min(i, result.size() - 1)];
        auto& b = expected[rank][std::min(i, expected[rank].size() - 1)];

        EXPECT_EQ(a, b);
    }
}


TEST_F(Overlap, CanCombineMatrices)
{
    auto recv_rows =
        gko::experimental::distributed::preconditioner::get_recv_rows(
            dist_mat.get(), imap);
    recv_rows =
        gko::experimental::distributed::preconditioner::filter_non_relevant(
            recv_rows, imap);

    auto result =
        gko::experimental::distributed::preconditioner::combine_overlap(
            dist_mat.get(), recv_rows, imap);
    std::sort(result.nonzeros.begin(), result.nonzeros.end());

    auto rank = comm.rank();
    matrix_data expected[] = {{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}},
                              {
                                  {2, -1, -1, 0},
                                  {-1, 2, 0, -1},
                                  {-1, 0, 2, 0},
                                  {0, -1, 0, 2},
                              },
                              {

                                  {2, -1, -1}, {-1, 2, 0}, {-1, 0, 2}}};
    auto result_op = dense_vec_type::create(exec);
    result_op->read(result);
    auto expected_op = dense_vec_type::create(exec);
    expected_op->read(expected[rank]);
    GKO_ASSERT_MTX_NEAR(result_op, expected_op, 0.0);
}


TEST_F(Overlap, CanCreateOverlapOp)
{
    using Csr = gko::matrix::Csr<value_type, local_index_type>;
    auto result =
        gko::experimental::distributed::preconditioner::OverlappingOperator<
            value_type, local_index_type>::create(dist_mat.get(), imap);

    auto rank = comm.rank();
    using Dense = dense_vec_type;
    std::unique_ptr<Dense> expected[] = {
        gko::initialize<Dense>({{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}}, exec),
        gko::initialize<Dense>(
            {
                {2, -1, -1, 0},
                {-1, 2, 0, -1},
                {-1, 0, 2, 0},
                {0, -1, 0, 2},
            },
            exec),
        gko::initialize<Dense>({{2, -1, -1}, {-1, 2, 0}, {-1, 0, 2}}, exec)};
    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(result->get_matrix()), expected[rank],
                        0.0);
}


TEST_F(Overlap, CanApplyOverlapOp)
{
    auto rank = comm.rank();
    using Dense = dense_vec_type;
    using Vector = dist_vec_type;
    std::unique_ptr<Dense> local_b[] = {
        gko::initialize<Dense>({1, 2}, exec),
        gko::initialize<Dense>({3, 4}, exec),
        gko::initialize<Dense>({5, 6}, exec),
    };
    auto b =
        Vector::create(exec, comm, gko::dim<2>{6, 1}, std::move(local_b[rank]));
    auto x = Vector::create(exec, comm, gko::dim<2>{6, 1}, gko::dim<2>{2, 1});
    auto ovlp =
        gko::experimental::distributed::preconditioner::OverlappingOperator<
            value_type, local_index_type>::create(dist_mat.get(), imap);

    ovlp->apply(b, x);

    std::unique_ptr<Dense> expected[] = {
        gko::initialize<Dense>({0, 0}, exec),
        gko::initialize<Dense>({0, 0}, exec),
        gko::initialize<Dense>({0, 7}, exec),
    };
    GKO_ASSERT_MTX_NEAR(x->get_local_vector(), expected[rank], 0.0);
}
