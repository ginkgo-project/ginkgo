// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/minres.hpp>
#include <ginkgo/core/solver/nullspace.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/nullspace_helpers.hpp"
#include "test/utils/mpi/common_fixture.hpp"


class NullspaceDistributed : public CommonMpiTestFixture {
protected:
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using dist_mtx =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec = gko::experimental::distributed::Vector<value_type>;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using dense = gko::matrix::Dense<value_type>;
    using Nullspace = gko::solver::Nullspace<value_type>;
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<
        value_type, local_index_type, global_index_type>;
    using jacobi = gko::preconditioner::Jacobi<value_type, local_index_type>;

    NullspaceDistributed()
        : part(gko::share(part_type::build_from_global_size_uniform(
              ref, comm.size(), static_cast<global_index_type>(n)))),
          x_star{{3, -1, -4, 2, 0, -5, 9, -2, -2}}
    {}

    // Neumann Laplacians of paths as diagonal blocks, spanning several ranks
    std::shared_ptr<dist_mtx> laplacian(std::vector<int> block_sizes)
    {
        gko::matrix_data<value_type, global_index_type> data(gko::dim<2>(n, n));
        int begin = 0;
        for (auto size : block_sizes) {
            for (int i = begin; i < begin + size; ++i) {
                value_type diag{};
                if (i > begin) {
                    data.nonzeros.emplace_back(i, i - 1, -1.0);
                    diag += 1.0;
                }
                if (i < begin + size - 1) {
                    data.nonzeros.emplace_back(i, i + 1, -1.0);
                    diag += 1.0;
                }
                data.nonzeros.emplace_back(i, i, diag);
            }
            begin += size;
        }
        data.sort_row_major();
        auto host = dist_mtx::create(ref, comm);
        host->read_distributed(data, part);
        return gko::share(gko::clone(exec, host));
    }

    // distributed n x k vector from the global columns
    std::unique_ptr<dist_vec> distributed(
        const std::vector<std::vector<double>>& cols)
    {
        const auto k = cols.size();
        gko::matrix_data<value_type, global_index_type> data(gko::dim<2>(n, k));
        for (gko::size_type j = 0; j < k; ++j) {
            for (gko::size_type i = 0; i < n; ++i) {
                data.nonzeros.emplace_back(i, j, cols[j][i]);
            }
        }
        auto host = dist_vec::create(ref, comm);
        host->read_distributed(data, part);
        return gko::clone(exec, host);
    }

    // the rows of a global n x k vector owned by this rank
    std::unique_ptr<dense> local_rows(
        const std::vector<std::vector<double>>& cols)
    {
        const auto begin = part->get_range_bounds()[comm.rank()];
        const auto end = part->get_range_bounds()[comm.rank() + 1];
        auto result = dense::create(ref, gko::dim<2>(end - begin, cols.size()));
        for (gko::size_type j = 0; j < cols.size(); ++j) {
            for (auto i = begin; i < end; ++i) {
                result->at(i - begin, j) = cols[j][i];
            }
        }
        return result;
    }

    std::unique_ptr<dist_vec> rhs(const gko::LinOp* mtx, double offset)
    {
        auto b = distributed(x_star);
        mtx->apply(distributed(x_star), b);
        b->add_scaled(gko::initialize<dense>({offset}, exec),
                      distributed({std::vector<double>(n, 1.0)}));
        return b;
    }

    std::unique_ptr<gko::LinOpFactory> factory(
        const std::string& name, std::shared_ptr<const Nullspace> nullspace,
        std::shared_ptr<const Nullspace> left_nullspace)
    {
        auto configure = [&](auto params) {
            return params
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(200u),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(r<value_type>::value))
                .with_preconditioner(schwarz::build().with_local_solver(
                    jacobi::build().with_max_block_size(1u)))
                .with_nullspace(nullspace)
                .with_left_nullspace(left_nullspace)
                .on(exec);
        };
        if (name == "cg") {
            return configure(gko::solver::Cg<value_type>::build());
        } else if (name == "fcg") {
            return configure(gko::solver::Fcg<value_type>::build());
        } else if (name == "minres") {
            return configure(gko::solver::Minres<value_type>::build());
        } else if (name == "gmres") {
            return configure(gko::solver::Gmres<value_type>::build());
        } else {
            return configure(gko::solver::Bicgstab<value_type>::build());
        }
    }

    std::vector<std::vector<double>> initial_guess()
    {
        std::vector<double> col(n);
        for (gko::size_type i = 0; i < n; ++i) {
            col[i] = 2.0 + i % 3;
        }
        return {col};
    }

    static constexpr gko::size_type n = 9;
    const gko::remove_complex<value_type> tol = 1000 * r<value_type>::value;
    const std::vector<std::string> solvers{"cg", "fcg", "minres", "gmres",
                                           "bicgstab"};
    std::shared_ptr<part_type> part;
    std::vector<std::vector<double>> x_star;
};


TEST_F(NullspaceDistributed, ProjectionMatchesSerial)
{
    const std::vector<std::vector<double>> basis{{1, 1, 1, 1, 0, 0, 0, 0, 0},
                                                 {0, 1, 2, 3, 4, 5, 6, 7, 8}};
    const std::vector<std::vector<double>> data{{4, 1, -2, 9, 3, 3, 0, -7, 2},
                                                {1, 0, 0, 5, -1, 8, 2, 2, 6}};
    auto dist_ns = Nullspace::create(exec,
                                     {gko::share(distributed({basis[0]})),
                                      gko::share(distributed({basis[1]}))},
                                     true);
    auto serial_ns =
        Nullspace::create(ref,
                          {gko::share(gko::initialize<dense>(
                               {1., 1., 1., 1., 0., 0., 0., 0., 0.}, ref)),
                           gko::share(gko::initialize<dense>(
                               {0., 1., 2., 3., 4., 5., 6., 7., 8.}, ref))},
                          true);
    auto dist_v = distributed(data);
    auto serial_v = dense::create(ref, gko::dim<2>(n, 2));
    for (gko::size_type i = 0; i < n; ++i) {
        serial_v->at(i, 0) = data[0][i];
        serial_v->at(i, 1) = data[1][i];
    }

    dist_ns->project(dist_v);
    serial_ns->project(serial_v);

    ASSERT_EQ(dist_ns->get_dimension(), serial_ns->get_dimension());
    const auto begin = part->get_range_bounds()[comm.rank()];
    const auto end = part->get_range_bounds()[comm.rank() + 1];
    auto expected =
        serial_v->create_submatrix(gko::span(begin, end), gko::span(0, 2));
    GKO_ASSERT_MTX_NEAR(dist_v->get_local_vector(), expected,
                        r<value_type>::value);
}


TEST_F(NullspaceDistributed, GivesMinimumNormLeastSquaresSolution)
{
    auto mtx = laplacian({9});
    auto constant =
        gko::share(Nullspace::create_from_constant(exec, mtx->get_size()));
    ASSERT_LT(gko::test::compute_nullspace_residual(mtx.get(), constant.get(),
                                                    distributed(x_star).get()),
              tol);
    for (const bool consistent : {true, false}) {
        SCOPED_TRACE(consistent ? "consistent" : "inconsistent");
        for (const auto& name : solvers) {
            SCOPED_TRACE(name);
            auto solver = factory(name, constant, constant)->generate(mtx);
            auto logger =
                gko::share(gko::log::Convergence<value_type>::create());
            solver->add_logger(logger);
            auto b = rhs(mtx.get(), consistent ? 0.0 : 3.0);
            auto x = distributed(initial_guess());

            solver->apply(b, x);

            ASSERT_TRUE(logger->has_converged());
            GKO_ASSERT_MTX_NEAR(x->get_local_vector(), local_rows(x_star), tol);
        }
    }
}


TEST_F(NullspaceDistributed, ExplicitBasisOfDisconnectedGraph)
{
    auto mtx = laplacian({5, 4});
    auto nullspace = gko::share(Nullspace::create(
        exec, {gko::share(distributed({{1, 1, 1, 1, 1, 0, 0, 0, 0}})),
               gko::share(distributed({{0, 0, 0, 0, 0, 1, 1, 1, 1}}))}));
    ASSERT_LT(gko::test::compute_nullspace_residual(mtx.get(), nullspace.get(),
                                                    distributed(x_star).get()),
              tol);
    for (const auto& name : solvers) {
        SCOPED_TRACE(name);
        auto solver = factory(name, nullspace, nullspace)->generate(mtx);
        auto b = rhs(mtx.get(), 0.0);
        auto x = distributed(initial_guess());

        solver->apply(b, x);

        GKO_ASSERT_MTX_NEAR(x->get_local_vector(), local_rows(x_star), tol);
    }
}


TEST_F(NullspaceDistributed, RejectsNonDistributedBasis)
{
    auto mtx = laplacian({5, 4});
    auto nullspace = gko::share(Nullspace::create(
        exec, {gko::share(gko::initialize<dense>(
                  {1., 1., 1., 1., 1., 0., 0., 0., 0.}, exec))}));
    auto v = distributed(x_star);

    ASSERT_THROW(nullspace->project(v), gko::NotSupported);
    ASSERT_THROW(factory("cg", nullspace, nullspace)->generate(mtx),
                 gko::InvalidStateError);
}
