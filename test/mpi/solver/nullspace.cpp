// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/null_space.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


class NullspaceDistributedCg : public CommonMpiTestFixture {
protected:
    using value_type = double;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using dist_mtx =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec = gko::experimental::distributed::Vector<value_type>;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using Cg = gko::solver::Cg<value_type>;
    using dense = gko::matrix::Dense<value_type>;

    NullspaceDistributedCg() : n{9} {}

    // 1-D Neumann graph Laplacian: singular, SPD, nullspace = span{ (1,..,1) }.
    gko::matrix_data<value_type, global_index_type> laplacian_data()
    {
        gko::matrix_data<value_type, global_index_type> data(gko::dim<2>{
            static_cast<gko::size_type>(n), static_cast<gko::size_type>(n)});
        for (int i = 0; i < n; ++i) {
            value_type diag{};
            if (i > 0) {
                data.nonzeros.emplace_back(i, i - 1, value_type{-1});
                diag += value_type{1};
            }
            if (i + 1 < n) {
                data.nonzeros.emplace_back(i, i + 1, value_type{-1});
                diag += value_type{1};
            }
            data.nonzeros.emplace_back(i, i, diag);
        }
        data.sort_row_major();
        return data;
    }

    value_type host_scalar(const dense* d)
    {
        return gko::clone(this->ref, d)->at(0, 0);
    }

    int n;
};


TEST_F(NullspaceDistributedCg, SolvesConsistentSingularSystem)
{
    ASSERT_EQ(comm.size(), 3);
    const auto gn = static_cast<gko::size_type>(n);
    auto part = gko::share(
        part_type::build_from_global_size_uniform(ref, comm.size(), gn));
    const auto local_n =
        static_cast<gko::size_type>(part->get_part_size(comm.rank()));

    // system matrix
    auto A_host = dist_mtx::create(ref, comm);
    A_host->read_distributed(laplacian_data(), part);
    auto A = gko::share(gko::clone(exec, A_host));

    // consistent RHS with zero global mean: b_i = i - (n-1)/2.
    gko::matrix_data<value_type, global_index_type> b_data(gko::dim<2>{gn, 1});
    for (gko::size_type i = 0; i < gn; ++i) {
        b_data.nonzeros.emplace_back(
            static_cast<global_index_type>(i), 0,
            static_cast<value_type>(i) - static_cast<value_type>(n - 1) / 2);
    }
    auto b_host = dist_vec::create(ref, comm, gko::dim<2>{gn, 1},
                                   gko::dim<2>{local_n, 1});
    b_host->read_distributed(b_data, part);
    auto b = gko::clone(exec, b_host);

    auto x = dist_vec::create(exec, comm, gko::dim<2>{gn, 1},
                              gko::dim<2>{local_n, 1});
    x->fill(gko::zero<value_type>());

    auto ns = gko::share(
        gko::NullSpace<value_type>::create_from_constant(exec, A->get_size()));
    auto solver =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(1e-12))
            .with_nullspace(ns)
            .with_left_nullspace(ns)
            .on(exec)
            ->generate(A);

    solver->apply(b, x);

    // residual r = b - A x must be small.
    auto r = gko::clone(b);
    auto one = gko::initialize<dense>({gko::one<value_type>()}, exec);
    auto neg_one = gko::initialize<dense>({-gko::one<value_type>()}, exec);
    A->apply(neg_one, x, one, r);
    auto rnorm = dense::create(exec, gko::dim<2>{1, 1});
    r->compute_norm2(rnorm);
    EXPECT_LT(host_scalar(rnorm.get()), 1e-9);

    // minimum-norm solution: global mean of x ~ 0.
    auto mean = dense::create(exec, gko::dim<2>{1, 1});
    x->compute_mean(mean);
    EXPECT_LT(gko::abs(host_scalar(mean.get())), 1e-9);
}
