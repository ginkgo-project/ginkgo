// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common.hpp"


using namespace gko_codspeed;


constexpr index_type grid_size = 64;
// Only an iteration limit is used, so every run performs the same amount of
// work independently of the convergence behavior.
constexpr gko::size_type num_iterations = 50;


std::shared_ptr<const gko::stop::CriterionFactory> iteration_criterion()
{
    return gko::share(gko::stop::Iteration::build()
                          .with_max_iters(num_iterations)
                          .on(exec()));
}


// Solves A x = b with a fixed number of iterations. The solver (including
// the preconditioner setup) is generated outside of the measured region.
template <typename FactoryBuilder>
void run_solver(benchmark::State& state, FactoryBuilder&& build_factory)
{
    auto mtx = laplacian_matrix<csr>(grid_size);
    const auto n = mtx->get_size()[0];
    auto solver = build_factory()->generate(mtx);
    auto b = make_vector(n);
    auto x0 = vec::create(exec(), gko::dim<2>{n, 1});
    x0->fill(0.0);
    auto x = x0->clone();
    for (auto _ : state) {
        x->copy_from(x0);
        solver->apply(b, x);
        benchmark::DoNotOptimize(x->get_values());
        benchmark::ClobberMemory();
    }
}


void BM_Cg(benchmark::State& state)
{
    run_solver(state, [] {
        return gko::solver::Cg<value_type>::build()
            .with_criteria(iteration_criterion())
            .on(exec());
    });
}


void BM_CgJacobi(benchmark::State& state)
{
    run_solver(state, [] {
        return gko::solver::Cg<value_type>::build()
            .with_criteria(iteration_criterion())
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type, index_type>::build()
                    .with_max_block_size(1u))
            .on(exec());
    });
}


void BM_Bicgstab(benchmark::State& state)
{
    run_solver(state, [] {
        return gko::solver::Bicgstab<value_type>::build()
            .with_criteria(iteration_criterion())
            .on(exec());
    });
}


void BM_Cgs(benchmark::State& state)
{
    run_solver(state, [] {
        return gko::solver::Cgs<value_type>::build()
            .with_criteria(iteration_criterion())
            .on(exec());
    });
}


void BM_Gmres(benchmark::State& state)
{
    run_solver(state, [] {
        return gko::solver::Gmres<value_type>::build()
            .with_criteria(iteration_criterion())
            .with_krylov_dim(30u)
            .on(exec());
    });
}


void BM_GmresIlu(benchmark::State& state)
{
    run_solver(state, [] {
        return gko::solver::Gmres<value_type>::build()
            .with_criteria(iteration_criterion())
            .with_krylov_dim(30u)
            .with_preconditioner(
                gko::preconditioner::Ilu<value_type, false, index_type>::build()
                    .with_factorization(
                        gko::factorization::Ilu<value_type,
                                                index_type>::build()))
            .on(exec());
    });
}


void BM_Idr(benchmark::State& state)
{
    run_solver(state, [] {
        return gko::solver::Idr<value_type>::build()
            .with_criteria(iteration_criterion())
            .with_subspace_dim(4u)
            .with_deterministic(true)
            .on(exec());
    });
}


std::unique_ptr<gko::LinOpFactory> multigrid_cg_factory()
{
    auto mg =
        gko::solver::Multigrid::build()
            .with_mg_level(gko::multigrid::Pgm<value_type, index_type>::build()
                               .with_deterministic(true))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec());
    return gko::solver::Cg<value_type>::build()
        .with_criteria(gko::stop::Iteration::build().with_max_iters(10u))
        .with_preconditioner(gko::share(std::move(mg)))
        .on(exec());
}


// CG preconditioned with an aggregation-based (PGM) algebraic multigrid.
void BM_CgMultigrid(benchmark::State& state)
{
    run_solver(state, multigrid_cg_factory);
}


// Setup phase of the AMG hierarchy (coarsening + Galerkin products).
void BM_MultigridGenerate(benchmark::State& state)
{
    auto mtx = laplacian_matrix<csr>(grid_size);
    auto factory = multigrid_cg_factory();
    for (auto _ : state) {
        auto solver = factory->generate(mtx);
        benchmark::DoNotOptimize(solver.get());
    }
}


// Sparse direct triangular solve on the lower triangle of the Laplacian.
void BM_LowerTrs(benchmark::State& state)
{
    run_solver(state, [] {
        return gko::solver::LowerTrs<value_type, index_type>::build().on(
            exec());
    });
}


BENCHMARK(BM_Cg);
BENCHMARK(BM_CgJacobi);
BENCHMARK(BM_Bicgstab);
BENCHMARK(BM_Cgs);
BENCHMARK(BM_Gmres);
BENCHMARK(BM_GmresIlu);
BENCHMARK(BM_Idr);
BENCHMARK(BM_CgMultigrid);
BENCHMARK(BM_MultigridGenerate);
BENCHMARK(BM_LowerTrs);
