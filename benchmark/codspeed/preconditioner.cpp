// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common.hpp"


using namespace gko_codspeed;


constexpr index_type grid_size = 64;


// Generation (setup) of a block-Jacobi preconditioner with the given maximum
// block size.
void BM_JacobiGenerate(benchmark::State& state)
{
    auto mtx = laplacian_matrix<csr>(grid_size);
    auto factory =
        gko::preconditioner::Jacobi<value_type, index_type>::build()
            .with_max_block_size(static_cast<gko::uint32>(state.range(0)))
            .on(exec());
    for (auto _ : state) {
        auto precond = factory->generate(mtx);
        benchmark::DoNotOptimize(precond.get());
    }
}


void BM_JacobiApply(benchmark::State& state)
{
    auto mtx = laplacian_matrix<csr>(grid_size);
    auto precond =
        gko::preconditioner::Jacobi<value_type, index_type>::build()
            .with_max_block_size(static_cast<gko::uint32>(state.range(0)))
            .on(exec())
            ->generate(mtx);
    const auto n = mtx->get_size()[0];
    auto b = make_vector(n);
    auto x = vec::create(exec(), gko::dim<2>{n, 1});
    for (auto _ : state) {
        precond->apply(b, x);
        benchmark::DoNotOptimize(x->get_values());
        benchmark::ClobberMemory();
    }
}


// Incomplete LU factorization ILU(0).
void BM_IluFactorization(benchmark::State& state)
{
    auto mtx = laplacian_matrix<csr>(grid_size);
    auto factory =
        gko::factorization::Ilu<value_type, index_type>::build().on(exec());
    for (auto _ : state) {
        auto factors = factory->generate(mtx);
        benchmark::DoNotOptimize(factors.get());
    }
}


// Incomplete Cholesky factorization IC(0).
void BM_IcFactorization(benchmark::State& state)
{
    auto mtx = laplacian_matrix<csr>(grid_size);
    auto factory =
        gko::factorization::Ic<value_type, index_type>::build().on(exec());
    for (auto _ : state) {
        auto factors = factory->generate(mtx);
        benchmark::DoNotOptimize(factors.get());
    }
}


// Iterative incomplete LU with threshold (ParILUT).
void BM_ParIlutFactorization(benchmark::State& state)
{
    auto mtx = laplacian_matrix<csr>(grid_size);
    auto factory = gko::factorization::ParIlut<value_type, index_type>::build()
                       .with_fill_in_limit(2.0)
                       .on(exec());
    for (auto _ : state) {
        auto factors = factory->generate(mtx);
        benchmark::DoNotOptimize(factors.get());
    }
}


// Application of an ILU(0) preconditioner, i.e. two sparse triangular solves.
void BM_IluApply(benchmark::State& state)
{
    auto mtx = laplacian_matrix<csr>(grid_size);
    auto factors =
        gko::share(gko::factorization::Ilu<value_type, index_type>::build()
                       .on(exec())
                       ->generate(mtx));
    auto precond =
        gko::preconditioner::Ilu<value_type, false, index_type>::build()
            .on(exec())
            ->generate(factors);
    const auto n = mtx->get_size()[0];
    auto b = make_vector(n);
    auto x = vec::create(exec(), gko::dim<2>{n, 1});
    for (auto _ : state) {
        precond->apply(b, x);
        benchmark::DoNotOptimize(x->get_values());
        benchmark::ClobberMemory();
    }
}


BENCHMARK(BM_JacobiGenerate)->Arg(1)->Arg(8);
BENCHMARK(BM_JacobiApply)->Arg(1)->Arg(8);
BENCHMARK(BM_IluFactorization);
BENCHMARK(BM_IcFactorization);
BENCHMARK(BM_ParIlutFactorization);
BENCHMARK(BM_IluApply);
