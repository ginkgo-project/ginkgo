// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common.hpp"


using namespace gko_codspeed;


// Sparse matrix-vector product y = A * x for the different storage formats,
// on a 2D Laplacian with range(0)^2 rows.
template <typename MatrixType>
void BM_Spmv(benchmark::State& state)
{
    const auto grid = static_cast<index_type>(state.range(0));
    auto mtx = laplacian_matrix<MatrixType>(grid);
    const auto n = mtx->get_size()[0];
    auto x = make_vector(n);
    auto y = vec::create(exec(), gko::dim<2>{n, 1});
    for (auto _ : state) {
        mtx->apply(x, y);
        benchmark::DoNotOptimize(y->get_values());
        benchmark::ClobberMemory();
    }
}


void BM_SpmvCsr(benchmark::State& state) { BM_Spmv<csr>(state); }


void BM_SpmvCoo(benchmark::State& state)
{
    BM_Spmv<gko::matrix::Coo<value_type, index_type>>(state);
}


void BM_SpmvEll(benchmark::State& state)
{
    BM_Spmv<gko::matrix::Ell<value_type, index_type>>(state);
}


void BM_SpmvSellp(benchmark::State& state)
{
    BM_Spmv<gko::matrix::Sellp<value_type, index_type>>(state);
}


void BM_SpmvHybrid(benchmark::State& state)
{
    BM_Spmv<gko::matrix::Hybrid<value_type, index_type>>(state);
}


void BM_SpmvFbcsr(benchmark::State& state)
{
    BM_Spmv<gko::matrix::Fbcsr<value_type, index_type>>(state);
}


// Advanced SpMV y = alpha * A * x + beta * y with multiple right-hand sides.
void BM_CsrAdvancedSpmvMultiRhs(benchmark::State& state)
{
    const auto grid = static_cast<index_type>(state.range(0));
    auto mtx = laplacian_matrix<csr>(grid);
    const auto n = mtx->get_size()[0];
    auto x = make_vector(n, 8);
    auto y = make_vector(n, 8);
    auto alpha = gko::initialize<vec>({0.5}, exec());
    auto beta = gko::initialize<vec>({0.25}, exec());
    for (auto _ : state) {
        mtx->apply(alpha, x, beta, y);
        benchmark::DoNotOptimize(y->get_values());
        benchmark::ClobberMemory();
    }
}


// Dense matrix-vector product (GEMV) on an n x n dense matrix.
void BM_DenseGemv(benchmark::State& state)
{
    const auto n = static_cast<gko::size_type>(state.range(0));
    auto mtx = make_vector(n, n);
    auto x = make_vector(n);
    auto y = vec::create(exec(), gko::dim<2>{n, 1});
    for (auto _ : state) {
        mtx->apply(x, y);
        benchmark::DoNotOptimize(y->get_values());
        benchmark::ClobberMemory();
    }
}


BENCHMARK(BM_SpmvCsr)->Arg(64)->Arg(128);
BENCHMARK(BM_SpmvCoo)->Arg(64)->Arg(128);
BENCHMARK(BM_SpmvEll)->Arg(64)->Arg(128);
BENCHMARK(BM_SpmvSellp)->Arg(64)->Arg(128);
BENCHMARK(BM_SpmvHybrid)->Arg(64)->Arg(128);
BENCHMARK(BM_SpmvFbcsr)->Arg(64);
BENCHMARK(BM_CsrAdvancedSpmvMultiRhs)->Arg(64);
BENCHMARK(BM_DenseGemv)->Arg(512);
