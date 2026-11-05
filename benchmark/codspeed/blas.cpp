// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common.hpp"


using namespace gko_codspeed;


constexpr gko::size_type vector_size = 1 << 16;


void BM_DenseDot(benchmark::State& state)
{
    auto x = make_vector(vector_size);
    auto y = make_vector(vector_size);
    auto result = vec::create(exec(), gko::dim<2>{1, 1});
    gko::array<char> tmp{exec()};
    for (auto _ : state) {
        x->compute_dot(y, result, tmp);
        benchmark::DoNotOptimize(result->get_values());
        benchmark::ClobberMemory();
    }
}


void BM_DenseNorm2(benchmark::State& state)
{
    auto x = make_vector(vector_size);
    auto result = vec::create(exec(), gko::dim<2>{1, 1});
    gko::array<char> tmp{exec()};
    for (auto _ : state) {
        x->compute_norm2(result, tmp);
        benchmark::DoNotOptimize(result->get_values());
        benchmark::ClobberMemory();
    }
}


void BM_DenseAxpy(benchmark::State& state)
{
    auto x = make_vector(vector_size);
    auto y = make_vector(vector_size);
    auto alpha = gko::initialize<vec>({1e-3}, exec());
    for (auto _ : state) {
        y->add_scaled(alpha, x);
        benchmark::DoNotOptimize(y->get_values());
        benchmark::ClobberMemory();
    }
}


void BM_DenseScale(benchmark::State& state)
{
    auto x = make_vector(vector_size);
    auto alpha = gko::initialize<vec>({1.0}, exec());
    for (auto _ : state) {
        x->scale(alpha);
        benchmark::DoNotOptimize(x->get_values());
        benchmark::ClobberMemory();
    }
}


// Dense matrix-matrix product (GEMM) on n x n matrices.
void BM_DenseGemm(benchmark::State& state)
{
    const auto n = static_cast<gko::size_type>(state.range(0));
    auto a = make_vector(n, n);
    auto b = make_vector(n, n);
    auto c = vec::create(exec(), gko::dim<2>{n, n});
    for (auto _ : state) {
        a->apply(b, c);
        benchmark::DoNotOptimize(c->get_values());
        benchmark::ClobberMemory();
    }
}


// Sparse matrix-matrix product (SpGEMM) C = A * A.
void BM_CsrSpgemm(benchmark::State& state)
{
    const auto grid = static_cast<index_type>(state.range(0));
    auto mtx = laplacian_matrix<csr>(grid);
    auto result = csr::create(exec(), mtx->get_size());
    for (auto _ : state) {
        mtx->apply(mtx, result);
        benchmark::DoNotOptimize(result->get_values());
        benchmark::ClobberMemory();
    }
}


// Explicit transposition of a Csr matrix.
void BM_CsrTranspose(benchmark::State& state)
{
    const auto grid = static_cast<index_type>(state.range(0));
    auto mtx = laplacian_matrix<csr>(grid);
    for (auto _ : state) {
        auto trans = mtx->transpose();
        benchmark::DoNotOptimize(trans.get());
    }
}


// Assembling a Csr matrix from (unsorted) coordinate data.
void BM_CsrRead(benchmark::State& state)
{
    const auto grid = static_cast<index_type>(state.range(0));
    auto data = laplacian_2d(grid);
    auto mtx = csr::create(exec());
    for (auto _ : state) {
        mtx->read(data);
        benchmark::DoNotOptimize(mtx->get_values());
        benchmark::ClobberMemory();
    }
}


// Format conversion from Csr to the other sparse formats.
template <typename TargetType>
void BM_CsrConvertTo(benchmark::State& state)
{
    const auto grid = static_cast<index_type>(state.range(0));
    auto mtx = laplacian_matrix<csr>(grid);
    auto result = TargetType::create(exec());
    for (auto _ : state) {
        mtx->convert_to(result);
        benchmark::DoNotOptimize(result.get());
        benchmark::ClobberMemory();
    }
}


void BM_CsrConvertToCoo(benchmark::State& state)
{
    BM_CsrConvertTo<gko::matrix::Coo<value_type, index_type>>(state);
}


void BM_CsrConvertToEll(benchmark::State& state)
{
    BM_CsrConvertTo<gko::matrix::Ell<value_type, index_type>>(state);
}


void BM_CsrConvertToSellp(benchmark::State& state)
{
    BM_CsrConvertTo<gko::matrix::Sellp<value_type, index_type>>(state);
}


void BM_CsrConvertToHybrid(benchmark::State& state)
{
    BM_CsrConvertTo<gko::matrix::Hybrid<value_type, index_type>>(state);
}


void BM_CsrConvertToDense(benchmark::State& state)
{
    BM_CsrConvertTo<vec>(state);
}


BENCHMARK(BM_DenseDot);
BENCHMARK(BM_DenseNorm2);
BENCHMARK(BM_DenseAxpy);
BENCHMARK(BM_DenseScale);
BENCHMARK(BM_DenseGemm)->Arg(64);
BENCHMARK(BM_CsrSpgemm)->Arg(64);
BENCHMARK(BM_CsrTranspose)->Arg(128);
BENCHMARK(BM_CsrRead)->Arg(128);
BENCHMARK(BM_CsrConvertToCoo)->Arg(128);
BENCHMARK(BM_CsrConvertToEll)->Arg(128);
BENCHMARK(BM_CsrConvertToSellp)->Arg(128);
BENCHMARK(BM_CsrConvertToHybrid)->Arg(128);
BENCHMARK(BM_CsrConvertToDense)->Arg(32);
