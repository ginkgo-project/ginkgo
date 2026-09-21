// SPDX-FileCopyrightText: 2024 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>


// ---------------------------------------------------------------------------
// CPU reference helpers
// ---------------------------------------------------------------------------

template <typename T, typename IndexT = int>
void cpu_spmv_ref(int nrows, const std::vector<IndexT>& row_ptr,
                  const std::vector<IndexT>& col_idx, const std::vector<T>& val,
                  const std::vector<T>& x, std::vector<T>& y, T alpha = T{1},
                  T beta = T{0})
{
    for (int r = 0; r < nrows; ++r) {
        T acc{};
        for (auto i = row_ptr[r]; i < row_ptr[r + 1]; ++i)
            acc += val[i] * x[col_idx[i]];
        y[r] = alpha * acc + beta * y[r];
    }
}

template <typename T>
double inf_norm_diff(const std::vector<T>& a, const std::vector<T>& b)
{
    double mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        mx = std::max(mx, std::abs(static_cast<double>(a[i]) -
                                   static_cast<double>(b[i])));
    return mx;
}

template <typename T>
double rel_inf_error(const std::vector<T>& ref, const std::vector<T>& got)
{
    double ref_norm = 0.0;
    for (auto v : ref) ref_norm = std::max(ref_norm, std::abs(double(v)));
    if (ref_norm == 0.0) return inf_norm_diff(ref, got);
    return inf_norm_diff(ref, got) / ref_norm;
}

template <typename T>
std::vector<T> to_host(const gko::matrix::Dense<T>* d)
{
    auto host = d->get_executor()->get_master();
    auto h = gko::matrix::Dense<T>::create(host, d->get_size());
    h->copy_from(d);
    std::vector<T> v(h->get_size()[0] * h->get_size()[1]);
    std::copy(h->get_const_values(), h->get_const_values() + v.size(),
              v.begin());
    return v;
}


// ---------------------------------------------------------------------------
// make_csr_mp — creates a CSR matrix on `exec` with merge_path strategy.
// This is critical: without explicit merge_path strategy the default
// automatical strategy selects classical or load_balance, bypassing the
// compute_items_per_thread / select_merge_path_spmv code path entirely.
// ---------------------------------------------------------------------------

template <typename T, typename IndexT = int>
std::unique_ptr<gko::matrix::Csr<T, IndexT>> make_csr_mp(
    std::shared_ptr<gko::Executor> exec, int nrows, int ncols,
    const std::vector<IndexT>& row_ptr, const std::vector<IndexT>& col_idx,
    const std::vector<T>& val)
{
    using Csr = gko::matrix::Csr<T, IndexT>;
    auto host = exec->get_master();
    auto strategy = std::make_shared<typename Csr::merge_path>();
    int nnz = static_cast<int>(val.size());

    auto m_host =
        Csr::create(host, gko::dim<2>(nrows, ncols),
                    gko::array<T>::view(host, nnz, const_cast<T*>(val.data())),
                    gko::array<IndexT>::view(
                        host, nnz, const_cast<IndexT*>(col_idx.data())),
                    gko::array<IndexT>::view(
                        host, nrows + 1, const_cast<IndexT*>(row_ptr.data())),
                    strategy);

    auto m_dev = Csr::create(exec, strategy);
    m_dev->copy_from(m_host);
    return m_dev;
}

template <typename T>
std::unique_ptr<gko::matrix::Dense<T>> make_vec(
    std::shared_ptr<gko::Executor> exec, const std::vector<T>& data,
    int ncols = 1)
{
    auto host = exec->get_master();
    int nrows = static_cast<int>(data.size()) / ncols;
    auto h = gko::matrix::Dense<T>::create(host, gko::dim<2>(nrows, ncols));
    std::copy(data.begin(), data.end(), h->get_values());
    auto d = gko::matrix::Dense<T>::create(exec, gko::dim<2>(nrows, ncols));
    d->copy_from(h);
    return d;
}


// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class CsrMergePath : public ::testing::Test {
protected:
    void SetUp() override
    {
        if (gko::CudaExecutor::get_num_devices() == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        exec = gko::CudaExecutor::create(0, gko::ReferenceExecutor::create());
    }

    std::shared_ptr<gko::CudaExecutor> exec;

    // Tolerances
    static constexpr double tol_f32 = 1e-4;
    static constexpr double tol_f64 = 1e-11;
};


// ===========================================================================
// GROUP 0: Architecture / version diagnostics
//
// This test never fails. It prints the GPU compute capability so we know
// which architecture is under test.  It also reports the version encoding
// (major << 4) + minor used by compute_items_per_thread, so that an
// unsupported architecture is immediately visible in the test log.
// ===========================================================================

TEST_F(CsrMergePath, ArchitectureInfo)
{
    int major = exec->get_major_version();
    int minor = exec->get_minor_version();
    int version_encoding = (major << 4) + minor;

    std::string coverage;
    // These are the cases handled by compute_items_per_thread.
    // Architectures beyond Volta (0x70) use the default: num_item=12 (same as
    // Volta).
    switch (version_encoding) {
    case 0x37:
        coverage = "Kepler GK210 (sm_37) — num_item=14";
        break;
    case 0x50:
        coverage = "Maxwell (sm_50) — num_item=8";
        break;
    case 0x52:
        coverage = "Maxwell (sm_52) — num_item=12";
        break;
    case 0x53:
        coverage = "Maxwell (sm_53) — num_item=8";
        break;
    case 0x60:
        coverage = "Pascal GP100 (sm_60) — num_item=8";
        break;
    case 0x61:
        coverage = "Pascal GP102/104 (sm_61) — num_item=12";
        break;
    case 0x62:
        coverage = "Pascal GP106/108 (sm_62) — num_item=8";
        break;
    case 0x70:
        coverage = "Volta (sm_70) — num_item=12";
        break;
    default:
        if (version_encoding > 0x70)
            coverage =
                "Post-Volta (Turing/Ampere/Ada/Hopper/Blackwell) — "
                "num_item=12 (default for unknown sm_75+)";
        else
            coverage = "Pre-Maxwell — num_item=6 (default)";
        break;
    }

    std::cout << "\n[CsrMergePath/ArchitectureInfo]\n"
              << "  GPU: sm_" << major * 10 + minor << "\n"
              << "  version_encoding: 0x" << std::hex << version_encoding
              << std::dec << "\n"
              << "  compute_items_per_thread coverage: " << coverage << "\n\n";

    SUCCEED();
}


// ===========================================================================
// GROUP 1: Correctness — simple known matrices (merge_path strategy)
// ===========================================================================

TEST_F(CsrMergePath, IdentityMatrixFloat)
{
    const int n = 64;
    std::vector<int> rp(n + 1), ci(n);
    std::vector<float> val(n, 1.f), x(n), y_ref(n);
    std::iota(ci.begin(), ci.end(), 0);
    for (int i = 0; i <= n; ++i) rp[i] = i;
    for (int i = 0; i < n; ++i) x[i] = float(i + 1);
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32);
}

TEST_F(CsrMergePath, IdentityMatrixDouble)
{
    const int n = 64;
    std::vector<int> rp(n + 1), ci(n);
    std::vector<double> val(n, 1.0), x(n), y_ref(n);
    std::iota(ci.begin(), ci.end(), 0);
    for (int i = 0; i <= n; ++i) rp[i] = i;
    for (int i = 0; i < n; ++i) x[i] = double(i + 1);
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<double>(exec, n, n, rp, ci, val);
    auto gx = make_vec<double>(exec, x);
    auto gy = gko::matrix::Dense<double>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f64);
}

TEST_F(CsrMergePath, DenseAllOnesRowsFloat)
{
    const int n = 128;
    std::vector<int> rp(n + 1), ci(n * n);
    std::vector<float> val(n * n, 1.f), x(n, 1.f), y_ref(n);
    for (int r = 0; r <= n; ++r) rp[r] = r * n;
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) ci[r * n + c] = c;
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32)
        << "Dense all-ones rows (float): merge_path_reduce error";
}

TEST_F(CsrMergePath, DenseAllOnesRowsDouble)
{
    const int n = 128;
    std::vector<int> rp(n + 1), ci(n * n);
    std::vector<double> val(n * n, 1.), x(n, 1.), y_ref(n);
    for (int r = 0; r <= n; ++r) rp[r] = r * n;
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) ci[r * n + c] = c;
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<double>(exec, n, n, rp, ci, val);
    auto gx = make_vec<double>(exec, x);
    auto gy = gko::matrix::Dense<double>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f64);
}


// ===========================================================================
// GROUP 2: Zero-sum / cancellation rows
//
// These expose the merge_path_reduce zero-suppression bug reported as
// "BUG 1" in the original analysis: a row whose accumulated partial sum
// happens to be exactly zero is incorrectly discarded, leaving the output
// element with the wrong value (usually the initial value of y).
//
// The pattern is: row with [+a, -a] entries applied to x=[1,1,...] gives 0.
// ===========================================================================

TEST_F(CsrMergePath, CancellationRowsFloat)
{
    //  [+1  -1   0   0] [1]   [0]
    //  [ 0   0  +2  -2] [1] = [0]
    //  [ 1   0   0   0] [1]   [1]
    //  [ 0   0   0   3] [1]   [3]
    std::vector<int> rp = {0, 2, 4, 5, 6};
    std::vector<int> ci = {0, 1, 2, 3, 0, 3};
    std::vector<float> v = {1.f, -1.f, 2.f, -2.f, 1.f, 3.f};
    std::vector<float> x = {1.f, 1.f, 1.f, 1.f};
    std::vector<float> y_ref(4);
    cpu_spmv_ref(4, rp, ci, v, x, y_ref);

    auto A = make_csr_mp<float>(exec, 4, 4, rp, ci, v);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(4, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());

    EXPECT_LT(inf_norm_diff(y_ref, y_gpu), tol_f32)
        << "CancellationRowsFloat: zero partial-sum rows dropped.\n"
           "y_ref=[0,0,1,3] y_gpu=["
        << y_gpu[0] << "," << y_gpu[1] << "," << y_gpu[2] << "," << y_gpu[3]
        << "]";
}

TEST_F(CsrMergePath, CancellationRowsDouble)
{
    std::vector<int> rp = {0, 2, 4, 5, 6};
    std::vector<int> ci = {0, 1, 2, 3, 0, 3};
    std::vector<double> v = {1., -1., 2., -2., 1., 3.};
    std::vector<double> x = {1., 1., 1., 1.};
    std::vector<double> y_ref(4);
    cpu_spmv_ref(4, rp, ci, v, x, y_ref);

    auto A = make_csr_mp<double>(exec, 4, 4, rp, ci, v);
    auto gx = make_vec<double>(exec, x);
    auto gy = gko::matrix::Dense<double>::create(exec, gko::dim<2>(4, 1));
    gy->fill(0.);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(inf_norm_diff(y_ref, y_gpu), tol_f64);
}

// Large cancellation: forces merge_path across multiple thread-blocks so
// the inter-block reduce (abstract_reduce) is exercised.
TEST_F(CsrMergePath, LargeCancellationFloat)
{
    const int n = 2048;
    std::vector<int> rp(n + 1), ci(2 * n);
    std::vector<float> val(2 * n), x(2 * n, 1.f), y_ref(n, 0.f);
    for (int r = 0; r < n; ++r) {
        rp[r] = 2 * r;
        ci[2 * r] = 2 * r;
        ci[2 * r + 1] = 2 * r + 1;
        val[2 * r] = 1.f;
        val[2 * r + 1] = -1.f;
    }
    rp[n] = 2 * n;

    auto A = make_csr_mp<float>(exec, n, 2 * n, rp, ci, val);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());

    EXPECT_LT(inf_norm_diff(y_ref, y_gpu), tol_f32)
        << "LargeCancellationFloat: max_err=" << inf_norm_diff(y_ref, y_gpu);
}

TEST_F(CsrMergePath, LargeCancellationDouble)
{
    const int n = 2048;
    std::vector<int> rp(n + 1), ci(2 * n);
    std::vector<double> val(2 * n), x(2 * n, 1.), y_ref(n, 0.);
    for (int r = 0; r < n; ++r) {
        rp[r] = 2 * r;
        ci[2 * r] = 2 * r;
        ci[2 * r + 1] = 2 * r + 1;
        val[2 * r] = 1.;
        val[2 * r + 1] = -1.;
    }
    rp[n] = 2 * n;

    auto A = make_csr_mp<double>(exec, n, 2 * n, rp, ci, val);
    auto gx = make_vec<double>(exec, x);
    auto gy = gko::matrix::Dense<double>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(inf_norm_diff(y_ref, y_gpu), tol_f64);
}


// ===========================================================================
// GROUP 3: Block-boundary stress
//
// These sizes are chosen so that the merge path cuts fall exactly at
// thread-block boundaries, stressing the inter-block reduction.
//
// With items_per_thread = 6 (the Blackwell fallback):
//   block_items = 128 * 6 = 768
//   grid_num = ceil((num_rows + nnz) / 768)
//
// With items_per_thread = 12 (Volta):
//   block_items = 128 * 12 = 1536
//   grid_num = ceil((num_rows + nnz) / 1536)
//
// By sizing the matrix so that num_rows + nnz is a multiple of 768 we
// ensure that at least one thread-block ends exactly at a row boundary,
// which is the scenario where merge_path_reduce must correctly combine
// the last partial from one block with the first partial from the next.
// ===========================================================================

// 3a. Diagonal (1 nnz/row), n=384: total=768=1*block_items(6)
TEST_F(CsrMergePath, BlockBoundaryDiagonalN384Float)
{
    const int n = 384;  // n + nnz = 768 = 128*6
    std::vector<int> rp(n + 1), ci(n);
    std::vector<float> val(n), x(n), y_ref(n);
    for (int r = 0; r < n; ++r) {
        rp[r] = r;
        ci[r] = r;
        val[r] = float(r + 1);
        x[r] = float(r + 1);
    }
    rp[n] = n;
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32)
        << "BlockBoundaryDiagonalN384: n+nnz exactly fills one block of 768";
}

// 3b. Diagonal, n=768: total=1536=2*block_items(6) / 1*block_items(12)
TEST_F(CsrMergePath, BlockBoundaryDiagonalN768Float)
{
    const int n = 768;  // n + nnz = 1536 = 2*128*6 = 1*128*12
    std::vector<int> rp(n + 1), ci(n);
    std::vector<float> val(n, 1.f), x(n, 1.f), y_ref(n, 1.f);
    for (int r = 0; r < n; ++r) {
        rp[r] = r;
        ci[r] = r;
    }
    rp[n] = n;

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32);
}

// 3c. n=255, nnz_per_row=2: total=255+510=765, just under one full block
TEST_F(CsrMergePath, JustUnderBlockItemsFloat)
{
    const int n = 255, nnz_row = 2;
    std::vector<int> rp(n + 1), ci(n * nnz_row);
    std::vector<float> val(n * nnz_row, 1.f), x(n, 1.f), y_ref(n);
    for (int r = 0; r <= n; ++r) rp[r] = r * nnz_row;
    for (int r = 0; r < n; ++r) {
        ci[r * nnz_row] = r % n;
        ci[r * nnz_row + 1] = (r + 1) % n;
    }
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32);
}

// 3d. n=256, nnz_per_row=2: total=256+512=768, exactly one block (6-item)
TEST_F(CsrMergePath, ExactlyOneBlockItemsFloat)
{
    const int n = 256, nnz_row = 2;
    std::vector<int> rp(n + 1), ci(n * nnz_row);
    std::vector<float> val(n * nnz_row, 1.f), x(n, 1.f), y_ref(n);
    for (int r = 0; r <= n; ++r) rp[r] = r * nnz_row;
    for (int r = 0; r < n; ++r) {
        ci[r * nnz_row] = r % n;
        ci[r * nnz_row + 1] = (r + 1) % n;
    }
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32);
}


// ===========================================================================
// GROUP 4: shared_row_ptrs aliasing stress
//
// In merge_path_spmv the shared array shared_row_ptrs[block_items] is
// first used to cache row end-pointers (up to block_num_rows entries), then
// after a __syncthreads() it is reused as:
//
//   IndexType*      tmp_ind = shared_row_ptrs;          // [0..127]
//   arithmetic_type* tmp_val = (arithmetic_type*)(shared_row_ptrs + 128);
//
// For correct aliasing we need:
//   128*sizeof(IndexType) + 128*sizeof(arithmetic_type)
//     <= block_items * sizeof(IndexType)
// i.e. items_per_thread >=
// (sizeof(IndexType)+sizeof(arithmetic_type))/sizeof(IndexType)
//
// This is guaranteed by the minimal_num check in compute_items_per_thread.
// These tests verify the property holds empirically, especially for the
// double/int64 type combination with the fallback items_per_thread=3.
// ===========================================================================

TEST_F(CsrMergePath, AliasingSafetyDoubleInt32)
{
    // double/int32: minimal_num = ceil((4+8)/4) = 3, fallback = max(3,6) = 6
    const int n = 512, nnz_row = 5;
    std::vector<int> rp(n + 1), ci(n * nnz_row);
    std::vector<double> val(n * nnz_row), x(n), y_ref(n);
    std::mt19937 rng(111);
    std::uniform_real_distribution<double> dist(-1., 1.);
    for (int r = 0; r <= n; ++r) rp[r] = r * nnz_row;
    for (int r = 0; r < n; ++r)
        for (int j = 0; j < nnz_row; ++j) {
            ci[r * nnz_row + j] = (r + j * 7) % n;
            val[r * nnz_row + j] = dist(rng);
        }
    for (auto& v : x) v = dist(rng);
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<double>(exec, n, n, rp, ci, val);
    auto gx = make_vec<double>(exec, x);
    auto gy = gko::matrix::Dense<double>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f64)
        << "AliasingSafetyDoubleInt32: shared_row_ptrs reuse violated";
}

// Same but for float/int32 — the common production case.
TEST_F(CsrMergePath, AliasingSafetyFloatInt32)
{
    const int n = 512, nnz_row = 5;
    std::vector<int> rp(n + 1), ci(n * nnz_row);
    std::vector<float> val(n * nnz_row), x(n), y_ref(n);
    std::mt19937 rng(222);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int r = 0; r <= n; ++r) rp[r] = r * nnz_row;
    for (int r = 0; r < n; ++r)
        for (int j = 0; j < nnz_row; ++j) {
            ci[r * nnz_row + j] = (r + j * 7) % n;
            val[r * nnz_row + j] = dist(rng);
        }
    for (auto& v : x) v = dist(rng);
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32);
}


// ===========================================================================
// GROUP 5: Warp-boundary rows
//
// Rows whose nnz count is exactly warp_size (32) or warp_size±1 sit at the
// boundary between warp-internal and warp-crossing segment scans.  These
// are the highest-risk inputs for warp_atomic_add masking bugs on Blackwell.
// ===========================================================================

TEST_F(CsrMergePath, RowsOfExactlyWarpSizeFloat)
{
    const int warp = 32, n = 64;
    std::vector<int> rp(n + 1), ci(n * warp);
    std::vector<float> val(n * warp), x(n, 1.f), y_ref(n);
    for (int r = 0; r <= n; ++r) rp[r] = r * warp;
    for (int r = 0; r < n; ++r)
        for (int j = 0; j < warp; ++j) {
            ci[r * warp + j] = j % n;
            val[r * warp + j] = (j % 2 == 0) ? 1.f : -1.f;
        }
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(inf_norm_diff(y_ref, y_gpu), tol_f32)
        << "RowsOfExactlyWarpSize: warp boundary segment-scan error";
}

TEST_F(CsrMergePath, RowsStraddlingWarpBoundaryFloat)
{
    const int warp = 32, nnz_row = warp + 1, ncols = 128, n = 32;
    std::vector<int> rp(n + 1), ci(n * nnz_row);
    std::vector<float> val(n * nnz_row, 1.f), x(ncols, 1.f), y_ref(n);
    for (int r = 0; r <= n; ++r) rp[r] = r * nnz_row;
    for (int r = 0; r < n; ++r)
        for (int j = 0; j < nnz_row; ++j) ci[r * nnz_row + j] = j % ncols;
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<float>(exec, n, ncols, rp, ci, val);
    auto gx = make_vec<float>(exec, x);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32)
        << "RowsStraddlingWarpBoundary: warp+1 nnz/row reduction error";
}

TEST_F(CsrMergePath, RowsStraddlingWarpBoundaryDouble)
{
    const int warp = 32, nnz_row = warp + 1, ncols = 128, n = 32;
    std::vector<int> rp(n + 1), ci(n * nnz_row);
    std::vector<double> val(n * nnz_row, 1.), x(ncols, 1.), y_ref(n);
    for (int r = 0; r <= n; ++r) rp[r] = r * nnz_row;
    for (int r = 0; r < n; ++r)
        for (int j = 0; j < nnz_row; ++j) ci[r * nnz_row + j] = j % ncols;
    cpu_spmv_ref(n, rp, ci, val, x, y_ref);

    auto A = make_csr_mp<double>(exec, n, ncols, rp, ci, val);
    auto gx = make_vec<double>(exec, x);
    auto gy = gko::matrix::Dense<double>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f64);
}


// ===========================================================================
// GROUP 6: Edge cases
// ===========================================================================

TEST_F(CsrMergePath, EmptyMatrix)
{
    std::vector<int> rp = {0}, ci = {};
    std::vector<float> val = {}, x_v = {};
    auto A = make_csr_mp<float>(exec, 0, 0, rp, ci, val);
    auto gx = gko::matrix::Dense<float>::create(exec, gko::dim<2>(0, 1));
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(0, 1));
    EXPECT_NO_THROW(A->apply(gx, gy));
}

TEST_F(CsrMergePath, SingleElementFloat)
{
    std::vector<int> rp = {0, 1}, ci = {0};
    std::vector<float> val = {3.14f}, x_v = {2.f};
    std::vector<float> y_ref(1);
    cpu_spmv_ref(1, rp, ci, val, x_v, y_ref);

    auto A = make_csr_mp<float>(exec, 1, 1, rp, ci, val);
    auto gx = make_vec<float>(exec, x_v);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(1, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32);
}

// Row with structurally present but numerically zero entries.
// BUG: if merge_path_reduce suppresses zero contributions, y[0] stays at
// its initial value (99) instead of being overwritten with 0.
TEST_F(CsrMergePath, AllZeroValuesInRowFloat)
{
    std::vector<int> rp = {0, 2, 3};
    std::vector<int> ci = {0, 1, 0};
    std::vector<float> val = {0.f, 0.f, 1.f};
    std::vector<float> x_v = {5.f, 7.f};
    std::vector<float> y_ref(2);
    cpu_spmv_ref(2, rp, ci, val, x_v, y_ref);

    auto A = make_csr_mp<float>(exec, 2, 2, rp, ci, val);
    auto gx = make_vec<float>(exec, x_v);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(2, 1));
    gy->fill(99.f);  // sentinel: detect missing writes
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(inf_norm_diff(y_ref, y_gpu), tol_f32)
        << "AllZeroValuesInRow: y[0] should be 0 but got " << y_gpu[0]
        << " (initial=99); zero-suppression bug?";
}

TEST_F(CsrMergePath, AllZeroValuesInRowDouble)
{
    std::vector<int> rp = {0, 2, 3};
    std::vector<int> ci = {0, 1, 0};
    std::vector<double> val = {0., 0., 1.};
    std::vector<double> x_v = {5., 7.};
    std::vector<double> y_ref(2);
    cpu_spmv_ref(2, rp, ci, val, x_v, y_ref);

    auto A = make_csr_mp<double>(exec, 2, 2, rp, ci, val);
    auto gx = make_vec<double>(exec, x_v);
    auto gy = gko::matrix::Dense<double>::create(exec, gko::dim<2>(2, 1));
    gy->fill(99.);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(inf_norm_diff(y_ref, y_gpu), tol_f64)
        << "AllZeroValuesInRow double: y[0]=" << y_gpu[0];
}

TEST_F(CsrMergePath, EmptyRowsFloat)
{
    // rows: [dense], [empty], [dense], [empty], [dense]
    std::vector<int> rp = {0, 2, 2, 4, 4, 5};
    std::vector<int> ci = {0, 1, 0, 1, 0};
    std::vector<float> val = {1.f, 2.f, 3.f, 4.f, 5.f};
    std::vector<float> x_v = {1.f, 1.f};
    std::vector<float> y_ref(5);
    cpu_spmv_ref(5, rp, ci, val, x_v, y_ref);

    auto A = make_csr_mp<float>(exec, 5, 2, rp, ci, val);
    auto gx = make_vec<float>(exec, x_v);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(5, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(inf_norm_diff(y_ref, y_gpu), tol_f32)
        << "EmptyRows: empty rows should produce y=0";
}


// ===========================================================================
// GROUP 7: advanced_spmv — y = alpha * A * x + beta * y
//          Tests the scaled code path (separate kernel overload).
// ===========================================================================

TEST_F(CsrMergePath, AdvancedSpMVAlphaBetaFloat)
{
    const int n = 128, nnz_row = 13;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<int> rp(n + 1, 0);
    for (int r = 0; r < n; ++r) rp[r + 1] = rp[r] + nnz_row;
    int nnz = rp[n];
    std::vector<int> ci(nnz);
    std::vector<float> val(nnz), x_v(n), y_init(n);
    for (int r = 0; r < n; ++r)
        for (int j = 0; j < nnz_row; ++j) {
            ci[rp[r] + j] = (r * 31 + j * 17) % n;
            val[rp[r] + j] = dist(rng);
        }
    for (auto& v : x_v) v = dist(rng);
    for (auto& v : y_init) v = dist(rng);

    const float alpha = 2.5f, beta = -0.5f;
    std::vector<float> y_ref = y_init;
    cpu_spmv_ref(n, rp, ci, val, x_v, y_ref, alpha, beta);

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x_v);
    auto gy = make_vec<float>(exec, y_init);
    auto galpha = make_vec<float>(exec, {alpha});
    auto gbeta = make_vec<float>(exec, {beta});
    A->apply(galpha, gx, gbeta, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32)
        << "AdvancedSpMV alpha/beta (float) failed";
}

TEST_F(CsrMergePath, AdvancedSpMVAlphaBetaDouble)
{
    const int n = 128, nnz_row = 13;
    std::mt19937 rng(43);
    std::uniform_real_distribution<double> dist(-1., 1.);
    std::vector<int> rp(n + 1, 0);
    for (int r = 0; r < n; ++r) rp[r + 1] = rp[r] + nnz_row;
    int nnz = rp[n];
    std::vector<int> ci(nnz);
    std::vector<double> val(nnz), x_v(n), y_init(n);
    for (int r = 0; r < n; ++r)
        for (int j = 0; j < nnz_row; ++j) {
            ci[rp[r] + j] = (r * 31 + j * 17) % n;
            val[rp[r] + j] = dist(rng);
        }
    for (auto& v : x_v) v = dist(rng);
    for (auto& v : y_init) v = dist(rng);

    const double alpha = 2.5, beta = -0.5;
    std::vector<double> y_ref = y_init;
    cpu_spmv_ref(n, rp, ci, val, x_v, y_ref, alpha, beta);

    auto A = make_csr_mp<double>(exec, n, n, rp, ci, val);
    auto gx = make_vec<double>(exec, x_v);
    auto gy = make_vec<double>(exec, y_init);
    auto galpha = make_vec<double>(exec, {alpha});
    auto gbeta = make_vec<double>(exec, {beta});
    A->apply(galpha, gx, gbeta, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f64);
}

TEST_F(CsrMergePath, AdvancedSpMVBetaZeroFloat)
{
    const int n = 64;
    std::vector<int> rp(n + 1), ci(n);
    std::vector<float> val(n, 2.f), x_v(n, 1.f), y_init(n, 999.f), y_ref(n);
    std::iota(ci.begin(), ci.end(), 0);
    for (int r = 0; r <= n; ++r) rp[r] = r;
    cpu_spmv_ref(n, rp, ci, val, x_v, y_ref, 3.f, 0.f);

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x_v);
    auto gy = make_vec<float>(exec, y_init);
    auto galpha = make_vec<float>(exec, {3.f});
    auto gbeta = make_vec<float>(exec, {0.f});
    A->apply(galpha, gx, gbeta, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32)
        << "beta=0: old y should be completely ignored";
}


// ===========================================================================
// GROUP 8: Multi-column RHS (SpMM)
//
// The outer loop `for (column_id = 0; column_id < b.size[1]; column_id++)`
// in merge_path_spmv runs each column independently.  Multi-column RHS
// stresses the correctness of that loop.
// ===========================================================================

TEST_F(CsrMergePath, MultiColumnRHSFloat)
{
    const int nrows = 64, ncols = 64, nvecs = 4, nnz_row = 8;
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<int> rp(nrows + 1, 0), ci(nrows * nnz_row);
    std::vector<float> val(nrows * nnz_row);
    for (int r = 0; r < nrows; ++r) rp[r + 1] = rp[r] + nnz_row;
    for (int r = 0; r < nrows; ++r)
        for (int j = 0; j < nnz_row; ++j) {
            ci[rp[r] + j] = (r * 13 + j * 7) % ncols;
            val[rp[r] + j] = dist(rng);
        }

    // Build x as ncols*nvecs row-major dense matrix (make_vec stores row-major)
    std::vector<float> x_data(ncols * nvecs), y_ref_data(nrows * nvecs, 0.f);
    for (auto& v : x_data) v = dist(rng);

    // CPU reference per column — x_data is row-major (r*nvecs+col) as stored by
    // make_vec
    for (int col = 0; col < nvecs; ++col) {
        std::vector<float> x_col(nrows);
        for (int r = 0; r < nrows; ++r) x_col[r] = x_data[r * nvecs + col];
        std::vector<float> y_col(nrows, 0.f);
        cpu_spmv_ref(nrows, rp, ci, val, x_col, y_col);
        for (int r = 0; r < nrows; ++r) y_ref_data[r * nvecs + col] = y_col[r];
    }

    auto A = make_csr_mp<float>(exec, nrows, ncols, rp, ci, val);
    auto gx = make_vec<float>(exec, x_data, nvecs);
    auto gy =
        gko::matrix::Dense<float>::create(exec, gko::dim<2>(nrows, nvecs));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref_data, y_gpu), tol_f32)
        << "MultiColumnRHS (float): SpMM result mismatch";
}

TEST_F(CsrMergePath, MultiColumnRHSDouble)
{
    const int nrows = 64, ncols = 64, nvecs = 4, nnz_row = 8;
    std::mt19937 rng(66);
    std::uniform_real_distribution<double> dist(-1., 1.);
    std::vector<int> rp(nrows + 1, 0), ci(nrows * nnz_row);
    std::vector<double> val(nrows * nnz_row);
    for (int r = 0; r < nrows; ++r) rp[r + 1] = rp[r] + nnz_row;
    for (int r = 0; r < nrows; ++r)
        for (int j = 0; j < nnz_row; ++j) {
            ci[rp[r] + j] = (r * 13 + j * 7) % ncols;
            val[rp[r] + j] = dist(rng);
        }

    std::vector<double> x_data(ncols * nvecs), y_ref_data(nrows * nvecs, 0.);
    for (auto& v : x_data) v = dist(rng);
    // x_data is row-major (r*nvecs+col) as stored by make_vec
    for (int col = 0; col < nvecs; ++col) {
        std::vector<double> x_col(nrows);
        for (int r = 0; r < nrows; ++r) x_col[r] = x_data[r * nvecs + col];
        std::vector<double> y_col(nrows, 0.);
        cpu_spmv_ref(nrows, rp, ci, val, x_col, y_col);
        for (int r = 0; r < nrows; ++r) y_ref_data[r * nvecs + col] = y_col[r];
    }

    auto A = make_csr_mp<double>(exec, nrows, ncols, rp, ci, val);
    auto gx = make_vec<double>(exec, x_data, nvecs);
    auto gy =
        gko::matrix::Dense<double>::create(exec, gko::dim<2>(nrows, nvecs));
    gy->fill(0.);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref_data, y_gpu), tol_f64);
}


// ===========================================================================
// GROUP 9: Determinism — same inputs must produce identical results across
//          repeated launches.  A printf buffer overflow (BUG 2) or race
//          condition would break this.
// ===========================================================================

TEST_F(CsrMergePath, DeterminismFloat)
{
    const int n = 4096, nnz_row = 16;
    std::vector<int> rp(n + 1), ci(n * nnz_row);
    std::vector<float> val(n * nnz_row, 1.f), x_v(n, 1.f);
    for (int r = 0; r <= n; ++r) rp[r] = r * nnz_row;
    for (int r = 0; r < n; ++r)
        for (int j = 0; j < nnz_row; ++j)
            ci[r * nnz_row + j] = (r * 7 + j * 13) % n;

    auto A = make_csr_mp<float>(exec, n, n, rp, ci, val);
    auto gx = make_vec<float>(exec, x_v);

    auto gy1 = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy1->fill(0.f);
    A->apply(gx, gy1);
    auto y1 = to_host(gy1.get());

    auto gy2 = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy2->fill(0.f);
    A->apply(gx, gy2);
    auto y2 = to_host(gy2.get());

    EXPECT_EQ(y1, y2)
        << "Determinism (float, n=4096): non-deterministic results — "
           "possible printf buffer overflow or race condition on Blackwell";
}

TEST_F(CsrMergePath, DeterminismDouble)
{
    const int n = 4096, nnz_row = 16;
    std::vector<int> rp(n + 1), ci(n * nnz_row);
    std::vector<double> val(n * nnz_row, 1.), x_v(n, 1.);
    for (int r = 0; r <= n; ++r) rp[r] = r * nnz_row;
    for (int r = 0; r < n; ++r)
        for (int j = 0; j < nnz_row; ++j)
            ci[r * nnz_row + j] = (r * 7 + j * 13) % n;

    auto A = make_csr_mp<double>(exec, n, n, rp, ci, val);
    auto gx = make_vec<double>(exec, x_v);

    auto gy1 = gko::matrix::Dense<double>::create(exec, gko::dim<2>(n, 1));
    gy1->fill(0.);
    A->apply(gx, gy1);
    auto y1 = to_host(gy1.get());

    auto gy2 = gko::matrix::Dense<double>::create(exec, gko::dim<2>(n, 1));
    gy2->fill(0.);
    A->apply(gx, gy2);
    auto y2 = to_host(gy2.get());

    EXPECT_EQ(y1, y2) << "Determinism (double, n=4096)";
}


// ===========================================================================
// GROUP 10: Randomised stress — many sizes and densities
// ===========================================================================

TEST_F(CsrMergePath, RandomisedStressFloat)
{
    std::mt19937 rng(12345);
    // {nrows, ncols, nnz_per_row}
    const std::vector<std::tuple<int, int, int>> configs = {
        {1, 1, 1},
        {32, 32, 4},
        {128, 128, 8},
        {513, 513, 1},  // prime-ish, diagonal
        {1024, 512, 7},
        {2048, 2048, 3},
        {100, 100, 100},  // fully dense
        // block-boundary-critical sizes
        {384, 384, 1},  // n+nnz = 768 = 128*6
        {256, 256, 2},  // n+nnz = 768
        {768, 768, 1},  // n+nnz = 1536 = 128*12
    };

    for (auto [nr, nc, nnz_row] : configs) {
        nnz_row = std::min(nnz_row, nc);
        std::vector<int> rp(nr + 1, 0), ci(nr * nnz_row);
        std::vector<float> val(nr * nnz_row), x_v(nc), y_ref(nr);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (int r = 0; r < nr; ++r) rp[r + 1] = rp[r] + nnz_row;
        for (int r = 0; r < nr; ++r)
            for (int j = 0; j < nnz_row; ++j) {
                ci[rp[r] + j] = (r * 37 + j * 19) % nc;
                val[rp[r] + j] = dist(rng);
            }
        for (auto& v : x_v) v = dist(rng);
        cpu_spmv_ref(nr, rp, ci, val, x_v, y_ref);

        auto A = make_csr_mp<float>(exec, nr, nc, rp, ci, val);
        auto gx = make_vec<float>(exec, x_v);
        auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(nr, 1));
        gy->fill(0.f);
        A->apply(gx, gy);
        auto y_gpu = to_host(gy.get());
        EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32)
            << "RandomisedStressFloat: nr=" << nr << " nc=" << nc
            << " nnz_row=" << nnz_row
            << " max_err=" << rel_inf_error(y_ref, y_gpu);
    }
}

TEST_F(CsrMergePath, RandomisedStressDouble)
{
    std::mt19937 rng(99999);
    const std::vector<std::tuple<int, int, int>> configs = {
        {1, 1, 1},      {32, 32, 4},     {128, 128, 8},   {513, 513, 1},
        {1024, 512, 7}, {2048, 2048, 3}, {100, 100, 100}, {384, 384, 1},
        {256, 256, 2},  {768, 768, 1},
    };

    for (auto [nr, nc, nnz_row] : configs) {
        nnz_row = std::min(nnz_row, nc);
        std::vector<int> rp(nr + 1, 0), ci(nr * nnz_row);
        std::vector<double> val(nr * nnz_row), x_v(nc), y_ref(nr);
        std::uniform_real_distribution<double> dist(-1., 1.);
        for (int r = 0; r < nr; ++r) rp[r + 1] = rp[r] + nnz_row;
        for (int r = 0; r < nr; ++r)
            for (int j = 0; j < nnz_row; ++j) {
                ci[rp[r] + j] = (r * 37 + j * 19) % nc;
                val[rp[r] + j] = dist(rng);
            }
        for (auto& v : x_v) v = dist(rng);
        cpu_spmv_ref(nr, rp, ci, val, x_v, y_ref);

        auto A = make_csr_mp<double>(exec, nr, nc, rp, ci, val);
        auto gx = make_vec<double>(exec, x_v);
        auto gy = gko::matrix::Dense<double>::create(exec, gko::dim<2>(nr, 1));
        gy->fill(0.);
        A->apply(gx, gy);
        auto y_gpu = to_host(gy.get());
        EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f64)
            << "RandomisedStressDouble: nr=" << nr << " nc=" << nc
            << " nnz_row=" << nnz_row;
    }
}


// ===========================================================================
// GROUP 11: Regression — exact value checks
// ===========================================================================

TEST_F(CsrMergePath, Tridiagonal3x3Float)
{
    //  [2 -1  0]   [1]   [ 0]   <- 2*1 + (-1)*2 = 0
    //  [-1 2 -1] * [2] = [ 0]   <- -1*1 + 2*2 + (-1)*3 = 0
    //  [0 -1  2]   [3]   [ 4]   <- (-1)*2 + 2*3 = 4
    std::vector<int> rp = {0, 2, 5, 7};
    std::vector<int> ci = {0, 1, 0, 1, 2, 1, 2};
    std::vector<float> val = {2, -1, -1, 2, -1, -1, 2};
    std::vector<float> x_v = {1, 2, 3};

    auto A = make_csr_mp<float>(exec, 3, 3, rp, ci, val);
    auto gx = make_vec<float>(exec, x_v);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(3, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());

    EXPECT_NEAR(y_gpu[0], 0.f, 1e-5f);
    EXPECT_NEAR(y_gpu[1], 0.f, 1e-5f);
    EXPECT_NEAR(y_gpu[2], 4.f, 1e-5f);
}

TEST_F(CsrMergePath, Tridiagonal3x3Double)
{
    std::vector<int> rp = {0, 2, 5, 7};
    std::vector<int> ci = {0, 1, 0, 1, 2, 1, 2};
    std::vector<double> val = {2, -1, -1, 2, -1, -1, 2};
    std::vector<double> x_v = {1, 2, 3};

    auto A = make_csr_mp<double>(exec, 3, 3, rp, ci, val);
    auto gx = make_vec<double>(exec, x_v);
    auto gy = gko::matrix::Dense<double>::create(exec, gko::dim<2>(3, 1));
    gy->fill(0.);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());

    EXPECT_NEAR(y_gpu[0], 0., 1e-13);
    EXPECT_NEAR(y_gpu[1], 0., 1e-13);
    EXPECT_NEAR(y_gpu[2], 4., 1e-13);
}

TEST_F(CsrMergePath, VariableRowLengthsFloat)
{
    // Row i has (i % 64 + 1) nonzeros — highly skewed, stresses merge_path
    // load balancing across blocks.
    const int n = 256;
    std::vector<int> rp(n + 1, 0);
    for (int r = 0; r < n; ++r) rp[r + 1] = rp[r] + (r % 64 + 1);
    int nnz = rp[n];
    const int ncols = 64;
    std::vector<int> ci(nnz);
    std::vector<float> val(nnz), x_v(ncols, 1.f), y_ref(n);
    for (int r = 0; r < n; ++r) {
        int len = rp[r + 1] - rp[r];
        for (int j = 0; j < len; ++j) {
            ci[rp[r] + j] = j % ncols;
            val[rp[r] + j] = 1.f / float(len);
        }
    }
    cpu_spmv_ref(n, rp, ci, val, x_v, y_ref);

    auto A = make_csr_mp<float>(exec, n, ncols, rp, ci, val);
    auto gx = make_vec<float>(exec, x_v);
    auto gy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(n, 1));
    gy->fill(0.f);
    A->apply(gx, gy);
    auto y_gpu = to_host(gy.get());
    EXPECT_LT(rel_inf_error(y_ref, y_gpu), tol_f32)
        << "VariableRowLengths: merge_path load-balance error";
}
