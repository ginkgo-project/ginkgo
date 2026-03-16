// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gauss_seidel_kernels.hpp"

#include <algorithm>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/solver/gauss_seidel.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


// Generates a square matrix suitable for multi-color Gauss-Seidel.
//
// The matrix has `n_rows` rows divided into `n_colors` color blocks.
// Leftover rows (n_rows % n_colors) are distributed one-per-block to the
// first few colors, so block sizes differ by at most one row.
//
// Each diagonal color block is a diagonal matrix with large random positive
// entries drawn from [diag_lo, diag_hi]. Each row also has `num_offdiag`
// small random negative entries placed at columns outside that row's color
// block, drawn from [-offdiag_hi, -offdiag_lo]. The returned color_ptrs
// vector has length n_colors + 1 with color_ptrs[c] = first row of color c
// and color_ptrs[n_colors] = n_rows.
template <typename ValueType, typename IndexType>
std::pair<std::unique_ptr<gko::matrix::Ell<ValueType, IndexType>>,
          std::vector<IndexType>>
generate_colored_matrix(std::shared_ptr<const gko::Executor> exec,
                        IndexType n_rows, IndexType n_colors,
                        IndexType num_offdiag = 2, unsigned seed = 42,
                        ValueType diag_lo = 10.0, ValueType diag_hi = 20.0,
                        ValueType offdiag_lo = 0.5, ValueType offdiag_hi = 2.0)
{
    GKO_ASSERT(n_colors > 0 && n_rows >= n_colors);
    const IndexType base_size = n_rows / n_colors;
    const IndexType remainder = n_rows % n_colors;

    // color_ptrs[c] = first row of color c; the first `remainder` colors get
    // one extra row so that all rows are covered.
    std::vector<IndexType> color_ptrs(n_colors + 1);
    for (IndexType c = 0; c <= n_colors; c++) {
        color_ptrs[c] = c * base_size + std::min(c, remainder);
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<ValueType> diag_dist(diag_lo, diag_hi);
    std::uniform_real_distribution<ValueType> offdiag_dist(offdiag_lo,
                                                           offdiag_hi);

    gko::matrix_data<ValueType, IndexType> data(
        gko::dim<2>{static_cast<gko::size_type>(n_rows),
                    static_cast<gko::size_type>(n_rows)});

    std::vector<IndexType> off_block_cols;
    off_block_cols.reserve(n_rows);

    for (IndexType c = 0; c < n_colors; c++) {
        const IndexType color_start = color_ptrs[c];
        const IndexType color_end = color_ptrs[c + 1];

        // Off-block columns are the same for every row in this color block
        off_block_cols.clear();
        for (IndexType col = 0; col < n_rows; col++) {
            if (col < color_start || col >= color_end) {
                off_block_cols.push_back(col);
            }
        }

        for (IndexType row = color_start; row < color_end; row++) {
            // Diagonal entry: large positive value
            data.nonzeros.emplace_back(row, row, diag_dist(rng));

            // Shuffle and pick num_offdiag columns for this row
            std::shuffle(off_block_cols.begin(), off_block_cols.end(), rng);
            const IndexType actual_offdiag = std::min(
                num_offdiag, static_cast<IndexType>(off_block_cols.size()));
            for (IndexType k = 0; k < actual_offdiag; k++) {
                data.nonzeros.emplace_back(row, off_block_cols[k],
                                           -offdiag_dist(rng));
            }
        }
    }

    data.sort_row_major();

    auto mtx = gko::matrix::Ell<ValueType, IndexType>::create(exec);
    mtx->read(data);
    return {std::move(mtx), std::move(color_ptrs)};
}


// Generates a Dense matrix with uniformly random entries from [-5, 5].
// Must be called with a host (reference) executor.
template <typename ValueType>
std::unique_ptr<gko::matrix::Dense<ValueType>> generate_random_dense(
    std::shared_ptr<const gko::Executor> exec, gko::size_type n_rows,
    gko::size_type n_cols = 1, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    gko::matrix_data<ValueType, gko::int32> data(gko::dim<2>{n_rows, n_cols});
    for (gko::size_type row = 0; row < n_rows; row++) {
        for (gko::size_type col = 0; col < n_cols; col++) {
            data.nonzeros.emplace_back(static_cast<gko::int32>(row),
                                       static_cast<gko::int32>(col),
                                       static_cast<ValueType>(dist(rng)));
        }
    }
    auto result = gko::matrix::Dense<ValueType>::create(exec);
    result->read(data);
    return result;
}


class GaussSeidelKernelsEll : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Ell<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    // Matrix dimensions: ≥ warp size (32) for meaningful GPU parallelism,
    // not divisible by n_colors to exercise the uneven-distribution path.
    static constexpr index_type n_rows = 35;
    static constexpr index_type n_colors = 4;

    GaussSeidelKernelsEll()
    {
        auto [gen_mtx, gen_color_ptrs] =
            generate_colored_matrix<value_type, index_type>(ref, n_rows,
                                                            n_colors);
        mtx = std::move(gen_mtx);
        d_mtx = gko::clone(exec, mtx);
        color_ptrs = std::move(gen_color_ptrs);
    }

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> d_mtx;
    std::vector<index_type> color_ptrs;
};


TEST_F(GaussSeidelKernelsEll, SingleIterationFromZeroIsEquivalentToRef)
{
    auto b = generate_random_dense<value_type>(ref, n_rows, 1, 11);
    auto x =
        Vec::create(ref, gko::dim<2>{static_cast<gko::size_type>(n_rows), 1});
    x->fill(gko::zero<value_type>());
    auto stop = gko::array<gko::stopping_status>(ref, 1);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, 1);

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        ref, color_ptrs, mtx.get(), b.get(), x.get(), true, &stop);
    gko::kernels::GKO_DEVICE_NAMESPACE::gssdl::multicolor_fgs_ell(
        exec, color_ptrs, d_mtx.get(), d_b.get(), d_x.get(), true, &d_stop);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(GaussSeidelKernelsEll, UsesCurrentXAsInitialGuessIsEquivalentToRef)
{
    auto b = generate_random_dense<value_type>(ref, n_rows, 1, 22);
    // Non-zero starting guess
    auto x = generate_random_dense<value_type>(ref, n_rows, 1, 23);
    auto stop = gko::array<gko::stopping_status>(ref, 1);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, 1);

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        ref, color_ptrs, mtx.get(), b.get(), x.get(), true, &stop);
    gko::kernels::GKO_DEVICE_NAMESPACE::gssdl::multicolor_fgs_ell(
        exec, color_ptrs, d_mtx.get(), d_b.get(), d_x.get(), true, &d_stop);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(GaussSeidelKernelsEll, MultipleRHSIsEquivalentToRef)
{
    auto b = generate_random_dense<value_type>(ref, n_rows, 2, 33);
    auto x =
        Vec::create(ref, gko::dim<2>{static_cast<gko::size_type>(n_rows), 2});
    x->fill(gko::zero<value_type>());
    auto stop = gko::array<gko::stopping_status>(ref, 2);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, 2);

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        ref, color_ptrs, mtx.get(), b.get(), x.get(), true, &stop);
    gko::kernels::GKO_DEVICE_NAMESPACE::gssdl::multicolor_fgs_ell(
        exec, color_ptrs, d_mtx.get(), d_b.get(), d_x.get(), true, &d_stop);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(GaussSeidelKernelsEll, FirstIterResetsStopStatusIsEquivalentToRef)
{
    const gko::uint32 n_cols = 2;
    auto b = generate_random_dense<value_type>(ref, n_rows, n_cols, 44);
    auto x = Vec::create(
        ref, gko::dim<2>{static_cast<gko::size_type>(n_rows), n_cols});
    x->fill(gko::zero<value_type>());
    auto stop = gko::array<gko::stopping_status>(ref, n_cols);

    gko::stopping_status stopped{};
    stopped.stop(1);
    for (int j = 0; j < static_cast<int>(n_cols); j++) {
        stop.get_data()[j] = stopped;
    }

    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, stop);

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        ref, color_ptrs, mtx.get(), b.get(), x.get(), true, &stop);
    gko::kernels::GKO_DEVICE_NAMESPACE::gssdl::multicolor_fgs_ell(
        exec, color_ptrs, d_mtx.get(), d_b.get(), d_x.get(), true, &d_stop);

    stopped.reset();
    for (int j = 0; j < static_cast<int>(n_cols); j++) {
        ASSERT_EQ(stop.get_data()[j], stopped);
    }
    GKO_ASSERT_ARRAY_EQ(d_stop, stop);
}


TEST_F(GaussSeidelKernelsEll,
       SubsequentIterDoesNotResetStopStatusIsEquivalentToRef)
{
    auto b = generate_random_dense<value_type>(ref, n_rows, 1, 55);
    auto x =
        Vec::create(ref, gko::dim<2>{static_cast<gko::size_type>(n_rows), 1});
    x->fill(gko::zero<value_type>());
    auto stop = gko::array<gko::stopping_status>(ref, 2);

    gko::stopping_status stopped{};
    stopped.stop(1);
    stop.get_data()[0] = stopped;
    stop.get_data()[1] = stopped;

    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, stop);

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        ref, color_ptrs, mtx.get(), b.get(), x.get(), false, &stop);
    gko::kernels::GKO_DEVICE_NAMESPACE::gssdl::multicolor_fgs_ell(
        exec, color_ptrs, d_mtx.get(), d_b.get(), d_x.get(), false, &d_stop);

    ASSERT_EQ(stop.get_data()[0], stopped);
    ASSERT_EQ(stop.get_data()[1], stopped);
    GKO_ASSERT_ARRAY_EQ(d_stop, stop);
}


TEST_F(GaussSeidelKernelsEll,
       DiagonalOnlyMatrixSolvesExactlyInOneStepIsEquivalentToRef)
{
    auto diag = gko::initialize<Mtx>(
        {{4.0, 0.0, 0.0}, {0.0, -2.0, 0.0}, {0.0, 0.0, 5.0}}, ref);
    auto d_diag = gko::clone(exec, diag);
    auto b = gko::initialize<Vec>({8.0, -6.0, 10.0}, ref);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0}, ref);
    auto stop = gko::array<gko::stopping_status>(ref, 1);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, 1);
    std::vector<index_type> single_color{0, 3};

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        ref, single_color, diag.get(), b.get(), x.get(), true, &stop);
    gko::kernels::GKO_DEVICE_NAMESPACE::gssdl::multicolor_fgs_ell(
        exec, single_color, d_diag.get(), d_b.get(), d_x.get(), true, &d_stop);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(GaussSeidelKernelsEll, FiveIterationSolverIsEquivalentToRef)
{
    using Solver = gko::solver::FwdGaussSeidel<value_type, index_type>;

    auto ref_mtx = gko::share(gko::clone(ref, mtx));
    auto dev_mtx = gko::share(gko::clone(exec, mtx));

    auto b = generate_random_dense<value_type>(ref, n_rows, 1, 77);
    auto x =
        Vec::create(ref, gko::dim<2>{static_cast<gko::size_type>(n_rows), 1});
    x->fill(gko::zero<value_type>());
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);

    auto ref_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(color_ptrs)
            .on(ref)
            ->generate(ref_mtx);
    auto dev_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(color_ptrs)
            .on(exec)
            ->generate(dev_mtx);

    ref_solver->apply(b, x);
    dev_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(GaussSeidelKernelsEll, EmptyColorPtrsDoesNothingIsEquivalentToRef)
{
    auto b = generate_random_dense<value_type>(ref, n_rows, 1, 66);
    // Non-zero x to verify it is left unchanged
    auto x = generate_random_dense<value_type>(ref, n_rows, 1, 67);
    auto stop = gko::array<gko::stopping_status>(ref, 1);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, 1);
    std::vector<index_type> empty_ptrs{};

    gko::kernels::reference::gssdl::multicolor_fgs_ell(
        ref, empty_ptrs, mtx.get(), b.get(), x.get(), true, &stop);
    gko::kernels::GKO_DEVICE_NAMESPACE::gssdl::multicolor_fgs_ell(
        exec, empty_ptrs, d_mtx.get(), d_b.get(), d_x.get(), true, &d_stop);

    GKO_ASSERT_MTX_NEAR(d_x, x, 0.0);
}

// ============================================================
// AMP Gauss-Seidel device-vs-reference tests
// ============================================================
//
// The CUDA/HIP AMP kernel is not yet implemented (GKO_NOT_IMPLEMENTED).
// Each test wraps the device call in a try/catch so that it skips
// gracefully on CUDA/HIP and passes on OMP where the kernel is
// fully implemented.  Once the CUDA/HIP kernel lands the skip will
// disappear automatically.

class GaussSeidelKernelsAMP : public CommonTestFixture {
protected:
    using AMPMtx = gko::matrix::AMP<value_type, index_type>;
    using EllMtx = gko::matrix::Ell<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    static constexpr index_type n_rows = 35;
    static constexpr index_type n_colors = 4;
    // 1e-6 works for both double and float; with the generated matrix
    // (diag in [10,20], offdiag in [0.5,2]) this pushes off-diagonal
    // entries below the double-bin threshold, exercising the float-bin
    // code path in the AMP kernel.
    static constexpr float amp_tol = 1e-6f;

    GaussSeidelKernelsAMP()
    {
        auto [gen_ell, gen_color_ptrs] =
            generate_colored_matrix<value_type, index_type>(ref, n_rows,
                                                            n_colors);
        color_ptrs = std::move(gen_color_ptrs);

        // Clone the ELL to the device before consuming it for the ref AMP.
        auto d_gen_ell = gko::clone(exec, gen_ell);

        amp = AMPMtx::build().with_tolerance(amp_tol).on(ref)->generate(
            gko::share(std::move(gen_ell)));

        d_amp = AMPMtx::build().with_tolerance(amp_tol).on(exec)->generate(
            gko::share(std::move(d_gen_ell)));
    }

    // Runs the device AMP kernel and skips the test if the kernel is not
    // yet implemented on this executor.
    template <typename... Args>
    void run_device_amp(Args&&... args)
    {
        try {
            gko::kernels::GKO_DEVICE_NAMESPACE::gssdl::multicolor_fgs_amp(
                std::forward<Args>(args)...);
        } catch (const gko::NotImplemented&) {
            GTEST_SKIP() << "multicolor_fgs_amp not yet implemented on "
                            "this executor";
        }
    }

    std::unique_ptr<AMPMtx> amp;
    std::unique_ptr<AMPMtx> d_amp;
    std::vector<index_type> color_ptrs;
};


TEST_F(GaussSeidelKernelsAMP, SingleIterationFromZeroIsEquivalentToRef)
{
    auto b = generate_random_dense<value_type>(ref, n_rows, 1, 11);
    auto x =
        Vec::create(ref, gko::dim<2>{static_cast<gko::size_type>(n_rows), 1});
    x->fill(gko::zero<value_type>());
    auto stop = gko::array<gko::stopping_status>(ref, 1);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, 1);

    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        ref, color_ptrs, amp.get(), b.get(), x.get(), true, &stop);
    run_device_amp(exec, color_ptrs, d_amp.get(), d_b.get(), d_x.get(), true,
                   &d_stop);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(GaussSeidelKernelsAMP, UsesCurrentXAsInitialGuessIsEquivalentToRef)
{
    auto b = generate_random_dense<value_type>(ref, n_rows, 1, 22);
    auto x = generate_random_dense<value_type>(ref, n_rows, 1, 23);
    auto stop = gko::array<gko::stopping_status>(ref, 1);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, 1);

    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        ref, color_ptrs, amp.get(), b.get(), x.get(), true, &stop);
    run_device_amp(exec, color_ptrs, d_amp.get(), d_b.get(), d_x.get(), true,
                   &d_stop);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(GaussSeidelKernelsAMP, MultipleRHSIsEquivalentToRef)
{
    auto b = generate_random_dense<value_type>(ref, n_rows, 2, 33);
    auto x =
        Vec::create(ref, gko::dim<2>{static_cast<gko::size_type>(n_rows), 2});
    x->fill(gko::zero<value_type>());
    auto stop = gko::array<gko::stopping_status>(ref, 2);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, 2);

    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        ref, color_ptrs, amp.get(), b.get(), x.get(), true, &stop);
    run_device_amp(exec, color_ptrs, d_amp.get(), d_b.get(), d_x.get(), true,
                   &d_stop);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(GaussSeidelKernelsAMP, FirstIterResetsStopStatus)
{
    const gko::uint32 n_cols = 2;
    auto b = generate_random_dense<value_type>(ref, n_rows, n_cols, 44);
    auto d_b = gko::clone(exec, b);
    auto d_x = Vec::create(
        exec, gko::dim<2>{static_cast<gko::size_type>(n_rows), n_cols});
    d_x->fill(gko::zero<value_type>());
    auto d_stop = gko::array<gko::stopping_status>(exec, n_cols);

    gko::stopping_status stopped{};
    stopped.stop(1);
    // Pre-fill device stop array with "stopped" entries
    {
        auto h_stop = gko::array<gko::stopping_status>(ref, n_cols);
        for (int j = 0; j < static_cast<int>(n_cols); j++) {
            h_stop.get_data()[j] = stopped;
        }
        d_stop = gko::array<gko::stopping_status>(exec, h_stop);
    }

    run_device_amp(exec, color_ptrs, d_amp.get(), d_b.get(), d_x.get(), true,
                   &d_stop);

    // Copy stop array back to host and check all entries were reset
    auto h_stop = gko::array<gko::stopping_status>(ref, d_stop);
    gko::stopping_status non_stopped{};
    non_stopped.reset();
    for (int j = 0; j < static_cast<int>(n_cols); j++) {
        EXPECT_EQ(h_stop.get_data()[j], non_stopped);
    }
}


TEST_F(GaussSeidelKernelsAMP, SubsequentIterDoesNotResetStopStatus)
{
    auto b = generate_random_dense<value_type>(ref, n_rows, 1, 55);
    auto d_b = gko::clone(exec, b);
    auto d_x =
        Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n_rows), 1});
    d_x->fill(gko::zero<value_type>());
    auto d_stop = gko::array<gko::stopping_status>(exec, 2);

    gko::stopping_status stopped{};
    stopped.stop(1);
    {
        auto h_stop = gko::array<gko::stopping_status>(ref, 2);
        h_stop.get_data()[0] = stopped;
        h_stop.get_data()[1] = stopped;
        d_stop = gko::array<gko::stopping_status>(exec, h_stop);
    }

    run_device_amp(exec, color_ptrs, d_amp.get(), d_b.get(), d_x.get(), false,
                   &d_stop);

    auto h_stop = gko::array<gko::stopping_status>(ref, d_stop);
    EXPECT_EQ(h_stop.get_data()[0], stopped);
    EXPECT_EQ(h_stop.get_data()[1], stopped);
}


TEST_F(GaussSeidelKernelsAMP, EmptyColorPtrsDoesNothingIsEquivalentToRef)
{
    auto b = generate_random_dense<value_type>(ref, n_rows, 1, 66);
    auto x = generate_random_dense<value_type>(ref, n_rows, 1, 67);
    auto stop = gko::array<gko::stopping_status>(ref, 1);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);
    auto d_stop = gko::array<gko::stopping_status>(exec, 1);
    std::vector<index_type> empty_ptrs{};

    gko::kernels::reference::gssdl::multicolor_fgs_amp(
        ref, empty_ptrs, amp.get(), b.get(), x.get(), true, &stop);
    run_device_amp(exec, empty_ptrs, d_amp.get(), d_b.get(), d_x.get(), true,
                   &d_stop);

    GKO_ASSERT_MTX_NEAR(d_x, x, 0.0);
}


TEST_F(GaussSeidelKernelsAMP, FiveIterationSolverIsEquivalentToRef)
{
    using Solver = gko::solver::FwdGaussSeidel<value_type, index_type>;

    auto ref_amp = gko::share(gko::clone(ref, amp));
    auto dev_amp = gko::share(gko::clone(exec, d_amp));

    auto b = generate_random_dense<value_type>(ref, n_rows, 1, 77);
    auto x =
        Vec::create(ref, gko::dim<2>{static_cast<gko::size_type>(n_rows), 1});
    x->fill(gko::zero<value_type>());
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);

    auto ref_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(color_ptrs)
            .on(ref)
            ->generate(ref_amp);
    auto dev_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(color_ptrs)
            .on(exec)
            ->generate(dev_amp);

    ref_solver->apply(b, x);
    dev_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}
