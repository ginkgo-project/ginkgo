/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/sparsity_csr_kernels.hpp"


#include <memory>
#include <random>
#include <utility>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils.hpp"


namespace {


class SparsityCsr : public ::testing::Test {
protected:
    using Mtx = gko::matrix::SparsityCsr<>;
    using Vec = gko::matrix::Dense<>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;
    using ComplexMtx = gko::matrix::SparsityCsr<std::complex<double>>;

    SparsityCsr() : mtx_size(532, 231), rand_engine(42) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown()
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_sparsity_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols), 1.0,
            rand_engine, ref);
    }

    void set_up_apply_data(int num_vectors = 1)
    {
        mtx = Mtx::create(ref);
        mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[1], 1));
        complex_mtx = ComplexMtx::create(ref);
        complex_mtx->copy_from(
            gen_mtx<ComplexVec>(mtx_size[0], mtx_size[1], 1));
        expected = gen_mtx<Vec>(mtx_size[0], num_vectors, 1);
        y = gen_mtx<Vec>(mtx_size[1], num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(omp);
        dmtx->copy_from(mtx.get());
        complex_dmtx = ComplexMtx::create(omp);
        complex_dmtx->copy_from(complex_mtx.get());
        dresult = Vec::create(omp);
        dresult->copy_from(expected.get());
        dy = Vec::create(omp);
        dy->copy_from(y.get());
        dalpha = Vec::create(omp);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(omp);
        dbeta->copy_from(beta.get());
    }

    struct matrix_pair {
        std::unique_ptr<Mtx> ref;
        std::unique_ptr<Mtx> omp;
    };

    matrix_pair gen_unsorted_mtx()
    {
        constexpr int min_nnz_per_row = 2;  // Must be at least 2
        auto local_mtx_ref =
            gen_mtx<Mtx>(mtx_size[0], mtx_size[1], min_nnz_per_row);
        for (size_t row = 0; row < mtx_size[0]; ++row) {
            const auto row_ptrs = local_mtx_ref->get_const_row_ptrs();
            const auto start_row = row_ptrs[row];
            auto col_idx = local_mtx_ref->get_col_idxs() + start_row;
            const auto nnz_in_this_row = row_ptrs[row + 1] - row_ptrs[row];
            auto swap_idx_dist =
                std::uniform_int_distribution<>(0, nnz_in_this_row - 1);
            // shuffle `nnz_in_this_row / 2` times
            for (size_t perm = 0; perm < nnz_in_this_row; perm += 2) {
                const auto idx1 = swap_idx_dist(rand_engine);
                const auto idx2 = swap_idx_dist(rand_engine);
                std::swap(col_idx[idx1], col_idx[idx2]);
            }
        }
        auto local_mtx_omp = Mtx::create(omp);
        local_mtx_omp->copy_from(local_mtx_ref.get());

        return {std::move(local_mtx_ref), std::move(local_mtx_omp)};
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    const gko::dim<2> mtx_size;
    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<ComplexMtx> complex_mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<ComplexMtx> complex_dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(SparsityCsr, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(SparsityCsr, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(SparsityCsr, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(SparsityCsr, AdvancedApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(SparsityCsr, TransposeIsEquivalentToRef)
{
    set_up_apply_data();

    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(d_trans.get()),
                        static_cast<Mtx *>(trans.get()), 0.0);
}


TEST_F(SparsityCsr, CountsNumberOfDiagElementsIsEqualToRef)
{
    set_up_apply_data();
    gko::size_type num_diags = 0;
    gko::size_type d_num_diags = 0;

    gko::kernels::reference::sparsity_csr::count_num_diagonal_elements(
        ref, mtx.get(), &num_diags);
    gko::kernels::omp::sparsity_csr::count_num_diagonal_elements(
        omp, dmtx.get(), &d_num_diags);

    ASSERT_EQ(d_num_diags, num_diags);
}


TEST_F(SparsityCsr, RemovesDiagElementsKernelIsEquivalentToRef)
{
    set_up_apply_data();
    gko::size_type num_diags = 0;
    gko::kernels::reference::sparsity_csr::count_num_diagonal_elements(
        ref, mtx.get(), &num_diags);
    auto tmp =
        Mtx::create(ref, mtx->get_size(), mtx->get_num_nonzeros() - num_diags);
    auto d_tmp = Mtx::create(omp, dmtx->get_size(),
                             dmtx->get_num_nonzeros() - num_diags);

    gko::kernels::reference::sparsity_csr::remove_diagonal_elements(
        ref, tmp.get(), mtx->get_const_row_ptrs(), mtx->get_const_col_idxs());
    gko::kernels::omp::sparsity_csr::remove_diagonal_elements(
        omp, d_tmp.get(), dmtx->get_const_row_ptrs(),
        dmtx->get_const_col_idxs());

    GKO_ASSERT_MTX_NEAR(tmp.get(), d_tmp.get(), 0.0);
}


TEST_F(SparsityCsr, RecognizeSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data();
    bool is_sorted_omp{};
    bool is_sorted_ref{};

    is_sorted_ref = mtx->is_sorted_by_column_index();
    is_sorted_omp = dmtx->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_omp);
}


TEST_F(SparsityCsr, RecognizeUnsortedMatrixIsEquivalentToRef)
{
    auto uns_mtx = gen_unsorted_mtx();
    bool is_sorted_omp{};
    bool is_sorted_ref{};

    is_sorted_ref = uns_mtx.ref->is_sorted_by_column_index();
    is_sorted_omp = uns_mtx.omp->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_omp);
}


TEST_F(SparsityCsr, SortSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(SparsityCsr, SortUnsortedMatrixIsEquivalentToRef)
{
    auto uns_mtx = gen_unsorted_mtx();

    uns_mtx.ref->sort_by_column_index();
    uns_mtx.omp->sort_by_column_index();

    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(uns_mtx.ref, uns_mtx.omp, 0);
}


}  // namespace
