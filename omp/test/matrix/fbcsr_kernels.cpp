/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/matrix/fbcsr.hpp>


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/fbcsr_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"


namespace {


class Fbcsr : public ::testing::Test {
protected:
    using real_type = double;
    using index_type = int;
    using Arr = gko::Array<index_type>;
    using Mtx = gko::matrix::Fbcsr<real_type, index_type>;
    using Vec = gko::matrix::Dense<real_type>;
    using ComplexVec = gko::matrix::Dense<std::complex<real_type>>;
    using ComplexMtx = gko::matrix::Fbcsr<std::complex<real_type>>;

    Fbcsr() : num_brows{232}, num_bcols{131}, blk_sz{3}, rand_engine(42) {}

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
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(int num_vectors = 1)
    {
        mtx = gko::test::generate_random_fbcsr<real_type>(
            ref, rand_engine, num_brows, num_bcols, blk_sz, false, false);
        complex_mtx = gko::test::generate_random_fbcsr<std::complex<real_type>>(
            ref, rand_engine, num_brows, num_bcols, blk_sz, false, false);
        square_mtx = gko::test::generate_random_fbcsr<real_type>(
            ref, rand_engine, num_brows, num_brows, blk_sz, false, false);
        dmtx = Mtx::create(omp);
        dmtx->copy_from(mtx.get());
        complex_dmtx = ComplexMtx::create(omp);
        complex_dmtx->copy_from(complex_mtx.get());
        square_dmtx = Mtx::create(omp);
        square_dmtx->copy_from(square_mtx.get());
    }

    struct matrix_pair {
        std::unique_ptr<Mtx> ref;
        std::unique_ptr<Mtx> omp;
    };

    matrix_pair gen_unsorted_mtx()
    {
        constexpr int min_nnz_per_row{2};
        auto local_mtx_ref = gko::test::generate_random_fbcsr<real_type>(
            ref, rand_engine, num_brows, num_bcols, blk_sz, false, true);

        auto local_mtx_omp = Mtx::create(omp);
        local_mtx_omp->copy_from(local_mtx_ref.get());

        return matrix_pair{std::move(local_mtx_ref), std::move(local_mtx_omp)};
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    const index_type num_brows;
    const index_type num_bcols;
    const int blk_sz;
    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<ComplexMtx> complex_mtx;
    std::unique_ptr<Mtx> square_mtx;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<ComplexMtx> complex_dmtx;
    std::unique_ptr<Mtx> square_dmtx;
};


TEST_F(Fbcsr, TransposeIsEquivalentToRef)
{
    set_up_apply_data();

    auto trans = gko::as<Mtx>(mtx->transpose());
    auto d_trans = gko::as<Mtx>(dmtx->transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    // ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Fbcsr, ConjugateTransposeIsEquivalentToRef)
{
    set_up_apply_data();

    auto trans = gko::as<ComplexMtx>(complex_mtx->conj_transpose());
    auto d_trans = gko::as<ComplexMtx>(complex_dmtx->conj_transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    // ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Fbcsr, CalculatesNonzerosPerRow)
{
    set_up_apply_data();
    gko::Array<gko::size_type> row_nnz(ref, mtx->get_size()[0]);
    gko::Array<gko::size_type> drow_nnz(omp, dmtx->get_size()[0]);

    gko::kernels::reference::fbcsr::calculate_nonzeros_per_row(ref, mtx.get(),
                                                               &row_nnz);
    gko::kernels::omp::fbcsr::calculate_nonzeros_per_row(omp, dmtx.get(),
                                                         &drow_nnz);

    GKO_ASSERT_ARRAY_EQ(row_nnz, drow_nnz);
}


TEST_F(Fbcsr, RecognizeSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data();
    bool is_sorted_omp{};
    bool is_sorted_ref{};

    is_sorted_ref = mtx->is_sorted_by_column_index();
    is_sorted_omp = dmtx->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_omp);
}


TEST_F(Fbcsr, RecognizeUnsortedMatrixIsEquivalentToRef)
{
    auto uns_mtx = gen_unsorted_mtx();
    bool is_sorted_omp{};
    bool is_sorted_ref{};

    is_sorted_ref = uns_mtx.ref->is_sorted_by_column_index();
    is_sorted_omp = uns_mtx.omp->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_omp);
}


TEST_F(Fbcsr, SortSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Fbcsr, SortUnsortedMatrixIsEquivalentToRef)
{
    auto uns_mtx = gen_unsorted_mtx();

    uns_mtx.ref->sort_by_column_index();
    uns_mtx.omp->sort_by_column_index();

    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(uns_mtx.ref, uns_mtx.omp, 0);
}


TEST_F(Fbcsr, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data();

    auto diag = mtx->extract_diagonal();
    auto ddiag = dmtx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Fbcsr, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->compute_absolute_inplace();
    dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 1e-14);
}


TEST_F(Fbcsr, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_mtx = mtx->compute_absolute();
    auto dabs_mtx = dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, 1e-14);
}


TEST_F(Fbcsr, InplaceAbsoluteComplexMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    complex_mtx->compute_absolute_inplace();
    complex_dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(complex_mtx, complex_dmtx, 1e-14);
}


TEST_F(Fbcsr, OutplaceAbsoluteComplexMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_mtx = complex_mtx->compute_absolute();
    auto dabs_mtx = complex_dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, 1e-14);
}


}  // namespace
