/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/fbcsr_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class Fbcsr : public ::testing::Test {
protected:
    using Arr = gko::Array<int>;
    using Mtx = gko::matrix::Fbcsr<>;
    using Vec = gko::matrix::Dense<>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;
    using ComplexMtx = gko::matrix::Fbcsr<std::complex<double>>;

    Fbcsr() : mtx_size(532, 231), rand_engine(42) {}

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
        mtx = Mtx::create(ref);
        mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[1], 1));
        complex_mtx = ComplexMtx::create(ref);
        complex_mtx->copy_from(
            gen_mtx<ComplexVec>(mtx_size[0], mtx_size[1], 1));
        square_mtx = Mtx::create(ref);
        square_mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[0], 1));
        expected = gen_mtx<Vec>(mtx_size[0], num_vectors, 1);
        y = gen_mtx<Vec>(mtx_size[1], num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(omp);
        dmtx->copy_from(mtx.get());
        complex_dmtx = ComplexMtx::create(omp);
        complex_dmtx->copy_from(complex_mtx.get());
        square_dmtx = Mtx::create(omp);
        square_dmtx->copy_from(square_mtx.get());
        dresult = Vec::create(omp);
        dresult->copy_from(expected.get());
        dy = Vec::create(omp);
        dy->copy_from(y.get());
        dalpha = Vec::create(omp);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(omp);
        dbeta->copy_from(beta.get());

        std::vector<int> tmp(mtx->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<int> tmp2(mtx->get_size()[1], 0);
        std::iota(tmp2.begin(), tmp2.end(), 0);
        std::shuffle(tmp2.begin(), tmp2.end(), rng);
        rpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp.begin(), tmp.end()});
        drpermute_idxs =
            std::unique_ptr<Arr>(new Arr{omp, tmp.begin(), tmp.end()});
        cpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp2.begin(), tmp2.end()});
        dcpermute_idxs =
            std::unique_ptr<Arr>(new Arr{omp, tmp2.begin(), tmp2.end()});
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
            auto vals = local_mtx_ref->get_values() + start_row;
            const auto nnz_in_this_row = row_ptrs[row + 1] - row_ptrs[row];
            auto swap_idx_dist =
                std::uniform_int_distribution<>(0, nnz_in_this_row - 1);
            // shuffle `nnz_in_this_row / 2` times
            for (size_t perm = 0; perm < nnz_in_this_row; perm += 2) {
                const auto idx1 = swap_idx_dist(rand_engine);
                const auto idx2 = swap_idx_dist(rand_engine);
                std::swap(col_idx[idx1], col_idx[idx2]);
                std::swap(vals[idx1], vals[idx2]);
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
    std::unique_ptr<Mtx> square_mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<ComplexMtx> complex_dmtx;
    std::unique_ptr<Mtx> square_dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> drpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
    std::unique_ptr<Arr> dcpermute_idxs;
};


TEST_F(Fbcsr, SimpleApplyIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//
//    mtx->apply(y.get(), expected.get());
//    dmtx->apply(dy.get(), dresult.get());
//
//    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
//}


TEST_F(Fbcsr, AdvancedApplyIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//
//    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
//    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());
//
//    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
//}


TEST_F(Fbcsr, SimpleApplyToDenseMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data(3);
//
//    mtx->apply(y.get(), expected.get());
//    dmtx->apply(dy.get(), dresult.get());
//
//    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
//}


TEST_F(Fbcsr, AdvancedApplyToFbcsrMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto trans = mtx->transpose();
//    auto d_trans = dmtx->transpose();
//
//    mtx->apply(alpha.get(), trans.get(), beta.get(), square_mtx.get());
//    dmtx->apply(dalpha.get(), d_trans.get(), dbeta.get(), square_dmtx.get());
//
//    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, 1e-14);
//    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
//    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
//}


TEST_F(Fbcsr, SimpleApplyToFbcsrMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto trans = mtx->transpose();
//    auto d_trans = dmtx->transpose();
//
//    mtx->apply(trans.get(), square_mtx.get());
//    dmtx->apply(d_trans.get(), square_dmtx.get());
//
//    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, 1e-14);
//    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
//    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
//}


TEST_F(Fbcsr, AdvancedApplyToIdentityMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto a = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
//    auto b = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
//    auto da = Mtx::create(omp);
//    auto db = Mtx::create(omp);
//    da->copy_from(a.get());
//    db->copy_from(b.get());
//    auto id = gko::matrix::Identity<Mtx::value_type>::create(ref,
//    mtx_size[1]); auto did =
//    gko::matrix::Identity<Mtx::value_type>::create(omp, mtx_size[1]);
//
//    a->apply(alpha.get(), id.get(), beta.get(), b.get());
//    da->apply(dalpha.get(), did.get(), dbeta.get(), db.get());
//
//    GKO_ASSERT_MTX_NEAR(b, db, 1e-14);
//    GKO_ASSERT_MTX_EQ_SPARSITY(b, db);
//    ASSERT_TRUE(db->is_sorted_by_column_index());
//}


TEST_F(Fbcsr, AdvancedApplyToDenseMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data(3);
//
//    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
//    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());
//
//    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
//}


TEST_F(Fbcsr, TransposeIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//
//    auto trans = mtx->transpose();
//    auto d_trans = dmtx->transpose();
//
//    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(d_trans.get()),
//                        static_cast<Mtx *>(trans.get()), 0.0);
//    ASSERT_TRUE(static_cast<Mtx
//    *>(d_trans.get())->is_sorted_by_column_index());
//}


TEST_F(Fbcsr, ConjugateTransposeIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//
//    auto trans = complex_mtx->conj_transpose();
//    auto d_trans = complex_dmtx->conj_transpose();
//
//    GKO_ASSERT_MTX_NEAR(static_cast<ComplexMtx *>(d_trans.get()),
//                        static_cast<ComplexMtx *>(trans.get()), 0.0);
//    ASSERT_TRUE(
//        static_cast<ComplexMtx
//        *>(d_trans.get())->is_sorted_by_column_index());
//}


TEST_F(Fbcsr, ConvertToCooIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto coo_mtx = gko::matrix::Coo<>::create(ref);
//    auto dcoo_mtx = gko::matrix::Coo<>::create(omp);
//
//    mtx->convert_to(coo_mtx.get());
//    dmtx->convert_to(dcoo_mtx.get());
//
//    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 1e-14);
//}


TEST_F(Fbcsr, MoveToCooIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto coo_mtx = gko::matrix::Coo<>::create(ref);
//    auto dcoo_mtx = gko::matrix::Coo<>::create(omp);
//
//    mtx->move_to(coo_mtx.get());
//    dmtx->move_to(dcoo_mtx.get());
//
//    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 1e-14);
//}


TEST_F(Fbcsr, ConvertToDenseIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto dense_mtx = gko::matrix::Dense<>::create(ref);
//    auto ddense_mtx = gko::matrix::Dense<>::create(omp);
//
//    mtx->convert_to(dense_mtx.get());
//    dmtx->convert_to(ddense_mtx.get());
//
//    GKO_ASSERT_MTX_NEAR(ddense_mtx.get(), dense_mtx.get(), 1e-14);
//}


TEST_F(Fbcsr, MoveToDenseIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto dense_mtx = gko::matrix::Dense<>::create(ref);
//    auto ddense_mtx = gko::matrix::Dense<>::create(omp);
//
//    mtx->move_to(dense_mtx.get());
//    dmtx->move_to(ddense_mtx.get());
//
//    GKO_ASSERT_MTX_NEAR(ddense_mtx.get(), dense_mtx.get(), 1e-14);
//}


TEST_F(Fbcsr, ConvertToSparsityFbcsrIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto sparsity_mtx = gko::matrix::SparsityFbcsr<>::create(ref);
//    auto d_sparsity_mtx = gko::matrix::SparsityFbcsr<>::create(omp);
//
//    mtx->convert_to(sparsity_mtx.get());
//    dmtx->convert_to(d_sparsity_mtx.get());
//
//    GKO_ASSERT_MTX_NEAR(d_sparsity_mtx.get(), sparsity_mtx.get(), 1e-14);
//}


TEST_F(Fbcsr, MoveToSparsityFbcsrIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto sparsity_mtx = gko::matrix::SparsityFbcsr<>::create(ref);
//    auto d_sparsity_mtx = gko::matrix::SparsityFbcsr<>::create(omp);
//
//    mtx->move_to(sparsity_mtx.get());
//    dmtx->move_to(d_sparsity_mtx.get());
//
//    GKO_ASSERT_MTX_NEAR(d_sparsity_mtx.get(), sparsity_mtx.get(), 1e-14);
//}


TEST_F(Fbcsr, CalculatesNonzerosPerRow)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    gko::Array<gko::size_type> row_nnz(ref, mtx->get_size()[0]);
//    gko::Array<gko::size_type> drow_nnz(omp, dmtx->get_size()[0]);
//
//    gko::kernels::reference::fbcsr::calculate_nonzeros_per_row(ref, mtx.get(),
//                                                             &row_nnz);
//    gko::kernels::omp::fbcsr::calculate_nonzeros_per_row(omp, dmtx.get(),
//                                                       &drow_nnz);
//
//    GKO_ASSERT_ARRAY_EQ(row_nnz, drow_nnz);
//}


TEST_F(Fbcsr, ConvertToHybridIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Hybrid_type = gko::matrix::Hybrid<>;
//    set_up_apply_data();
//    auto hybrid_mtx = Hybrid_type::create(
//        ref, std::make_shared<Hybrid_type::column_limit>(2));
//    auto dhybrid_mtx = Hybrid_type::create(
//        omp, std::make_shared<Hybrid_type::column_limit>(2));
//
//    mtx->convert_to(hybrid_mtx.get());
//    dmtx->convert_to(dhybrid_mtx.get());
//
//    GKO_ASSERT_MTX_NEAR(hybrid_mtx.get(), dhybrid_mtx.get(), 1e-14);
//}


TEST_F(Fbcsr, MoveToHybridIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Hybrid_type = gko::matrix::Hybrid<>;
//    set_up_apply_data();
//    auto hybrid_mtx = Hybrid_type::create(
//        ref, std::make_shared<Hybrid_type::column_limit>(2));
//    auto dhybrid_mtx = Hybrid_type::create(
//        omp, std::make_shared<Hybrid_type::column_limit>(2));
//
//    mtx->move_to(hybrid_mtx.get());
//    dmtx->move_to(dhybrid_mtx.get());
//
//    GKO_ASSERT_MTX_NEAR(hybrid_mtx.get(), dhybrid_mtx.get(), 1e-14);
//}


TEST_F(Fbcsr, IsRowPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto r_permute = mtx->row_permute(rpermute_idxs.get());
//    auto dr_permute = dmtx->row_permute(drpermute_idxs.get());
//
//    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(r_permute.get()),
//                        static_cast<Mtx *>(dr_permute.get()), 0);
//}


TEST_F(Fbcsr, IsColPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto c_permute = mtx->column_permute(cpermute_idxs.get());
//    auto dc_permute = dmtx->column_permute(dcpermute_idxs.get());
//
//    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(c_permute.get()),
//                        static_cast<Mtx *>(dc_permute.get()), 0);
//}


TEST_F(Fbcsr, IsInverseRowPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto inverse_r_permute = mtx->inverse_row_permute(rpermute_idxs.get());
//    auto d_inverse_r_permute =
//    dmtx->inverse_row_permute(drpermute_idxs.get());
//
//    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(inverse_r_permute.get()),
//                        static_cast<Mtx *>(d_inverse_r_permute.get()), 0);
//}


TEST_F(Fbcsr, IsInverseColPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    auto inverse_c_permute = mtx->inverse_column_permute(cpermute_idxs.get());
//    auto d_inverse_c_permute =
//        dmtx->inverse_column_permute(dcpermute_idxs.get());
//
//    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(inverse_c_permute.get()),
//                        static_cast<Mtx *>(d_inverse_c_permute.get()), 0);
//}


TEST_F(Fbcsr, RecognizeSortedMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//    bool is_sorted_omp{};
//    bool is_sorted_ref{};
//
//    is_sorted_ref = mtx->is_sorted_by_column_index();
//    is_sorted_omp = dmtx->is_sorted_by_column_index();
//
//    ASSERT_EQ(is_sorted_ref, is_sorted_omp);
//}


TEST_F(Fbcsr, RecognizeUnsortedMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto uns_mtx = gen_unsorted_mtx();
//    bool is_sorted_omp{};
//    bool is_sorted_ref{};
//
//    is_sorted_ref = uns_mtx.ref->is_sorted_by_column_index();
//    is_sorted_omp = uns_mtx.omp->is_sorted_by_column_index();
//
//    ASSERT_EQ(is_sorted_ref, is_sorted_omp);
//}


TEST_F(Fbcsr, SortSortedMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//
//    mtx->sort_by_column_index();
//    dmtx->sort_by_column_index();
//
//    // Values must be unchanged, therefore, tolerance is `0`
//    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
//}


TEST_F(Fbcsr, SortUnsortedMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto uns_mtx = gen_unsorted_mtx();
//
//    uns_mtx.ref->sort_by_column_index();
//    uns_mtx.omp->sort_by_column_index();
//
//    // Values must be unchanged, therefore, tolerance is `0`
//    GKO_ASSERT_MTX_NEAR(uns_mtx.ref, uns_mtx.omp, 0);
//}


TEST_F(Fbcsr, ExtractDiagonalIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//
//    auto diag = mtx->extract_diagonal();
//    auto ddiag = dmtx->extract_diagonal();
//
//    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
//}


TEST_F(Fbcsr, InplaceAbsoluteMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//
//    mtx->compute_absolute_inplace();
//    dmtx->compute_absolute_inplace();
//
//    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 1e-14);
//}


TEST_F(Fbcsr, OutplaceAbsoluteMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//
//    auto abs_mtx = mtx->compute_absolute();
//    auto dabs_mtx = dmtx->compute_absolute();
//
//    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, 1e-14);
//}


TEST_F(Fbcsr, InplaceAbsoluteComplexMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//
//    complex_mtx->compute_absolute_inplace();
//    complex_dmtx->compute_absolute_inplace();
//
//    GKO_ASSERT_MTX_NEAR(complex_mtx, complex_dmtx, 1e-14);
//}


TEST_F(Fbcsr, OutplaceAbsoluteComplexMatrixIsEquivalentToRef)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    set_up_apply_data();
//
//    auto abs_mtx = complex_mtx->compute_absolute();
//    auto dabs_mtx = complex_dmtx->compute_absolute();
//
//    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, 1e-14);
//}


}  // namespace
