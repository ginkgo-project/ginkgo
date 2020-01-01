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

#include "core/matrix/dense_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils.hpp"


namespace {


class Dense : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Arr = gko::Array<int>;
    using ComplexMtx = gko::matrix::Dense<std::complex<double>>;

    Dense() : rand_engine(15) {}

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
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
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

    void set_up_vector_data(gko::size_type num_vecs,
                            bool different_alpha = false)
    {
        x = gen_mtx<Mtx>(1000, num_vecs);
        y = gen_mtx<Mtx>(1000, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<Mtx>(1, num_vecs);
        } else {
            alpha = gko::initialize<Mtx>({2.0}, ref);
        }
        dx = Mtx::create(omp);
        dx->copy_from(x.get());
        dy = Mtx::create(omp);
        dy->copy_from(y.get());
        dalpha = Mtx::create(omp);
        dalpha->copy_from(alpha.get());
        expected = Mtx::create(ref, gko::dim<2>{1, num_vecs});
        dresult = Mtx::create(omp, gko::dim<2>{1, num_vecs});
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(40, 25);
        c_x = gen_mtx<ComplexMtx>(40, 25);
        y = gen_mtx<Mtx>(25, 35);
        expected = gen_mtx<Mtx>(40, 35);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-1.0}, ref);
        dx = Mtx::create(omp);
        dx->copy_from(x.get());
        dc_x = ComplexMtx::create(omp);
        dc_x->copy_from(c_x.get());
        dy = Mtx::create(omp);
        dy->copy_from(y.get());
        dresult = Mtx::create(omp);
        dresult->copy_from(expected.get());
        dalpha = Mtx::create(omp);
        dalpha->copy_from(alpha.get());
        dbeta = Mtx::create(omp);
        dbeta->copy_from(beta.get());

        std::vector<int> tmp(x->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<int> tmp2(x->get_size()[1], 0);
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

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<ComplexMtx> c_x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> expected;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<ComplexMtx> dc_x;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> drpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
    std::unique_ptr<Arr> dcpermute_idxs;
};


TEST_F(Dense, SingleVectorOmpScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    auto result = Mtx::create(ref);
    result->copy_from(dx.get());
    GKO_ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, MultipleVectorOmpScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, MultipleVectorOmpScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, SingleVectorOmpAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, MultipleVectorOmpAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, MultipleVectorOmpAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, SingleVectorOmpComputeDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, MultipleVectorOmpComputeDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, ComputesNorm2IsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_norm2(expected.get());
    dx->compute_norm2(dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(y.get(), expected.get());
    dx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, ConvertToCooIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::Coo<>::create(ref);
    auto somtx = gko::matrix::Coo<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());
    srmtx->convert_to(drmtx.get());
    somtx->convert_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 1e-14);
}


TEST_F(Dense, MoveToCooIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::Coo<>::create(ref);
    auto somtx = gko::matrix::Coo<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->move_to(srmtx.get());
    omtx->move_to(somtx.get());
    srmtx->move_to(drmtx.get());
    somtx->move_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 1e-14);
}


TEST_F(Dense, ConvertToCsrIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::Csr<>::create(ref);
    auto somtx = gko::matrix::Csr<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());
    srmtx->convert_to(drmtx.get());
    somtx->convert_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 1e-14);
}


TEST_F(Dense, MoveToCsrIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::Csr<>::create(ref);
    auto somtx = gko::matrix::Csr<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->move_to(srmtx.get());
    omtx->move_to(somtx.get());
    srmtx->move_to(drmtx.get());
    somtx->move_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 1e-14);
}


TEST_F(Dense, ConvertToSparsityCsrIsEquivalentToRef)
{
    auto mtx = gen_mtx<Mtx>(532, 231);
    auto dmtx = Mtx::create(omp);
    dmtx->copy_from(mtx.get());
    auto sparsity_mtx = gko::matrix::SparsityCsr<>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<>::create(omp);

    mtx->convert_to(sparsity_mtx.get());
    dmtx->convert_to(d_sparsity_mtx.get());

    GKO_ASSERT_MTX_NEAR(d_sparsity_mtx.get(), sparsity_mtx.get(), 1e-14);
}


TEST_F(Dense, MoveToSparsityCsrIsEquivalentToRef)
{
    auto mtx = gen_mtx<Mtx>(532, 231);
    auto dmtx = Mtx::create(omp);
    dmtx->copy_from(mtx.get());
    auto sparsity_mtx = gko::matrix::SparsityCsr<>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<>::create(omp);

    mtx->move_to(sparsity_mtx.get());
    dmtx->move_to(d_sparsity_mtx.get());

    GKO_ASSERT_MTX_NEAR(d_sparsity_mtx.get(), sparsity_mtx.get(), 1e-14);
}


TEST_F(Dense, ConvertToEllIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::Ell<>::create(ref);
    auto somtx = gko::matrix::Ell<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());
    srmtx->convert_to(drmtx.get());
    somtx->convert_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 1e-14);
}


TEST_F(Dense, MoveToEllIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::Ell<>::create(ref);
    auto somtx = gko::matrix::Ell<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->move_to(srmtx.get());
    omtx->move_to(somtx.get());
    srmtx->move_to(drmtx.get());
    somtx->move_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 1e-14);
}


TEST_F(Dense, ConvertToHybridIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::Hybrid<>::create(ref);
    auto somtx = gko::matrix::Hybrid<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());
    srmtx->convert_to(drmtx.get());
    somtx->convert_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
    // Test between `srmtx` and `somtx` may fail due to the OpenMP
    // implementation not sorting the Coo matrix part.
    // Therefore, it is not performed.
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 1e-14);
}


TEST_F(Dense, MoveToHybridIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::Hybrid<>::create(ref);
    auto somtx = gko::matrix::Hybrid<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->move_to(srmtx.get());
    omtx->move_to(somtx.get());
    srmtx->move_to(drmtx.get());
    somtx->move_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
    // Test between `srmtx` and `somtx` may fail due to the OpenMP
    // implementation not sorting the Coo matrix part.
    // Therefore, it is not performed.
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 1e-14);
}


TEST_F(Dense, ConvertToSellpIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::Sellp<>::create(ref);
    auto somtx = gko::matrix::Sellp<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());
    srmtx->convert_to(drmtx.get());
    somtx->convert_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 1e-14);
}


TEST_F(Dense, MoveToSellpIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::Sellp<>::create(ref);
    auto somtx = gko::matrix::Sellp<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->move_to(srmtx.get());
    omtx->move_to(somtx.get());
    srmtx->move_to(drmtx.get());
    somtx->move_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 1e-14);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 1e-14);
}


TEST_F(Dense, CalculateMaxNNZPerRowIsEquivalentToRef)
{
    std::size_t ref_max_nnz_per_row = 0;
    std::size_t omp_max_nnz_per_row = 0;
    auto rmtx = gen_mtx<Mtx>(100, 100, 1);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());

    gko::kernels::reference::dense::calculate_max_nnz_per_row(
        ref, rmtx.get(), &ref_max_nnz_per_row);
    gko::kernels::omp::dense::calculate_max_nnz_per_row(omp, omtx.get(),
                                                        &omp_max_nnz_per_row);

    ASSERT_EQ(ref_max_nnz_per_row, omp_max_nnz_per_row);
}


TEST_F(Dense, CalculateTotalColsIsEquivalentToRef)
{
    std::size_t ref_total_cols = 0;
    std::size_t omp_total_cols = 0;
    auto rmtx = gen_mtx<Mtx>(100, 100, 1);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());

    gko::kernels::reference::dense::calculate_total_cols(
        ref, rmtx.get(), &ref_total_cols, 1, gko::matrix::default_slice_size);
    gko::kernels::omp::dense::calculate_total_cols(
        omp, omtx.get(), &omp_total_cols, 1, gko::matrix::default_slice_size);

    ASSERT_EQ(ref_total_cols, omp_total_cols);
}


TEST_F(Dense, IsTransposable)
{
    set_up_apply_data();

    auto trans = x->transpose();
    auto dtrans = dx->transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(dtrans.get()),
                        static_cast<Mtx *>(trans.get()), 0);
}


TEST_F(Dense, IsConjugateTransposable)
{
    set_up_apply_data();

    auto trans = c_x->conj_transpose();
    auto dtrans = dc_x->conj_transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<ComplexMtx *>(dtrans.get()),
                        static_cast<ComplexMtx *>(trans.get()), 0);
}


TEST_F(Dense, IsRowPermutable)
{
    set_up_apply_data();
    auto r_permute = x->row_permute(rpermute_idxs.get());
    auto dr_permute = dx->row_permute(drpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(r_permute.get()),
                        static_cast<Mtx *>(dr_permute.get()), 0);
}


TEST_F(Dense, IsColPermutable)
{
    set_up_apply_data();
    auto c_permute = x->column_permute(cpermute_idxs.get());
    auto dc_permute = dx->column_permute(dcpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(c_permute.get()),
                        static_cast<Mtx *>(dc_permute.get()), 0);
}


TEST_F(Dense, IsInverseRowPermutable)
{
    set_up_apply_data();
    auto inverse_r_permute = x->inverse_row_permute(rpermute_idxs.get());
    auto d_inverse_r_permute = dx->inverse_row_permute(drpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(inverse_r_permute.get()),
                        static_cast<Mtx *>(d_inverse_r_permute.get()), 0);
}


TEST_F(Dense, IsInverseColPermutable)
{
    set_up_apply_data();
    auto inverse_c_permute = x->inverse_column_permute(cpermute_idxs.get());
    auto d_inverse_c_permute = dx->inverse_column_permute(dcpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(inverse_c_permute.get()),
                        static_cast<Mtx *>(d_inverse_c_permute.get()), 0);
}


}  // namespace
