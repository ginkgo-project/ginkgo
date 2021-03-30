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

#include <ginkgo/core/matrix/dense.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "core/components/fill_array.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


class Dense : public ::testing::Test {
protected:
    using itype = int;
    using vtype = double;
    using Mtx = gko::matrix::Dense<vtype>;
    using MixedMtx = gko::matrix::Dense<gko::next_precision<vtype>>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<vtype>>;
    using Arr = gko::Array<itype>;
    using ComplexMtx = gko::matrix::Dense<std::complex<vtype>>;
    using MixedComplexMtx =
        gko::matrix::Dense<gko::next_precision<std::complex<vtype>>>;

    Dense() : rand_engine(15) {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        hip = gko::HipExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (hip != nullptr) {
            ASSERT_NO_THROW(hip->synchronize());
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
        dx = Mtx::create(hip);
        dx->copy_from(x.get());
        dy = Mtx::create(hip);
        dy->copy_from(y.get());
        dalpha = Mtx::create(hip);
        dalpha->copy_from(alpha.get());
        expected = Mtx::create(ref, gko::dim<2>{1, num_vecs});
        dresult = Mtx::create(hip, gko::dim<2>{1, num_vecs});
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(65, 25);
        y = gen_mtx<Mtx>(25, 35);
        expected = gen_mtx<Mtx>(65, 35);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-1.0}, ref);
        square = gen_mtx<Mtx>(x->get_size()[0], x->get_size()[0]);
        dx = Mtx::create(hip);
        dx->copy_from(x.get());
        dy = Mtx::create(hip);
        dy->copy_from(y.get());
        dresult = Mtx::create(hip);
        dresult->copy_from(expected.get());
        dalpha = Mtx::create(hip);
        dalpha->copy_from(alpha.get());
        dbeta = Mtx::create(hip);
        dbeta->copy_from(beta.get());
        dsquare = Mtx::create(hip);
        dsquare->copy_from(square.get());

        std::vector<itype> tmp(x->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<itype> tmp2(x->get_size()[1], 0);
        std::iota(tmp2.begin(), tmp2.end(), 0);
        std::shuffle(tmp2.begin(), tmp2.end(), rng);
        std::vector<itype> tmp3(x->get_size()[0] / 10);
        std::uniform_int_distribution<itype> row_dist(0, x->get_size()[0] - 1);
        for (auto &i : tmp3) {
            i = row_dist(rng);
        }
        rpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp.begin(), tmp.end()});
        cpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp2.begin(), tmp2.end()});
        rgather_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp3.begin(), tmp3.end()});
    }

    template <typename ConvertedType, typename InputType>
    std::unique_ptr<ConvertedType> convert(InputType &&input)
    {
        auto result = ConvertedType::create(input->get_executor());
        input->convert_to(result.get());
        return result;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> hip;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> expected;
    std::unique_ptr<Mtx> square;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
    std::unique_ptr<Mtx> dsquare;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
    std::unique_ptr<Arr> rgather_idxs;
};


TEST_F(Dense, HipFillIsEquivalentToRef)
{
    set_up_vector_data(3);
    auto result = Mtx::create(ref);

    x->fill(42);
    dx->fill(42);
    result->copy_from(dx.get());

    GKO_ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, HipStridedFillIsEquivalentToRef)
{
    using T = double;
    auto x = gko::initialize<gko::matrix::Dense<T>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, ref);
    auto dx = gko::initialize<gko::matrix::Dense<T>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, hip);
    auto result = Mtx::create(ref);

    x->fill(42);
    dx->fill(42);
    result->copy_from(dx.get());

    GKO_ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, SingleVectorHipScaleIsEquivalentToRef)
{
    set_up_vector_data(1);
    auto result = Mtx::create(ref);

    x->scale(alpha.get());
    dx->scale(dalpha.get());
    result->copy_from(dx.get());

    GKO_ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, MultipleVectorHipScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, MultipleVectorHipScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, SingleVectorHipAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, MultipleVectorHipAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, MultipleVectorHipAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, AddsScaledDiagIsEquivalentToRef)
{
    auto mat = gen_mtx<Mtx>(532, 532);
    gko::Array<Mtx::value_type> diag_values(ref, 532);
    gko::kernels::reference::components::fill_array(ref, diag_values.get_data(),
                                                    532, Mtx::value_type{2.0});
    auto diag =
        gko::matrix::Diagonal<Mtx::value_type>::create(ref, 532, diag_values);
    alpha = gko::initialize<Mtx>({2.0}, ref);
    auto dmat = Mtx::create(hip);
    dmat->copy_from(mat.get());
    auto ddiag = gko::matrix::Diagonal<Mtx::value_type>::create(hip);
    ddiag->copy_from(diag.get());
    dalpha = Mtx::create(hip);
    dalpha->copy_from(alpha.get());

    mat->add_scaled(alpha.get(), diag.get());
    dmat->add_scaled(dalpha.get(), ddiag.get());

    GKO_ASSERT_MTX_NEAR(mat, dmat, 1e-14);
}


TEST_F(Dense, SingleVectorHipComputeDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, MultipleVectorHipComputeDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, HipComputeNorm2IsEquivalentToRef)
{
    set_up_vector_data(20);
    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->hip, norm_size);

    x->compute_norm2(norm_expected.get());
    dx->compute_norm2(dnorm.get());

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, 1e-14);
}


TEST_F(Dense, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(y.get(), expected.get());
    dx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, SimpleApplyMixedIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(convert<MixedMtx>(y).get(), convert<MixedMtx>(expected).get());
    dx->apply(convert<MixedMtx>(dy).get(), convert<MixedMtx>(dresult).get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-7);
}


TEST_F(Dense, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, AdvancedApplyMixedIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(convert<MixedMtx>(alpha).get(), convert<MixedMtx>(y).get(),
             convert<MixedMtx>(beta).get(), convert<MixedMtx>(expected).get());
    dx->apply(convert<MixedMtx>(dalpha).get(), convert<MixedMtx>(dy).get(),
              convert<MixedMtx>(dbeta).get(), convert<MixedMtx>(dresult).get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-7);
}


TEST_F(Dense, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(25, 1);
    auto dcomplex_b = ComplexMtx::create(hip);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexMtx>(65, 1);
    auto dcomplex_x = ComplexMtx::create(hip);
    dcomplex_x->copy_from(complex_x.get());

    x->apply(complex_b.get(), complex_x.get());
    dx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Dense, ApplyToMixedComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<MixedComplexMtx>(25, 1);
    auto dcomplex_b = MixedComplexMtx::create(hip);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<MixedComplexMtx>(65, 1);
    auto dcomplex_x = MixedComplexMtx::create(hip);
    dcomplex_x->copy_from(complex_x.get());

    x->apply(complex_b.get(), complex_x.get());
    dx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-7);
}


TEST_F(Dense, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(25, 1);
    auto dcomplex_b = ComplexMtx::create(hip);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexMtx>(65, 1);
    auto dcomplex_x = ComplexMtx::create(hip);
    dcomplex_x->copy_from(complex_x.get());

    x->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dx->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Dense, AdvancedApplyToMixedComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<MixedComplexMtx>(25, 1);
    auto dcomplex_b = MixedComplexMtx::create(hip);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<MixedComplexMtx>(65, 1);
    auto dcomplex_x = MixedComplexMtx::create(hip);
    dcomplex_x->copy_from(complex_x.get());

    x->apply(convert<MixedMtx>(alpha).get(), complex_b.get(),
             convert<MixedMtx>(beta).get(), complex_x.get());
    dx->apply(convert<MixedMtx>(dalpha).get(), dcomplex_b.get(),
              convert<MixedMtx>(dbeta).get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-7);
}


TEST_F(Dense, IsTransposable)
{
    set_up_apply_data();

    auto trans = x->transpose();
    auto dtrans = dx->transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(dtrans.get()),
                        static_cast<Mtx *>(trans.get()), 0);
}


TEST_F(Dense, ConvertToCooIsEquivalentToRef)
{
    set_up_apply_data();
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(hip);

    x->convert_to(coo_mtx.get());
    dx->convert_to(dcoo_mtx.get());

    ASSERT_EQ(dcoo_mtx->get_num_stored_elements(),
              coo_mtx->get_num_stored_elements());
    GKO_ASSERT_MTX_NEAR(dcoo_mtx.get(), coo_mtx.get(), 1e-14);
}


TEST_F(Dense, MoveToCooIsEquivalentToRef)
{
    set_up_apply_data();
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(hip);

    x->move_to(coo_mtx.get());
    dx->move_to(dcoo_mtx.get());

    ASSERT_EQ(dcoo_mtx->get_num_stored_elements(),
              coo_mtx->get_num_stored_elements());
    GKO_ASSERT_MTX_NEAR(dcoo_mtx.get(), coo_mtx.get(), 1e-14);
}


TEST_F(Dense, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data();
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(hip);

    x->convert_to(csr_mtx.get());
    dx->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(dcsr_mtx.get(), csr_mtx.get(), 1e-14);
}


TEST_F(Dense, MoveToCsrIsEquivalentToRef)
{
    set_up_apply_data();
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(hip);

    x->move_to(csr_mtx.get());
    dx->move_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(dcsr_mtx.get(), csr_mtx.get(), 1e-14);
}


TEST_F(Dense, ConvertToEllIsEquivalentToRef)
{
    set_up_apply_data();
    auto ell_mtx = gko::matrix::Ell<>::create(ref);
    auto dell_mtx = gko::matrix::Ell<>::create(hip);

    x->convert_to(ell_mtx.get());
    dx->convert_to(dell_mtx.get());

    GKO_ASSERT_MTX_NEAR(dell_mtx.get(), ell_mtx.get(), 1e-14);
}


TEST_F(Dense, MoveToEllIsEquivalentToRef)
{
    set_up_apply_data();
    auto ell_mtx = gko::matrix::Ell<>::create(ref);
    auto dell_mtx = gko::matrix::Ell<>::create(hip);

    x->move_to(ell_mtx.get());
    dx->move_to(dell_mtx.get());

    GKO_ASSERT_MTX_NEAR(dell_mtx.get(), ell_mtx.get(), 1e-14);
}


TEST_F(Dense, ConvertToSellpIsEquivalentToRef)
{
    set_up_apply_data();
    auto sellp_mtx = gko::matrix::Sellp<>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<>::create(hip);

    x->convert_to(sellp_mtx.get());
    dx->convert_to(dsellp_mtx.get());

    GKO_ASSERT_MTX_NEAR(sellp_mtx, dsellp_mtx, 1e-14);
}


TEST_F(Dense, MoveToSellpIsEquivalentToRef)
{
    set_up_apply_data();
    auto sellp_mtx = gko::matrix::Sellp<>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<>::create(hip);

    x->move_to(sellp_mtx.get());
    dx->move_to(dsellp_mtx.get());

    GKO_ASSERT_MTX_NEAR(sellp_mtx, dsellp_mtx, 1e-14);
}


TEST_F(Dense, ConvertsEmptyToSellp)
{
    auto dempty_mtx = Mtx::create(hip);
    auto dsellp_mtx = gko::matrix::Sellp<>::create(hip);

    dempty_mtx->convert_to(dsellp_mtx.get());

    ASSERT_EQ(hip->copy_val_to_host(dsellp_mtx->get_const_slice_sets()), 0);
    ASSERT_FALSE(dsellp_mtx->get_size());
}


TEST_F(Dense, CountNNZIsEquivalentToRef)
{
    set_up_apply_data();
    gko::size_type nnz;
    gko::size_type dnnz;

    gko::kernels::reference::dense::count_nonzeros(ref, x.get(), &nnz);
    gko::kernels::hip::dense::count_nonzeros(hip, dx.get(), &dnnz);

    ASSERT_EQ(nnz, dnnz);
}


TEST_F(Dense, CalculateNNZPerRowIsEquivalentToRef)
{
    set_up_apply_data();
    gko::Array<gko::size_type> nnz_per_row(ref);
    nnz_per_row.resize_and_reset(x->get_size()[0]);
    gko::Array<gko::size_type> dnnz_per_row(hip);
    dnnz_per_row.resize_and_reset(dx->get_size()[0]);

    gko::kernels::reference::dense::calculate_nonzeros_per_row(ref, x.get(),
                                                               &nnz_per_row);
    gko::kernels::hip::dense::calculate_nonzeros_per_row(hip, dx.get(),
                                                         &dnnz_per_row);

    auto tmp = gko::Array<gko::size_type>(ref, dnnz_per_row);
    for (auto i = 0; i < nnz_per_row.get_num_elems(); i++) {
        ASSERT_EQ(nnz_per_row.get_const_data()[i], tmp.get_const_data()[i]);
    }
}


TEST_F(Dense, CalculateMaxNNZPerRowIsEquivalentToRef)
{
    set_up_apply_data();
    gko::size_type max_nnz;
    gko::size_type dmax_nnz;

    gko::kernels::reference::dense::calculate_max_nnz_per_row(ref, x.get(),
                                                              &max_nnz);
    gko::kernels::hip::dense::calculate_max_nnz_per_row(hip, dx.get(),
                                                        &dmax_nnz);

    ASSERT_EQ(max_nnz, dmax_nnz);
}


TEST_F(Dense, CalculateTotalColsIsEquivalentToRef)
{
    set_up_apply_data();
    gko::size_type total_cols;
    gko::size_type dtotal_cols;

    gko::kernels::reference::dense::calculate_total_cols(
        ref, x.get(), &total_cols, 2, gko::matrix::default_slice_size);
    gko::kernels::hip::dense::calculate_total_cols(
        hip, dx.get(), &dtotal_cols, 2, gko::matrix::default_slice_size);

    ASSERT_EQ(total_cols, dtotal_cols);
}


TEST_F(Dense, CanGatherRows)
{
    set_up_apply_data();

    auto r_gather = x->row_gather(rgather_idxs.get());
    auto dr_gather = dx->row_gather(rgather_idxs.get());

    GKO_ASSERT_MTX_NEAR(r_gather.get(), dr_gather.get(), 0);
}


TEST_F(Dense, CanGatherRowsIntoDense)
{
    set_up_apply_data();
    auto gather_size =
        gko::dim<2>{rgather_idxs->get_num_elems(), x->get_size()[1]};
    auto r_gather = Mtx::create(ref, gather_size);
    // test make_temporary_clone and non-default stride
    auto dr_gather = Mtx::create(ref, gather_size, x->get_size()[1] + 2);

    x->row_gather(rgather_idxs.get(), r_gather.get());
    dx->row_gather(rgather_idxs.get(), dr_gather.get());

    GKO_ASSERT_MTX_NEAR(r_gather.get(), dr_gather.get(), 0);
}


TEST_F(Dense, IsPermutable)
{
    set_up_apply_data();

    auto permuted = square->permute(rpermute_idxs.get());
    auto dpermuted = dsquare->permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(permuted.get()),
                        static_cast<Mtx *>(dpermuted.get()), 0);
}


TEST_F(Dense, IsInversePermutable)
{
    set_up_apply_data();

    auto permuted = square->inverse_permute(rpermute_idxs.get());
    auto dpermuted = dsquare->inverse_permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(permuted.get()),
                        static_cast<Mtx *>(dpermuted.get()), 0);
}


TEST_F(Dense, IsRowPermutable)
{
    set_up_apply_data();

    auto r_permute = x->row_permute(rpermute_idxs.get());
    auto dr_permute = dx->row_permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(r_permute.get()),
                        static_cast<Mtx *>(dr_permute.get()), 0);
}


TEST_F(Dense, IsColPermutable)
{
    set_up_apply_data();

    auto c_permute = x->column_permute(cpermute_idxs.get());
    auto dc_permute = dx->column_permute(cpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(c_permute.get()),
                        static_cast<Mtx *>(dc_permute.get()), 0);
}


TEST_F(Dense, IsInverseRowPermutable)
{
    set_up_apply_data();

    auto inverse_r_permute = x->inverse_row_permute(rpermute_idxs.get());
    auto d_inverse_r_permute = dx->inverse_row_permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(inverse_r_permute.get()),
                        static_cast<Mtx *>(d_inverse_r_permute.get()), 0);
}


TEST_F(Dense, IsInverseColPermutable)
{
    set_up_apply_data();

    auto inverse_c_permute = x->inverse_column_permute(cpermute_idxs.get());
    auto d_inverse_c_permute = dx->inverse_column_permute(cpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(inverse_c_permute.get()),
                        static_cast<Mtx *>(d_inverse_c_permute.get()), 0);
}


TEST_F(Dense, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data();

    auto diag = x->extract_diagonal();
    auto ddiag = dx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Dense, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    x->compute_absolute_inplace();
    dx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(x, dx, 1e-14);
}


TEST_F(Dense, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_x = x->compute_absolute();
    auto dabs_x = dx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_x, dabs_x, 1e-14);
}


TEST_F(Dense, MakeComplexIsEquivalentToRef)
{
    set_up_apply_data();

    auto complex_x = x->make_complex();
    auto dcomplex_x = dx->make_complex();

    GKO_ASSERT_MTX_NEAR(complex_x, dcomplex_x, 0);
}


TEST_F(Dense, MakeComplexWithGivenResultIsEquivalentToRef)
{
    set_up_apply_data();

    auto complex_x = ComplexMtx::create(ref, x->get_size());
    x->make_complex(complex_x.get());
    auto dcomplex_x = ComplexMtx::create(hip, x->get_size());
    dx->make_complex(dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(complex_x, dcomplex_x, 0);
}


TEST_F(Dense, GetRealIsEquivalentToRef)
{
    set_up_apply_data();

    auto real_x = x->get_real();
    auto dreal_x = dx->get_real();

    GKO_ASSERT_MTX_NEAR(real_x, dreal_x, 0);
}


TEST_F(Dense, GetRealWithGivenResultIsEquivalentToRef)
{
    set_up_apply_data();

    auto real_x = Mtx::create(ref, x->get_size());
    x->get_real(real_x.get());
    auto dreal_x = Mtx::create(hip, dx->get_size());
    dx->get_real(dreal_x.get());

    GKO_ASSERT_MTX_NEAR(real_x, dreal_x, 0);
}


TEST_F(Dense, GetImagIsEquivalentToRef)
{
    set_up_apply_data();

    auto imag_x = x->get_imag();
    auto dimag_x = dx->get_imag();

    GKO_ASSERT_MTX_NEAR(imag_x, dimag_x, 0);
}


TEST_F(Dense, GetImagWithGivenResultIsEquivalentToRef)
{
    set_up_apply_data();

    auto imag_x = Mtx::create(ref, x->get_size());
    x->get_imag(imag_x.get());
    auto dimag_x = Mtx::create(hip, dx->get_size());
    dx->get_imag(dimag_x.get());

    GKO_ASSERT_MTX_NEAR(imag_x, dimag_x, 0);
}


}  // namespace
