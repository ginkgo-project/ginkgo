/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class Dense : public ::testing::Test {
protected:
    using itype = int;
    using vtype = double;
    using Mtx = gko::matrix::Dense<vtype>;
    using MixedMtx = gko::matrix::Dense<gko::next_precision<vtype>>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<vtype>>;
    using Arr = gko::array<itype>;
    using ComplexMtx = gko::matrix::Dense<std::complex<vtype>>;
    using Diagonal = gko::matrix::Diagonal<vtype>;
    using MixedComplexMtx =
        gko::matrix::Dense<gko::next_precision<std::complex<vtype>>>;

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
        dx = gko::clone(omp, x);
        dy = gko::clone(omp, y);
        dalpha = gko::clone(omp, alpha);
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
        square = gen_mtx<Mtx>(x->get_size()[0], x->get_size()[0]);
        dx = gko::clone(omp, x);
        dc_x = gko::clone(omp, c_x);
        dy = gko::clone(omp, y);
        dresult = gko::clone(omp, expected);
        dalpha = gko::clone(omp, alpha);
        dbeta = gko::clone(omp, beta);
        dsquare = gko::clone(omp, square);

        std::vector<int> tmp(x->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<int> tmp2(x->get_size()[1], 0);
        std::iota(tmp2.begin(), tmp2.end(), 0);
        std::shuffle(tmp2.begin(), tmp2.end(), rng);
        std::vector<int> tmp3(x->get_size()[0] / 10);
        std::uniform_int_distribution<int> row_dist(0, x->get_size()[0] - 1);
        for (auto& i : tmp3) {
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
    std::unique_ptr<ConvertedType> convert(InputType&& input)
    {
        auto result = ConvertedType::create(input->get_executor());
        input->convert_to(result.get());
        return result;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<ComplexMtx> c_x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> expected;
    std::unique_ptr<Mtx> square;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<ComplexMtx> dc_x;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
    std::unique_ptr<Mtx> dsquare;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
    std::unique_ptr<Arr> rgather_idxs;
};


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


TEST_F(Dense, SingleVectorOmpComputeConjDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_conj_dot(y.get(), expected.get());
    dx->compute_conj_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, MultipleVectorOmpComputeConjDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_conj_dot(y.get(), expected.get());
    dx->compute_conj_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, SingleVectorComputesNorm2IsEquivalentToRef)
{
    set_up_vector_data(1);
    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->omp, norm_size);

    x->compute_norm2(norm_expected.get());
    dx->compute_norm2(dnorm.get());

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, 1e-14);
}


TEST_F(Dense, MultipleVectorComputesNorm2IsEquivalentToRef)
{
    set_up_vector_data(20);
    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->omp, norm_size);

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
    auto dcomplex_b = gko::clone(omp, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(40, 1);
    auto dcomplex_x = gko::clone(omp, complex_x);

    x->apply(complex_b.get(), complex_x.get());
    dx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Dense, ApplyToMixedComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<MixedComplexMtx>(25, 1);
    auto dcomplex_b = gko::clone(omp, complex_b);
    auto complex_x = gen_mtx<MixedComplexMtx>(40, 1);
    auto dcomplex_x = gko::clone(omp, complex_x);

    x->apply(complex_b.get(), complex_x.get());
    dx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-7);
}


TEST_F(Dense, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(25, 1);
    auto dcomplex_b = gko::clone(omp, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(40, 1);
    auto dcomplex_x = gko::clone(omp, complex_x);

    x->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dx->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Dense, AdvancedApplyToMixedComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<MixedComplexMtx>(25, 1);
    auto dcomplex_b = gko::clone(omp, complex_b);
    auto complex_x = gen_mtx<MixedComplexMtx>(40, 1);
    auto dcomplex_x = gko::clone(omp, complex_x);

    x->apply(convert<MixedMtx>(alpha).get(), complex_b.get(),
             convert<MixedMtx>(beta).get(), complex_x.get());
    dx->apply(convert<MixedMtx>(dalpha).get(), dcomplex_b.get(),
              convert<MixedMtx>(dbeta).get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-7);
}


TEST_F(Dense, ComputeDotComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_b = gko::clone(omp, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_x = gko::clone(omp, complex_x);
    auto result = ComplexMtx::create(ref, gko::dim<2>{1, 2});
    auto dresult = ComplexMtx::create(omp, gko::dim<2>{1, 2});

    complex_b->compute_dot(complex_x.get(), result.get());
    dcomplex_b->compute_dot(dcomplex_x.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(result, dresult, 1e-14);
}


TEST_F(Dense, ComputeConjDotComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_b = gko::clone(omp, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_x = gko::clone(omp, complex_x);
    auto result = ComplexMtx::create(ref, gko::dim<2>{1, 2});
    auto dresult = ComplexMtx::create(omp, gko::dim<2>{1, 2});

    complex_b->compute_conj_dot(complex_x.get(), result.get());
    dcomplex_b->compute_conj_dot(dcomplex_x.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(result, dresult, 1e-14);
}


TEST_F(Dense, ConvertToCooIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = gko::clone(omp, rmtx);
    auto srmtx = gko::matrix::Coo<>::create(ref);
    auto somtx = gko::matrix::Coo<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());
    srmtx->convert_to(drmtx.get());
    somtx->convert_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 0);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 0);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 0);
}


TEST_F(Dense, MoveToCooIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = gko::clone(omp, rmtx);
    auto srmtx = gko::matrix::Coo<>::create(ref);
    auto somtx = gko::matrix::Coo<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->move_to(srmtx.get());
    omtx->move_to(somtx.get());
    srmtx->move_to(drmtx.get());
    somtx->move_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 0);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 0);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 0);
}


TEST_F(Dense, ConvertToCsrIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = gko::clone(omp, rmtx);
    auto srmtx = gko::matrix::Csr<>::create(ref);
    auto somtx = gko::matrix::Csr<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());
    srmtx->convert_to(drmtx.get());
    somtx->convert_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 0);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 0);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 0);
}


TEST_F(Dense, MoveToCsrIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = gko::clone(omp, rmtx);
    auto srmtx = gko::matrix::Csr<>::create(ref);
    auto somtx = gko::matrix::Csr<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->move_to(srmtx.get());
    omtx->move_to(somtx.get());
    srmtx->move_to(drmtx.get());
    somtx->move_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 0);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 0);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 0);
}


TEST_F(Dense, ConvertToSparsityCsrIsEquivalentToRef)
{
    auto mtx = gen_mtx<Mtx>(532, 231);
    auto dmtx = gko::clone(omp, mtx);
    auto sparsity_mtx = gko::matrix::SparsityCsr<>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<>::create(omp);

    mtx->convert_to(sparsity_mtx.get());
    dmtx->convert_to(d_sparsity_mtx.get());

    GKO_ASSERT_MTX_NEAR(d_sparsity_mtx.get(), sparsity_mtx.get(), 0);
}


TEST_F(Dense, MoveToSparsityCsrIsEquivalentToRef)
{
    auto mtx = gen_mtx<Mtx>(532, 231);
    auto dmtx = gko::clone(omp, mtx);
    auto sparsity_mtx = gko::matrix::SparsityCsr<>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<>::create(omp);

    mtx->move_to(sparsity_mtx.get());
    dmtx->move_to(d_sparsity_mtx.get());

    GKO_ASSERT_MTX_NEAR(d_sparsity_mtx.get(), sparsity_mtx.get(), 0);
}


TEST_F(Dense, ConvertToEllIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = gko::clone(omp, rmtx);
    auto srmtx = gko::matrix::Ell<>::create(ref);
    auto somtx = gko::matrix::Ell<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());
    srmtx->convert_to(drmtx.get());
    somtx->convert_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 0);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 0);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 0);
}


TEST_F(Dense, MoveToEllIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = gko::clone(omp, rmtx);
    auto srmtx = gko::matrix::Ell<>::create(ref);
    auto somtx = gko::matrix::Ell<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->move_to(srmtx.get());
    omtx->move_to(somtx.get());
    srmtx->move_to(drmtx.get());
    somtx->move_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 0);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 0);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 0);
}


TEST_F(Dense, ConvertToHybridIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = gko::clone(omp, rmtx);
    auto srmtx = gko::matrix::Hybrid<>::create(ref);
    auto somtx = gko::matrix::Hybrid<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());
    srmtx->convert_to(drmtx.get());
    somtx->convert_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 0);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 0);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 0);
}


TEST_F(Dense, MoveToHybridIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = gko::clone(omp, rmtx);
    auto srmtx = gko::matrix::Hybrid<>::create(ref);
    auto somtx = gko::matrix::Hybrid<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->move_to(srmtx.get());
    omtx->move_to(somtx.get());
    srmtx->move_to(drmtx.get());
    somtx->move_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 0);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 0);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 0);
}


TEST_F(Dense, ConvertToSellpIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = gko::clone(omp, rmtx);
    auto srmtx = gko::matrix::Sellp<>::create(ref);
    auto somtx = gko::matrix::Sellp<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());
    srmtx->convert_to(drmtx.get());
    somtx->convert_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 0);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 0);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 0);
}


TEST_F(Dense, MoveToSellpIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(532, 231);
    auto omtx = gko::clone(omp, rmtx);
    auto srmtx = gko::matrix::Sellp<>::create(ref);
    auto somtx = gko::matrix::Sellp<>::create(omp);
    auto drmtx = Mtx::create(ref);
    auto domtx = Mtx::create(omp);

    rmtx->move_to(srmtx.get());
    omtx->move_to(somtx.get());
    srmtx->move_to(drmtx.get());
    somtx->move_to(domtx.get());

    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 0);
    GKO_ASSERT_MTX_NEAR(srmtx, somtx, 0);
    GKO_ASSERT_MTX_NEAR(domtx, omtx, 0);
}


TEST_F(Dense, ConvertsEmptyToSellp)
{
    auto dempty_mtx = Mtx::create(omp);
    auto dsellp_mtx = gko::matrix::Sellp<>::create(omp);

    dempty_mtx->convert_to(dsellp_mtx.get());

    ASSERT_EQ(*dsellp_mtx->get_const_slice_sets(), 0);
    ASSERT_FALSE(dsellp_mtx->get_size());
}


TEST_F(Dense, CalculateNNZPerRowIsEquivalentToRef)
{
    set_up_apply_data();
    gko::array<gko::size_type> nnz_per_row(ref);
    nnz_per_row.resize_and_reset(x->get_size()[0]);
    gko::array<gko::size_type> dnnz_per_row(omp);
    dnnz_per_row.resize_and_reset(dx->get_size()[0]);

    gko::kernels::reference::dense::count_nonzeros_per_row(
        ref, x.get(), nnz_per_row.get_data());
    gko::kernels::omp::dense::count_nonzeros_per_row(omp, dx.get(),
                                                     dnnz_per_row.get_data());

    GKO_ASSERT_ARRAY_EQ(nnz_per_row, dnnz_per_row);
}


TEST_F(Dense, ComputeMaxNNZPerRowIsEquivalentToRef)
{
    std::size_t ref_max_nnz_per_row = 0;
    std::size_t omp_max_nnz_per_row = 0;
    auto rmtx = gen_mtx<Mtx>(100, 100, 1);
    auto omtx = gko::clone(omp, rmtx);

    gko::kernels::reference::dense::compute_max_nnz_per_row(
        ref, rmtx.get(), ref_max_nnz_per_row);
    gko::kernels::omp::dense::compute_max_nnz_per_row(omp, omtx.get(),
                                                      omp_max_nnz_per_row);

    ASSERT_EQ(ref_max_nnz_per_row, omp_max_nnz_per_row);
}


TEST_F(Dense, IsTransposable)
{
    set_up_apply_data();

    auto trans = x->transpose();
    auto dtrans = dx->transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx*>(dtrans.get()),
                        static_cast<Mtx*>(trans.get()), 0);
}


TEST_F(Dense, IsTransposableIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto row_span = gko::span{0, x->get_size()[0] - 2};
    auto col_span = gko::span{0, x->get_size()[1] - 2};
    auto sub_x = x->create_submatrix(row_span, col_span);
    auto sub_dx = dx->create_submatrix(row_span, col_span);
    // create the target matrices on another executor to
    // force temporary clone
    auto trans = Mtx::create(ref, gko::transpose(sub_x->get_size()));
    auto dtrans = Mtx::create(ref, gko::transpose(sub_x->get_size()),
                              sub_x->get_size()[0] + 4);

    sub_x->transpose(trans.get());
    sub_dx->transpose(dtrans.get());

    GKO_ASSERT_MTX_NEAR(dtrans, trans, 0);
}


TEST_F(Dense, IsConjugateTransposable)
{
    set_up_apply_data();

    auto trans = c_x->conj_transpose();
    auto dtrans = dc_x->conj_transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<ComplexMtx*>(dtrans.get()),
                        static_cast<ComplexMtx*>(trans.get()), 0);
}


TEST_F(Dense, IsConjugateTransposableIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto row_span = gko::span{0, c_x->get_size()[0] - 2};
    auto col_span = gko::span{0, c_x->get_size()[1] - 2};
    auto sub_x = c_x->create_submatrix(row_span, col_span);
    auto sub_dx = dc_x->create_submatrix(row_span, col_span);
    // create the target matrices on another executor to
    // force temporary clone
    auto trans = ComplexMtx::create(ref, gko::transpose(sub_x->get_size()));
    auto dtrans = ComplexMtx::create(ref, gko::transpose(sub_x->get_size()),
                                     sub_x->get_size()[0] + 4);

    sub_x->conj_transpose(trans.get());
    sub_dx->conj_transpose(dtrans.get());

    GKO_ASSERT_MTX_NEAR(dtrans, trans, 0);
}


}  // namespace
