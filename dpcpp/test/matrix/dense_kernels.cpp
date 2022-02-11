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


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/test/utils.hpp"
#include "dpcpp/test/utils.hpp"


namespace {


class Dense : public ::testing::Test {
protected:
    using itype = int;
#if GINKGO_DPCPP_SINGLE_MODE
    using vtype = float;
#else
    using vtype = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE
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
        ASSERT_GT(gko::DpcppExecutor::get_num_devices("all"), 0);
        ref = gko::ReferenceExecutor::create();
        dpcpp = gko::DpcppExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (dpcpp != nullptr) {
            ASSERT_NO_THROW(dpcpp->synchronize());
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
        dx = gko::clone(dpcpp, x);
        dy = gko::clone(dpcpp, y);
        dalpha = gko::clone(dpcpp, alpha);
        expected = Mtx::create(ref, gko::dim<2>{1, num_vecs});
        dresult = Mtx::create(dpcpp, gko::dim<2>{1, num_vecs});
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(65, 25);
        c_x = gen_mtx<ComplexMtx>(65, 25);
        y = gen_mtx<Mtx>(25, 35);
        expected = gen_mtx<Mtx>(65, 35);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-1.0}, ref);
        square = gen_mtx<Mtx>(x->get_size()[0], x->get_size()[0]);
        dx = gko::clone(dpcpp, x);
        dc_x = gko::clone(dpcpp, c_x);
        dy = gko::clone(dpcpp, y);
        dresult = gko::clone(dpcpp, expected);
        dalpha = gko::clone(dpcpp, alpha);
        dbeta = gko::clone(dpcpp, beta);
        dsquare = gko::clone(dpcpp, square);

        std::vector<itype> tmp(x->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<itype> tmp2(x->get_size()[1], 0);
        std::iota(tmp2.begin(), tmp2.end(), 0);
        std::shuffle(tmp2.begin(), tmp2.end(), rng);
        std::vector<itype> tmp3(x->get_size()[0] / 10);
        std::uniform_int_distribution<itype> row_dist(0, x->get_size()[0] - 1);
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
    std::shared_ptr<const gko::DpcppExecutor> dpcpp;

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


TEST_F(Dense, SingleVectorDpcppComputeDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorDpcppComputeDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Dense, SingleVectorDpcppComputeConjDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_conj_dot(y.get(), expected.get());
    dx->compute_conj_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorDpcppComputeConjDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_conj_dot(y.get(), expected.get());
    dx->compute_conj_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Dense, SingleVectorDpcppComputeNorm2IsEquivalentToRef)
{
    set_up_vector_data(1);
    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->dpcpp, norm_size);

    x->compute_norm2(norm_expected.get());
    dx->compute_norm2(dnorm.get());

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorDpcppComputeNorm2IsEquivalentToRef)
{
    set_up_vector_data(20);
    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->dpcpp, norm_size);

    x->compute_norm2(norm_expected.get());
    dx->compute_norm2(dnorm.get());

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, r<vtype>::value);
}


TEST_F(Dense, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(y.get(), expected.get());
    dx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Dense, SimpleApplyMixedIsEquivalentToRef)
{
    SKIP_IF_SINGLE_MODE;
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

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Dense, AdvancedApplyMixedIsEquivalentToRef)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data();

    x->apply(convert<MixedMtx>(alpha).get(), convert<MixedMtx>(y).get(),
             convert<MixedMtx>(beta).get(), convert<MixedMtx>(expected).get());
    dx->apply(convert<MixedMtx>(dalpha).get(), convert<MixedMtx>(dy).get(),
              convert<MixedMtx>(dbeta).get(), convert<MixedMtx>(dresult).get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-7);
}


TEST_F(Dense, ApplyToComplexIsEquivalentToRef)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(25, 1);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(65, 1);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);

    x->apply(complex_b.get(), complex_x.get());
    dx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<vtype>::value);
}


TEST_F(Dense, ApplyToMixedComplexIsEquivalentToRef)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data();
    auto complex_b = gen_mtx<MixedComplexMtx>(25, 1);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<MixedComplexMtx>(65, 1);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);

    x->apply(complex_b.get(), complex_x.get());
    dx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-7);
}


TEST_F(Dense, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(25, 1);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(65, 1);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);

    x->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dx->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<vtype>::value);
}


TEST_F(Dense, AdvancedApplyToMixedComplexIsEquivalentToRef)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data();
    auto complex_b = gen_mtx<MixedComplexMtx>(25, 1);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<MixedComplexMtx>(65, 1);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);

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
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);
    auto result = ComplexMtx::create(ref, gko::dim<2>{1, 2});
    auto dresult = ComplexMtx::create(dpcpp, gko::dim<2>{1, 2});

    complex_b->compute_dot(complex_x.get(), result.get());
    dcomplex_b->compute_dot(dcomplex_x.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(result, dresult, r<vtype>::value);
}


TEST_F(Dense, ComputeConjDotComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);
    auto result = ComplexMtx::create(ref, gko::dim<2>{1, 2});
    auto dresult = ComplexMtx::create(dpcpp, gko::dim<2>{1, 2});

    complex_b->compute_conj_dot(complex_x.get(), result.get());
    dcomplex_b->compute_conj_dot(dcomplex_x.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(result, dresult, r<vtype>::value);
}


TEST_F(Dense, IsTransposable)
{
    set_up_apply_data();

    auto trans = x->transpose();
    auto dtrans = dx->transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx*>(dtrans.get()),
                        static_cast<Mtx*>(trans.get()), 0);
}


TEST_F(Dense, IsConjugateTransposable)
{
    set_up_apply_data();

    auto trans = c_x->conj_transpose();
    auto dtrans = dc_x->conj_transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<ComplexMtx*>(dtrans.get()),
                        static_cast<ComplexMtx*>(trans.get()), 0);
}


TEST_F(Dense, ConvertToCooIsEquivalentToRef)
{
    set_up_apply_data();
    auto coo_mtx = gko::matrix::Coo<vtype>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<vtype>::create(dpcpp);

    x->convert_to(coo_mtx.get());
    dx->convert_to(dcoo_mtx.get());

    ASSERT_EQ(dcoo_mtx->get_num_stored_elements(),
              coo_mtx->get_num_stored_elements());
    GKO_ASSERT_MTX_NEAR(dcoo_mtx.get(), coo_mtx.get(), 0);
}


TEST_F(Dense, MoveToCooIsEquivalentToRef)
{
    set_up_apply_data();
    auto coo_mtx = gko::matrix::Coo<vtype>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<vtype>::create(dpcpp);

    x->move_to(coo_mtx.get());
    dx->move_to(dcoo_mtx.get());

    ASSERT_EQ(dcoo_mtx->get_num_stored_elements(),
              coo_mtx->get_num_stored_elements());
    GKO_ASSERT_MTX_NEAR(dcoo_mtx.get(), coo_mtx.get(), 0);
}


TEST_F(Dense, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data();
    auto csr_mtx = gko::matrix::Csr<vtype>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<vtype>::create(dpcpp);

    x->convert_to(csr_mtx.get());
    dx->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(dcsr_mtx.get(), csr_mtx.get(), 0);
}


TEST_F(Dense, MoveToCsrIsEquivalentToRef)
{
    set_up_apply_data();
    auto csr_mtx = gko::matrix::Csr<vtype>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<vtype>::create(dpcpp);

    x->move_to(csr_mtx.get());
    dx->move_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(dcsr_mtx.get(), csr_mtx.get(), 0);
}


TEST_F(Dense, ConvertToSparsityCsrIsEquivalentToRef)
{
    set_up_apply_data();
    auto sparsity_mtx = gko::matrix::SparsityCsr<vtype>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<vtype>::create(dpcpp);

    x->convert_to(sparsity_mtx.get());
    dx->convert_to(d_sparsity_mtx.get());

    GKO_ASSERT_MTX_NEAR(d_sparsity_mtx.get(), sparsity_mtx.get(), 0);
}


TEST_F(Dense, MoveToSparsityCsrIsEquivalentToRef)
{
    set_up_apply_data();
    auto sparsity_mtx = gko::matrix::SparsityCsr<vtype>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<vtype>::create(dpcpp);

    x->move_to(sparsity_mtx.get());
    dx->move_to(d_sparsity_mtx.get());

    GKO_ASSERT_MTX_NEAR(d_sparsity_mtx.get(), sparsity_mtx.get(), 0);
}


TEST_F(Dense, ConvertToEllIsEquivalentToRef)
{
    set_up_apply_data();
    auto ell_mtx = gko::matrix::Ell<vtype>::create(ref);
    auto dell_mtx = gko::matrix::Ell<vtype>::create(dpcpp);

    x->convert_to(ell_mtx.get());
    dx->convert_to(dell_mtx.get());

    GKO_ASSERT_MTX_NEAR(dell_mtx.get(), ell_mtx.get(), 0);
}


TEST_F(Dense, MoveToEllIsEquivalentToRef)
{
    set_up_apply_data();
    auto ell_mtx = gko::matrix::Ell<vtype>::create(ref);
    auto dell_mtx = gko::matrix::Ell<vtype>::create(dpcpp);

    x->move_to(ell_mtx.get());
    dx->move_to(dell_mtx.get());

    GKO_ASSERT_MTX_NEAR(dell_mtx.get(), ell_mtx.get(), 0);
}


TEST_F(Dense, ConvertToSellpIsEquivalentToRef)
{
    set_up_apply_data();
    auto sellp_mtx = gko::matrix::Sellp<vtype>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<vtype>::create(dpcpp);

    x->convert_to(sellp_mtx.get());
    dx->convert_to(dsellp_mtx.get());

    GKO_ASSERT_MTX_NEAR(sellp_mtx, dsellp_mtx, 0);
}


TEST_F(Dense, MoveToSellpIsEquivalentToRef)
{
    set_up_apply_data();
    auto sellp_mtx = gko::matrix::Sellp<vtype>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<vtype>::create(dpcpp);

    x->move_to(sellp_mtx.get());
    dx->move_to(dsellp_mtx.get());

    GKO_ASSERT_MTX_NEAR(sellp_mtx, dsellp_mtx, 0);
}


TEST_F(Dense, ConvertsEmptyToSellp)
{
    auto dempty_mtx = Mtx::create(dpcpp);
    auto dsellp_mtx = gko::matrix::Sellp<vtype>::create(dpcpp);

    dempty_mtx->convert_to(dsellp_mtx.get());

    ASSERT_EQ(dpcpp->copy_val_to_host(dsellp_mtx->get_const_slice_sets()), 0);
    ASSERT_FALSE(dsellp_mtx->get_size());
}


TEST_F(Dense, CalculateNNZPerRowIsEquivalentToRef)
{
    set_up_apply_data();
    gko::Array<gko::size_type> nnz_per_row(ref);
    nnz_per_row.resize_and_reset(x->get_size()[0]);
    gko::Array<gko::size_type> dnnz_per_row(dpcpp);
    dnnz_per_row.resize_and_reset(dx->get_size()[0]);

    gko::kernels::reference::dense::count_nonzeros_per_row(
        ref, x.get(), nnz_per_row.get_data());
    gko::kernels::dpcpp::dense::count_nonzeros_per_row(dpcpp, dx.get(),
                                                       dnnz_per_row.get_data());

    auto tmp = gko::Array<gko::size_type>(ref, dnnz_per_row);
    for (gko::size_type i = 0; i < nnz_per_row.get_num_elems(); i++) {
        ASSERT_EQ(nnz_per_row.get_const_data()[i], tmp.get_const_data()[i]);
    }
}


TEST_F(Dense, ComputeMaxNNZPerRowIsEquivalentToRef)
{
    set_up_apply_data();
    gko::size_type max_nnz;
    gko::size_type dmax_nnz;

    gko::kernels::reference::dense::compute_max_nnz_per_row(ref, x.get(),
                                                            max_nnz);
    gko::kernels::dpcpp::dense::compute_max_nnz_per_row(dpcpp, dx.get(),
                                                        dmax_nnz);

    ASSERT_EQ(max_nnz, dmax_nnz);
}


}  // namespace
